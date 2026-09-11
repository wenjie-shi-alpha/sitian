#!/usr/bin/env python3
"""Derive city and regional source-composition evidence from open CAMS fields."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio

from sitian.open_evidence import (DEFAULT_EVIDENCE_ROOT, EVIDENCE_VERSION,
                                  available_at, issue_asof)

UTC = timezone.utc
AEROSOL_NAMES = {
    "total": "aod550",
    "fine": "aodfm550",
    "dust": "duaod550",
    "organic_matter": "omaod550",
    "black_carbon": "bcaod550",
    "sulphate": "suaod550",
    "nitrate": "niaod550",
}
GAS_NAMES = {"carbon_monoxide": "co", "nitrogen_dioxide": "no2", "sulphur_dioxide": "so2"}
MOLAR_MASS_G_MOL = {"carbon_monoxide": 28.0101, "nitrogen_dioxide": 46.0055,
                    "sulphur_dioxide": 64.066}
AIR_MOLAR_MASS_G_MOL = 28.9647
FILE_RE = re.compile(r"^(aerosol|trace_gases)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.nc$")
COLUMN_SCHEMA_VERSION = "cams-earthengine-columns-v1"


def _open_netcdf(path: Path):
    """Keep the NetCDF C extension out of Earth-Engine-only code paths/tests."""
    from netCDF4 import Dataset

    return Dataset(path)
COLUMN_GAS_NAMES = ("carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide")


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value, digits: int = 3):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _find_cycle_file(root: Path, kind: str, cycle: datetime) -> tuple[Path, int] | None:
    for path in sorted((root / "raw" / "cams").glob(f"{kind}_*.nc")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        d0, d1 = (datetime.fromisoformat(value).date() for value in match.groups()[1:])
        if not (d0 <= cycle.date() <= d1):
            continue
        with _open_netcdf(path) as dataset:
            references = np.asarray(dataset.variables["forecast_reference_time"][:], dtype=float)
            expected = cycle.timestamp()
            indexes = np.where(np.isclose(references, expected, atol=1))[0]
            if len(indexes):
                return path, int(indexes[0])
    return None


def _sidecar(path: Path, root: Path) -> dict:
    meta_path = path.with_suffix(path.suffix + ".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return {"relative_path": str(path.relative_to(root)), "sha256": meta.get("sha256") or _sha256(path),
            "bytes": path.stat().st_size}


def _nearest_indexes(latitudes: np.ndarray, longitudes: np.ndarray, coords: dict) -> dict:
    output = {}
    for city, location in coords.items():
        if isinstance(location, list) and len(location) == 2:
            lat, lon = map(float, location)
            output[city] = (int(np.argmin(np.abs(latitudes - lat))),
                            int(np.argmin(np.abs(longitudes - lon))))
    return output


def _grid(variable, period_index: int, reference_index: int) -> np.ndarray:
    dimensions = variable.dimensions
    selectors = []
    for name in dimensions:
        if name == "forecast_period":
            selectors.append(period_index)
        elif name == "forecast_reference_time":
            selectors.append(reference_index)
        elif name == "model_level":
            selectors.append(0)
        elif name in ("latitude", "longitude"):
            selectors.append(slice(None))
        else:
            raise RuntimeError(f"unsupported CAMS dimension {name!r} on {variable.name}")
    return np.asarray(np.ma.filled(variable[tuple(selectors)], np.nan), dtype=float)


def _regional_centers(values: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray,
                      count: int = 4) -> list[dict]:
    """Rank 2-degree block means, keeping spatially separated source signals."""
    lat_step = max(1, int(round(2.0 / max(abs(float(latitudes[1] - latitudes[0])), 0.01))))
    lon_step = max(1, int(round(2.0 / max(abs(float(longitudes[1] - longitudes[0])), 0.01))))
    cells = []
    for i in range(0, len(latitudes), lat_step):
        for j in range(0, len(longitudes), lon_step):
            block = values[i:i + lat_step, j:j + lon_step]
            if np.isfinite(block).any():
                cells.append((float(np.nanmean(block)), float(np.nanmean(latitudes[i:i + lat_step])),
                              float(np.nanmean(longitudes[j:j + lon_step]))))
    cells.sort(reverse=True)
    chosen = []
    for value, lat, lon in cells:
        if any(math.hypot(lat - row["lat"],
                          (lon - row["lon"]) * math.cos(math.radians(lat))) < 5.0
               for row in chosen):
            continue
        chosen.append({"lat": round(lat, 2), "lon": round(lon, 2), "aod550": round(value, 3)})
        if len(chosen) == count:
            break
    return chosen


def _aerosol(root: Path, cycle: datetime, coords: dict) -> dict:
    located = _find_cycle_file(root, "aerosol", cycle)
    if not located:
        return {"available": False, "reason": "raw_cams_aerosol_missing"}
    path, reference_index = located
    with _open_netcdf(path) as dataset:
        latitudes = np.asarray(dataset.variables["latitude"][:], dtype=float)
        longitudes = np.asarray(dataset.variables["longitude"][:], dtype=float)
        city_indexes = _nearest_indexes(latitudes, longitudes, coords)
        periods = np.asarray(dataset.variables["forecast_period"][:], dtype=float)
        records = []
        for period_index, lead in enumerate(periods):
            fields = {name: _grid(dataset.variables[short], period_index, reference_index)
                      for name, short in AEROSOL_NAMES.items()}
            valid = cycle + timedelta(hours=float(lead))
            cities = {}
            for city, (i, j) in city_indexes.items():
                values = {name: float(field[i, j]) for name, field in fields.items()}
                total = values["total"]
                components = {name: values[name] for name in
                              ("dust", "organic_matter", "black_carbon", "sulphate", "nitrate")}
                cities[city] = {
                    "aod550": {name: _finite(value) for name, value in values.items()},
                    "fraction_of_total": {name: _finite(value / total, 3) if total > 1e-6 else None
                                          for name, value in components.items()},
                    "fine_fraction": _finite(values["fine"] / total, 3) if total > 1e-6 else None,
                    "dominant_component": max(components, key=components.get),
                }
            records.append({
                "lead_hour": int(round(float(lead))), "valid_time": _iso(valid),
                "issue_relative_hour": int(round(float(lead))) - 12,
                "available_at": _iso(available_at("cams", cycle)),
                "cities": cities,
                "regional_source_signals": {name: _regional_centers(fields[name], latitudes, longitudes)
                                            for name in ("dust", "organic_matter", "black_carbon",
                                                         "sulphate", "nitrate")},
            })
    return {"available": True, "cycle": _iso(cycle), "records": records,
            "raw": _sidecar(path, root)}


def _trace_gases(root: Path, cycle: datetime, coords: dict) -> dict:
    located = _find_cycle_file(root, "trace_gases", cycle)
    if not located:
        return {"available": False, "reason": "raw_cams_trace_gases_missing"}
    path, reference_index = located
    with _open_netcdf(path) as dataset:
        latitudes = np.asarray(dataset.variables["latitude"][:], dtype=float)
        longitudes = np.asarray(dataset.variables["longitude"][:], dtype=float)
        city_indexes = _nearest_indexes(latitudes, longitudes, coords)
        periods = np.asarray(dataset.variables["forecast_period"][:], dtype=float)
        records = []
        for period_index, lead in enumerate(periods):
            fields = {name: _grid(dataset.variables[short], period_index, reference_index)
                      for name, short in GAS_NAMES.items()}
            factors = {name: AIR_MOLAR_MASS_G_MOL / MOLAR_MASS_G_MOL[name] * 1e9
                       for name in GAS_NAMES}
            cities = {city: {name + "_ppbv_approx": _finite(fields[name][i, j] * factors[name], 2)
                             for name in GAS_NAMES}
                      for city, (i, j) in city_indexes.items()}
            record = {
                "lead_hour": int(round(float(lead))),
                "issue_relative_hour": int(round(float(lead))) - 12,
                "valid_time": _iso(cycle + timedelta(hours=float(lead))),
                "available_at": _iso(available_at("cams", cycle)),
                "cities": cities,
            }
            if int(round(float(lead))) % 24 == 12:
                record["regional_summary"] = {
                    name: {"p90_ppbv_approx": _finite(np.nanpercentile(field * factors[name], 90), 2),
                           "max_ppbv_approx": _finite(np.nanmax(field * factors[name]), 2)}
                    for name, field in fields.items()
                }
            records.append(record)
    return {"available": True, "cycle": _iso(cycle),
            "conversion_note": "Mass mixing ratio converted to approximate dry-air ppbv using molar masses.",
            "records": records, "raw": _sidecar(path, root)}


def _column_gases(root: Path, cycle: datetime, coords: dict) -> dict:
    """Derive total-column gas evidence without conflating it with model level 137."""
    stamp = cycle.strftime("%Y%m%dT%H%M%S")
    path = root / "raw" / "cams_earthengine" / f"cams_column_trajectory_{stamp}.tif"
    metadata_path = Path(str(path) + ".json")
    if not path.is_file() or not metadata_path.is_file():
        return {"available": False, "reason": "raw_cams_earthengine_columns_missing"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != COLUMN_SCHEMA_VERSION or
            metadata.get("forecast_reference_time") != _iso(cycle) or
            metadata.get("semantics") != "total_column_mass"):
        return {"available": False, "reason": "raw_cams_earthengine_columns_schema_mismatch"}
    try:
        dataset = rasterio.open(path)
    except rasterio.errors.RasterioIOError:
        return {"available": False, "reason": "raw_cams_earthengine_columns_unreadable"}
    transform = dataset.transform.to_gdal()
    longitudes = transform[0] + (np.arange(dataset.width) + 0.5) * transform[1]
    latitudes = transform[3] + (np.arange(dataset.height) + 0.5) * transform[5]
    city_indexes = _nearest_indexes(latitudes, longitudes, coords)
    band_indexes = {name: index + 1 for index, name in enumerate(metadata["bands"])}
    records = []
    try:
        for lead in metadata["leadtime_hour"]:
            fields = {}
            for name in COLUMN_GAS_NAMES:
                band_name = f"fh{int(lead):03d}_{name}_column"
                index = band_indexes.get(band_name)
                if index is None:
                    raise RuntimeError(f"missing Earth Engine CAMS band {band_name} in {path}")
                fields[name] = dataset.read(index).astype(float)
            cities = {
                city: {name + "_kg_m2": _finite(fields[name][i, j], 9)
                       for name in COLUMN_GAS_NAMES}
                for city, (i, j) in city_indexes.items()
            }
            record = {
                "lead_hour": int(lead),
                "issue_relative_hour": int(lead) - 12,
                "valid_time": _iso(cycle + timedelta(hours=int(lead))),
                "available_at": metadata["available_at"],
                "cities": cities,
            }
            if int(lead) % 24 == 0:
                record["regional_summary"] = {
                    name: {"p90_kg_m2": _finite(np.nanpercentile(field, 90), 9),
                           "max_kg_m2": _finite(np.nanmax(field), 9)}
                    for name, field in fields.items()
                }
            records.append(record)
    finally:
        dataset.close()
    return {
        "available": True,
        "cycle": _iso(cycle),
        "semantics": "total_column_mass",
        "unit": "kg m-2",
        "non_substitution_note": (
            "This total-column trajectory is independent of CAMS model-level-137 "
            "near-surface mixing ratios."
        ),
        "records": records,
        "raw": _sidecar(path, root),
    }


def _complete_existing(path: Path, issue_date: str) -> bool:
    """True only when a previously derived file exists for exactly this issue date."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (document.get("issue_date") == issue_date
            and document.get("contract_version") == EVIDENCE_VERSION
            and isinstance(document.get("pollution", {}).get("composition"), dict))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=DEFAULT_EVIDENCE_ROOT / "manifests" / "open_evidence_v1.json")
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--coords", type=Path, default=Path("data/interim/city_coords.json"))
    parser.add_argument("--issue-date")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip issue dates whose derived file already exists for this issue date")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    coords = json.loads(args.coords.read_text(encoding="utf-8"))
    dates = manifest["issue_dates"]
    if args.issue_date:
        dates = [value for value in dates if value == args.issue_date]
    if args.limit is not None:
        dates = dates[:args.limit]
    for number, issue_date in enumerate(dates, 1):
        target = args.root / "derived" / "by_issue" / f"{issue_date}.composition.json"
        if args.skip_existing and _complete_existing(target, issue_date):
            print(f"[{number}/{len(dates)}] {issue_date}: complete existing file -> skip", flush=True)
            continue
        cycle = issue_asof(issue_date) - timedelta(hours=12)
        aerosol = _aerosol(args.root, cycle, coords)
        trace_gases = _trace_gases(args.root, cycle, coords)
        column_gases = _column_gases(args.root, cycle, coords)
        document = {"contract_version": EVIDENCE_VERSION, "issue_date": issue_date,
                    "pollution": {"composition": {"aerosol": aerosol,
                                                   "trace_gases": trace_gases,
                                                   "column_gases": column_gases}}}
        target = args.root / "derived" / "by_issue" / f"{issue_date}.composition.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
                          encoding="utf-8")
        print(f"[{number}/{len(dates)}] {issue_date}: aerosol={aerosol['available']} "
              f"trace_gases={trace_gases['available']} "
              f"column_gases={column_gases['available']} -> {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
