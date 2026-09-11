#!/usr/bin/env python3
"""Derive deterministic city/system synoptic evidence from open NWP GRIBs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from sitian.open_evidence import (
    DEFAULT_EVIDENCE_ROOT,
    EVIDENCE_VERSION,
    NWP_SNAPSHOTS_PER_ISSUE,
    NWP_TEMPORAL_PROFILE_VERSION,
    parse_utc,
)


def _round(value, digits=1):
    return round(float(value), digits) if np.isfinite(value) else None


def _wind(u: float, v: float) -> dict:
    speed = math.hypot(u, v)
    direction = (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0
    return {"speed_ms": round(speed, 1), "from_deg": round(direction),
            "u_ms": round(u, 1), "v_ms": round(v, 1)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GribFields:
    def __init__(self, path: Path):
        import pygrib

        handle = pygrib.open(str(path))
        try:
            messages = list(handle)
        finally:
            handle.close()
        if not messages:
            raise RuntimeError(f"empty GRIB: {path}")
        self.variables: dict[tuple[str, int | None], np.ndarray] = {}
        self.metadata: dict[tuple[str, int | None], dict] = {}
        self.point_indexes = {}
        shapes = set()
        for message in messages:
            name = "msl" if message.shortName == "prmsl" else message.shortName
            level = (int(message.level)
                     if message.typeOfLevel == "isobaricInhPa" else None)
            values = np.asarray(message.values)
            self.variables[(name, level)] = values
            self.metadata[(name, level)] = {
                "units": str(getattr(message, "units", "")),
                "step_type": str(getattr(message, "stepType", "")),
                "start_step": int(getattr(message, "startStep", 0)),
                "end_step": int(getattr(message, "endStep", 0)),
            }
            shapes.add(values.shape)
        latitudes, longitudes = messages[0].latlons()
        self.latitudes = np.asarray(latitudes[:, 0], dtype=float)
        self.longitudes = np.asarray(longitudes[0, :], dtype=float)
        if shapes != {(len(self.latitudes), len(self.longitudes))}:
            raise RuntimeError(f"mixed or unexpected GRIB grids in {path}: {sorted(shapes)}")

    def close(self):
        self.variables.clear()
        self.metadata.clear()

    def layer(self, name: str, level: int | None = None):
        return self.variables[(name, level)]

    def point(self, name: str, level: int | None, lat: float, lon: float) -> float:
        array = self.layer(name, level)
        lon_value = lon % 360 if float(self.longitudes.max()) > 180 else lon
        # Every field in one compact GRIB uses the same horizontal grid.  Cache
        # city indexes instead of repeating two nearest-neighbour searches for
        # every variable and pressure level.
        key = (float(lat), float(lon_value))
        indexes = self.point_indexes.get(key)
        if indexes is None:
            indexes = (int(np.abs(self.latitudes - lat).argmin()),
                       int(np.abs(self.longitudes - lon_value).argmin()))
            self.point_indexes[key] = indexes
        return float(array[indexes])

    def region(self, name: str, level: int | None, domain: dict):
        array = self.layer(name, level)
        lat, lon = self.latitudes, self.longitudes
        lon_min = domain["west"] % 360 if lon.max() > 180 else domain["west"]
        lon_max = domain["east"] % 360 if lon.max() > 180 else domain["east"]
        lat_indexes = np.where((lat >= domain["south"]) & (lat <= domain["north"]))[0]
        lon_indexes = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        return array[np.ix_(lat_indexes, lon_indexes)], lat[lat_indexes], lon[lon_indexes]


def _separated_extrema(values: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                       *, largest: bool, count: int = 3, separation_deg: float = 8.0) -> list[dict]:
    stride = max(1, int(round(2.0 / max(abs(float(lats[1] - lats[0])), 0.01)))) if len(lats) > 1 else 1
    sample = values[::stride, ::stride]
    sample_lats, sample_lons = lats[::stride], lons[::stride]
    flat = sample.ravel()
    order = np.argsort(flat)
    if largest:
        order = order[::-1]
    chosen = []
    for flat_index in order:
        value = float(flat[flat_index])
        if not np.isfinite(value):
            continue
        i, j = np.unravel_index(flat_index, sample.shape)
        lat, lon = float(sample_lats[i]), float(sample_lons[j])
        if any(math.hypot(lat - row["lat"], (lon - row["lon"]) * math.cos(math.radians(lat)))
               < separation_deg for row in chosen):
            continue
        chosen.append({"lat": round(lat, 2), "lon": round(lon, 2), "value": value})
        if len(chosen) == count:
            break
    return chosen


def _systems(fields: GribFields, domain: dict) -> dict:
    msl, lats, lons = fields.region("msl", None, domain)
    msl = msl / 100.0
    z500, zlats, zlons = fields.region("gh", 500, domain)
    # Remove the large meridional climatological slope before locating trough/ridge signals.
    z_anomaly = z500 - np.nanmean(z500, axis=1, keepdims=True)
    lows = _separated_extrema(msl, lats, lons, largest=False)
    highs = _separated_extrema(msl, lats, lons, largest=True)
    troughs = _separated_extrema(z_anomaly, zlats, zlons, largest=False)
    ridges = _separated_extrema(z_anomaly, zlats, zlons, largest=True)
    for rows, key in ((lows, "mslp_hpa"), (highs, "mslp_hpa"),
                      (troughs, "z500_zonal_anomaly_gpm"),
                      (ridges, "z500_zonal_anomaly_gpm")):
        for row in rows:
            row[key] = round(row.pop("value"), 1)
    return {"low_centers": lows, "high_centers": highs,
            "trough_signals": troughs, "ridge_signals": ridges}


def _city_features(fields: GribFields, coords: dict) -> dict:
    out = {}
    for city, location in coords.items():
        if not isinstance(location, list) or len(location) != 2:
            continue
        lat, lon = map(float, location)
        winds = {}
        for level in (925, 850, 700, 500, 200):
            winds[str(level)] = _wind(fields.point("u", level, lat, lon),
                                      fields.point("v", level, lat, lon))
        temperatures = {str(level): _round(fields.point("t", level, lat, lon) - 273.15)
                        for level in (925, 850, 700)}
        humidity = {str(level): _round(fields.point("r", level, lat, lon))
                    for level in (925, 850, 700)}
        omega = {str(level): _round(fields.point("w", level, lat, lon), 3)
                 for level in (850, 700, 500)}
        out[city] = {
            "mslp_hpa": _round(fields.point("msl", None, lat, lon) / 100.0),
            "z500_gpm": _round(fields.point("gh", 500, lat, lon)),
            "wind": winds,
            "temperature_c": temperatures,
            "relative_humidity_pct": humidity,
            "omega_pa_s": omega,
            "stability": {
                "t925_minus_t850_c": _round(temperatures["925"] - temperatures["850"]),
                "t850_minus_t700_c": _round(temperatures["850"] - temperatures["700"]),
                "low_level_inversion_signal": temperatures["925"] <= temperatures["850"],
            },
        }
    return out


def _raw_path(root: Path, source: str, record: dict) -> Path:
    cycle = parse_utc(record["cycle"])
    return (root / "raw" / "nwp" / source / f"{cycle:%Y%m%d%H}" /
            f"{source}_{cycle:%Y%m%d%H}_f{int(record['step_hour']):03d}.grib2")


def _radiation_raw_path(root: Path, source: str, record: dict) -> Path:
    cycle = parse_utc(record["cycle"])
    return (root / "raw" / "nwp_radiation" / source / f"{cycle:%Y%m%d%H}" /
            f"{source}_{cycle:%Y%m%d%H}_f{int(record['step_hour']):03d}.radiation.grib2")


def _radiation_features(path: Path, source: str, coords: dict) -> tuple[dict, dict]:
    fields = GribFields(path)
    try:
        # ecCodes exposes the GFS DSWRF field as sdswrf, while IFS retains ssrd.
        name = "sdswrf" if source == "gfs" else "ssrd"
        metadata = dict(fields.metadata[(name, None)])
        cities = {}
        for city, location in coords.items():
            if isinstance(location, list) and len(location) == 2:
                cities[city] = fields.point(name, None, *map(float, location))
    finally:
        fields.close()
    return cities, metadata


def _normalize_radiation(records: list[dict], source: str) -> None:
    """Normalize source-specific radiation to preceding-period mean W m-2."""
    previous_step = 0
    previous_energy: dict[str, float] = {}
    for row in records:
        if not row.get("radiation_available"):
            continue
        step = int(row["step_hour"])
        raw = row.pop("_radiation_city_raw")
        metadata = row.pop("_radiation_metadata")
        start_step = int(metadata.get("start_step", 0))
        end_step = int(metadata.get("end_step", step))
        if source == "gfs":
            duration_hours = max(1, end_step - start_step)
            normalized = raw
        else:
            duration_hours = max(1, step - previous_step)
            normalized = {
                city: max(0.0, (float(value) - previous_energy.get(city, 0.0)) /
                          (duration_hours * 3600.0))
                for city, value in raw.items()
            }
            previous_step = step
            previous_energy = {city: float(value) for city, value in raw.items()}
        for city, value in normalized.items():
            if city in row.get("cities", {}):
                row["cities"][city]["surface_downward_shortwave_w_m2"] = _round(value)
        row["radiation_period_hours"] = duration_hours
        row["radiation_source_units"] = metadata.get("units")


def _derive_radiation_record(root: Path, source: str, record: dict, coords: dict) -> dict:
    path = _radiation_raw_path(root, source, record)
    base = {key: record[key] for key in
            ("role", "cycle", "step_hour", "valid_time", "available_at")}
    if not path.is_file():
        return {**base, "radiation_available": False,
                "radiation_reason": "raw_nwp_radiation_missing"}
    sidecar = path.with_suffix(path.suffix + ".json")
    raw_meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    raw_cities, field_meta = _radiation_features(path, source, coords)
    digest = raw_meta.get("sha256") or _sha256(path)
    return {
        **base,
        "radiation_available": True,
        "cities": {city: {} for city in raw_cities},
        "_radiation_city_raw": raw_cities,
        "_radiation_metadata": field_meta,
        "radiation_raw": {
            "sha256": digest,
            "bytes": path.stat().st_size,
            "relative_path": str(path.relative_to(root)),
        },
    }


def _attach_radiation(synoptic_rows: list[dict], radiation_rows: list[dict]) -> None:
    by_time = {row["valid_time"]: row for row in radiation_rows}
    for row in synoptic_rows:
        if int(row.get("step_hour", 0)) == 0:
            continue
        radiation = by_time.get(row["valid_time"])
        if not radiation or not radiation.get("radiation_available"):
            row.update({"radiation_available": False,
                        "radiation_reason": "normalized_nwp_radiation_missing"})
            continue
        for city, values in radiation.get("cities", {}).items():
            flux = values.get("surface_downward_shortwave_w_m2")
            if city in row.get("cities", {}) and flux is not None:
                row["cities"][city]["surface_downward_shortwave_w_m2"] = flux
        row.update({
            "radiation_available": True,
            "radiation_period_hours": radiation.get("radiation_period_hours"),
            "radiation_source_units": radiation.get("radiation_source_units"),
            "radiation_raw": radiation.get("radiation_raw"),
        })


def _derive_record(root: Path, source: str, record: dict, coords: dict, domain: dict) -> dict:
    path = _raw_path(root, source, record)
    base = {key: record[key] for key in
            ("role", "cycle", "step_hour", "valid_time", "available_at")}
    if not path.is_file():
        return {**base, "available": False, "reason": "raw_nwp_missing"}
    sidecar = path.with_suffix(path.suffix + ".json")
    raw_meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    fields = GribFields(path)
    try:
        cities = _city_features(fields, coords)
        systems = _systems(fields, domain)
    finally:
        fields.close()
    digest = raw_meta.get("sha256") or _sha256(path)
    return {
        **base,
        "available": True,
        "evidence_id": f"synoptic:{source}:{digest[:16]}",
        "systems": systems,
        "cities": cities,
        "raw": {"sha256": digest, "bytes": path.stat().st_size,
                "relative_path": str(path.relative_to(root))},
    }


def _cross_model(sources: dict, cities: list[str]) -> dict:
    by_source = {source: {row["valid_time"]: row for row in rows if row.get("available")}
                 for source, rows in sources.items()}
    times = sorted(set(by_source.get("gfs", {})) & set(by_source.get("ifs", {})))
    out = {}
    for valid_time in times:
        city_rows = {}
        for city in cities:
            gfs = by_source["gfs"][valid_time]["cities"].get(city)
            ifs = by_source["ifs"][valid_time]["cities"].get(city)
            if not gfs or not ifs:
                continue
            gu, gv = gfs["wind"]["925"]["u_ms"], gfs["wind"]["925"]["v_ms"]
            iu, iv = ifs["wind"]["925"]["u_ms"], ifs["wind"]["925"]["v_ms"]
            city_rows[city] = {
                "mslp_abs_diff_hpa": round(abs(gfs["mslp_hpa"] - ifs["mslp_hpa"]), 1),
                "z500_abs_diff_gpm": round(abs(gfs["z500_gpm"] - ifs["z500_gpm"]), 1),
                "wind925_vector_diff_ms": round(math.hypot(gu - iu, gv - iv), 1),
                "shortwave_abs_diff_w_m2": round(abs(
                    float(gfs.get("surface_downward_shortwave_w_m2", float("nan"))) -
                    float(ifs.get("surface_downward_shortwave_w_m2", float("nan")))
                ), 1) if (gfs.get("surface_downward_shortwave_w_m2") is not None and
                           ifs.get("surface_downward_shortwave_w_m2") is not None) else None,
            }
        out[valid_time] = city_rows
    return out


def _complete_existing(path: Path, issue_date: str) -> bool:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if document.get("issue_date") != issue_date:
        return False
    sources = document.get("synoptic", {}).get("sources", {})
    return all(len(sources.get(source, [])) == NWP_SNAPSHOTS_PER_ISSUE and
               all(row.get("available") for row in sources[source]) and
               all(int(row.get("step_hour", 0)) == 0 or row.get("radiation_available")
                   for row in sources[source])
               for source in ("gfs", "ifs"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=DEFAULT_EVIDENCE_ROOT / "manifests" / "open_evidence_v1.json")
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--coords", type=Path, default=Path("data/interim/city_coords.json"))
    parser.add_argument("--issue-date")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    coords = json.loads(args.coords.read_text(encoding="utf-8"))
    cities = sorted(city for city, value in coords.items() if isinstance(value, list))
    dates = manifest["issue_dates"]
    if args.issue_date:
        dates = [value for value in dates if value == args.issue_date]
    if args.limit is not None:
        dates = dates[:args.limit]
    # Contiguous ranges retain mostly sequential disk access on archival HDDs.
    start = len(dates) * args.shard_index // args.num_shards
    stop = len(dates) * (args.shard_index + 1) // args.num_shards
    dates = dates[start:stop]
    for number, issue_date in enumerate(dates, 1):
        target = args.root / "derived" / "by_issue" / f"{issue_date}.synoptic.json"
        if args.skip_existing and _complete_existing(target, issue_date):
            print(f"[{number}/{len(dates)}] {issue_date}: complete existing file -> skip", flush=True)
            continue
        sources = {}
        for source in ("gfs", "ifs"):
            records = [row for row in manifest["nwp"][source] if row["issue_date"] == issue_date]
            sources[source] = [_derive_record(args.root, source, row, coords, manifest["domain"])
                               for row in records]
            radiation_records = [
                row for row in manifest["nwp_radiation"]["records"][source]
                if row["issue_date"] == issue_date
            ]
            radiation = [
                _derive_radiation_record(args.root, source, row, coords)
                for row in radiation_records
            ]
            _normalize_radiation(radiation, source)
            _attach_radiation(sources[source], radiation)
        document = {
            "contract_version": EVIDENCE_VERSION,
            "issue_date": issue_date,
            "synoptic": {
                "temporal_profile": manifest.get("nwp_temporal_profile", {
                    "version": NWP_TEMPORAL_PROFILE_VERSION,
                }),
                "sources": sources,
                "cross_model": _cross_model(sources, cities),
            },
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
                          encoding="utf-8")
        available = sum(row.get("available", False) for rows in sources.values() for row in rows)
        expected = 2 * NWP_SNAPSHOTS_PER_ISSUE
        print(f"[{number}/{len(dates)}] {issue_date}: {available}/{expected} raw snapshots -> {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
