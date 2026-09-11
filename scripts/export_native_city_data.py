#!/usr/bin/env python3
"""Run on the raw-data host: export unaggregated city series, never raw grids.

Requires numpy/netCDF4 for CAMS. CSV extraction uses the standard library.
Outputs gzip JSONL plus hashes, file metadata, and per-case failure reasons.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def clean(value):
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def extract_netcdf_point(dataset, variable, cycle, lat, lon):
    """Select by named dimensions, preserving all remaining vertical dimensions."""
    import netCDF4
    import numpy as np

    def pick(names):
        return next((dataset.variables[n] for n in names if n in dataset.variables), None)

    y = pick(("latitude", "lat"))
    x = pick(("longitude", "lon"))
    ref = pick(("forecast_reference_time", "time"))
    lead = pick(("forecast_period", "leadtime_hour", "step"))
    if any(v is None for v in (x, y, ref, lead)) or any(v.ndim != 1 for v in (x, y, ref, lead)):
        raise ValueError("requires explicit 1D lat/lon/reference_time/forecast_period axes")
    if cycle.tzinfo is None or cycle.utcoffset() != timedelta(0):
        raise ValueError("reference cycle must use UTC")
    if len({v.dimensions[0] for v in (x, y, ref, lead)}) != 4:
        raise ValueError("forecast axes must have distinct named dimensions")
    for axis in (x, y, ref, lead):
        values = np.ma.asarray(axis[:], dtype=float)
        if values.size == 0 or np.ma.getmaskarray(values).any() or not np.isfinite(values).all():
            raise ValueError(f"invalid coordinate axis: {axis.name}")
    if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90:
        raise ValueError("invalid city coordinates")
    if getattr(ref, "calendar", "standard") not in {"standard", "gregorian", "proleptic_gregorian"}:
        raise ValueError("unsupported reference calendar")
    references = netCDF4.num2date(ref[:], ref.units, calendar=getattr(ref, "calendar", "standard"))
    indices = [i for i, t in enumerate(references) if
               (t.year, t.month, t.day, t.hour, t.minute, t.second) ==
               (cycle.year, cycle.month, cycle.day, cycle.hour, cycle.minute, cycle.second)]
    if len(indices) != 1:
        raise ValueError("requested reference cycle missing or duplicated")
    lead_unit = getattr(lead, "units", "").lower()
    factors = {"hours": 1, "hour": 1, "h": 1, "seconds": 1/3600, "second": 1/3600,
               "s": 1/3600, "days": 24, "day": 24}
    if lead_unit not in factors:
        raise ValueError(f"unsupported lead units {lead_unit!r}; do not assume hours")
    hours = np.asarray(lead[:], dtype=float) * factors[lead_unit]
    if not np.isfinite(hours).all() or len(set(hours)) != len(hours) or np.any(np.diff(hours) <= 0):
        raise ValueError("invalid, unordered or duplicate forecast leads")
    li = int(np.abs(np.asarray(y[:]) - lat).argmin())
    lj = int(np.abs((np.asarray(x[:]) - lon + 180) % 360 - 180).argmin())
    fixed = {y.dimensions[0]: li, x.dimensions[0]: lj, ref.dimensions[0]: indices[0]}
    if not {v.dimensions[0] for v in (x, y, ref, lead)}.issubset(variable.dimensions):
        raise ValueError("field is not a forecast for the requested reference/lead axes")
    selection = tuple(fixed.get(dim, slice(None)) for dim in variable.dimensions)
    values = np.ma.asarray(variable[selection], dtype=float).filled(np.nan)
    dimensions = [dim for dim in variable.dimensions if dim not in fixed]
    coordinates = {}
    for dim in dimensions:
        coordinate = dataset.variables.get(dim)
        coordinates[dim] = {"values": clean(np.ma.asarray(coordinate[:], dtype=float).filled(np.nan).tolist()),
                            "unit": getattr(coordinate, "units", None)} if coordinate is not None else None
    return {"variable": variable.name, "unit": getattr(variable, "units", None),
            "standard_name": getattr(variable, "standard_name", None),
            "long_name": getattr(variable, "long_name", None),
            "grib_step_type": getattr(variable, "GRIB_stepType", None),
            "cell_methods": getattr(variable, "cell_methods", None),
            "selected_grid": {"lat": float(y[li]), "lon": float(x[lj])},
            "cycle": cycle.isoformat(), "available_at": None,
            "availability_note": "publication not inferred from cycle or file mtime; consult source sidecars",
            "lead_hours": hours.tolist(),
            "valid_times": [(cycle + timedelta(hours=float(h))).isoformat() for h in hours],
            "dimensions": dimensions, "lead_dimension": lead.dimensions[0], "coordinates": coordinates,
            "values": clean(values.tolist()), "missing_values": int((~np.isfinite(values)).sum()),
            "decoding": "netCDF4 mask/scale applied; units unchanged; no aggregation or interpolation"}


VARIABLES = {
    "sfc": [("pm2p5", "particulate_matter_2.5um"), ("pm10", "particulate_matter_10um"),
            ("u10", "10m_u_component_of_wind"), ("v10", "10m_v_component_of_wind"),
            ("t2m", "2m_temperature"), ("d2m", "2m_dewpoint_temperature"),
            ("blh", "boundary_layer_height"), ("tp", "total_precipitation"), ("tcc", "total_cloud_cover")],
    "o3": [("go3", "o3", "ozone")],
}


def export(request_path, cams_root, obs_root, out, *, limit_cases=0, preserve_overlaps=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    cases = request["cases"][:limit_cases] if limit_cases else request["cases"]
    coords = request["coordinates"]
    sources, failures, overlaps = {}, [], []
    count = 0
    index = {}
    if cams_root:
        import netCDF4
        for path in sorted(Path(cams_root).rglob("*.nc")):
            parts = path.stem.split("_")
            if len(parts) != 3 or parts[0] not in VARIABLES:
                continue
            try:
                start, end = date.fromisoformat(parts[1]), date.fromisoformat(parts[2])
            except ValueError:
                continue
            while start <= end:
                index.setdefault((parts[0], start.isoformat()), []).append(path)
                start += timedelta(days=1)
        with gzip.open(out / "cams_native.jsonl.gz", "wt", encoding="utf-8") as stream:
            grouped = {}
            for case in cases:
                ref = (date.fromisoformat(case["issue_date"]) - timedelta(days=1)).isoformat()
                for kind in VARIABLES:
                    matches = index.get((kind, ref), [])
                    if not matches or (len(matches) > 1 and not preserve_overlaps):
                        failures.append({"city": case["city"], "issue_date": case["issue_date"], "kind": kind,
                                         "reason": "file_missing" if not matches else "ambiguous_overlapping_files"})
                        continue
                    if len(matches) > 1:
                        overlaps.append({"city": case["city"], "issue_date": case["issue_date"],
                                         "kind": kind, "paths": [str(p) for p in matches],
                                         "policy": "all source variants retained; no implicit deduplication"})
                    for match in matches:
                        grouped.setdefault(match, []).append((case, kind, ref))
            for path, selections in grouped.items():
                source = identity(path); sources[source["sha256"]] = source
                try:
                    with netCDF4.Dataset(path) as dataset:
                        source["variables"] = {name: {"dimensions": list(var.dimensions), "shape": list(var.shape),
                                                       "units": getattr(var, "units", None)}
                                               for name, var in dataset.variables.items()}
                        for case, kind, ref in selections:
                            for names in VARIABLES[kind]:
                                try:
                                    var = next((dataset.variables[n] for n in names if n in dataset.variables), None)
                                    if var is None:
                                        raise ValueError("variable missing")
                                    cycle = datetime.fromisoformat(ref).replace(hour=12, tzinfo=UTC)
                                    record = extract_netcdf_point(dataset, var, cycle, *coords[case["city"]])
                                    record.update({"city": case["city"], "issue_date": case["issue_date"],
                                                   "canonical_variable": names[0], "source_sha256": source["sha256"],
                                                   "source_path": str(path.resolve())})
                                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                                    count += 1
                                except (ValueError, KeyError, IndexError) as exc:
                                    failures.append({"city": case["city"], "issue_date": case["issue_date"],
                                                     "variable": names[0], "reason": str(exc)})
                except OSError as exc:
                    failures.append({"path": str(path), "reason": f"unreadable_netcdf: {exc}"})
    obs_count = 0
    if obs_root:
        cities = {case["city"] for case in cases}
        required_days = set()
        for case in cases:
            issue = date.fromisoformat(case["issue_date"])
            required_days.update((issue + timedelta(days=i)).strftime("%Y%m%d") for i in range(-3, 6))
        obs_index = {}
        for path in Path(obs_root).rglob("china_cities_*.csv"):
            obs_index.setdefault(path.stem.removeprefix("china_cities_"), []).append(path)
        with gzip.open(out / "observations_native.jsonl.gz", "wt", encoding="utf-8") as stream:
            for day in sorted(required_days):
                paths = obs_index.get(day, [])
                if len(paths) != 1:
                    failures.append({"date": day, "reason": "observation_file_missing_or_ambiguous"}); continue
                path = paths[0]; source = identity(path); sources[source["sha256"]] = source
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        if row.get("type") not in {"PM2.5", "PM10", "O3", "O3_8h", "SO2", "NO2", "CO"}:
                            continue
                        # Retain raw strings: invalid values remain auditable at the receiver.
                        stream.write(json.dumps({"date": row.get("date"), "hour": row.get("hour"),
                            "type": row["type"], "city_values": {city: row.get(city) for city in sorted(cities)},
                            "source_sha256": source["sha256"]}, ensure_ascii=False) + "\n")
                        obs_count += 1
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    manifest = {"code_git_commit": git_commit, "overlapping_sources": overlaps,
                "overlap_policy": "preserve_all" if preserve_overlaps else "reject",
                "contract_version": "native-city-export-v1", "request": identity(request_path),
                "extractor": identity(__file__), "requested_cases": len(cases), "cams_series": count,
                "observation_records": obs_count, "raw_files_transferred": False,
                "sources": list(sources.values()), "failures": failures,
                "outputs": [identity(p) for p in sorted(out.glob("*.gz"))]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return {key: manifest[key] for key in ("requested_cases", "cams_series", "observation_records", "raw_files_transferred")} | {"failures": len(failures)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--cams-root", type=Path)
    parser.add_argument("--obs-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--preserve-overlaps", action="store_true",
                        help="retain every overlapping file as a separately identified source variant")
    args = parser.parse_args()
    if not args.cams_root and not args.obs_root:
        parser.error("provide --cams-root and/or --obs-root")
    if args.limit_cases < 0:
        parser.error("limit-cases must be nonnegative")
    print(json.dumps(export(args.request, args.cams_root, args.obs_root, args.out,
                            limit_cases=args.limit_cases, preserve_overlaps=args.preserve_overlaps), ensure_ascii=False))


if __name__ == "__main__":
    main()
