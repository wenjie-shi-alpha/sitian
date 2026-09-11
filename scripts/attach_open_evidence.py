#!/usr/bin/env python3
"""Attach target-city slices of shared open evidence to national case bundles."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
import math

from sitian.case import CaseBundle
from sitian.open_evidence import (
    DEFAULT_EVIDENCE_ROOT,
    EVIDENCE_VERSION,
    PRESSURE_LEVEL_APPROX_HEIGHT_M,
    PRESSURE_LEVELS,
    TERRAIN_CLEARANCE_M,
    lowest_pressure_level_above_terrain,
)


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _city_map(value, city: str):
    if not isinstance(value, dict):
        return value
    return {city: value[city]} if city in value else {}


def _vector_difference(first: dict, second: dict) -> float | None:
    try:
        return round(math.hypot(
            float(first["u_ms"]) - float(second["u_ms"]),
            float(first["v_ms"]) - float(second["v_ms"]),
        ), 1)
    except (KeyError, TypeError, ValueError):
        return None


def _synoptic(document: dict, city: str, elevation_m: float | None) -> dict:
    """Attach one city while masking pressure surfaces below model terrain."""
    result = deepcopy(document["synoptic"])
    transport_level = lowest_pressure_level_above_terrain(elevation_m)
    valid_levels = {
        str(level): (
            elevation_m is not None
            and PRESSURE_LEVEL_APPROX_HEIGHT_M[level]
                >= float(elevation_m) + TERRAIN_CLEARANCE_M
        )
        for level in PRESSURE_LEVELS
    }
    for rows in result.get("sources", {}).values():
        for row in rows:
            if "cities" in row:
                row["cities"] = _city_map(row["cities"], city)
                features = row["cities"].get(city)
                if not isinstance(features, dict):
                    continue
                features["terrain_elevation_m"] = elevation_m
                features["pressure_level_clearance_m"] = TERRAIN_CLEARANCE_M
                features["pressure_level_above_terrain"] = valid_levels
                features["transport_pressure_level_hpa"] = transport_level
                for section in ("wind", "temperature_c", "relative_humidity_pct",
                                "omega_pa_s"):
                    values = features.get(section)
                    if not isinstance(values, dict):
                        continue
                    for level in list(values):
                        if valid_levels.get(str(level)) is False:
                            values[level] = None
                stability = features.get("stability")
                if isinstance(stability, dict):
                    if not (valid_levels.get("925") and valid_levels.get("850")):
                        stability["t925_minus_t850_c"] = None
                        stability["low_level_inversion_signal"] = None
                    if not (valid_levels.get("850") and valid_levels.get("700")):
                        stability["t850_minus_t700_c"] = None
    cross = result.get("cross_model", {})
    for valid_time, cities in list(cross.items()):
        city_cross = _city_map(cities, city)
        values = city_cross.get(city)
        if isinstance(values, dict):
            source_rows = [
                rows_by_source
                for rows in document.get("synoptic", {}).get("sources", {}).values()
                for rows_by_source in rows
                if rows_by_source.get("valid_time") == valid_time
                and rows_by_source.get("available")
            ]
            winds = [
                row.get("cities", {}).get(city, {}).get("wind", {}).get(
                    str(transport_level), {}
                )
                for row in source_rows
            ] if transport_level is not None else []
            values["transport_pressure_level_hpa"] = transport_level
            values["transport_wind_vector_diff_ms"] = (
                _vector_difference(winds[0], winds[1]) if len(winds) >= 2 else None
            )
            if not valid_levels.get("925"):
                values["wind925_vector_diff_ms"] = None
        cross[valid_time] = city_cross
    return result


def _composition(document: dict, city: str) -> dict:
    result = deepcopy(document["pollution"]["composition"])
    for kind in ("aerosol", "trace_gases", "column_gases"):
        for row in result.get(kind, {}).get("records", []):
            if "cities" in row:
                row["cities"] = _city_map(row["cities"], city)
    return result


def _fires(document: dict, city: str) -> dict:
    result = deepcopy(document["pollution"]["fires"])
    if "cities" in result:
        result["cities"] = _city_map(result["cities"], city)
    return result


def _distance_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    distance = 2 * 6371.0088 * math.asin(min(1.0, math.sqrt(a)))
    bearing = (math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
               math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))) + 360) % 360
    return distance, bearing


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _current_wind_from(
    synoptic: dict | None, city: str, pressure_level: int | None
) -> float | None:
    values = []
    if not synoptic or pressure_level is None:
        return None
    for rows in synoptic.get("synoptic", {}).get("sources", {}).values():
        row = next((item for item in rows if item.get("role") == "current" and item.get("available")), None)
        wind = ((row or {}).get("cities", {}).get(city, {}).get("wind", {})
                .get(str(pressure_level), {}))
        if wind.get("from_deg") is not None:
            values.append(float(wind["from_deg"]))
    if not values:
        return None
    # Circular mean of the two forecast systems.
    x = sum(math.cos(math.radians(value)) for value in values)
    y = sum(math.sin(math.radians(value)) for value in values)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _source_context(
    document: dict, city: str, coords: dict, wind_from: float | None,
    pressure_level: int | None,
) -> dict:
    rows = document["pollution"]["spatial_observations"]
    target = coords.get(city)
    if not isinstance(target, list):
        return {"available": False, "reason": "target_city_coordinate_missing"}
    candidates = []
    for other, values in rows.get("cities", {}).items():
        location = coords.get(other)
        pm25 = values.get("PM2.5", {})
        if other == city or not isinstance(location, list) or pm25.get("latest") is None:
            continue
        distance, bearing = _distance_bearing(*map(float, target), *map(float, location))
        if distance > 1200:
            continue
        angle = _angle_difference(bearing, wind_from) if wind_from is not None else None
        candidates.append({"city": other, "distance_km": round(distance),
                           "bearing_from_target_deg": round(bearing),
                           "angle_to_inflow_deg": round(angle) if angle is not None else None,
                           **values})
    upwind = [row for row in candidates if row["angle_to_inflow_deg"] is not None
              and row["angle_to_inflow_deg"] <= 50]
    upwind.sort(key=lambda row: (-float(row["PM2.5"]["latest"]), row["distance_km"]))
    regional = sorted(candidates, key=lambda row: (-float(row["PM2.5"]["latest"]),
                                                    row["distance_km"]))
    # Preserve a bounded, directionally balanced pool so a forecast wind shift
    # can nominate a different upstream city without attaching all 120 cities.
    # Selection uses only issue-time observations and static geometry.
    transport_pool = {}
    for sector in range(8):
        center = sector * 45.0
        sector_rows = [row for row in candidates
                       if _angle_difference(row["bearing_from_target_deg"], center) <= 30]
        selected = sorted(sector_rows, key=lambda row: row["distance_km"])[:2]
        for pollutant in ("PM2.5", "PM10", "O3"):
            ranked = sorted(
                (row for row in sector_rows
                 if row.get(pollutant, {}).get("latest") is not None),
                key=lambda row: (-float(row[pollutant]["latest"]), row["distance_km"]),
            )[:2]
            selected.extend(ranked)
        for row in selected:
            transport_pool[row["city"]] = row
    return {"available": True, "observation_cutoff": rows.get("cutoff"),
            "cutoff_note": rows.get("cutoff_note"),
            "inflow_from_deg": round(wind_from) if wind_from is not None else None,
            "inflow_pressure_level_hpa": pressure_level,
            "target": {city: rows.get("cities", {}).get(city, {})},
            "upwind_candidates": upwind[:12], "regional_hotspots": regional[:12],
            "transport_candidate_pool": [transport_pool[key]
                                         for key in sorted(transport_pool)]}


def _static_context(document: dict, city: str, wind_from: float | None) -> dict:
    context = document["static_context"]
    row = deepcopy(context.get("cities", {}).get(city, {}))
    direction = None if wind_from is None else ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[
        int((wind_from + 22.5) // 45) % 8]
    inflow = {"direction": direction, "terrain": None, "emissions": {}}
    if direction and row:
        inflow["terrain"] = (row.get("terrain", {}).get("directional_terrain_30_300km", {})
                             .get(direction))
        for pollutant, values in row.get("anthropogenic_emissions", {}).items():
            inflow["emissions"][pollutant] = {
                "directional_30_800km_tonnes_per_year":
                    values.get("directional_total_30_800km_tonnes_per_year", {}).get(direction),
                "dominant_sectors_within_300km": values.get("dominant_sectors"),
            }
    return {"notes": context.get("notes"), "target": row,
            "inflow_aligned": inflow, "raw": context.get("raw")}


def build_case_evidence(derived_root: Path, issue_date: str, city: str,
                        coords: dict | None = None) -> dict:
    evidence = {"contract_version": EVIDENCE_VERSION,
                "issue_date": issue_date, "target_city": city}
    if coords and isinstance(coords.get(city), list):
        evidence["target_location"] = {
            "lat": float(coords[city][0]), "lon": float(coords[city][1])
        }
    synoptic = _read(derived_root / f"{issue_date}.synoptic.json")
    composition = _read(derived_root / f"{issue_date}.composition.json")
    fires = _read(derived_root / f"{issue_date}.fires.json")
    spatial = _read(derived_root / f"{issue_date}.spatial_obs.json")
    static = _read(derived_root.parent / "static_context.json")
    static_row = ((static or {}).get("static_context", {}).get("cities", {}).get(city, {}))
    elevation_m = (static_row.get("terrain", {}).get("elevation_m")
                   if isinstance(static_row, dict) else None)
    transport_level = lowest_pressure_level_above_terrain(elevation_m)
    if synoptic:
        evidence["synoptic"] = _synoptic(synoptic, city, elevation_m)
    pollution = {}
    if composition:
        pollution["composition"] = _composition(composition, city)
    if fires:
        pollution["fires"] = _fires(fires, city)
    wind_from = _current_wind_from(synoptic, city, transport_level)
    if spatial and coords:
        pollution["source_context"] = _source_context(
            spatial, city, coords, wind_from, transport_level
        )
    if static:
        pollution.setdefault("source_context", {"available": True})["static"] = _static_context(
            static, city, wind_from)
    if pollution:
        evidence["pollution"] = pollution
    return evidence


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-root", type=Path, default=Path("cases/national"))
    parser.add_argument("--derived-root", type=Path,
                        default=DEFAULT_EVIDENCE_ROOT / "derived" / "by_issue")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--issue-date")
    parser.add_argument("--coords", type=Path, default=Path("data/interim/city_coords.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    coords = json.loads(args.coords.read_text(encoding="utf-8"))
    roots = [args.cases_root / args.split] if args.split else [args.cases_root / value
                                                               for value in ("train", "val", "test")]
    cases = sorted(path for root in roots if root.is_dir() for path in root.iterdir()
                   if path.is_dir() and (path / "case.json").is_file())
    if args.issue_date:
        cases = [path for path in cases
                 if json.loads((path / "case.json").read_text(encoding="utf-8"))["issue_date"] == args.issue_date]
    if args.limit is not None:
        cases = cases[:args.limit]
    counts = {"attached": 0, "no_derived_evidence": 0, "failed_audit": 0}
    for path in cases:
        bundle = CaseBundle.load(path)
        evidence = build_case_evidence(args.derived_root, bundle.issue_date, bundle.region, coords)
        if len(evidence) == 3:
            counts["no_derived_evidence"] += 1
            continue
        candidate = deepcopy(bundle)
        candidate.evidence = evidence
        violations = candidate.audit_time_gate()
        if violations:
            counts["failed_audit"] += 1
            print(f"FAILED {bundle.case_id}: {violations[:3]}", flush=True)
            continue
        if not args.dry_run:
            _atomic_json(path / "evidence.json", evidence)
        counts["attached"] += 1
    print(json.dumps({"cases": len(cases), **counts, "dry_run": args.dry_run}, ensure_ascii=False))
    return 1 if counts["failed_audit"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
