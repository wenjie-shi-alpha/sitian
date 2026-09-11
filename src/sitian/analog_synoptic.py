"""Comparable synoptic features at fixed hours relative to forecast issue.

Keep GFS/IFS separate, mask underground pressure levels, and express nearby
systems relative to the target city. This is a sparse feature comparison, not
full-field pattern matching or trajectory modelling.
"""
from __future__ import annotations

import math
from datetime import datetime

from .data_contract import finite_number, issue_time
from .open_evidence import PRESSURE_LEVEL_APPROX_HEIGHT_M, TERRAIN_CLEARANCE_M

SYNOPTIC_HOURS = {-24, 0, 12, 24, 48, 72, 96, 120}
SYNOPTIC_SCALES = {"mslp_hpa": 12.0, "z500_gpm": 150.0, "shortwave_w_m2": 250.0}
for level in (925, 850, 700, 500):
    SYNOPTIC_SCALES.update({f"u{level}_ms": 8.0, f"v{level}_ms": 8.0})
for level in (925, 850, 700):
    SYNOPTIC_SCALES.update({f"t{level}_c": 12.0, f"rh{level}_pct": 30.0})
for level in (850, 700, 500):
    SYNOPTIC_SCALES[f"omega{level}_pa_s"] = 0.3
for pair in ("925_850", "850_700"):
    SYNOPTIC_SCALES[f"temperature_difference_{pair}_c"] = 6.0
for kind in ("low_centers", "high_centers", "trough_signals", "ridge_signals"):
    SYNOPTIC_SCALES.update({f"{kind}_east_km": 1000.0, f"{kind}_north_km": 1000.0,
                            f"{kind}_intensity": 12.0 if "centers" in kind else 150.0})


def _above_terrain(city: dict, level: int) -> bool:
    declared = city.get("pressure_level_above_terrain", {}).get(str(level))
    if isinstance(declared, bool):
        return declared
    elevation = city.get("terrain_elevation_m")
    return finite_number(elevation) and PRESSURE_LEVEL_APPROX_HEIGHT_M[level] >= elevation + TERRAIN_CLEARANCE_M


def synoptic_features(bundle) -> dict[str, float]:
    cutoff = issue_time(bundle.issue_date)
    result = {}
    location = bundle.evidence.get("target_location", {})
    for source, rows in bundle.evidence.get("synoptic", {}).get("sources", {}).items():
        seen = set()
        for row in rows:
            if not row.get("available") or not row.get("available_at") or not row.get("valid_time"):
                continue
            valid = datetime.fromisoformat(row["valid_time"])
            if valid.tzinfo is None:
                raise ValueError("synoptic valid time must have a timezone")
            relative_hour = (valid - cutoff).total_seconds() / 3600
            if relative_hour not in SYNOPTIC_HOURS or relative_hour > bundle.horizon * 24:
                continue
            if relative_hour in seen:
                raise ValueError("multiple synoptic cycles for the same source/valid time")
            seen.add(relative_hour)
            prefix = f"synoptic/{source}/h{int(relative_hour)}/"
            city = row.get("cities", {}).get(bundle.region, {})
            values = {"mslp_hpa": city.get("mslp_hpa"),
                      "shortwave_w_m2": city.get("surface_downward_shortwave_w_m2")}

            if _above_terrain(city, 500):
                values["z500_gpm"] = city.get("z500_gpm")
            for level in (925, 850, 700, 500):
                if not _above_terrain(city, level):
                    continue
                wind = city.get("wind", {}).get(str(level), {})
                for component in ("u", "v"):
                    values[f"{component}{level}_ms"] = wind.get(f"{component}_ms")
                for field, key in (("temperature_c", f"t{level}_c"),
                                   ("relative_humidity_pct", f"rh{level}_pct"),
                                   ("omega_pa_s", f"omega{level}_pa_s")):
                    if key in SYNOPTIC_SCALES:
                        values[key] = city.get(field, {}).get(str(level))
            for lower, upper in ((925, 850), (850, 700)):
                low, high = values.get(f"t{lower}_c"), values.get(f"t{upper}_c")
                if finite_number(low) and finite_number(high):
                    values[f"temperature_difference_{lower}_{upper}_c"] = low - high
            lat, lon = location.get("lat"), location.get("lon")
            if finite_number(lat) and finite_number(lon):
                for kind in ("low_centers", "high_centers", "trough_signals", "ridge_signals"):
                    nearby = []
                    for center in row.get("systems", {}).get(kind, []):
                        cy, cx = center.get("lat"), center.get("lon")
                        if not finite_number(cy) or not finite_number(cx):
                            continue
                        # Equirectangular offsets (approx km), not transport distances.
                        east = 111.2 * ((cx - lon + 180) % 360 - 180) * math.cos(math.radians((cy + lat) / 2))
                        north = 111.2 * (cy - lat)
                        intensity = center.get("mslp_hpa" if "centers" in kind else "z500_zonal_anomaly_gpm")
                        if finite_number(intensity):
                            nearby.append((east * east + north * north, east, north, intensity))
                    if nearby:
                        _, east, north, intensity = min(nearby)
                        values.update({f"{kind}_east_km": east, f"{kind}_north_km": north,
                                       f"{kind}_intensity": intensity})
            result.update({prefix + key: value for key, value in values.items() if finite_number(value)})
    return result
