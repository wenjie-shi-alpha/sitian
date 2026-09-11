"""Compact, deterministic process view over issue-time-legal evidence.

This module is a presentation layer, not an expert system: it preserves the
measured/modelled state and a few explicitly defined diagnostic flags, while
leaving the forecast decision to the policy.  It never reads truth or expert
files.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from .open_evidence import (
    PRESSURE_LEVEL_APPROX_HEIGHT_M,
    TERRAIN_CLEARANCE_M,
    lowest_pressure_level_above_terrain,
)


def _numbers(values) -> list[float]:
    output = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def _mean(values, digits: int = 1):
    values = _numbers(values)
    return round(sum(values) / len(values), digits) if values else None


def _circular_mean(values):
    values = _numbers(values)
    if not values:
        return None
    x = sum(math.cos(math.radians(value)) for value in values)
    y = sum(math.sin(math.radians(value)) for value in values)
    return round((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _direction_spread(values):
    values = _numbers(values)
    center = _circular_mean(values)
    return (round(max(_angle_difference(value, center) for value in values))
            if values and center is not None else None)


def _distance_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(value)))


def _nearby_systems(source_rows: list[tuple[str, dict]], target: dict) -> dict:
    """Closest deterministic system signal per model, encoded compactly."""
    if not isinstance(target, dict) or target.get("lat") is None or target.get("lon") is None:
        return {}
    target_point = (float(target["lat"]), float(target["lon"]))
    specifications = {
        "low": ("low_centers", "mslp_hpa"),
        "high": ("high_centers", "mslp_hpa"),
        "trough": ("trough_signals", "z500_zonal_anomaly_gpm"),
        "ridge": ("ridge_signals", "z500_zonal_anomaly_gpm"),
    }
    output = {}
    for label, (field, value_field) in specifications.items():
        by_source = {}
        for source, row in source_rows:
            candidates = row.get("systems", {}).get(field, [])
            if not candidates:
                continue
            closest = min(candidates, key=lambda item: _distance_km(
                target_point, (float(item["lat"]), float(item["lon"]))
            ))
            by_source[source] = [
                closest.get("lat"), closest.get("lon"), closest.get(value_field),
                round(_distance_km(
                    target_point, (float(closest["lat"]), float(closest["lon"]))
                )),
            ]
        if by_source:
            output[label] = by_source
    return output


def _bjt(utc_text: str) -> str:
    parsed = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
    return parsed.astimezone(timezone(timedelta(hours=8))).isoformat(timespec="minutes")


def _wind(city: dict, level: str) -> dict:
    value = city.get("wind", {}).get(level, {}) if isinstance(city, dict) else {}
    return value if isinstance(value, dict) else {}


def _trajectory(bundle) -> dict[str, dict]:
    synoptic = (bundle.evidence or {}).get("synoptic", {})
    terrain = (((bundle.evidence or {}).get("pollution", {}).get("source_context", {})
                .get("static", {}).get("target", {}).get("terrain", {})))
    elevation_m = terrain.get("elevation_m")
    transport_level = lowest_pressure_level_above_terrain(elevation_m)
    thermodynamic_level = lowest_pressure_level_above_terrain(
        elevation_m, levels=(925, 850, 700)
    )
    vertical_motion_level = lowest_pressure_level_above_terrain(
        elevation_m, levels=(850, 700, 500)
    )
    sources = synoptic.get("sources", {})
    by_source = {
        source: {row.get("valid_time"): row for row in rows if row.get("available")}
        for source, rows in sources.items()
    }
    valid_times = sorted({valid_time for rows in by_source.values() for valid_time in rows
                          if valid_time})
    output = {}
    issue_bjt = datetime.fromisoformat(f"{bundle.issue_date}T08:00:00+08:00")
    for valid_time in valid_times:
        source_rows = [(source, rows[valid_time]) for source, rows in by_source.items()
                       if valid_time in rows]
        cities = [row.get("cities", {}).get(bundle.region, {}) for _, row in source_rows]
        wind925 = [_wind(city, "925") for city in cities]
        wind850 = [_wind(city, "850") for city in cities]
        wind700 = [_wind(city, "700") for city in cities]
        transport_winds = [
            _wind(city, str(transport_level)) for city in cities
        ] if transport_level is not None else []
        cross = synoptic.get("cross_model", {}).get(valid_time, {}).get(bundle.region, {})
        wind925_mean = {
            "speed": _mean(row.get("speed_ms") for row in wind925),
            "from": _circular_mean(row.get("from_deg") for row in wind925),
            "spread": _direction_spread(row.get("from_deg") for row in wind925),
        }
        wind850_mean = {
            "speed": _mean(row.get("speed_ms") for row in wind850),
            "from": _circular_mean(row.get("from_deg") for row in wind850),
        }
        wind700_mean = {
            "speed": _mean(row.get("speed_ms") for row in wind700),
            "from": _circular_mean(row.get("from_deg") for row in wind700),
        }
        transport_wind_mean = {
            "speed": _mean(row.get("speed_ms") for row in transport_winds),
            "from": _circular_mean(row.get("from_deg") for row in transport_winds),
            "spread": _direction_spread(row.get("from_deg") for row in transport_winds),
        }
        humidity = (
            _mean(city.get("relative_humidity_pct", {}).get(str(thermodynamic_level))
                  for city in cities)
            if thermodynamic_level is not None else None
        )
        vertical_motion = (
            _mean((city.get("omega_pa_s", {}).get(str(vertical_motion_level))
                   for city in cities), digits=3)
            if vertical_motion_level is not None else None
        )
        inversion_values = []
        inversion_pair = {
            925: ("925", "850"),
            850: ("850", "700"),
        }.get(thermodynamic_level)
        if inversion_pair:
            lower, upper = inversion_pair
            for city in cities:
                temperatures = city.get("temperature_c", {})
                if temperatures.get(lower) is not None and temperatures.get(upper) is not None:
                    inversion_values.append(
                        float(temperatures[lower]) <= float(temperatures[upper])
                    )
        valid_bjt = _bjt(valid_time)
        issue_relative_hour = round(
            (datetime.fromisoformat(valid_bjt) - issue_bjt).total_seconds() / 3600
        )
        record = {
            "role": next((row.get("role") for _, row in source_rows if row.get("role")), None),
            "issue_relative_hour": issue_relative_hour,
            "n_models": len(source_rows),
            "wind925_ms_from_spread": [wind925_mean["speed"], wind925_mean["from"],
                                        wind925_mean["spread"]],
            "wind850_ms_from": [wind850_mean["speed"], wind850_mean["from"]],
            "wind700_ms_from": [wind700_mean["speed"], wind700_mean["from"]],
            "terrain_adaptive_low_level": {
                "terrain_elevation_m": elevation_m,
                "pressure_level_hpa": transport_level,
                "approx_level_height_m": (
                    PRESSURE_LEVEL_APPROX_HEIGHT_M.get(transport_level)
                    if transport_level is not None else None
                ),
                "minimum_clearance_m": TERRAIN_CLEARANCE_M,
                "wind_ms_from_spread": [
                    transport_wind_mean["speed"], transport_wind_mean["from"],
                    transport_wind_mean["spread"],
                ],
                "thermodynamic_level_hpa": thermodynamic_level,
                "rh_pct": humidity,
                "vertical_motion_level_hpa": vertical_motion_level,
                "omega_pa_s": vertical_motion,
                "inversion_models": sum(inversion_values) if inversion_values else None,
                "inversion_models_available": len(inversion_values),
            },
            "mslp_hpa_z500_gpm": [
                _mean(city.get("mslp_hpa") for city in cities),
                _mean(city.get("z500_gpm") for city in cities),
            ],
            "temperature_925_850_700_c": [
                _mean(city.get("temperature_c", {}).get(level) for city in cities)
                for level in ("925", "850", "700")
            ],
            "rh925_700_pct": [
                _mean(city.get("relative_humidity_pct", {}).get("925") for city in cities),
                _mean(city.get("relative_humidity_pct", {}).get("700") for city in cities),
            ],
            "omega700_pa_s": _mean(
                (city.get("omega_pa_s", {}).get("700") for city in cities), digits=3),
            "inversion_models": sum(
                city.get("stability", {}).get("low_level_inversion_signal") is True
                for city in cities),
            "shortwave_w_m2": _mean(
                city.get("surface_downward_shortwave_w_m2") for city in cities),
            "model_diff_wind_msl_z500_sw": [
                cross.get("transport_wind_vector_diff_ms"), cross.get("mslp_abs_diff_hpa"),
                cross.get("z500_abs_diff_gpm"), cross.get("shortwave_abs_diff_w_m2"),
            ],
            "nearest_system_signal_source_lat_lon_value_km": _nearby_systems(
                source_rows, (bundle.evidence or {}).get("target_location", {})
            ),
        }
        speed = transport_wind_mean["speed"]
        humid = humidity
        inversions = sum(inversion_values) if inversion_values else None
        if speed is None:
            signal = "unavailable"
        elif inversions is None and humid is None:
            signal = "wind_only"
        elif speed < 2.0 and ((inversions or 0) > 0 or (humid is not None and humid >= 75)):
            signal = "very_poor"
        elif speed < 3.0 or (inversions is not None and inversions == len(inversion_values)):
            signal = "poor"
        elif inversions is None:
            # High terrain can leave RH/omega available but no valid adjacent
            # temperature pair. Do not silently turn missing stability
            # information into "no inversion" and overclaim favorable mixing.
            signal = "partial_profile"
        elif speed >= 5.0 and inversions == 0:
            signal = "favorable"
        else:
            signal = "mixed"
        record["ventilation"] = signal
        output[valid_bjt] = record
    return output


def _observation_state(bundle) -> dict:
    from .assess import obs_trend

    trends = obs_trend(bundle.observations, bundle.region, bundle.issue_date)
    output = {}
    for pollutant, values in trends.items():
        block = bundle.observations.get(pollutant, {})
        output[pollutant] = {"unit": block.get("unit") or
                            ("mg/m³" if pollutant == "CO" else "µg/m³"), **values}
    return output


def _daily_weather(bundle, trajectory: dict | None = None) -> dict:
    allowed = ("wind_speed_ms", "wind_dir_deg", "wind_dir", "blh_max_m", "blh_m",
               "rh_pct", "rain_mm", "precip_mm", "tmax_c", "cloud_pct")
    output = {
        day: {key: values.get(key) for key in allowed if values.get(key) is not None}
        for day, values in sorted(bundle.diagnostics.get("daily", {}).items())
    }
    # Forecast days without CAMS-derived surface diagnostics (the fifth target
    # day under the previous-day 12UTC cycle) still have GFS/IFS pressure-level
    # snapshots.  Give every forecast day a row of the same shape so the policy
    # never has to read "no row" as "no information"; CAMS-only fields stay
    # explicit nulls.
    try:
        forecast_dates = list(bundle.forecast_dates())
    except Exception:  # pragma: no cover - legacy bundles without the helper
        forecast_dates = []
    for day in forecast_dates:
        if day in output or not trajectory:
            continue
        points = [record for stamp, record in trajectory.items() if stamp[:10] == day]
        if not points:
            continue
        winds = [record.get("terrain_adaptive_low_level", {}).get("wind_ms_from_spread")
                 or record.get("wind925_ms_from_spread") or [] for record in points]
        speed = _mean(wind[0] for wind in winds if len(wind) >= 1)
        direction = _circular_mean(wind[1] for wind in winds if len(wind) >= 2)
        humidity = _mean(record.get("terrain_adaptive_low_level", {}).get("rh_pct")
                         for record in points)
        inversions = [record.get("terrain_adaptive_low_level", {}).get("inversion_models")
                      for record in points]
        shortwave = [record.get("shortwave_w_m2") for record in points
                     if record.get("shortwave_w_m2") is not None]
        output[day] = {
            "source": "nwp_levels_only",
            "wind_speed_ms": speed,
            "wind_dir_deg": direction,
            "wind_dir": _direction_label(direction) if direction is not None else None,
            "rh_pct": humidity,
            "inversion_models_max": max((value for value in inversions if value is not None),
                                        default=None),
            "shortwave_w_m2_max": max(shortwave) if shortwave else None,
            "blh_max_m": None, "rain_mm": None, "tmax_c": None, "cloud_pct": None,
            "note": ("terrain-adaptive low-level wind/RH/inversion from GFS+IFS snapshots; "
                     "no CAMS surface diagnostics or pollution guidance for this day"),
        }
    return dict(sorted(output.items()))


def _guidance(bundle) -> dict:
    output = {}
    for source, values in sorted(bundle.guidance.get("sources", {}).items()):
        row = {}
        for key in ("daily_pm25", "daily_pm10", "daily_o3max"):
            if values.get(key):
                row[key] = values[key]
        if row:
            output[source] = row
    try:
        forecast_dates = list(bundle.forecast_dates())
    except Exception:  # pragma: no cover
        forecast_dates = []
    covered = {day for row in output.values() for series in row.values() for day in series}
    missing = [day for day in forecast_dates if day not in covered]
    if missing:
        output["days_without_guidance"] = missing
    return output


def _city_summary(row: dict) -> dict:
    output = {key: row.get(key) for key in
              ("city", "distance_km", "bearing_from_target_deg", "angle_to_inflow_deg")}
    for pollutant in ("PM2.5", "PM10", "O3", "SO2", "NO2", "CO"):
        values = row.get(pollutant, {})
        if values:
            output[pollutant] = {key: values.get(key) for key in
                                 ("unit", "latest", "mean_24h", "change_6h")
                                 if values.get(key) is not None}
    return output


def _transport_candidate(row: dict) -> dict:
    output = {key: row.get(key) for key in
              ("city", "distance_km", "bearing_from_target_deg", "angle_to_inflow_deg")}
    for pollutant in ("PM2.5", "PM10", "O3", "NO2", "CO"):
        values = row.get(pollutant, {})
        if values:
            output[pollutant] = [values.get("latest"), values.get("change_6h")]
    return output


_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _direction_label(value: float) -> str:
    return _DIRECTIONS[int((float(value) + 22.5) // 45) % 8]


def _future_inflow_scenarios(context: dict, trajectory: dict[str, dict]) -> dict:
    pool = context.get("transport_candidate_pool", [])
    static_target = context.get("static", {}).get("target", {})
    grouped: dict[str, list[tuple[str, float, float | None, int | None]]] = {}
    for valid_time, values in trajectory.items():
        wind = (values.get("terrain_adaptive_low_level", {})
                .get("wind_ms_from_spread", []))
        if not wind:
            wind = values.get("wind925_ms_from_spread", [])
        if len(wind) >= 2 and wind[1] is not None:
            grouped.setdefault(_direction_label(float(wind[1])), []).append(
                (valid_time, float(wind[1]),
                 float(wind[0]) if wind[0] is not None else None,
                 values.get("issue_relative_hour"))
            )
    output = {}
    for direction, times_and_winds in grouped.items():
        center = _DIRECTIONS.index(direction) * 45.0
        mean_speed = _mean(item[2] for item in times_and_winds)
        aligned = [row for row in pool
                   if row.get("bearing_from_target_deg") is not None
                   and _angle_difference(float(row["bearing_from_target_deg"]), center) <= 50]
        selected = []
        for pollutant in ("PM2.5", "PM10", "O3"):
            selected.extend(sorted(
                (row for row in aligned if row.get(pollutant, {}).get("latest") is not None),
                key=lambda row: (-float(row[pollutant]["latest"]), row.get("distance_km", 9999)),
            )[:2])
        unique = {}
        for row in selected:
            unique.setdefault(row.get("city"), row)
        candidates = {}
        for city, row in list(unique.items())[:5]:
            compact = _transport_candidate(row)
            compact.pop("city", None)
            distance = compact.get("distance_km")
            compact["estimated_travel_hours_at_mean_target_wind"] = (
                round(float(distance) / (3.6 * mean_speed), 1)
                if distance is not None and mean_speed is not None and mean_speed > 0
                else None
            )
            candidates[city] = compact

        terrain = (static_target.get("terrain", {})
                   .get("directional_terrain_30_300km", {}).get(direction))
        emissions = {}
        for pollutant in ("PM2.5", "NOx", "SO2", "NMVOC"):
            values = static_target.get("anthropogenic_emissions", {}).get(pollutant, {})
            directional = values.get("directional_total_30_800km_tonnes_per_year", {})
            if direction in directional:
                emissions[pollutant] = {
                    "directional_30_800km_tonnes_per_year": directional[direction],
                    "dominant_sectors_within_300km": (values.get("dominant_sectors") or [])[:2],
                }
        output[direction] = {
            "valid_bjt": [item[0] for item in times_and_winds],
            "mean_wind_from_deg": _circular_mean(item[1] for item in times_and_winds),
            "mean_target_wind_speed_ms": mean_speed,
            "issue_relative_hours": [item[3] for item in times_and_winds
                                     if item[3] is not None],
            "idealized_advective_reach_km": {
                f"{hours}h": round(mean_speed * 3.6 * hours)
                for hours in (6, 12, 24) if mean_speed is not None
            },
            "issue_time_observation_candidates": candidates,
            "static_directional_context": {"terrain": terrain, "emissions": emissions},
            "screening_note": (
                "target-point wind alignment and constant-speed travel-time screen only; "
                "not a Lagrangian trajectory or source attribution"
            ),
        }
    return output


def _transport(bundle, trajectory: dict[str, dict]) -> dict:
    context = (bundle.evidence or {}).get("pollution", {}).get("source_context", {})
    candidates = {}
    roles: dict[str, set[str]] = {}
    by_city = {}
    for role, rows in (("upwind", context.get("upwind_candidates", [])[:3]),
                       ("hotspot", context.get("regional_hotspots", [])[:3])):
        for row in rows:
            city = row.get("city")
            if not city:
                continue
            roles.setdefault(city, set()).add(role)
            by_city.setdefault(city, row)
    for city, row in by_city.items():
        candidate = _transport_candidate(row)
        candidate.pop("city", None)
        candidates[city] = {"roles": sorted(roles[city]), **candidate}
    aligned = context.get("static", {}).get("inflow_aligned", {})
    emissions = aligned.get("emissions", {})
    compact_emissions = {
        name: (values.get("dominant_sectors_within_300km") or [])[:2]
        for name, values in emissions.items()
        if name in {"PM2.5", "NOx", "SO2", "NMVOC"}
    }
    return {
        "observation_cutoff": context.get("observation_cutoff"),
        "inflow_from_deg_at_issue": context.get("inflow_from_deg"),
        "inflow_pressure_level_hpa": context.get("inflow_pressure_level_hpa"),
        "candidate_value_order": "each pollutant is [latest, change_6h] in its native unit",
        "upwind_or_hotspot_candidates": candidates,
        "inflow_aligned_static": {
            "direction": aligned.get("direction"), "terrain": aligned.get("terrain"),
            "dominant_emission_sectors": compact_emissions,
        },
        "forecast_inflow_scenarios": _future_inflow_scenarios(context, trajectory),
    }


def build_process_evidence(bundle) -> dict[str, Any]:
    """Return the compact first-screen evidence used to route deeper queries."""
    if not bundle.evidence:
        return {"available": False, "reason": "open_evidence_unavailable"}
    trajectory = _trajectory(bundle)
    return {
        "available": True,
        "contract_version": bundle.evidence.get("contract_version"),
        "issue_date": bundle.issue_date,
        "target_city": bundle.region,
        "temporal_profile": bundle.evidence.get("synoptic", {}).get("temporal_profile"),
        "definitions": {
            "ventilation_signal": (
                "uses the closest pressure level >=150m above model terrain; "
                "very_poor: adaptive-low-level wind<2m/s and inversion or RH>=75%; "
                "poor: wind<3m/s or all models invert; favorable: wind>=5m/s and no inversion; "
                "wind_only: both thermodynamic profile and RH are unavailable; partial_profile: "
                "RH/omega may exist but no valid adjacent-level inversion pair; otherwise mixed. "
                "It is a diagnostic flag, not a pollution forecast."
            ),
            "wind_from_deg": "meteorological direction from which wind blows",
            "trajectory_tuple_fields": {
                "wind925_ms_from_spread": "[mean speed m/s, circular-mean from-deg, max model direction spread deg]",
                "terrain_adaptive_low_level": (
                    "closest pressure surface >=150m above terrain, its wind/RH/inversion availability; "
                    "use this—not 925 hPa—for national transport and ventilation"
                ),
                "wind850_ms_from": "[mean speed m/s, circular-mean from-deg]",
                "wind700_ms_from": "[mean speed m/s, circular-mean from-deg]",
                "mslp_hpa_z500_gpm": "[model-mean MSLP hPa, model-mean 500-hPa geopotential height gpm]",
                "temperature_925_850_700_c": "[model-mean T925, T850, T700 in degC]",
                "rh925_700_pct": "[RH925%, RH700%]",
                "model_diff_wind_msl_z500_sw": "[terrain-adaptive wind-vector m/s, MSLP hPa, Z500 gpm, shortwave W/m2]",
                "nearest_system_signal_source_lat_lon_value_km": (
                    "for each low/high/trough/ridge and source: [lat, lon, pressure-or-zonal-Z500 anomaly, "
                    "great-circle distance to target km]; signals are deterministic extrema, not named systems"
                ),
            },
        },
        "initial_pollution_state": _observation_state(bundle),
        "weather_trajectory_6h_to_72h_then_12h": trajectory,
        "daily_surface_dispersion": _daily_weather(bundle, trajectory),
        "transport_context": _transport(bundle, trajectory),
        "pollution_guidance_anchor": _guidance(bundle),
        "deep_query_routes": {
            "system_positions_or_model_disagreement": "get_synoptic_evidence",
            "aerosol_gases_dust_fire": "get_pollution_evidence(kind=composition|fires)",
            "all_upwind_or_static_sources": "get_pollution_evidence(kind=source_context)",
            "historical_guidance_error": "get_guidance_bias",
        },
    }


# --------------------------------------------------------------------------
# Presentation-budget view.  The builder above is the shared information
# contract consumed by the tabular baseline and the audits; the tool view
# returned to the policy must additionally fit the rollout context.  With the
# hybrid-6h72-12h132-v2 contract the raw builder output measured ~21.5K
# Qwen tokens for one call, so the view hoists per-city constants, shortens
# tuple keys and trims the inflow scenarios.  No signal is recomputed here.
_STATIC_LOW_LEVEL_KEYS = (
    "terrain_elevation_m", "pressure_level_hpa", "approx_level_height_m",
    "minimum_clearance_m", "thermodynamic_level_hpa", "vertical_motion_level_hpa",
)

COMPACT_TRAJECTORY_LEGEND = {
    "h": "hours from issue (08 BJT); h%12!=0 points keep only ll/w850/mslp_z500/sw/vent",
    "ll": ("terrain-adaptive low level [wind m/s, from-deg, model spread deg, RH %, omega Pa/s, "
           "models with inversion, models with T pair]; level per terrain_adaptive_levels"),
    "w925/w850/w700": "[m/s, from-deg(, spread)] per level; w925 omitted when ll is 925",
    "mslp_z500": "[MSLP hPa, Z500 gpm]",
    "t": "[T925, T850, T700 degC]",
    "rh": "[RH925, RH700 %]",
    "omega700": "Pa/s, negative = ascent",
    "inv": "models flagging low-level inversion",
    "sw": "downward shortwave W/m2",
    "diff": "GFS-IFS [ll wind vector m/s, MSLP hPa, Z500 gpm, shortwave W/m2]",
    "vent": "see definitions.ventilation_signal",
    "systems": "daily nearest GFS low/high [lat, lon, MSLP hPa, km]; more in get_synoptic_evidence",
}


def _compact_trajectory_point(record: dict, ll_level) -> dict:
    ll = record.get("terrain_adaptive_low_level", {})
    out = {
        "h": record.get("issue_relative_hour"),
        "role": record.get("role"),
        "ll": [
            *(ll.get("wind_ms_from_spread") or [None, None, None]),
            ll.get("rh_pct"), ll.get("omega_pa_s"),
            ll.get("inversion_models"), ll.get("inversion_models_available"),
        ],
    }
    if ll_level != 925:
        out["w925"] = record.get("wind925_ms_from_spread")
    out.update({
        "w850": record.get("wind850_ms_from"),
        "w700": record.get("wind700_ms_from"),
        "mslp_z500": record.get("mslp_hpa_z500_gpm"),
        "t": record.get("temperature_925_850_700_c"),
        "rh": record.get("rh925_700_pct"),
        "omega700": record.get("omega700_pa_s"),
        "inv": record.get("inversion_models"),
        "sw": record.get("shortwave_w_m2"),
        "diff": record.get("model_diff_wind_msl_z500_sw"),
        "vent": record.get("ventilation"),
    })
    systems = record.get("nearest_system_signal_source_lat_lon_value_km")
    hour = record.get("issue_relative_hour")
    # System positions evolve slowly; the overview keeps them daily (and at
    # the analyses), the synoptic tool carries every snapshot.
    if systems and isinstance(hour, (int, float)) and int(hour) % 24 == 0:
        # GFS positions of the nearest low/high only; IFS offsets are covered by
        # ``diff`` and every system class/source stays in get_synoptic_evidence.
        daily = {}
        for name in ("low", "high"):
            entry = systems.get(name)
            if isinstance(entry, dict) and entry.get("gfs"):
                daily[name] = entry["gfs"]
            elif isinstance(entry, dict) and entry:
                daily[name] = next(iter(entry.values()))
        if daily:
            out["systems"] = daily
    diff = out.get("diff")
    if isinstance(diff, list) and all(value is None for value in diff):
        out.pop("diff", None)
    mslp_z500 = out.get("mslp_z500")
    if isinstance(mslp_z500, list) and len(mslp_z500) == 2 and mslp_z500[1] is not None:
        out["mslp_z500"] = [mslp_z500[0], round(float(mslp_z500[1]))]
    if isinstance(out.get("omega700"), (int, float)):
        out["omega700"] = round(float(out["omega700"]), 2)
    if isinstance(out.get("sw"), (int, float)):
        out["sw"] = round(float(out["sw"]))
    if hour not in (None, 0) and str(out.get("role", "")).startswith("forecast"):
        out.pop("role", None)
    if isinstance(hour, (int, float)) and int(hour) % 12 != 0:
        # Intermediate six-hourly points keep the transport/ventilation core;
        # the full thermodynamic profile is retained at the 12-hourly skeleton
        # and in get_synoptic_evidence.
        out = {key: out[key] for key in ("h", "ll", "w850", "mslp_z500", "sw", "vent")
               if key in out}
    return out


def _compact_scenario(direction: str, scenario: dict) -> dict:
    candidates = {}
    for city, row in list((scenario.get("issue_time_observation_candidates") or {}).items())[:2]:
        candidates[city] = {
            "distance_km": row.get("distance_km"),
            "travel_h": row.get("estimated_travel_hours_at_mean_target_wind"),
            **{pollutant: row.get(pollutant) for pollutant in ("PM2.5", "PM10", "O3")
               if row.get(pollutant) is not None},
        }
    static = scenario.get("static_directional_context") or {}
    terrain = static.get("terrain")
    emissions = static.get("emissions") or {}
    return {
        "issue_relative_hours": scenario.get("issue_relative_hours"),
        "mean_wind_from_deg": scenario.get("mean_wind_from_deg"),
        "mean_target_wind_speed_ms": scenario.get("mean_target_wind_speed_ms"),
        "reach_km": scenario.get("idealized_advective_reach_km"),
        "candidates": candidates,
        "terrain_barrier_above_city_m": (
            terrain.get("barrier_above_city_m") if isinstance(terrain, dict) else terrain
        ),
        "pm25_nox_sectors": [
            (emissions.get(name) or {}).get("dominant_sectors_within_300km")
            for name in ("PM2.5", "NOx")
        ],
    }


def compact_process_view(document: dict) -> dict:
    """Return the context-budgeted tool view of ``build_process_evidence`` output."""
    if not document.get("available"):
        return document
    view = dict(document)
    trajectory = document.get("weather_trajectory_6h_to_72h_then_12h") or {}
    first = next(iter(trajectory.values()), {})
    static_ll = {key: first.get("terrain_adaptive_low_level", {}).get(key)
                 for key in _STATIC_LOW_LEVEL_KEYS}
    ll_level = static_ll.get("pressure_level_hpa")
    view["terrain_adaptive_levels"] = static_ll
    view["weather_trajectory_6h_to_72h_then_12h"] = {
        valid_bjt: _compact_trajectory_point(record, ll_level)
        for valid_bjt, record in trajectory.items()
    }
    definitions = dict(document.get("definitions") or {})
    definitions.pop("trajectory_tuple_fields", None)
    definitions["trajectory_fields"] = COMPACT_TRAJECTORY_LEGEND
    definitions["ventilation_signal"] = (
        "ll level >=150 m above terrain; very_poor: wind<2 m/s and (inversion or RH>=75); "
        "poor: wind<3 or all models invert; favorable: wind>=5 and no inversion; wind_only / "
        "partial_profile: thermodynamic profile unavailable / no valid inversion pair; else mixed. "
        "Diagnostic flag, not a forecast."
    )
    view["definitions"] = definitions
    view.pop("temporal_profile", None)
    transport = dict(document.get("transport_context") or {})
    scenarios = transport.get("forecast_inflow_scenarios") or {}
    # Keep the sectors that govern the most forecast hours (at most four).
    ranked = sorted(scenarios.items(),
                    key=lambda item: -len(item[1].get("issue_relative_hours") or []))
    transport["forecast_inflow_scenarios"] = {
        direction: _compact_scenario(direction, scenario)
        for direction, scenario in ranked[:2]
    }
    if len(ranked) > 2:
        transport["minor_inflow_sectors"] = {
            direction: scenario.get("issue_relative_hours") for direction, scenario in ranked[2:]
        }
    transport["scenario_note"] = (
        "per future low-level inflow sector: hours it applies, mean wind, idealized "
        "constant-speed reach, up to 2 issue-time upstream candidates [latest, change_6h], "
        "dominant PM2.5/NOx emission sectors in that sector; "
        "a direction/speed/distance screen, not a Lagrangian trajectory"
    )
    view["transport_context"] = transport
    return view
