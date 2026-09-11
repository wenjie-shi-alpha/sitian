#!/usr/bin/env python3
"""非 LLM 的监督学习强基线：只使用 episode 起报时可见的结构化特征。

目的不是替代 agent，而是回答“RL 是否学到了超过普通数值回归的工具综合能力”。
开放形势/成分/火点接入后，这里读取同一份确定性数值摘要；不能把新增信息只给
agent，否则所谓“推理增值”会与信息集增值混淆。
训练只读 national/train，评估只读 val/test；不使用 stratum、truth 派生标签或未来实况特征。
需要 numpy/scikit-learn（不加入 sitian 核心零依赖安装）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
from collections import defaultdict
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.case import CaseBundle  # noqa: E402
from sitian.assess import compute_assessment  # noqa: E402
from sitian.hard_metrics import (  # noqa: E402
    forecast_hard_counts, sum_counts, summarize_hard_counts,
)
from sitian.guidance_bias import load_guidance_bias_index  # noqa: E402
from sitian.open_evidence import (  # noqa: E402
    CAMS_AEROSOL_LEAD_HOURS,
    CAMS_TRACE_GAS_LEAD_HOURS,
    NWP_SNAPSHOTS_PER_ISSUE,
)
from sitian.provenance import case_bundle_snapshot, file_identity  # noqa: E402
from sitian.process_evidence import build_process_evidence  # noqa: E402
from sitian.schema import aqi_standard_for_date, daily_aqi  # noqa: E402
from sitian.scoring import extract_event, reward_spec, score_forecast  # noqa: E402
from sitian.splits import purged_selection_calibration_split  # noqa: E402

TARGETS = ("PM2.5", "PM10", "O3", "SO2", "NO2", "CO")
GUIDANCE_SOURCES = ("cams", "cmaq", "naqp")
GUIDANCE_POLLUTANTS = ("PM2.5", "PM10", "O3")
GUIDANCE_FIELDS = {"PM2.5": "daily_pm25", "PM10": "daily_pm10", "O3": "daily_o3max"}
GUIDANCE_BIAS_SCOPES = (
    "city_season", "city_all_seasons", "national_season", "national_all_seasons",
)
GUIDANCE_BIAS_PATH = Path(os.environ.get(
    "FH_GUIDANCE_BIAS_INDEX",
    REPO_ROOT / "data/interim/guidance_bias_history_v1.json",
))
GUIDANCE_BIAS_INDEX = (
    load_guidance_bias_index(str(GUIDANCE_BIAS_PATH))
    if GUIDANCE_BIAS_PATH.is_file() else None
)
@cache
def _city_coords() -> dict:
    # Feature helpers and test collection do not require the national snapshot.
    # Load coordinates only when constructing actual city rows; missing data
    # still fails rather than inventing coordinates or dropping spatial features.
    path = REPO_ROOT / "data/interim/city_coords.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _software_versions() -> dict[str, str | None]:
    result = {}
    for package in ("numpy", "scikit-learn", "lightgbm", "xgboost"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _stats(values) -> list[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return [np.nan] * 5
    return [float(np.mean(vals)), float(np.max(vals)), float(np.min(vals)),
            float(np.std(vals)), float(vals[-1])]


def _obs_features(bundle: CaseBundle) -> list[float]:
    out = []
    for pollutant in TARGETS:
        series = ((bundle.observations.get(pollutant) or {}).get("series", {})
                  .get(bundle.region, []))
        out += _stats(series[-24:])
        recent = [float(v) for v in series[-24:] if v is not None]
        previous = [float(v) for v in series[-48:-24] if v is not None]
        out.append(float(np.mean(recent) - np.mean(previous)) if recent and previous else np.nan)
        # get_observations can expose the unaggregated 48-hour sequence at
        # stride=1. Preserve it after the robust summaries so the baseline is
        # not forced to infer timing from means unavailable to the agent.
        raw = list(series[-48:])
        out += [np.nan] * (48 - len(raw))
        out += [np.nan if value is None else float(value) for value in raw]
    return out


def _carry_map(values: dict, dates: list[str]) -> dict[str, float]:
    known = [float(values[d]) for d in dates if d in values]
    if not known:
        return {d: np.nan for d in dates}
    last = known[0]
    out = {}
    for day in dates:
        if day in values:
            last = float(values[day])
        out[day] = last
    return out


def _guidance_features(bundle: CaseBundle, dates: list[str]) -> dict[str, list[float]]:
    """Expose every pollution-guidance and time-safe bias signal available to the agent."""
    sources = bundle.guidance.get("sources", {})
    trajectories = {
        (source, pollutant): _carry_map(
            (sources.get(source) or {}).get(GUIDANCE_FIELDS[pollutant], {}), dates
        )
        for source in GUIDANCE_SOURCES for pollutant in GUIDANCE_POLLUTANTS
    }
    bias_by_key = {}
    if GUIDANCE_BIAS_INDEX is not None:
        bias = GUIDANCE_BIAS_INDEX.query(
            region=bundle.region, issue_date=bundle.issue_date,
            horizon=bundle.horizon, window_days=365,
        )
        bias_by_key = {
            (row.get("source"), row.get("pollutant"), int(row.get("lead_days", -1))): row
            for row in bias.get("series", [])
        }

    output = {}
    for lead, day in enumerate(dates, 1):
        values = [
            trajectories[(source, pollutant)][day]
            for source in GUIDANCE_SOURCES for pollutant in GUIDANCE_POLLUTANTS
        ]
        # Give a finite-capacity tree the same explicit disagreement summary
        # shown by get_assessment, while preserving every source value.
        for pollutant in GUIDANCE_POLLUTANTS:
            ensemble = [
                trajectories[(source, pollutant)][day] for source in GUIDANCE_SOURCES
            ]
            finite = [value for value in ensemble if math.isfinite(value)]
            values.extend([
                float(np.mean(finite)) if finite else np.nan,
                float(np.std(finite)) if finite else np.nan,
                float(max(finite) - min(finite)) if finite else np.nan,
            ])
        for source in GUIDANCE_SOURCES:
            for pollutant in GUIDANCE_POLLUTANTS:
                row = bias_by_key.get((source, pollutant, lead), {})
                quantiles = row.get("error_quantiles") or {}
                event = row.get("event") or {}
                scope = row.get("scope")
                values.extend([
                    float(bool(row.get("available"))), row.get("n"),
                    row.get("mean_error"), row.get("median_error"), row.get("mae"),
                    quantiles.get("p10"), quantiles.get("p25"), quantiles.get("p50"),
                    quantiles.get("p75"), quantiles.get("p90"),
                    event.get("miss_rate_given_truth_event"),
                    event.get("false_alarm_rate_given_truth_non_event"),
                    event.get("truth_event_count"),
                    event.get("guidance_event_count"),
                    event.get("miss_count"), event.get("false_alarm_count"),
                    *(float(scope == candidate) for candidate in GUIDANCE_BIAS_SCOPES),
                ])
        output[day] = [np.nan if value is None else float(value) for value in values]
    return output


def _number(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _nested(value: dict | None, *path):
    current = value or {}
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return np.nan
        current = current[key]
    return _number(current)


def _centers(rows, count: int, value_key: str) -> list[float]:
    rows = rows if isinstance(rows, list) else []
    output = []
    for index in range(count):
        row = rows[index] if index < len(rows) else {}
        output.extend([_nested(row, "lat"), _nested(row, "lon"), _nested(row, value_key)])
    return output


def _evidence_features(bundle: CaseBundle, day: str) -> list[float]:
    """Fixed-width projection of exactly the open evidence exposed to the agent."""
    evidence = bundle.evidence or {}
    synoptic = evidence.get("synoptic", {})
    output = []
    system_spec = (("low_centers", "mslp_hpa"), ("high_centers", "mslp_hpa"),
                   ("trough_signals", "z500_zonal_anomaly_gpm"),
                   ("ridge_signals", "z500_zonal_anomaly_gpm"))
    # The interactive policy can inspect all 20 issue-time-legal synoptic
    # snapshots, not only the midnight snapshot corresponding to this target
    # day. Give the fitted baseline the complete trajectory as a fixed-width
    # projection so any agent gain cannot be explained by privileged timing
    # context. Missing rows are padded, never filled from a future analysis.
    source_records = {}
    for source in ("gfs", "ifs"):
        records = sorted(
            (row for row in synoptic.get("sources", {}).get(source, [])
             if row.get("available")),
            key=lambda row: row.get("valid_time", ""),
        )[:NWP_SNAPSHOTS_PER_ISSUE]
        records += [{}] * (NWP_SNAPSHOTS_PER_ISSUE - len(records))
        source_records[source] = records
        for row in records:
            city = row.get("cities", {}).get(bundle.region, {})
            output += [_nested(city, "mslp_hpa"), _nested(city, "z500_gpm")]
            for level in ("925", "850", "700", "500", "200"):
                output += [_nested(city, "wind", level, "u_ms"),
                           _nested(city, "wind", level, "v_ms")]
            for section, levels in (("temperature_c", ("925", "850", "700")),
                                    ("relative_humidity_pct", ("925", "850", "700")),
                                    ("omega_pa_s", ("850", "700", "500"))):
                output += [_nested(city, section, level) for level in levels]
            output += [_nested(city, "stability", "t925_minus_t850_c"),
                       _nested(city, "stability", "t850_minus_t700_c"),
                       _nested(city, "stability", "low_level_inversion_signal"),
                       _nested(city, "surface_downward_shortwave_w_m2")]
            for system_name, value_key in system_spec:
                output += _centers(row.get("systems", {}).get(system_name), 3, value_key)

    trajectory_times = [row.get("valid_time") for row in source_records["gfs"]]
    if not any(trajectory_times):
        trajectory_times = [row.get("valid_time") for row in source_records["ifs"]]
    for valid_time in trajectory_times:
        disagreement = (synoptic.get("cross_model", {}).get(valid_time, {})
                        .get(bundle.region, {})) if valid_time else {}
        transport_wind_difference = _nested(
            disagreement, "transport_wind_vector_diff_ms"
        )
        if not np.isfinite(transport_wind_difference):
            transport_wind_difference = _nested(disagreement, "wind925_vector_diff_ms")
        output += [_nested(disagreement, "mslp_abs_diff_hpa"),
                   _nested(disagreement, "z500_abs_diff_gpm"),
                   transport_wind_difference,
                   _nested(disagreement, "shortwave_abs_diff_w_m2")]

    pollution = evidence.get("pollution", {})
    aerosol = pollution.get("composition", {}).get("aerosol", {})
    aerosol_by_lead = {int(row["lead_hour"]): row for row in aerosol.get("records", [])
                       if row.get("lead_hour") is not None}
    for aerosol_lead in CAMS_AEROSOL_LEAD_HOURS:
        aerosol_row = aerosol_by_lead.get(aerosol_lead, {})
        aerosol_city = aerosol_row.get("cities", {}).get(bundle.region, {})
        for name in ("total", "fine", "dust", "organic_matter", "black_carbon", "sulphate", "nitrate"):
            output.append(_nested(aerosol_city, "aod550", name))
        for name in ("dust", "organic_matter", "black_carbon", "sulphate", "nitrate"):
            output.append(_nested(aerosol_city, "fraction_of_total", name))
        output.append(_nested(aerosol_city, "fine_fraction"))
        dominant = aerosol_city.get("dominant_component")
        output += [float(dominant == name) if dominant is not None else np.nan
                   for name in ("dust", "organic_matter", "black_carbon", "sulphate", "nitrate")]
        for name in ("dust", "organic_matter", "black_carbon", "sulphate", "nitrate"):
            output += _centers(
                aerosol_row.get("regional_source_signals", {}).get(name), 3, "aod550"
            )

    gases = pollution.get("composition", {}).get("trace_gases", {})
    gas_by_lead = {int(row["lead_hour"]): row for row in gases.get("records", [])
                   if row.get("lead_hour") is not None}
    # The agent sees the complete issue-time-legal CAMS forecast trajectory.
    # Give the strong tabular baseline the identical 21 x 3 values rather than
    # silently collapsing the process to one daily snapshot.
    for gas_lead in CAMS_TRACE_GAS_LEAD_HOURS:
        gas_city = gas_by_lead.get(gas_lead, {}).get("cities", {}).get(bundle.region, {})
        output += [_nested(gas_city, name + "_ppbv_approx") for name in
                   ("carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide")]

    # Earth Engine mirrors the same CAMS forecast cycles but exposes total
    # columns, not model-level-137 mixing ratios. Keep the 21 x 3 column fields
    # separate and give them to the baseline exactly as shown to the agent.
    columns = pollution.get("composition", {}).get("column_gases", {})
    columns_by_lead = {int(row["lead_hour"]): row for row in columns.get("records", [])
                       if row.get("lead_hour") is not None}
    for gas_lead in CAMS_TRACE_GAS_LEAD_HOURS:
        column_city = (columns_by_lead.get(gas_lead, {}).get("cities", {})
                       .get(bundle.region, {}))
        output += [_nested(column_city, name + "_kg_m2") for name in
                   ("carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide")]

    fires = pollution.get("fires", {})
    fire_city = fires.get("cities", {}).get(bundle.region, {})
    fire_count = _nested(fires, "active_fire_candidate_detections")
    if not np.isfinite(fire_count):
        fire_count = _nested(fires, "vegetation_fire_detections")  # pre-NRT evidence bundles
    output += [fire_count, _nested(fires, "frp_sum_mw"),
               _nested(fire_city, "within_100km"), _nested(fire_city, "within_300km"),
               _nested(fire_city, "within_800km"), _nested(fire_city, "frp_within_300km_mw"),
               _nested(fire_city, "frp_within_800km_mw"),
               _nested(fire_city, "nearest", "distance_km"),
               _nested(fire_city, "nearest", "bearing_from_city_deg"),
               _nested(fire_city, "nearest", "frp_mw")]
    output += _centers(fires.get("regional_clusters"), 12, "frp_sum_mw")

    context = pollution.get("source_context", {})
    inflow = _nested(context, "inflow_from_deg")
    if not np.isfinite(inflow):
        inflow = _nested(context, "inflow_925_from_deg")
    output += ([math.sin(math.radians(inflow)), math.cos(math.radians(inflow))]
               if np.isfinite(inflow) else [np.nan, np.nan])

    def observation_values(row: dict) -> list[float]:
        values = []
        for pollutant in TARGETS:
            for statistic in ("n_24h", "latest", "mean_24h", "max_24h",
                              "change_6h", "slope_24h_per_hour"):
                values.append(_nested(row, pollutant, statistic))
        return values

    output += observation_values(context.get("target", {}).get(bundle.region, {}))
    for group in ("upwind_candidates", "regional_hotspots"):
        rows = context.get(group, []) if isinstance(context.get(group), list) else []
        for index in range(12):
            row = rows[index] if index < len(rows) else {}
            inflow_angle = _nested(row, "angle_to_inflow_deg")
            if not np.isfinite(inflow_angle):
                inflow_angle = _nested(row, "angle_to_925_inflow_deg")
            output += [_nested(row, "distance_km"), _nested(row, "bearing_from_target_deg"),
                       inflow_angle, *observation_values(row)]

    # Deep source-context queries expose a directionally balanced pool beyond
    # the current-wind top lists. Preserve all bounded slots (<=64 by the
    # attachment contract), including future-wind alternatives.
    transport_pool = (context.get("transport_candidate_pool", [])
                      if isinstance(context.get("transport_candidate_pool"), list) else [])
    for index in range(64):
        row = transport_pool[index] if index < len(transport_pool) else {}
        inflow_angle = _nested(row, "angle_to_inflow_deg")
        if not np.isfinite(inflow_angle):
            inflow_angle = _nested(row, "angle_to_925_inflow_deg")
        output += [_nested(row, "distance_km"), _nested(row, "bearing_from_target_deg"),
                   inflow_angle, *observation_values(row)]

    static = context.get("static", {}).get("target", {})
    terrain = static.get("terrain", {})
    output += [_nested(terrain, "elevation_m"), _nested(terrain, "basin_index_m")]
    for radius in ("100", "300"):
        output += [_nested(terrain, "radius", radius, name) for name in
                   ("mean_elevation_m", "max_elevation_m", "relief_m")]
    for direction in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
        output += [_nested(terrain, "directional_terrain_30_300km", direction, name) for name in
                   ("mean_elevation_m", "max_elevation_m", "barrier_above_city_m")]
    sector_groups = ("shipping", "aviation", "energy", "industry", "ground_transport",
                     "residential", "waste", "agriculture", "other")
    emissions = static.get("anthropogenic_emissions", {})
    for pollutant in ("PM2.5", "PM10", "NOx", "SO2", "CO", "NMVOC", "NH3", "BC", "OC"):
        values = emissions.get(pollutant, {})
        output += [_nested(values, "total_tonnes_per_year_by_radius_km", radius)
                   for radius in ("100", "300", "800")]
        output += [_nested(values, "sector_fraction_within_300km", sector)
                   for sector in sector_groups]
        output += [_nested(values, "directional_total_30_800km_tonnes_per_year", direction)
                   for direction in ("N", "NE", "E", "SE", "S", "SW", "W", "NW")]
    return output


_VENTILATION_SIGNALS = (
    "unavailable", "wind_only", "very_poor", "poor",
    "partial_profile", "mixed", "favorable",
)
_SYSTEM_SIGNALS = ("low", "high", "trough", "ridge")
_PROCESS_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _fixed_numbers(values, count: int) -> list[float]:
    values = values if isinstance(values, (list, tuple)) else []
    return [_number(values[index]) if index < len(values) else np.nan
            for index in range(count)]


def _process_features(bundle: CaseBundle) -> list[float]:
    """Project the deterministic process view exposed to the policy.

    Raw fields remain in :func:`_evidence_features`. This closes a subtler
    fairness gap: a fitted tree should not have to rediscover the harness'
    terrain masking, circular wind means, nearest-system distances,
    ventilation labels, and future-inflow screen while the agent receives
    those transforms explicitly.
    """
    process = build_process_evidence(bundle)
    trajectory = process.get("weather_trajectory_6h_to_72h_then_12h", {})
    rows = [trajectory[key] for key in sorted(trajectory)][:NWP_SNAPSHOTS_PER_ISSUE]
    rows += [{}] * (NWP_SNAPSHOTS_PER_ISSUE - len(rows))
    output: list[float] = []
    for row in rows:
        output += [_number(row.get("issue_relative_hour")), _number(row.get("n_models"))]
        output += _fixed_numbers(row.get("wind925_ms_from_spread"), 3)
        output += _fixed_numbers(row.get("wind850_ms_from"), 2)
        output += _fixed_numbers(row.get("wind700_ms_from"), 2)
        adaptive = row.get("terrain_adaptive_low_level", {})
        output += [
            _number(adaptive.get("terrain_elevation_m")),
            _number(adaptive.get("pressure_level_hpa")),
            _number(adaptive.get("approx_level_height_m")),
            _number(adaptive.get("minimum_clearance_m")),
            *_fixed_numbers(adaptive.get("wind_ms_from_spread"), 3),
            _number(adaptive.get("thermodynamic_level_hpa")),
            _number(adaptive.get("rh_pct")),
            _number(adaptive.get("vertical_motion_level_hpa")),
            _number(adaptive.get("omega_pa_s")),
            _number(adaptive.get("inversion_models")),
            _number(adaptive.get("inversion_models_available")),
        ]
        output += _fixed_numbers(row.get("mslp_hpa_z500_gpm"), 2)
        output += _fixed_numbers(row.get("temperature_925_850_700_c"), 3)
        output += _fixed_numbers(row.get("rh925_700_pct"), 2)
        output += [_number(row.get("omega700_pa_s")),
                   _number(row.get("inversion_models")),
                   _number(row.get("shortwave_w_m2"))]
        output += _fixed_numbers(row.get("model_diff_wind_msl_z500_sw"), 4)
        ventilation = row.get("ventilation")
        output += [float(ventilation == label) if ventilation is not None else np.nan
                   for label in _VENTILATION_SIGNALS]
        nearby = row.get("nearest_system_signal_source_lat_lon_value_km", {})
        for signal in _SYSTEM_SIGNALS:
            for source in ("gfs", "ifs"):
                output += _fixed_numbers((nearby.get(signal) or {}).get(source), 4)

    scenarios = (process.get("transport_context", {})
                 .get("forecast_inflow_scenarios", {}))
    for direction in _PROCESS_DIRECTIONS:
        scenario = scenarios.get(direction, {})
        wind_from = _number(scenario.get("mean_wind_from_deg"))
        issue_hours = [_number(value) for value in scenario.get("issue_relative_hours", [])]
        issue_hours = [value for value in issue_hours if np.isfinite(value)]
        reach = scenario.get("idealized_advective_reach_km", {})
        output += [
            float(bool(scenario)),
            math.sin(math.radians(wind_from)) if np.isfinite(wind_from) else np.nan,
            math.cos(math.radians(wind_from)) if np.isfinite(wind_from) else np.nan,
            _number(scenario.get("mean_target_wind_speed_ms")),
            min(issue_hours) if issue_hours else np.nan,
            max(issue_hours) if issue_hours else np.nan,
            float(len(issue_hours)),
            _number(reach.get("6h")), _number(reach.get("12h")),
            _number(reach.get("24h")),
        ]
        candidates = scenario.get("issue_time_observation_candidates", {})
        ordered = sorted(
            candidates.values(),
            key=lambda candidate: (
                _number(candidate.get("distance_km"))
                if np.isfinite(_number(candidate.get("distance_km"))) else math.inf,
                _number(candidate.get("bearing_from_target_deg"))
                if np.isfinite(_number(candidate.get("bearing_from_target_deg"))) else math.inf,
            ),
        )[:5]
        ordered += [{}] * (5 - len(ordered))
        for candidate in ordered:
            output += [
                _number(candidate.get("distance_km")),
                _number(candidate.get("bearing_from_target_deg")),
                _number(candidate.get("angle_to_inflow_deg")),
                _number(candidate.get("estimated_travel_hours_at_mean_target_wind")),
            ]
            for pollutant in ("PM2.5", "PM10", "O3", "NO2", "CO"):
                output += _fixed_numbers(candidate.get(pollutant), 2)
    return output


_ASSESSMENT_FLAGS = ("cold_air", "stagnation", "rain", "o3_potential")
_CLIMATOLOGY_POSITIONS = ("below_p50", "above_p50", "above_p75", "above_p90")


def _assessment_features(bundle: CaseBundle) -> list[float]:
    """Expose the same deterministic get_assessment flags to the baseline."""
    assessment = compute_assessment(
        bundle.observations, bundle.diagnostics, bundle.guidance,
        bundle.region, bundle.issue_date, (bundle.meta or {}).get("climatology"),
    )
    signals = assessment.get("daily_signals", {})
    output = []
    for day in bundle.forecast_dates():
        row = signals.get(day, {})
        flags = set(row.get("flags") or [])
        output.append(float(day in signals))
        output += [float(flag in flags) for flag in _ASSESSMENT_FLAGS]
        output.append(_number(row.get("wind_change_ms")))
    position = (assessment.get("climatology") or {}).get("pm25_position")
    output += [float(position == value) if position is not None else np.nan
               for value in _CLIMATOLOGY_POSITIONS]
    return output


def _case_rows(bundle: CaseBundle) -> list[dict]:
    dates = bundle.forecast_dates()
    obs = _obs_features(bundle)
    guidance = _guidance_features(bundle, dates)
    guidance_trajectory = [value for target_day in dates for value in guidance[target_day]]
    diag_daily = bundle.diagnostics.get("daily", {})
    last_diag = diag_daily[min(diag_daily)] if diag_daily else {}
    diag_by_day = {}
    for day in dates:
        if day in diag_daily:
            last_diag = diag_daily[day]
        diag_by_day[day] = last_diag
    diagnostic_keys = (
        "wind_speed_ms", "wind_dir_deg", "blh_max_m", "rh_pct",
        "rain_mm", "tmax_c", "cloud_pct",
    )
    diagnostic_trajectory = [
        diag_by_day[target_day].get(key)
        for target_day in dates for key in diagnostic_keys
    ]
    climate = (bundle.meta.get("climatology", {}).get(bundle.issue_date[5:7], {}))
    lat, lon = _city_coords()[bundle.region]
    truth = bundle.truth_daily_full()
    evidence = _evidence_features(bundle, dates[0])
    process = _process_features(bundle)
    assessment = _assessment_features(bundle)
    rows = []
    for lead, day in enumerate(dates, 1):
        month = int(day[5:7])
        angle = 2 * math.pi * (month - 1) / 12
        features = [
            lead, math.sin(angle), math.cos(angle), lat, lon,
            *guidance_trajectory,
            *diagnostic_trajectory,
            (climate.get("pm25") or {}).get("p50"),
            (climate.get("pm25") or {}).get("p75"),
            (climate.get("pm25") or {}).get("p90"),
            (climate.get("o3_8h") or {}).get("p50"),
            (climate.get("o3_8h") or {}).get("p90"),
            *obs,
            *evidence,
            *process,
            *assessment,
        ]
        # Store rows compactly from construction time. Keeping ~10k Python
        # float objects per lead would waste several GiB before fitting.
        features = np.asarray(
            [np.nan if v is None else v for v in features], dtype=np.float32
        )
        rows.append({"case_id": bundle.case_id, "region": bundle.region,
                     "stratum": bundle.meta.get("stratum"),
                     "issue_date": bundle.issue_date,
                     "day": day, "lead": lead, "x": features, "y": truth[day]})
    return rows


def _paths(split: str, manifest_override: str | None = None) -> list[Path]:
    manifest = (REPO_ROOT / manifest_override if manifest_override else
                REPO_ROOT / "data" / "interim" / f"valid_cases_{split}.json")
    if manifest.exists():
        return [REPO_ROOT / p for p in json.loads(manifest.read_text(encoding="utf-8"))]
    return sorted(p for p in (REPO_ROOT / "cases" / "national" / split).iterdir() if p.is_dir())


def load_split(split: str, manifest_override: str | None = None) -> tuple[list[dict], list[Path]]:
    rows = []
    paths = _paths(split, manifest_override)
    for path in paths:
        rows.extend(_case_rows(CaseBundle.load(path)))
    return rows, paths


def _new_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=180, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
    )


def _candidate_builders(seed: int) -> dict[str, Callable[[], object]]:
    """Return the reproducible boosted-tree suite available in this runtime."""
    builders: dict[str, Callable[[], object]] = {
        "hist_gradient_boosting": lambda: _new_model(seed),
    }
    try:
        from lightgbm import LGBMRegressor
        builders["lightgbm"] = lambda: LGBMRegressor(
            n_estimators=400, learning_rate=0.04, num_leaves=31,
            min_child_samples=40, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=8, verbosity=-1,
            deterministic=True, force_col_wise=True,
        )
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor
        builders["xgboost"] = lambda: XGBRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=6,
            min_child_weight=5, subsample=0.85, colsample_bytree=0.8,
            reg_lambda=1.0, objective="reg:squarederror", eval_metric="mae",
            random_state=seed, n_jobs=8, tree_method="hist",
        )
    except ImportError:
        pass
    return builders


def fit_models(
    train: list[dict], seed: int, purge_days: int = 5,
    selection_fraction_of_development: float = 0.125,
):
    """Select boosted trees inside train, then calibrate on a later untouched slice.

    Candidate families never see calibration, val, challenge or test outcomes.
    The selected family is refit on the pre-calibration development fold; a
    temporal embargo prevents rolling five-day target overlap at both internal
    boundaries.
    """
    selection_fit, selection, final_fit, calibration, split_meta = (
        purged_selection_calibration_split(
            train,
            calibration_fraction=0.2,
            selection_fraction_of_development=selection_fraction_of_development,
            purge_days=purge_days,
        )
    )
    x_selection_fit = np.asarray([r["x"] for r in selection_fit], dtype=np.float32)
    x_selection = np.asarray([r["x"] for r in selection], dtype=np.float32)
    x_final_fit = np.asarray([r["x"] for r in final_fit], dtype=np.float32)
    x_cal = np.asarray([r["x"] for r in calibration], dtype=np.float32)
    builders = _candidate_builders(seed)
    models, half_width = {}, {}
    selection_results = {}
    for pollutant in TARGETS:
        y_selection_fit = np.asarray(
            [r["y"][pollutant] for r in selection_fit], dtype=np.float32
        )
        y_selection = np.asarray([r["y"][pollutant] for r in selection], dtype=np.float32)
        candidate_mae = {}
        for name, build in builders.items():
            candidate = build()
            candidate.fit(x_selection_fit, y_selection_fit)
            prediction = np.maximum(0.0, candidate.predict(x_selection))
            candidate_mae[name] = float(np.mean(np.abs(y_selection - prediction)))
        selected = min(candidate_mae, key=lambda name: (candidate_mae[name], name))
        final_model = builders[selected]()
        final_model.fit(
            x_final_fit,
            np.asarray([r["y"][pollutant] for r in final_fit], dtype=np.float32),
        )
        y_cal = np.asarray([r["y"][pollutant] for r in calibration], dtype=np.float32)
        pred_cal = np.maximum(0.0, final_model.predict(x_cal))
        for lead in range(1, 6):
            mask = np.asarray([r["lead"] == lead for r in calibration])
            half_width[(pollutant, lead)] = float(
                np.quantile(np.abs(y_cal[mask] - pred_cal[mask]), 0.8))
        models[pollutant] = final_model
        selection_results[pollutant] = {
            "selected": selected,
            "selection_mae": {
                name: round(value, 6) for name, value in sorted(candidate_mae.items())
            },
        }
    split_meta["candidate_algorithms"] = sorted(builders)
    split_meta["candidate_count"] = len(builders)
    split_meta["selected_by_pollutant"] = selection_results
    split_meta["selection_metric"] = "MAE on train-only chronological selection fold"
    split_meta["calibration_used_for_model_selection"] = False
    return models, half_width, split_meta


def evaluate(rows: list[dict], paths: list[Path], models, half_width) -> dict:
    x = np.asarray([r["x"] for r in rows], dtype=np.float32)
    predictions = {p: np.maximum(0.0, models[p].predict(x)) for p in TARGETS}
    by_case = defaultdict(list)
    for i, row in enumerate(rows):
        by_case[row["case_id"]].append((row, {p: float(predictions[p][i]) for p in TARGETS}))

    results = []
    path_by_id = {p.name: p for p in paths}
    for case_id, items in by_case.items():
        items.sort(key=lambda item: item[0]["lead"])
        bundle = CaseBundle.load(path_by_id[case_id])
        daily, levels, magnitudes = [], {}, {}
        for row, pred in items:
            day, lead = row["day"], row["lead"]
            standard = (bundle.meta or {}).get("aqi_standard") or aqi_standard_for_date(day)
            aq = daily_aqi(pred, standard=standard)
            levels[day], magnitudes[day] = aq["level"], aq["aqi"]
            pm_hw = half_width[("PM2.5", lead)]
            pm10_hw = half_width[("PM10", lead)]
            o3_hw = half_width[("O3", lead)]
            daily.append({
                "date": day,
                "aqi_level": aq["level"],
                "primary_pollutant": aq["primary"][0] if aq["primary"] else None,
                "pm25_range": [round(max(0.0, pred["PM2.5"] - pm_hw), 1),
                               round(pred["PM2.5"] + pm_hw, 1)],
                "pm10_range": [round(max(0.0, pred["PM10"] - pm10_hw), 1),
                               round(pred["PM10"] + pm10_hw, 1)],
                "o3_range": [round(max(0.0, pred["O3"] - o3_hw), 1),
                             round(pred["O3"] + o3_hw, 1)],
            })
        event = extract_event(magnitudes, 3, levels=levels)
        process = ({"has_event": False, "start": None, "peak": None, "end": None}
                   if event is None else
                   {"has_event": True, "start": event[0], "peak": event[1], "end": event[2]})
        forecast = {"issue_date": bundle.issue_date, "region": bundle.region, "daily": daily,
                    "process": process,
                    "evidence": [{"type": "model_guidance",
                                  "claim": "监督基线使用起报时可见的模式、诊断、实况和气候态特征"}]}
        score = score_forecast(
            forecast, bundle.truth_daily_full(), issue_date=bundle.issue_date,
            horizon=bundle.horizon, region=bundle.region,
            # A non-interactive fitted baseline has no tool trajectory; semantic
            # grounding is therefore inapplicable and must abstain, not receive
            # synthetic credit or a penalty.
            tools_called=None,
            aqi_standard=(bundle.meta or {}).get("aqi_standard"),
        )
        hard_counts = forecast_hard_counts(
            forecast, bundle.truth_daily_full(),
            aqi_standard=(bundle.meta or {}).get("aqi_standard"),
        )
        results.append({
            "case_id": case_id,
            "issue_date": bundle.issue_date,
            "stratum": items[0][0]["stratum"],
            # All model-vs-baseline comparisons use the evidence-neutral
            # forecast outcome score.  The training composite is retained only
            # for auditing; for this non-interactive baseline they are normally
            # identical because grounding/evidence abstain.
            "composite": score.outcome_composite,
            "outcome_composite": score.outcome_composite,
            "training_composite": score.composite,
            "components": score.components,
            "forecast": forecast,
            "interval_diagnostics": score.details["interval_diagnostics"],
            "hard_counts": hard_counts,
        })

    def mean(values):
        return round(sum(values) / len(values), 4) if values else None

    def ratio(numerator, denominator):
        return round(numerator / denominator, 4) if denominator else None

    def hard_metrics(subset: list[dict]) -> dict:
        return summarize_hard_counts(sum_counts([
            row["hard_counts"] for row in subset
        ]))

    strata = {}
    for stratum in sorted({r["stratum"] for r in results}):
        subset = [r for r in results if r["stratum"] == stratum]
        strata[stratum] = {
            "n": len(subset),
            "composite": mean([r["composite"] for r in subset]),
            "hard_metrics": hard_metrics(subset),
        }
    components = {}
    for component in next(iter(results))["components"]:
        vals = [r["components"][component] for r in results if r["components"][component] is not None]
        components[component] = mean(vals)
    return {"n": len(results), "composite": mean([r["composite"] for r in results]),
            "macro_stratum": mean([v["composite"] for v in strata.values()]),
            "components": components, "hard_metrics": hard_metrics(results),
            "strata": strata, "rows": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--calibration-purge-days", type=int, default=5)
    parser.add_argument("--selection-fraction-of-development", type=float, default=0.125)
    parser.add_argument(
        "--train-manifest",
        default="data/interim/train_without_winter_or_spatial_holdout.json",
    )
    parser.add_argument("--val-manifest", default="data/interim/valid_cases_val.json")
    parser.add_argument("--test-manifest", default="data/interim/valid_cases_test.json")
    parser.add_argument("--challenge-manifest", default="data/interim/challenge_winter.json")
    parser.add_argument("--spatial-ood-val-manifest",
                        default="data/interim/spatial_ood_val.json")
    parser.add_argument("--spatial-ood-test-manifest",
                        default="data/interim/spatial_ood_test.json")
    parser.add_argument("--out", default="data/interim/eval_tabular.json")
    args = parser.parse_args()
    train, train_paths = load_split("train", args.train_manifest)
    print(f"train rows={len(train)}")
    feature_count = len(train[0]["x"])
    if any(len(row["x"]) != feature_count for row in train):
        raise RuntimeError("tabular feature projection is not fixed-width")
    models, widths, calibration_meta = fit_models(
        train, args.seed, args.calibration_purge_days,
        args.selection_fraction_of_development,
    )
    report = {"artifact_type": "strong_tabular_baseline",
              "baseline_version": "2.2.0",
              "model": "train-only selected boosted-tree suite", "seed": args.seed,
              "software_versions": _software_versions(),
              "feature_contract": (
                  "base-visible-v2 + open-evidence-v1; complete 20-step GFS/IFS and "
                  "5-step aerosol trajectories, all 21-step gas/column trajectories, "
                  "bounded transport pool, all CAMS/CMAQ/NAQP values, cross-model "
                  "disagreement, and the same deterministic process/assessment views "
                  "and time-safe guidance-bias aggregates available to the agent"
              ),
              "feature_information_set": {
                  "full_trajectory_repeated_for_each_target_lead": True,
                  "nwp_sources": ["gfs", "ifs"],
                  "nwp_snapshots_per_source": NWP_SNAPSHOTS_PER_ISSUE,
                  "aerosol_snapshots": len(CAMS_AEROSOL_LEAD_HOURS),
                  "trace_gas_snapshots": len(CAMS_TRACE_GAS_LEAD_HOURS),
                  "column_gas_snapshots": len(CAMS_TRACE_GAS_LEAD_HOURS),
                  "transport_candidate_pool_slots": 64,
                  "raw_observation_hours_per_pollutant": 48,
                  "guidance_and_bias_target_slots": 5,
                  "surface_diagnostic_target_slots": 5,
                  "cams_pollution_guidance_source_days": 4,
                  "cams_surface_diagnostic_source_days": 4,
                  "lead5_pollution_guidance_policy": (
                      "no CAMS value; deterministic carry projection for fitted baseline, "
                      "with lead index and bias availability retained"
                  ),
                  "lead5_surface_diagnostic_policy": (
                      "no CAMS value; deterministic carry projection for fitted baseline, "
                      "while assessment-presence remains false"
                  ),
                  "lead5_nwp_weather_available": True,
                  "deterministic_process_view_projection": True,
                  "deterministic_assessment_projection": True,
                  "matrix_dtype": "float32",
                  "missing_value_policy": "NaN; no future-analysis or truth fill",
                  "truth_or_stratum_features": False,
              },
              "guidance_bias_feature_contract": {
                  "available": GUIDANCE_BIAS_INDEX is not None,
                  "sources": list(GUIDANCE_SOURCES),
                  "pollutants": list(GUIDANCE_POLLUTANTS),
                  "strict_query_time_gate": "verification_available_at < issue_time",
                  "window_days": 365,
              },
              "feature_count": feature_count,
              "train_cases": len(train) // 5,
              "interval": "symmetric 80% residual quantile from a later, purged, model-held-out calibration slice",
              "calibration_split": calibration_meta,
              "provenance": {
                  "reward": reward_spec(),
                  "baseline_implementation": file_identity(
                      Path(__file__), relative_to=REPO_ROOT),
                  "hard_metric_implementation": file_identity(
                      REPO_ROOT / "src/sitian/hard_metrics.py", relative_to=REPO_ROOT),
                  "guidance_bias_implementation": file_identity(
                      REPO_ROOT / "src/sitian/guidance_bias.py", relative_to=REPO_ROOT),
                  "split_implementation": file_identity(
                      REPO_ROOT / "src/sitian/splits.py", relative_to=REPO_ROOT),
                  "scoring_implementation": file_identity(
                      REPO_ROOT / "src/sitian/scoring.py", relative_to=REPO_ROOT),
                  "schema_implementation": file_identity(
                      REPO_ROOT / "src/sitian/schema.py", relative_to=REPO_ROOT),
                  "standards_manifest": file_identity(
                      REPO_ROOT / "references" / "standards" / "manifest.json",
                      relative_to=REPO_ROOT),
                  "guidance_bias_index": (
                      file_identity(GUIDANCE_BIAS_PATH, relative_to=REPO_ROOT)
                      if GUIDANCE_BIAS_PATH.is_file() else None
                  ),
                  "case_manifests": {
                      name: file_identity(REPO_ROOT / path, relative_to=REPO_ROOT)
                      for name, path in {
                          "train": args.train_manifest,
                          "val": args.val_manifest,
                          "test": args.test_manifest,
                          "challenge_winter": args.challenge_manifest,
                          "spatial_ood_val": args.spatial_ood_val_manifest,
                          "spatial_ood_test": args.spatial_ood_test_manifest,
                      }.items()
                  },
                  "case_input_snapshots": {
                      "train": case_bundle_snapshot(
                          train_paths, relative_to=REPO_ROOT, include_truth=True
                      ),
                  },
              },
              "results": {}}
    eval_specs = {
        "val": ("val", args.val_manifest),
        "challenge_winter": ("train", args.challenge_manifest),
        "test": ("test", args.test_manifest),
        "spatial_ood_val": ("val", args.spatial_ood_val_manifest),
        "spatial_ood_test": ("test", args.spatial_ood_test_manifest),
    }
    for name, (split, manifest) in eval_specs.items():
        rows, paths = load_split(split, manifest)
        report["provenance"]["case_input_snapshots"][name] = case_bundle_snapshot(
            paths, relative_to=REPO_ROOT, include_truth=True
        )
        report["results"][name] = evaluate(rows, paths, models, widths)
        calibration_targets = {
            (r["region"], r["day"]) for r in train
            if r["issue_date"] >= calibration_meta["calibration_start_issue_date"]
        }
        training_targets = {(r["region"], r["day"]) for r in train}
        eval_targets = {(r["region"], r["day"]) for r in rows}
        report["calibration_split"].setdefault("target_overlap_with_evaluations", {})[name] = len(
            calibration_targets & eval_targets)
        report["calibration_split"].setdefault("training_target_overlap_with_evaluations", {})[
            name] = len(training_targets & eval_targets)
        print(name, {k: v for k, v in report["results"][name].items() if k != "rows"})
    out = REPO_ROOT / args.out
    report["runtime_memory"] = {
        "train_feature_matrix_mib_float32": round(
            len(train) * feature_count * np.dtype(np.float32).itemsize / 1024 ** 2, 1
        ),
        "process_peak_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "platform_note": "ru_maxrss is reported in KiB on the Linux evaluation host",
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
