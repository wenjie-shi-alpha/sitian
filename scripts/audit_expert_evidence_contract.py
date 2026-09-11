#!/usr/bin/env python3
"""Audit whether every national case can support the expert decision chain."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.case import CaseBundle  # noqa: E402
from sitian.provenance import file_identity, identity_matches_file  # noqa: E402
from sitian.open_evidence import (  # noqa: E402
    CAMS_AEROSOL_LEAD_HOURS,
    CAMS_TRACE_GAS_LEAD_HOURS,
    NWP_SNAPSHOTS_PER_ISSUE,
    PRESSURE_LEVEL_APPROX_HEIGHT_M,
    TERRAIN_CLEARANCE_M,
    lowest_pressure_level_above_terrain,
)
from sitian.process_evidence import build_process_evidence  # noqa: E402
from sitian.schema import (  # noqa: E402
    SCHEMA_VERSION,
    aqi_standard_for_date,
    daily_aqi,
)
from sitian.scoring import REWARD_VERSION, reward_spec  # noqa: E402

POLLUTANTS = ("PM2.5", "PM10", "O3", "SO2", "NO2", "CO")
PRESSURE_LEVELS = ("925", "850", "700", "500", "200")
DEFAULT_MANIFESTS = (
    "data/interim/valid_cases_train.json",
    "data/interim/valid_cases_val.json",
    "data/interim/valid_cases_test.json",
)


def _present(value, *path) -> bool:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return current is not None


def _synoptic_complete(bundle: CaseBundle) -> bool:
    sources = (bundle.evidence or {}).get("synoptic", {}).get("sources", {})
    terrain = (((bundle.evidence or {}).get("pollution", {}).get("source_context", {})
                .get("static", {}).get("target", {}).get("terrain", {})))
    elevation_m = terrain.get("elevation_m")
    expected_transport_level = lowest_pressure_level_above_terrain(elevation_m)
    if expected_transport_level is None:
        return False
    for source in ("gfs", "ifs"):
        rows = sources.get(source, [])
        if len(rows) != NWP_SNAPSHOTS_PER_ISSUE:
            return False
        for row in rows:
            city = row.get("cities", {}).get(bundle.region, {})
            validity = city.get("pressure_level_above_terrain", {})
            expected_validity = {
                level: PRESSURE_LEVEL_APPROX_HEIGHT_M[int(level)]
                    >= float(elevation_m) + TERRAIN_CLEARANCE_M
                for level in PRESSURE_LEVELS
            }
            if (
                not row.get("available")
                or city.get("transport_pressure_level_hpa") != expected_transport_level
                or validity != expected_validity
                or not all(
                    (not expected_validity[level])
                    or _present(city, "wind", level, "speed_ms")
                    for level in PRESSURE_LEVELS
                )
            ):
                return False
            if not all(
                (not expected_validity[level])
                or _present(city, "temperature_c", level)
                for level in ("925", "850", "700")
            ):
                return False
            if not all(
                (not expected_validity[level])
                or _present(city, "relative_humidity_pct", level)
                for level in ("925", "850", "700")
            ):
                return False
            if not all(
                (not expected_validity[level])
                or _present(city, "omega_pa_s", level)
                for level in ("850", "700", "500")
            ):
                return False
            if int(row.get("step_hour", 0)) and not row.get("radiation_available"):
                return False
            systems = row.get("systems", {})
            if not all(systems.get(name) for name in
                       ("low_centers", "high_centers", "trough_signals", "ridge_signals")):
                return False
    return True


def _composition_complete(bundle: CaseBundle, composition: dict) -> bool:
    specifications = {
        "aerosol": (
            CAMS_AEROSOL_LEAD_HOURS,
            ("aod550", "dominant_component"),
        ),
        "trace_gases": (
            CAMS_TRACE_GAS_LEAD_HOURS,
            ("carbon_monoxide_ppbv_approx", "nitrogen_dioxide_ppbv_approx",
             "sulphur_dioxide_ppbv_approx"),
        ),
        "column_gases": (
            CAMS_TRACE_GAS_LEAD_HOURS,
            ("carbon_monoxide_kg_m2", "nitrogen_dioxide_kg_m2",
             "sulphur_dioxide_kg_m2"),
        ),
    }
    for name, (expected_hours, required_fields) in specifications.items():
        block = composition.get(name, {})
        rows = block.get("records", [])
        if block.get("available") is not True:
            return False
        if tuple(row.get("lead_hour") for row in rows) != tuple(expected_hours):
            return False
        for row in rows:
            if row.get("issue_relative_hour") != row.get("lead_hour") - 12:
                return False
            city = row.get("cities", {}).get(bundle.region, {})
            if not city or not all(city.get(field) is not None for field in required_fields):
                return False
    return True


def _capabilities(bundle: CaseBundle) -> dict[str, bool]:
    evidence = bundle.evidence or {}
    pollution = evidence.get("pollution", {})
    context = pollution.get("source_context", {})
    target = context.get("target", {}).get(bundle.region, {})
    static_target = context.get("static", {}).get("target", {})
    composition = pollution.get("composition", {})
    process = build_process_evidence(bundle)
    trajectory = process.get("weather_trajectory_6h_to_72h_then_12h", {})
    observations = all(
        pollutant in bundle.observations and pollutant in target for pollutant in POLLUTANTS
    )
    composition_complete = _composition_complete(bundle, composition)
    guidance_complete = any(
        all(values.get(field) for field in ("daily_pm25", "daily_pm10", "daily_o3max"))
        for values in bundle.guidance.get("sources", {}).values()
    )
    transport_pool = context.get("transport_candidate_pool")
    expected_transport_level = lowest_pressure_level_above_terrain(
        static_target.get("terrain", {}).get("elevation_m")
    )
    terrain_adaptive = all(
        row.get("terrain_adaptive_low_level", {}).get("pressure_level_hpa")
            == expected_transport_level
        and len(row.get("terrain_adaptive_low_level", {}).get(
            "wind_ms_from_spread", [])) == 3
        for row in trajectory.values()
    )
    inflow_scenarios = process.get("transport_context", {}).get(
        "forecast_inflow_scenarios", {}
    )
    screened_transport = bool(inflow_scenarios) and all(
        row.get("screening_note")
        and "not a Lagrangian trajectory" in row["screening_note"]
        and row.get("idealized_advective_reach_km")
        and all(
            # Calm/zero-wind screens legitimately have no finite travel time.
            # Require transparent abstention rather than a fabricated speed.
            "estimated_travel_hours_at_mean_target_wind" in candidate
            for candidate in row.get("issue_time_observation_candidates", {}).values()
        )
        for row in inflow_scenarios.values()
    )
    return {
        "pollution_initial_state": observations,
        "synoptic_system_evolution": _synoptic_complete(bundle),
        "horizontal_transport": (
            isinstance(transport_pool, list)
            and context.get("inflow_pressure_level_hpa") == expected_transport_level
            and terrain_adaptive
            and screened_transport
        ),
        "vertical_dispersion": (
            bool(trajectory)
            and all(row.get("terrain_adaptive_low_level", {}).get(
                        "pressure_level_hpa") is not None
                    and row.get("terrain_adaptive_low_level", {}).get(
                        "vertical_motion_level_hpa") is not None
                    and row.get("terrain_adaptive_low_level", {}).get(
                        "omega_pa_s") is not None
                    for row in trajectory.values())
            and bool(process.get("daily_surface_dispersion"))
        ),
        "terrain_adaptive_pressure_levels": terrain_adaptive,
        "pollutant_mechanism_evidence": (
            composition_complete and pollution.get("fires", {}).get("available") is True
            and pollution.get("fires", {}).get("asof_nrt_source_available") is True
            and bool(context.get("static", {}).get("target"))
        ),
        "guidance_and_calibration_inputs": guidance_complete,
        "process_timing_view": len(trajectory) == NWP_SNAPSHOTS_PER_ISSUE,
        "forecast_output_contract": SCHEMA_VERSION == "0.6.4" and REWARD_VERSION == "0.8.1",
        "time_gate": not bundle.audit_time_gate(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-root", type=Path, default=Path("cases/national"))
    parser.add_argument("--manifests", nargs="+", default=list(DEFAULT_MANIFESTS),
                        help="exact valid-case manifests to audit; directory extras are ignored")
    parser.add_argument("--guidance-bias", type=Path,
                        default=Path("data/interim/guidance_bias_history_v1.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path,
                        default=Path("data/interim/expert_evidence_contract_audit.json"))
    args = parser.parse_args()
    manifest_paths = [
        path if path.is_absolute() else REPO_ROOT / path
        for path in map(Path, args.manifests)
    ]
    case_entries = [
        entry
        for manifest_path in manifest_paths
        for entry in json.loads(manifest_path.read_text(encoding="utf-8"))
    ]
    duplicate_case_paths = len(case_entries) - len(set(case_entries))
    paths = sorted(REPO_ROOT / entry for entry in dict.fromkeys(case_entries))
    missing_case_paths = [str(path) for path in paths if not (path / "case.json").is_file()]
    if missing_case_paths:
        raise FileNotFoundError(f"manifest case paths missing: {missing_case_paths[:5]}")
    if args.limit is not None:
        paths = paths[:args.limit]
    counts = Counter()
    primary_memberships = {
        split: Counter() for split in ("train", "val", "test")
    }
    tied_primary_days = Counter()
    failures = []
    for index, path in enumerate(paths, 1):
        bundle = CaseBundle.load(path)
        split = path.parent.name
        configured_standard = (bundle.meta or {}).get("aqi_standard")
        for day, concentrations in bundle.truth_daily_full().items():
            standard = (
                configured_standard.get(day)
                if isinstance(configured_standard, dict)
                else configured_standard
            ) or aqi_standard_for_date(day)
            truth_aqi = daily_aqi(concentrations, standard=standard)
            primary_memberships[split].update(truth_aqi["primary"])
            tied_primary_days[split] += int(len(truth_aqi["primary"]) > 1)
        result = _capabilities(bundle)
        counts.update(name for name, passed in result.items() if passed)
        missing = [name for name, passed in result.items() if not passed]
        if missing and len(failures) < 50:
            failures.append({"case_id": bundle.case_id, "missing": missing})
        if index % 500 == 0:
            print(f"[{index}/{len(paths)}] audited", flush=True)

    bias = json.loads(args.guidance_bias.read_text(encoding="utf-8"))
    bias_summary = bias.get("summary", {})
    purged_train_manifest = (
        REPO_ROOT / "data/interim/train_without_winter_or_spatial_holdout.json"
    )
    bias_manifests = (bias.get("provenance") or {}).get("manifests") or []
    bias_available = bool(bias_summary.get("rows") or bias_summary.get("records"))
    bias_train_only = bool(
        len(bias_manifests) == 1
        and identity_matches_file(
            bias_manifests[0], purged_train_manifest, relative_to=REPO_ROOT
        )
        and (bias.get("split_policy") or {}).get("main_experiment")
            == "train_only_city_day_purged"
        and (bias.get("split_policy") or {}).get("evaluation_truth_updates_index") is False
    )
    required = (
        "pollution_initial_state", "synoptic_system_evolution", "horizontal_transport",
        "vertical_dispersion", "pollutant_mechanism_evidence",
        "terrain_adaptive_pressure_levels",
        "guidance_and_calibration_inputs", "process_timing_view",
        "forecast_output_contract", "time_gate",
    )
    capability_rows = {
        name: {"passed_cases": counts[name], "cases": len(paths),
               "fraction": counts[name] / len(paths) if paths else 0.0,
               "required_for_preflight": name in required}
        for name in required
    }
    report = {
        "artifact_type": "expert_evidence_contract_audit",
        "audit_version": "1.5.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cases": len(paths),
        "schema_version": SCHEMA_VERSION,
        "reward": reward_spec(),
        "case_selection": {
            "policy": "exact_valid_case_manifests",
            "manifest_entries": len(case_entries),
            "unique_case_paths": len(paths),
            "duplicate_case_paths": duplicate_case_paths,
        },
        "provenance": {
            "audit_implementation": file_identity(Path(__file__), relative_to=REPO_ROOT),
            "process_evidence_implementation": file_identity(
                REPO_ROOT / "src/sitian/process_evidence.py", relative_to=REPO_ROOT),
            "open_evidence_implementation": file_identity(
                REPO_ROOT / "src/sitian/open_evidence.py", relative_to=REPO_ROOT),
            "guidance_bias_implementation": file_identity(
                REPO_ROOT / "src/sitian/guidance_bias.py", relative_to=REPO_ROOT),
            "case_manifests": [
                file_identity(path, relative_to=REPO_ROOT) for path in manifest_paths
            ],
        },
        "capabilities": capability_rows,
        "guidance_bias_index": {
            "available": bias_available,
            "train_only_city_day_purged": bias_train_only,
            "artifact": file_identity(args.guidance_bias, relative_to=REPO_ROOT),
            "source_manifests": bias_manifests,
            "summary": bias_summary,
        },
        "forecast_output_scope_audit": {
            "allowed_primary_pollutants": list(POLLUTANTS),
            "probabilistic_interval_heads": ["PM2.5", "PM10", "O3"],
            "unrepresented_primary_interval_heads": ["SO2", "NO2", "CO"],
            "truth_primary_memberships_by_split": {
                split: dict(sorted(primary_memberships[split].items()))
                for split in ("train", "val", "test")
            },
            "tied_primary_days_by_split": dict(tied_primary_days),
            "scientific_boundary": (
                "AQI level and six-class primary pollutant are scored for all pollutants; "
                "calibrated concentration intervals are currently requested only for the "
                "three dominant pollutants. NO2/CO primary memberships occur only in train "
                "and SO2 is absent in the current valid pool, so this is an explicit output-"
                "scope limitation rather than an unreported validation advantage."
            ),
        },
        "known_nonblocking_research_gaps": {
            "independent_pollution_mechanism_labels": False,
            "time_safe_historical_analog_index": False,
            "previous_operational_forecast_archive": False,
            "lagrangian_transport_trajectory": False,
            "time_varying_emissions_and_control_measures": False,
            "surface_speciated_aerosol_observations": False,
            "ensemble_spread_beyond_deterministic_gfs_ifs": False,
            "so2_no2_co_probabilistic_interval_heads": False,
            "free_text_causal_claim_verification": False,
            "decision_relevance_of_verified_citations": False,
            "all_fire_sensor_days_have_archived_nrt": False,
            "trained_policy_interval_calibration": False,
            "trained_policy_spatial_ood_evaluation": False,
            "prospective_2026_27_winter_evidence": False,
        },
        "failure_samples": failures,
    }
    report["preflight_contract_passed"] = (
        bool(paths) and duplicate_case_paths == 0 and bias_available and bias_train_only
        and all(row["fraction"] == 1.0 for row in capability_rows.values())
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failure_samples"},
                     ensure_ascii=False, indent=1))
    print(f"-> {args.out}")
    return 0 if report["preflight_contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
