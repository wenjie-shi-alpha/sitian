#!/usr/bin/env python3
"""RL 门禁：preflight 只判断能否安全开小训，graduation 判断训后能否扩到 8B。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.provenance import (  # noqa: E402
    case_bundle_snapshot_matches, file_identity, score_implementation_matches,
)
from sitian.open_evidence import (  # noqa: E402
    CAMS_AEROSOL_LEAD_HOURS,
    CAMS_TRACE_GAS_LEAD_HOURS,
    EVIDENCE_VERSION,
    NWP_SNAPSHOTS_PER_ISSUE,
    NWP_TEMPORAL_PROFILE_VERSION,
)
from sitian.scoring import reward_spec  # noqa: E402


def _load(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def _gpu() -> dict:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().splitlines()[0]
        name, memory = raw.rsplit(",", 1)
        return {"name": name.strip(), "memory_mib": int(memory.strip())}
    except Exception:
        return {"name": None, "memory_mib": None}


def _comparison(path: str | None) -> dict | None:
    if not path or not (REPO_ROOT / path).exists():
        return None
    return _load(path)


def _identity_valid(identity: dict | None) -> bool:
    if not identity or not identity.get("path"):
        return False
    stored = Path(identity["path"])
    path = stored if stored.is_absolute() else REPO_ROOT / stored
    if not path.exists():
        return False
    actual = file_identity(path)
    return (
        actual["sha256"] == identity.get("sha256")
        and actual["bytes"] == identity.get("bytes")
    )


def _identity_json(identity: dict | None) -> dict:
    if not _identity_valid(identity):
        return {}
    stored = Path(identity["path"])
    path = stored if stored.is_absolute() else REPO_ROOT / stored
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _standards_archive_audit() -> dict:
    """Verify that the frozen official documents still match their manifest."""
    required_standards = {
        "HJ 633-2012", "HJ 663-2013", "HJ 633-2026", "HJ 663-2026",
    }
    transition_date = "2026-03-01"
    manifest_path = REPO_ROOT / "references/standards/manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"pass": False, "error": str(exc), "documents": []}
    rows = []
    for document in manifest.get("documents") or []:
        checks = {}
        for file_key, hash_key, require_bytes in (
            ("file", "sha256", True),
            ("review_image", "review_image_sha256", False),
        ):
            path = manifest_path.parent / str(document.get(file_key) or "")
            try:
                identity = file_identity(path)
                checks[file_key] = bool(
                    identity["sha256"] == document.get(hash_key)
                    and (not require_bytes or identity["bytes"] == document.get("bytes"))
                )
            except OSError:
                checks[file_key] = False
        metadata_complete = all(document.get(key) for key in (
            "standard", "title", "effective_date", "official_url",
            "relevant_location", "implemented_claim",
        ))
        standard = document.get("standard")
        official_source = str(document.get("official_url") or "").startswith(
            "https://www.mee.gov.cn/"
        )
        if standard in {"HJ 633-2012", "HJ 663-2013"}:
            transition_metadata = document.get("superseded_on") == transition_date
        else:
            transition_metadata = document.get("effective_date") == transition_date
        rows.append({
            "standard": standard,
            "checks": checks,
            "metadata_complete": metadata_complete,
            "official_source": official_source,
            "transition_metadata": transition_metadata,
            "pass": (
                metadata_complete and official_source and transition_metadata
                and all(checks.values())
            ),
        })
    standards = [row["standard"] for row in rows]
    exact_required_set = len(standards) == len(set(standards)) and set(standards) == required_standards
    return {
        "pass": exact_required_set and all(row["pass"] for row in rows),
        "required_standards": sorted(required_standards),
        "exact_required_set": exact_required_set,
        "transition_date": transition_date,
        "manifest": file_identity(manifest_path, relative_to=REPO_ROOT),
        "documents": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="data/interim/probe_open_evidence_v079_n50_g8.json")
    ap.add_argument("--tabular-baseline",
                    default="data/interim/eval_tabular_open_evidence_v079.json")
    ap.add_argument("--evidence-coverage",
                    default="data/interim/open_evidence_coverage.json")
    ap.add_argument("--expert-contract",
                    default="data/interim/expert_evidence_contract_audit.json")
    ap.add_argument("--spatial-ood-audit",
                    default="data/interim/spatial_ood_audit.json")
    ap.add_argument("--reward-contract",
                    default="data/interim/reward_contract_audit.json")
    ap.add_argument("--verl-runtime-contract",
                    default="data/interim/verl_runtime_contract.json")
    ap.add_argument("--guidance-comparison", default=None)
    ap.add_argument("--event-comparison", default=None)
    ap.add_argument("--event-hard-comparison", default=None)
    ap.add_argument("--event-tabular-comparison", default=None)
    ap.add_argument("--event-tabular-hard-comparison", default=None)
    ap.add_argument("--challenge-comparison", default=None)
    ap.add_argument("--tabular-comparison", default=None)
    ap.add_argument("--tabular-hard-comparison", default=None)
    ap.add_argument("--spatial-ood-comparison", default=None)
    ap.add_argument("--spatial-ood-hard-comparison", default=None)
    ap.add_argument("--reward-sensitivity",
                    default="data/interim/reward_weight_sensitivity_v079.json")
    ap.add_argument("--evidence-dose-response",
                    default="data/interim/evidence_dose_response_trained.json")
    ap.add_argument("--training-smoke", default="data/interim/training_smoke.json")
    ap.add_argument("--min-smoke-steps", type=int, default=50)
    ap.add_argument("--min-probe-cases", type=int, default=50)
    ap.add_argument("--min-usable-group-rate", type=float, default=0.70,
                    help="动态过滤后可用 group 的预算门槛，不是科学效果门槛")
    ap.add_argument("--min-outcome-usable-group-rate", type=float, default=0.50,
                    help="有效提交中 outcome_composite 有变化的 group 预算门槛")
    ap.add_argument("--min-event-decision-usable-group-rate", type=float, default=0.50,
                    help="event case 中 level/event/turning 有变化的 group 预算门槛")
    ap.add_argument("--min-event-groups", type=int, default=8)
    ap.add_argument("--min-decision-sigma", type=float, default=0.001,
                    help="排除只有四位小数舍入噪声的实质变化门槛")
    ap.add_argument("--expected-rollout-temperature", type=float, default=1.0,
                    help="variance probe must match the train-time sampling regime")
    ap.add_argument("--max-void-turn-rate", type=float, default=0.20,
                    help=("rollouts with >=1 text-only assistant turn; 0.20 for the mandatory "
                          "thinking-on arm (in-think malformed tool JSON is a zero-reward policy "
                          "behaviour that veRL also scores 0), monitored again during training"))
    ap.add_argument("--min-semantic-grounding-rate", type=float, default=0.50)
    ap.add_argument("--min-process-tool-use-rate", type=float, default=0.80)
    ap.add_argument("--max-prompt-tokens", type=int, default=28_000,
                    help=("single-call prompt budget; leaves about 8k tokens for tool/answer "
                          "generation under the current 32,768-token rollout context"))
    ap.add_argument("--out", default="data/interim/rl_preflight.json")
    args = ap.parse_args()

    quality = _load("data/interim/data_quality_audit.json")
    challenge = _load("data/interim/challenge_winter_audit.json")
    spatial_ood = _load(args.spatial_ood_audit)
    probe = _load(args.probe)
    tabular = _load(args.tabular_baseline)
    evidence_coverage = _load(args.evidence_coverage)
    expert_contract = _load(args.expert_contract)
    reward_contract = _load(args.reward_contract)
    runtime_contract_path = REPO_ROOT / args.verl_runtime_contract
    runtime_contract = (_load(args.verl_runtime_contract)
                        if runtime_contract_path.is_file() else {})
    current_reward = reward_spec()
    valid_rows = [r for r in probe["rows"] if r.get("submitted")]
    groups: dict[str, list[float]] = defaultdict(list)
    valid_outcomes: dict[str, list[float]] = defaultdict(list)
    valid_decisions: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    group_truth_event: dict[str, bool] = {}
    for row in probe["rows"]:
        groups[row["case_id"]].append(float(row.get("reward", 0)))
        if isinstance(row.get("truth_has_event"), bool):
            group_truth_event[row["case_id"]] = row["truth_has_event"]
        if row.get("submitted") and row.get("outcome_composite") is not None:
            valid_outcomes[row["case_id"]].append(float(row["outcome_composite"]))
            for component in ("level", "event", "turning"):
                value = (row.get("components") or {}).get(component)
                if value is not None:
                    valid_decisions[row["case_id"]][component].append(float(value))
    zero_groups = sum(max(v) == min(v) for v in groups.values())
    usable_groups = len(groups) - zero_groups
    usable_rate = usable_groups / len(groups) if groups else 0.0
    outcome_sigmas = {
        case_id: (statistics.pstdev(values) if len(values) >= 2 else 0.0)
        for case_id, values in valid_outcomes.items()
    }
    outcome_usable_groups = sum(
        outcome_sigmas.get(case_id, 0.0) > args.min_decision_sigma
        for case_id in groups
    )
    outcome_usable_rate = outcome_usable_groups / len(groups) if groups else 0.0
    truth_event_audit_complete = (
        len(group_truth_event) == len(groups)
        and all(isinstance(row.get("truth_has_event"), bool) for row in probe["rows"])
    )
    event_group_ids = [case_id for case_id in groups
                       if group_truth_event.get(case_id) is True]
    event_decision_usable_groups = sum(
        any(
            len(values) >= 2
            and statistics.pstdev(values) > args.min_decision_sigma
            for values in valid_decisions.get(case_id, {}).values()
        )
        for case_id in event_group_ids
    )
    event_decision_usable_rate = (
        event_decision_usable_groups / len(event_group_ids) if event_group_ids else 0.0
    )
    observed_temperature = probe.get("temperature")
    runtime_sampling = runtime_contract.get("training_sampling_contract") or {}
    observed_group_size = int(probe.get("group_size") or 0)
    expected_training_group_size = int(runtime_sampling.get("rollout_group_size") or 0)
    sampling_group_size_matches = (
        expected_training_group_size >= 2
        and observed_group_size == expected_training_group_size
    )
    sampling_temperature_matches = (
        isinstance(observed_temperature, (int, float))
        and abs(float(observed_temperature) - args.expected_rollout_temperature) <= 1e-9
    )
    submit_rate = len(valid_rows) / len(probe["rows"]) if probe["rows"] else 0.0
    first_submit_rate = (
        sum(bool(row.get("submitted")) and int(row.get("invalid_submits", 0)) == 0
            for row in probe["rows"]) / len(probe["rows"])
        if probe["rows"] else 0.0
    )
    # Rows that ended in a transport/runtime error (e.g. context overflow) are
    # zero-reward endpoint failures with no assistant turn to audit; they count
    # as audited (no void turn) rather than as an incomplete audit.
    void_turn_audit_complete = bool(probe["rows"]) and all(
        isinstance(row.get("void_assistant_turns"), int) or row.get("error")
        for row in probe["rows"]
    )
    void_turns = sum(
        int(row.get("void_assistant_turns", 0)) for row in probe["rows"]
    )
    rollouts_with_void_turn = sum(
        int(row.get("void_assistant_turns", 0) > 0) for row in probe["rows"]
    )
    void_turn_rate = (
        rollouts_with_void_turn / len(probe["rows"]) if probe["rows"] else 0.0
    )
    usage_rows = [r.get("usage") for r in probe["rows"] if r.get("submitted")]
    usage_complete = bool(usage_rows) and all(
        isinstance(usage, dict) and isinstance(usage.get("max_prompt_tokens"), int)
        for usage in usage_rows
    )
    prompt_maxima = sorted(
        int(usage["max_prompt_tokens"]) for usage in usage_rows
        if isinstance(usage, dict) and isinstance(usage.get("max_prompt_tokens"), int)
    )
    max_single_prompt = prompt_maxima[-1] if prompt_maxima else None
    # A single over-fetching rollout is a zero-reward policy behaviour, not a
    # harness defect; the budget gate is judged on the 95th percentile while
    # the maximum and the over-budget fraction stay reported.
    p95_single_prompt = (
        prompt_maxima[min(len(prompt_maxima) - 1, int(0.95 * (len(prompt_maxima) - 1)))]
        if prompt_maxima else None
    )
    runtime_context_limit = int(runtime_sampling.get("max_model_len") or 0)
    completion_reserve = int(
        runtime_sampling.get("minimum_completion_reserve_tokens") or 4096
    )
    effective_prompt_limit = min(
        args.max_prompt_tokens,
        max(0, runtime_context_limit - completion_reserve),
    )

    raw = quality["raw_validity"]
    data_quality_fail_fields = (
        "truth_day_failures", "truth_field_failures",
        "six_pollutant_observation_failures", "observation_unit_failures",
        "observation_shape_failures", "late_observation_cases",
    )
    observation_contract_clean = all(
        quality.get("splits", {}).get(split, {}).get(name) == 0
        for split in ("train", "val", "test")
        for name in data_quality_fail_fields
    )
    raw_validity_complete = not (raw.get("missing_raw_files") or [])
    truth_event_population_audited = bool(
        quality.get("audit_version") == "1.4.0"
        and (quality.get("evaluation_risks") or {}).get(
            "event_population_definition"
        )
        and all(
            isinstance(
                (quality.get("splits", {}).get(split) or {}).get(
                    "valid_truth_event_cases"
                ),
                int,
            )
            and sum(
                ((quality.get("splits", {}).get(split) or {}).get(
                    "valid_truth_event_cases_by_sampling_stratum"
                ) or {}).values()
            ) == (quality.get("splits", {}).get(split) or {}).get(
                "valid_truth_event_cases"
            )
            for split in ("train", "val", "test")
        )
    )
    dataset_identity_clean = bool(
        quality.get("cases") == quality.get("unique_case_ids")
        and not (quality.get("duplicate_case_ids") or [])
        and _identity_valid(quality.get("audit_implementation"))
    )
    reward_specs = {
        "current": current_reward,
        "probe": (probe.get("provenance") or {}).get("reward") or {},
        "tabular": (tabular.get("provenance") or {}).get("reward") or {},
    }
    reward_identities = {
        name: {"version": spec.get("version"), "config_sha256": spec.get("config_sha256")}
        for name, spec in reward_specs.items()
    }
    version_match = (
        len({(v["version"], v["config_sha256"]) for v in reward_identities.values()}) == 1
        and all(v["version"] and v["config_sha256"] for v in reward_identities.values())
    )
    current_score_identities = {
        "scoring_implementation": file_identity(
            REPO_ROOT / "src/sitian/scoring.py", relative_to=REPO_ROOT
        ),
        "schema_implementation": file_identity(
            REPO_ROOT / "src/sitian/schema.py", relative_to=REPO_ROOT
        ),
    }
    score_identities = {
        "probe": probe.get("provenance") or {},
        "tabular": tabular.get("provenance") or {},
        "reward_contract": reward_contract,
    }
    score_implementation_match = all(
        score_implementation_matches(identities, REPO_ROOT)
        for identities in score_identities.values()
    )
    reward_contract_identity = reward_contract.get("reward") or {}
    reward_contract_ready = (
        reward_contract.get("passed") is True
        and score_implementation_matches(reward_contract, REPO_ROOT)
        and reward_contract_identity.get("version") == current_reward.get("version")
        and reward_contract_identity.get("config_sha256")
            == current_reward.get("config_sha256")
        and bool(reward_contract.get("checks"))
        and all(row.get("pass") is True for row in reward_contract["checks"].values())
    )
    runtime_reward = runtime_contract.get("reward") or {}
    runtime_contract_ready = (
        runtime_contract.get("passed") is True
        and runtime_reward.get("version") == current_reward.get("version")
        and runtime_reward.get("config_sha256") == current_reward.get("config_sha256")
        and bool(runtime_contract.get("checks"))
        and all((runtime_contract.get("checks") or {}).values())
        and bool(runtime_contract.get("inputs"))
        and all(_identity_valid(identity)
                for identity in (runtime_contract.get("inputs") or {}).values())
    )
    valid_manifest_identities = raw["valid_case_manifests"]
    manifests_valid = all(
        _identity_valid(valid_manifest_identities[s])
        for s in ("train", "val", "test")
    )
    valid_manifest_entries = {}
    if manifests_valid:
        for split in ("train", "val", "test"):
            stored = Path(valid_manifest_identities[split]["path"])
            path = stored if stored.is_absolute() else REPO_ROOT / stored
            valid_manifest_entries[split] = json.loads(path.read_text(encoding="utf-8"))
    manifest_case_ids = [
        Path(entry).name for entries in valid_manifest_entries.values() for entry in entries
    ]
    manifest_contents_clean = bool(
        manifests_valid
        and all(
            isinstance(valid_manifest_entries.get(split), list)
            and len(valid_manifest_entries[split])
                == valid_manifest_identities[split].get("cases")
            and len(valid_manifest_entries[split]) == len(set(valid_manifest_entries[split]))
            and all((REPO_ROOT / entry / "case.json").is_file()
                    and (REPO_ROOT / entry / "truth.json").is_file()
                    for entry in valid_manifest_entries[split])
            for split in ("train", "val", "test")
        )
        and len(manifest_case_ids) == len(set(manifest_case_ids))
        and len(manifest_case_ids) == quality.get("cases") - raw.get("affected_cases", 0)
    )
    challenge_artifacts_valid = all(_identity_valid(identity) for identity in (
        challenge.get("source_manifest"),
        (challenge.get("challenge") or {}).get("manifest"),
        (challenge.get("train_after_purge") or {}).get("manifest"),
    ))
    spatial_sources_valid = bool(spatial_ood.get("sources")) and all(
        _identity_valid(identity) for identity in spatial_ood.get("sources", {}).values()
    )
    spatial_outputs_valid = all(_identity_valid(identity) for identity in (
        (spatial_ood.get("train") or {}).get("manifest"),
        (spatial_ood.get("spatial_ood_val") or {}).get("manifest"),
        (spatial_ood.get("spatial_ood_test") or {}).get("manifest"),
    ))
    spatial_integrity = spatial_ood.get("integrity") or {}
    heldout_by_cluster = spatial_ood.get("heldout_by_cluster") or {}
    spatial_clean = bool(
        spatial_ood.get("artifact_type") == "spatial_ood_split_audit"
        and spatial_sources_valid
        and spatial_outputs_valid
        and (spatial_ood.get("selection_policy") or {}).get(
            "model_or_reward_results_used") is False
        and (spatial_ood.get("selection_policy") or {}).get(
            "validation_or_test_outcomes_used") is False
        and (spatial_ood.get("selection_policy") or {}).get(
            "power_rule_uses_outcomes") is False
        and (spatial_ood.get("train_only_regime_construction") or {}).get(
            "source_period") == "original_train_manifest_only"
        and len(heldout_by_cluster) == 8
        and len(set(heldout_by_cluster.values())) == 8
        and set(spatial_integrity) >= {
            "train_heldout_city_overlap",
            "train_winter_challenge_forecast_city_day_overlap",
        }
        and all(value == 0 for value in spatial_integrity.values())
        and (spatial_ood.get("spatial_ood_val") or {}).get("issue_dates", 0) >= 30
        and (spatial_ood.get("spatial_ood_test") or {}).get("issue_dates", 0) >= 30
    )
    tabular_provenance = tabular.get("provenance") or {}
    tabular_manifest_identities = tabular_provenance.get("case_manifests") or {}
    tabular_input_snapshots = tabular_provenance.get("case_input_snapshots") or {}
    tabular_implementation_keys = (
        "baseline_implementation", "hard_metric_implementation",
        "guidance_bias_implementation", "split_implementation",
    )
    tabular_artifacts_valid = (
        set(tabular_manifest_identities) >= {
            "train", "val", "test", "challenge_winter",
            "spatial_ood_val", "spatial_ood_test",
        }
        and all(_identity_valid(identity) for identity in tabular_manifest_identities.values())
        and _identity_valid(tabular_provenance.get("guidance_bias_index"))
        and all(_identity_valid(tabular_provenance.get(key))
                for key in tabular_implementation_keys)
        and set(tabular_input_snapshots) >= {
            "train", "val", "test", "challenge_winter",
            "spatial_ood_val", "spatial_ood_test",
        }
        and all(case_bundle_snapshot_matches(snapshot, REPO_ROOT)
                for snapshot in tabular_input_snapshots.values())
    )
    standards_identities = {
        "audit": quality.get("standards_manifest"),
        "probe": (probe.get("provenance") or {}).get("standards_manifest"),
        "tabular": (tabular.get("provenance") or {}).get("standards_manifest"),
    }
    standards_archive = _standards_archive_audit()
    standards_valid = (
        all(_identity_valid(identity) for identity in standards_identities.values())
        and standards_archive["pass"] is True
    )
    probe_provenance = probe.get("provenance") or {}
    probe_manifest_valid = _identity_valid(probe_provenance.get("case_manifest"))
    probe_implementation_keys = (
        "probe_script", "agent_driver", "environment_implementation",
        "process_evidence_implementation", "guidance_bias_implementation",
    )
    probe_artifacts_valid = (
        probe_manifest_valid
        and all(_identity_valid(probe_provenance.get(key))
                for key in probe_implementation_keys)
        and case_bundle_snapshot_matches(
            probe_provenance.get("case_input_snapshot"), REPO_ROOT
        )
    )
    challenge_integrity = challenge.get("integrity") or {}
    challenge_clean = (
        set(challenge_integrity) >= {
            "challenge_train_case_id_overlap",
            "challenge_train_forecast_city_day_overlap",
        }
        and all(v == 0 for v in challenge_integrity.values())
    )
    calibration_audit = tabular.get("calibration_split", {})
    calibration_overlaps = calibration_audit.get("target_overlap_with_evaluations", {})
    train_eval_overlaps = calibration_audit.get("training_target_overlap_with_evaluations", {})
    selection_audit = calibration_audit.get("model_selection") or {}
    selected_by_pollutant = calibration_audit.get("selected_by_pollutant") or {}
    expected_pollutants = {"PM2.5", "PM10", "O3", "NO2", "SO2", "CO"}
    candidate_algorithms = set(calibration_audit.get("candidate_algorithms") or [])
    tabular_information = tabular.get("feature_information_set") or {}
    strong_tabular_suite_ready = (
        tabular.get("baseline_version") == "2.2.0"
        and tabular_information.get(
            "full_trajectory_repeated_for_each_target_lead") is True
        and tabular_information.get("nwp_sources") == ["gfs", "ifs"]
        and tabular_information.get("nwp_snapshots_per_source") == NWP_SNAPSHOTS_PER_ISSUE
        and tabular_information.get("aerosol_snapshots") == len(CAMS_AEROSOL_LEAD_HOURS)
        and tabular_information.get("trace_gas_snapshots") == len(CAMS_TRACE_GAS_LEAD_HOURS)
        and tabular_information.get("column_gas_snapshots") == len(CAMS_TRACE_GAS_LEAD_HOURS)
        and tabular_information.get("transport_candidate_pool_slots") == 64
        and tabular_information.get("raw_observation_hours_per_pollutant") == 48
        and tabular_information.get("guidance_and_bias_target_slots") == 5
        and tabular_information.get("surface_diagnostic_target_slots") == 5
        and tabular_information.get("cams_pollution_guidance_source_days") == 4
        and tabular_information.get("cams_surface_diagnostic_source_days") == 4
        and tabular_information.get("lead5_nwp_weather_available") is True
        and tabular_information.get("deterministic_process_view_projection") is True
        and tabular_information.get("deterministic_assessment_projection") is True
        and tabular_information.get("matrix_dtype") == "float32"
        and tabular_information.get("truth_or_stratum_features") is False
        and calibration_audit.get("candidate_count", 0) >= 3
        and candidate_algorithms >= {
            "hist_gradient_boosting", "lightgbm", "xgboost",
        }
        and calibration_audit.get("calibration_used_for_model_selection") is False
        and (tabular.get("guidance_bias_feature_contract") or {}).get(
            "available") is True
        and set((tabular.get("guidance_bias_feature_contract") or {}).get(
            "sources") or []) == {"cams", "cmaq", "naqp"}
        and set((tabular.get("guidance_bias_feature_contract") or {}).get(
            "pollutants") or []) == {"PM2.5", "PM10", "O3"}
        and (tabular.get("guidance_bias_feature_contract") or {}).get(
            "strict_query_time_gate") == "verification_available_at < issue_time"
        and set(selected_by_pollutant) == expected_pollutants
        and all((tabular.get("software_versions") or {}).get(package)
                for package in ("numpy", "scikit-learn", "lightgbm", "xgboost"))
        and all(
            row.get("selected") in candidate_algorithms
            and set(row.get("selection_mae") or {}) == candidate_algorithms
            for row in selected_by_pollutant.values()
        )
    )
    calibration_clean = (
        calibration_audit.get("fit_calibration_city_day_overlap") == 0
        and selection_audit.get("fit_calibration_city_day_overlap") == 0
        and calibration_audit.get(
            "model_selection_calibration_city_day_overlap") == 0
        and set(calibration_overlaps) >= {
            "val", "challenge_winter", "test", "spatial_ood_val", "spatial_ood_test"
        }
        and all(v == 0 for v in calibration_overlaps.values())
        and set(train_eval_overlaps) >= {
            "val", "challenge_winter", "test", "spatial_ood_val", "spatial_ood_test"
        }
        and all(v == 0 for v in train_eval_overlaps.values())
    )
    derived_coverage = evidence_coverage.get("derived_by_issue", {})
    required_coverage = evidence_coverage.get("required_by_issue", {})
    coverage_provenance = evidence_coverage.get("provenance") or {}
    coverage_derivations = coverage_provenance.get("derivation_implementations") or {}
    coverage_artifacts_valid = (
        _identity_valid(coverage_provenance.get("audit_implementation"))
        and _identity_valid(
            coverage_provenance.get("open_evidence_contract_implementation")
        )
        and set(coverage_derivations) == {
            "synoptic", "composition", "spatial_observations", "fires",
            "static_context", "case_attachment",
        }
        and all(_identity_valid(identity) for identity in coverage_derivations.values())
    )
    evidence_ready = (
        evidence_coverage.get("ready_for_full_attachment") is True
        and evidence_coverage.get("contract_version") == EVIDENCE_VERSION
        and (evidence_coverage.get("nwp_temporal_profile") or {}).get("version")
            == NWP_TEMPORAL_PROFILE_VERSION
        and _identity_valid(evidence_coverage.get("source_manifest"))
        and coverage_artifacts_valid
        and evidence_coverage.get("case_attachments", {}).get("fraction") == 1.0
        and all(derived_coverage.get(name, {}).get("fraction") == 1.0
                for name in ("synoptic", "composition", "spatial_observations", "fires"))
        and all(evidence_coverage.get("raw", {}).get("nwp", {}).get(name, {})
                .get("fraction") == 1.0 for name in ("gfs", "ifs"))
        and all(evidence_coverage.get("raw", {}).get("nwp_radiation", {}).get(name, {})
                .get("fraction") == 1.0 for name in ("gfs", "ifs"))
        and all(
            required_coverage.get(name, {}).get("fraction") == 1.0
            for name in (
                "cams_aerosol_speciation",
                "cams_model_level_137_trace_gases",
                "cams_total_column_trace_gases",
            )
        )
    )
    expert_bias = expert_contract.get("guidance_bias_index") or {}
    guidance_bias_provenance = (
        _identity_json(expert_bias.get("artifact")).get("provenance") or {}
    )
    expert_bias_ready = (
        expert_bias.get("available") is True
        and expert_bias.get("train_only_city_day_purged") is True
        and _identity_valid(expert_bias.get("artifact"))
        and bool(expert_bias.get("source_manifests"))
        and all(_identity_valid(identity)
                for identity in expert_bias.get("source_manifests", []))
        and all(_identity_valid(guidance_bias_provenance.get(key))
                for key in ("builder", "implementation"))
    )
    expert_provenance = expert_contract.get("provenance") or {}
    expert_manifest_identities = expert_provenance.get("case_manifests") or []
    expert_artifacts_valid = (
        len(expert_manifest_identities) == 3
        and all(_identity_valid(identity) for identity in expert_manifest_identities)
        and all(_identity_valid(expert_provenance.get(key)) for key in (
            "audit_implementation", "process_evidence_implementation",
            "open_evidence_implementation", "guidance_bias_implementation",
        ))
        and (expert_contract.get("case_selection") or {}).get("policy")
            == "exact_valid_case_manifests"
        and (expert_contract.get("case_selection") or {}).get("duplicate_case_paths") == 0
    )
    expert_contract_ready = (
        expert_contract.get("preflight_contract_passed") is True
        and expert_contract.get("cases") == sum(
            raw["valid_case_manifests"][split]["cases"] for split in ("train", "val", "test")
        )
        and all(row.get("fraction") == 1.0 for row in
                (expert_contract.get("capabilities") or {}).values()
                if row.get("required_for_preflight"))
        and expert_bias_ready and expert_artifacts_valid
    )
    preflight = {
        "data_and_reward_integrity": {
            "pass": (
                manifests_valid and manifest_contents_clean
                and dataset_identity_clean and raw_validity_complete
                and challenge_artifacts_valid and spatial_clean
                and tabular_artifacts_valid
                and probe_artifacts_valid and standards_valid and challenge_clean
                and version_match and calibration_clean and strong_tabular_suite_ready
                and evidence_ready
                and score_implementation_match
                and observation_contract_clean and expert_contract_ready
                and reward_contract_ready and truth_event_population_audited
            ),
            "valid_cases": {s: raw["valid_case_manifests"][s]["cases"]
                            for s in ("train", "val", "test")},
            "invalid_cases_quarantined": raw["affected_cases"],
            "six_pollutant_observation_contract": {
                "pass": (observation_contract_clean and raw_validity_complete
                         and dataset_identity_clean and manifest_contents_clean),
                "dataset_cases": quality.get("cases"),
                "unique_case_ids": quality.get("unique_case_ids"),
                "missing_raw_files": raw.get("missing_raw_files"),
                "valid_manifest_contents_clean": manifest_contents_clean,
                "by_split": {
                    split: {name: quality.get("splits", {}).get(split, {}).get(name)
                            for name in data_quality_fail_fields}
                    for split in ("train", "val", "test")
                },
            },
            "truth_event_population_audit": {
                "pass": truth_event_population_audited,
                "definition": (quality.get("evaluation_risks") or {}).get(
                    "event_population_definition"
                ),
                "by_split": {
                    split: {
                        "cases": (quality.get("splits", {}).get(split) or {}).get(
                            "valid_truth_event_cases"
                        ),
                        "by_sampling_stratum": (
                            quality.get("splits", {}).get(split) or {}
                        ).get("valid_truth_event_cases_by_sampling_stratum"),
                        "sampling_event_stratum_recall": (
                            quality.get("splits", {}).get(split) or {}
                        ).get("valid_sampling_event_stratum_recall"),
                    }
                    for split in ("train", "val", "test")
                },
                "note": (
                    "event graduation and event probes use hidden truth flags; the mutually "
                    "exclusive sampling stratum is reported only as a coverage diagnostic"
                ),
            },
            "artifact_hashes_valid": {
                "data_quality_audit": dataset_identity_clean,
                "valid_manifests": manifests_valid and manifest_contents_clean,
                "challenge_manifests": challenge_artifacts_valid,
                "spatial_ood_manifests": spatial_sources_valid and spatial_outputs_valid,
                "tabular_manifests": tabular_artifacts_valid,
                "probe_manifest_and_implementations": probe_artifacts_valid,
                "standards_manifests": standards_valid,
                "standards_archive": standards_archive,
            },
            "train_after_winter_challenge_purge": challenge["train_after_purge"],
            "winter_challenge_integrity": challenge_integrity,
            "train_after_all_purges": spatial_ood.get("train"),
            "spatial_ood_integrity": {
                "pass": spatial_clean,
                "artifact": file_identity(REPO_ROOT / args.spatial_ood_audit,
                                          relative_to=REPO_ROOT),
                "selection_policy": spatial_ood.get("selection_policy"),
                "heldout_by_cluster": heldout_by_cluster,
                "validation": spatial_ood.get("spatial_ood_val"),
                "test": spatial_ood.get("spatial_ood_test"),
                "integrity": spatial_integrity,
            },
            "reward_identities": reward_identities,
            "score_implementation_identities": {
                "pass": score_implementation_match,
                "current": current_score_identities,
                **score_identities,
            },
            "reward_contract_audit": {
                "pass": reward_contract_ready,
                "artifact": file_identity(REPO_ROOT / args.reward_contract,
                                          relative_to=REPO_ROOT),
                "checks": reward_contract.get("checks"),
                "scientific_boundary": reward_contract.get("scientific_boundary"),
            },
            "tabular_calibration_integrity": {
                "pass": calibration_clean and strong_tabular_suite_ready,
                "fit_calibration_city_day_overlap": calibration_audit.get(
                    "fit_calibration_city_day_overlap"),
                "selection_fit_selection_city_day_overlap": selection_audit.get(
                    "fit_calibration_city_day_overlap"),
                "selection_calibration_city_day_overlap": calibration_audit.get(
                    "model_selection_calibration_city_day_overlap"),
                "calibration_target_overlap_with_evaluations": calibration_overlaps,
                "training_target_overlap_with_evaluations": train_eval_overlaps,
                "strong_baseline_suite": {
                    "pass": strong_tabular_suite_ready,
                    "minimum_candidates": 3,
                    "candidate_algorithms": calibration_audit.get(
                        "candidate_algorithms"),
                    "calibration_used_for_model_selection": calibration_audit.get(
                        "calibration_used_for_model_selection"),
                    "selected_by_pollutant": selected_by_pollutant,
                    "guidance_bias_feature_contract": tabular.get(
                        "guidance_bias_feature_contract"),
                    "feature_information_set": tabular_information,
                },
            },
            "open_evidence_coverage": {
                "pass": evidence_ready,
                "implementation_hashes_valid": coverage_artifacts_valid,
                "artifact": file_identity(REPO_ROOT / args.evidence_coverage,
                                          relative_to=REPO_ROOT),
                "issue_dates": evidence_coverage.get("issue_dates"),
                "contract_version": evidence_coverage.get("contract_version"),
                "nwp_temporal_profile": evidence_coverage.get("nwp_temporal_profile"),
                "source_manifest": evidence_coverage.get("source_manifest"),
                "derived_by_issue": derived_coverage,
                "case_attachments": evidence_coverage.get("case_attachments"),
                "required_by_issue": required_coverage,
            },
            "expert_evidence_contract": {
                "pass": expert_contract_ready,
                "artifact": file_identity(REPO_ROOT / args.expert_contract,
                                          relative_to=REPO_ROOT),
                "capabilities": expert_contract.get("capabilities"),
                "artifact_hashes_valid": expert_artifacts_valid,
                "known_nonblocking_research_gaps": expert_contract.get(
                    "known_nonblocking_research_gaps"),
                "guidance_bias_index": expert_bias,
            },
        },
        "format_and_trajectory": {
            "pass": (
                len(groups) >= args.min_probe_cases
                and submit_rate >= 0.95
                and first_submit_rate >= 0.95
                and void_turn_audit_complete
                and void_turn_rate <= args.max_void_turn_rate
            ),
            "eventual_submit_rate": round(submit_rate, 4),
            "first_submit_valid_rate": round(first_submit_rate, 4),
            "invalid_submit_attempts": sum(int(row.get("invalid_submits", 0))
                                           for row in probe["rows"]),
            "void_turn_audit_complete": void_turn_audit_complete,
            "void_assistant_turns": void_turns,
            "rollouts_with_void_turn": rollouts_with_void_turn,
            "void_turn_rate": round(void_turn_rate, 4),
            "cases": len(groups),
            "threshold": (
                f"cases >= {args.min_probe_cases}, eventual_submit_rate >= 0.95, "
                "first_submit_valid_rate >= 0.95, complete per-turn audit, and "
                f"rollout_void_turn_rate <= {args.max_void_turn_rate}"
            ),
            "evidence_grade": ("confirmed" if len(groups) >= args.min_probe_cases else
                               "provisional_small_sample"),
        },
        "context_budget": {
            "pass": (
                len(groups) >= args.min_probe_cases
                and usage_complete
                and p95_single_prompt is not None
                and runtime_context_limit > completion_reserve
                and p95_single_prompt <= effective_prompt_limit
            ),
            "cases": len(groups),
            "submitted_rollouts": len(usage_rows),
            "usage_complete": usage_complete,
            "max_single_call_prompt_tokens": max_single_prompt,
            "p95_single_call_prompt_tokens": p95_single_prompt,
            "rollouts_over_budget": sum(
                int(value > effective_prompt_limit) for value in prompt_maxima
            ),
            "gate_statistic": "p95 of per-rollout max single-call prompt tokens",
            "threshold": {
                "minimum_cases": args.min_probe_cases,
                "configured_maximum_single_call_prompt_tokens": args.max_prompt_tokens,
                "train_runtime_context_tokens": runtime_context_limit,
                "minimum_completion_reserve_tokens": completion_reserve,
                "effective_maximum_single_call_prompt_tokens": effective_prompt_limit,
            },
            "note": (
                "max_prompt_tokens is checked against the smaller of the explicit budget and "
                "the audited train runtime context minus a completion reserve; cumulative "
                "prompt_tokens is retained only for rollout cost accounting"
            ),
        },
        "sampling_efficiency": {
            "pass": (len(groups) >= args.min_probe_cases
                     and sampling_temperature_matches
                     and sampling_group_size_matches
                     and usable_rate >= args.min_usable_group_rate
                     and outcome_usable_rate >= args.min_outcome_usable_group_rate
                     and truth_event_audit_complete
                     and len(event_group_ids) >= args.min_event_groups
                     and event_decision_usable_rate
                        >= args.min_event_decision_usable_group_rate),
            "groups": len(groups),
            "zero_variance_groups": zero_groups,
            "usable_groups_after_filter": usable_groups,
            "usable_group_rate": round(usable_rate, 4),
            "outcome_variable_groups_valid_only": outcome_usable_groups,
            "outcome_usable_group_rate_valid_only": round(outcome_usable_rate, 4),
            "event_groups": len(event_group_ids),
            "truth_event_audit_complete": truth_event_audit_complete,
            "event_group_definition": "truth daily AQI level >=4; independent of sampling stratum",
            "event_decision_variable_groups_valid_only": event_decision_usable_groups,
            "event_decision_usable_group_rate_valid_only": round(
                event_decision_usable_rate, 4
            ),
            "effective_rollout_rate": round(
                sum(len(v) for v in groups.values() if max(v) != min(v)) /
                len(probe["rows"]), 4) if probe["rows"] else 0.0,
            "sampling_temperature": {
                "observed": observed_temperature,
                "expected_train_time": args.expected_rollout_temperature,
                "matches": sampling_temperature_matches,
            },
            "sampling_group_size": {
                "observed_probe": observed_group_size,
                "expected_train_time": expected_training_group_size,
                "matches": sampling_group_size_matches,
            },
            "budget_threshold": {
                "minimum_cases": args.min_probe_cases,
                "minimum_usable_group_rate": args.min_usable_group_rate,
                "minimum_outcome_usable_group_rate_valid_only":
                    args.min_outcome_usable_group_rate,
                "minimum_event_groups": args.min_event_groups,
                "minimum_event_decision_usable_group_rate_valid_only":
                    args.min_event_decision_usable_group_rate,
                "minimum_decision_sigma": args.min_decision_sigma,
            },
            "note": (
                "total reward, forecast outcome, and event decisions are audited separately; "
                "grounding-only variance cannot certify numerical RL learnability"
            ),
        },
    }
    grounding_summary = (probe.get("summary") or {}).get("semantic_grounding") or {}
    semantic_rate = float(grounding_summary.get("rate") or 0.0)
    full_grounding_rate = float(
        grounding_summary.get("full_credit_rollout_rate") or 0.0
    )
    process_use_rate = float((probe.get("summary") or {}).get("process_tool_use_rate") or 0.0)
    preflight["structured_evidence_grounding"] = {
        "pass": (len(groups) >= args.min_probe_cases
                 and full_grounding_rate >= args.min_semantic_grounding_rate
                 and process_use_rate >= args.min_process_tool_use_rate),
        "cases": len(groups),
        "semantic_grounding": grounding_summary,
        "semantic_assertion_rate": semantic_rate,
        "full_grounding_rollout_rate": full_grounding_rate,
        "process_tool_use_rate": process_use_rate,
        "thresholds": {
            "minimum_cases": args.min_probe_cases,
            "minimum_full_grounding_rollout_rate": args.min_semantic_grounding_rate,
            "minimum_process_tool_use_rate": args.min_process_tool_use_rate,
        },
        "failure_action": "run a small format/grounding tutor stage; do not weaken semantic verification",
    }

    smoke_path = REPO_ROOT / args.training_smoke
    smoke = _load(args.training_smoke) if smoke_path.exists() else {"success": False, "steps": 0}
    # Production training intentionally lives in an isolated veRL environment;
    # requiring packages in the harness/control Python would reintroduce ABI
    # conflicts with its CPU torch.  A successful smoke records the packages it
    # actually imported, so either source is acceptable and remains auditable.
    smoke_packages = smoke.get("packages") or {}
    stack = {
        name: (
            importlib.util.find_spec(name) is not None
            or bool((smoke_packages.get(name) or {}).get("available"))
        )
        for name in ("verl", "peft")
    }
    stack["trl"] = importlib.util.find_spec("trl") is not None
    trainable_model = (
        os.environ.get("FH_TRAINABLE_MODEL")
        or ((smoke.get("model") or {}).get("path"))
    )
    model_available = bool(trainable_model and Path(trainable_model).exists())
    gpu = _gpu()
    is_wsl = "microsoft" in platform.release().lower()
    local_8b_supported = bool(gpu["memory_mib"] and gpu["memory_mib"] >= 70000 and not is_wsl)
    smoke_checks = {
        "run_succeeded": bool(smoke.get("success")),
        "minimum_steps_reached": smoke.get("steps", 0) >= args.min_smoke_steps,
        "rollout_policy_sync_verified": bool(smoke.get("rollout_policy_sync_verified")),
        "tool_tokens_loss_masked": bool(smoke.get("tool_tokens_loss_masked")),
        "checkpoint_reload_verified": bool(smoke.get("checkpoint_reload_verified")),
        "finite_losses": bool(smoke.get("finite_losses")),
        "sampling_contract_matches_formal_probe": (
            smoke.get("rollout_group_size") == observed_group_size
            and smoke.get("sampling_temperature") == observed_temperature
        ),
    }
    smoke_passed = all(smoke_checks.values())
    preflight["trainable_stack_and_hardware_smoke"] = {
        "pass": (runtime_contract_ready and stack["verl"] and stack["peft"]
                 and model_available and smoke_passed),
        "native_runtime_contract": {
            "pass": runtime_contract_ready,
            "path": args.verl_runtime_contract,
            "exists": runtime_contract_path.is_file(),
            "checks": runtime_contract.get("checks"),
            "verl": runtime_contract.get("verl"),
        },
        "packages": stack,
        "training_environment_packages": smoke_packages,
        "trainable_model_env": trainable_model,
        "trainable_model_available": model_available,
        "host": {"wsl2": is_wsl, "gpu": gpu},
        "local_8b_long_train_supported": local_8b_supported,
        "training_smoke": {
            "path": args.training_smoke,
            "exists": smoke_path.exists(),
            "success": bool(smoke.get("success")),
            "steps": smoke.get("steps", 0),
            "minimum_steps": args.min_smoke_steps,
            "checks": smoke_checks,
        },
        "8b_target_hardware": "cloud >=80GB-class A100/H100; prefer >=2x80GB after a 100-step memory/throughput smoke",
        "awq_role": "frozen evaluation and data probing only; never generate rollout for updates to a different BF16 policy",
    }

    guidance_cmp = _comparison(args.guidance_comparison)
    event_cmp = _comparison(args.event_comparison)
    event_hard_cmp = _comparison(args.event_hard_comparison)
    event_tabular_cmp = _comparison(args.event_tabular_comparison)
    event_tabular_hard_cmp = _comparison(args.event_tabular_hard_comparison)
    challenge_cmp = _comparison(args.challenge_comparison)
    tabular_cmp = _comparison(args.tabular_comparison)
    tabular_hard_cmp = _comparison(args.tabular_hard_comparison)
    spatial_ood_cmp = _comparison(args.spatial_ood_comparison)
    spatial_ood_hard_cmp = _comparison(args.spatial_ood_hard_comparison)
    reward_sensitivity = _comparison(args.reward_sensitivity)
    evidence_dose_response = _comparison(args.evidence_dose_response)
    policy_id = probe.get("policy_id")
    trained_policy = probe.get("policy_stage") == "trained" and bool(policy_id)

    def current_reward_artifact(artifact: dict | None) -> bool:
        identity = ((artifact or {}).get("reward") or {})
        return bool(
            identity.get("version") == current_reward.get("version")
            and identity.get("config_sha256") == current_reward.get("config_sha256")
        )

    def comparison_artifact_current(artifact: dict | None) -> bool:
        return bool(
            artifact
            and current_reward_artifact(artifact)
            and score_implementation_matches(artifact, REPO_ROOT)
        )

    def graduated(cmp: dict | None, *, min_cases: int = 0) -> bool:
        return bool(
            trained_policy and cmp and cmp.get("policy_stage") == "trained"
            and cmp.get("policy_id") == policy_id
            and comparison_artifact_current(cmp)
            and _identity_valid(cmp.get("comparison_implementation"))
            and _identity_valid(cmp.get("probe"))
            and (
                cmp.get("baseline") != "tabular"
                or _identity_valid(cmp.get("baseline_artifact"))
            )
            and cmp.get("n_cases", 0) >= min_cases
            and cmp.get("inference_status") == "estimable"
            and cmp.get("cluster_bootstrap_95pct", [0])[0] > 0
        )

    def hard_graduated(cmp: dict | None, metric: str, *, min_cases: int = 0) -> bool:
        interval = ((cmp or {}).get("cluster_bootstrap_95pct") or {}).get(metric)
        return bool(
            trained_policy and cmp and cmp.get("policy_stage") == "trained"
            and cmp.get("policy_id") == policy_id
            and comparison_artifact_current(cmp)
            and _identity_valid(cmp.get("comparison_implementation"))
            and _identity_valid(cmp.get("hard_metric_implementation"))
            and _identity_valid(cmp.get("probe"))
            and (
                cmp.get("baseline") != "tabular"
                or _identity_valid(cmp.get("baseline_artifact"))
            )
            and cmp.get("n_cases", 0) >= min_cases
            and cmp.get("inference_status") == "estimable"
            and interval and interval[0] > 0
        )

    event_cases = len({r["case_id"] for r in probe["rows"]
                       if r.get("truth_has_event") is True})
    graduation = {
        "evaluated_on_trained_policy": trained_policy,
        "scale_up_to_8b": {
            "beats_cams_on_independent_val": {
                "pass": graduated(guidance_cmp),
                "criterion": "paired issue-date-block bootstrap lower 95% bound > 0",
                "comparison": guidance_cmp,
            },
            "event_layer_beats_cams": {
                "pass": (graduated(event_cmp, min_cases=100)
                         and event_cmp.get("truth_event_filter") is True
                         and hard_graduated(event_hard_cmp, "event_csi", min_cases=100)
                         and event_hard_cmp.get("truth_event_filter") is True),
                "criterion": (
                    ">=100 event cases; shaped/composite paired block CI lower bound >0 and "
                    "non-shaped hard event-CSI improvement CI lower bound >0 vs CAMS"
                ),
                "event_cases_in_probe": event_cases,
                "composite_comparison": event_cmp,
                "hard_metric_comparison": event_hard_cmp,
            },
            "winter_challenge_model_selection": {
                "pass": bool(
                    trained_policy and challenge_cmp
                    and challenge_cmp.get("policy_stage") == "trained"
                    and challenge_cmp.get("policy_id") == policy_id
                    and comparison_artifact_current(challenge_cmp)
                    and _identity_valid(challenge_cmp.get("comparison_implementation"))
                    and _identity_valid(challenge_cmp.get("probe"))
                    and challenge_cmp.get("n_cases", 0) >= 100
                    and challenge_cmp.get("paired_difference", -1) >= 0
                ),
                "criterion": (
                    "trained checkpoint paired mean >= CAMS on the purged winter challenge; "
                    "the small number of weather-process blocks is a selection guardrail, not a paper CI"
                ),
                "challenge_cases": challenge["challenge"]["cases"],
                "comparison": challenge_cmp,
            },
        },
        "paper_graduation": {
            "beats_strong_tabular": {
                "pass": graduated(tabular_cmp),
                "criterion": "paired issue-date-block CI lower bound > 0 vs strong tabular baseline",
                "comparison": tabular_cmp,
            },
            "event_and_turning_beats_strong_tabular": {
                "pass": bool(
                    graduated(event_tabular_cmp, min_cases=100)
                    and event_tabular_cmp.get("baseline") == "tabular"
                    and event_tabular_cmp.get("truth_event_filter") is True
                    and hard_graduated(
                        event_tabular_hard_cmp, "event_csi", min_cases=100
                    )
                    and hard_graduated(
                        event_tabular_hard_cmp,
                        "turning_peak_mae_days",
                        min_cases=100,
                    )
                    and event_tabular_hard_cmp.get("baseline") == "tabular"
                    and event_tabular_hard_cmp.get("truth_event_filter") is True
                ),
                "criterion": (
                    ">=100 hidden-truth event cases; composite, hard event CSI, and "
                    "intention-to-treat peak-timing MAE all have issue-date-block "
                    "95% CI lower bound >0 versus the same-information strong tabular baseline"
                ),
                "scientific_boundary": (
                    "this gate is required before claiming that multi-source reasoning adds "
                    "event/turning skill beyond a dedicated learner; beating CAMS alone is "
                    "insufficient"
                ),
                "composite_comparison": event_tabular_cmp,
                "hard_metric_comparison": event_tabular_hard_cmp,
            },
            "hard_metrics_vs_strong_tabular_reported": {
                "pass": bool(
                    trained_policy and tabular_hard_cmp
                    and tabular_hard_cmp.get("policy_stage") == "trained"
                    and tabular_hard_cmp.get("policy_id") == policy_id
                    and comparison_artifact_current(tabular_hard_cmp)
                    and _identity_valid(tabular_hard_cmp.get("comparison_implementation"))
                    and _identity_valid(tabular_hard_cmp.get("hard_metric_implementation"))
                    and _identity_valid(tabular_hard_cmp.get("probe"))
                    and tabular_hard_cmp.get("inference_status") == "estimable"
                ),
                "criterion": (
                    "report clustered hard level/event/primary/interval metrics vs tabular; "
                    "claim-specific superiority must use the corresponding pre-registered CI"
                ),
                "comparison": tabular_hard_cmp,
            },
            "unseen_city_spatial_ood": {
                "pass": bool(
                    graduated(spatial_ood_cmp, min_cases=100)
                    and spatial_ood_cmp.get("split") == "spatial_ood_test"
                    and spatial_ood_hard_cmp
                    and spatial_ood_hard_cmp.get("policy_stage") == "trained"
                    and spatial_ood_hard_cmp.get("policy_id") == policy_id
                    and comparison_artifact_current(spatial_ood_hard_cmp)
                    and spatial_ood_hard_cmp.get("split") == "spatial_ood_test"
                    and spatial_ood_hard_cmp.get("n_cases", 0) >= 100
                    and spatial_ood_hard_cmp.get("inference_status") == "estimable"
                    and _identity_valid(
                        spatial_ood_hard_cmp.get("comparison_implementation")
                    )
                    and _identity_valid(
                        spatial_ood_hard_cmp.get("hard_metric_implementation")
                    )
                ),
                "criterion": (
                    ">=100 cases and >=30 issue-date clusters from eight completely "
                    "held-out cities; composite block-bootstrap lower 95% bound >0 vs "
                    "strong tabular, with clustered hard metrics reported"
                ),
                "scientific_boundary": (
                    "this warm-season city holdout tests spatial OOD only; it cannot replace "
                    "the prospective 2026-27 winter event evaluation"
                ),
                "comparison": spatial_ood_cmp,
                "hard_metric_comparison": spatial_ood_hard_cmp,
            },
            "outcome_weight_robustness": {
                "pass": bool(
                    trained_policy and reward_sensitivity
                    and reward_sensitivity.get("policy_stage") == "trained"
                    and reward_sensitivity.get("policy_id") == policy_id
                    and current_reward_artifact(reward_sensitivity)
                    and score_implementation_matches(reward_sensitivity, REPO_ROOT)
                    and _identity_valid(reward_sensitivity.get("audit_implementation"))
                    and _identity_valid(reward_sensitivity.get("probe"))
                    and _identity_valid(reward_sensitivity.get("tabular"))
                    and reward_sensitivity.get("n_cases", 0) >= 50
                    and reward_sensitivity.get("inference_status") == "estimable"
                    and reward_sensitivity.get("direction_stable_across_grid") is True
                    and bool(reward_sensitivity.get("scenarios"))
                    and all(
                        row.get("direction") == "policy_above"
                        and (row.get("cluster_bootstrap_95pct") or [0])[0] > 0
                        for row in reward_sensitivity.get("scenarios", [])
                    )
                ),
                "criterion": (
                    "trained policy beats strong tabular with clustered CI lower bound >0 "
                    "under default, equal, and each outcome component +/-25% weights"
                ),
                "comparison": reward_sensitivity,
            },
            "core_evidence_dose_response": {
                "pass": bool(
                    trained_policy and evidence_dose_response
                    and evidence_dose_response.get("policy_stage") == "trained"
                    and evidence_dose_response.get("policy_id") == policy_id
                    and current_reward_artifact(evidence_dose_response)
                    and score_implementation_matches(evidence_dose_response, REPO_ROOT)
                    and _identity_valid(
                        evidence_dose_response.get("comparison_implementation")
                    )
                    and _identity_valid(evidence_dose_response.get("full"))
                    and _identity_valid(evidence_dose_response.get("masked"))
                    and evidence_dose_response.get("split") == probe.get("split")
                    and evidence_dose_response.get("masked_channels")
                    and {"synoptic", "composition"}.issubset(
                        set(evidence_dose_response.get("masked_channels") or [])
                    )
                    and (evidence_dose_response.get("outcome_composite") or {}).get(
                        "n_cases", 0) >= 50
                    and (evidence_dose_response.get("outcome_composite") or {}).get(
                        "supports_positive_evidence_dose_response") is True
                ),
                "criterion": (
                    "paired same-seed ablation masks at least synoptic+composition; "
                    "full-minus-masked outcome-composite issue-date-block CI lower bound >0"
                ),
                "comparison": evidence_dose_response,
            },
            "forward_winter_2026_27": {
                "pass": False,
                "available_from": "2026-12 onward",
                "criterion": "fully prospective winter event/process evaluation",
            },
        },
    }
    graduation["ready_for_8b_scale_up"] = all(
        item["pass"] for item in graduation["scale_up_to_8b"].values()
    )
    graduation["paper_ready"] = graduation["ready_for_8b_scale_up"] and all(
        item["pass"] for item in graduation["paper_graduation"].values()
    )

    report = {
        "artifact_type": "rl_readiness",
        "preflight_version": "2.3.0",
        "probe": file_identity(REPO_ROOT / args.probe, relative_to=REPO_ROOT),
        "model": probe.get("model"),
        "policy_id": policy_id,
        "policy_stage": probe.get("policy_stage", "unknown_legacy_artifact"),
        "reward": current_reward,
        "preflight": preflight,
        "preflight_passed": all(g["pass"] for g in preflight.values()),
        "graduation": graduation,
        "baseline_context": {
            s: {k: tabular["results"][s][k] for k in ("n", "composite", "macro_stratum")}
            for s in ("val", "challenge_winter", "test",
                      "spatial_ood_val", "spatial_ood_test")
            if s in tabular["results"]
        },
    }
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {out}")
    return 0 if report["preflight_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
