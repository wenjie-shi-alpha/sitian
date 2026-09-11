from pathlib import Path

from scripts.analyze_expert_forecast_requirements import (
    _formal_probe,
    _resolved_coverage_audit,
)
from scripts.prepare_verl_dataset import _input_snapshot, _manifest_path, _row
from scripts.probe_model import ablate_bundle
from scripts.compare_evidence_ablation import _inference, _paired_cases, _validate
from scripts.build_expert_harness_report import _effective_gates
from sitian.case import CaseBundle
from sitian.open_evidence import EVIDENCE_VERSION, NWP_TEMPORAL_PROFILE_VERSION
from sitian.provenance import case_bundle_snapshot, file_identity
from sitian.scoring import reward_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


def _scoring_identity() -> dict:
    return file_identity(
        REPO_ROOT / "src/sitian/scoring.py", relative_to=REPO_ROOT
    )


def _schema_identity() -> dict:
    return file_identity(
        REPO_ROOT / "src/sitian/schema.py", relative_to=REPO_ROOT
    )


def test_coverage_matrix_keeps_data_policy_and_reward_claims_separate(tmp_path):
    case_dir = tmp_path / "case_a"
    case_dir.mkdir()
    (case_dir / "case.json").write_text("{}", encoding="utf-8")
    (case_dir / "truth.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    manifest_identity = file_identity(manifest)
    probe = {
        "n_cases": 50,
        "group_size": 8,
        "n_rollouts": 400,
        "temperature": 1.0,
        "rows": [
            {"tools": ["get_process_evidence", "get_synoptic_evidence"]}
            for _ in range(400)
        ],
        "summary": {
            "semantic_grounding": {"full_credit_rollout_rate": 0.75}
        },
        "provenance": {
            "reward": reward_spec(),
            "scoring_implementation": _scoring_identity(),
            "schema_implementation": _schema_identity(),
            "probe_script": file_identity(REPO_ROOT / "scripts/probe_model.py"),
            "agent_driver": file_identity(
                REPO_ROOT / "src/sitian/agents/llm_openai.py"),
            "environment_implementation": file_identity(
                REPO_ROOT / "src/sitian/env.py"),
            "process_evidence_implementation": file_identity(
                REPO_ROOT / "src/sitian/process_evidence.py"),
            "guidance_bias_implementation": file_identity(
                REPO_ROOT / "src/sitian/guidance_bias.py"),
            "case_manifest": manifest_identity,
            "standards_manifest": file_identity(
                REPO_ROOT / "references/standards/manifest.json"),
            "case_input_snapshot": case_bundle_snapshot(
                [case_dir], relative_to=REPO_ROOT, include_truth=True),
        },
    }
    coverage_scripts = {
        name: file_identity(REPO_ROOT / path)
        for name, path in {
            "synoptic": "scripts/derive_open_synoptic.py",
            "composition": "scripts/derive_open_composition.py",
            "spatial_observations": "scripts/derive_spatial_observations.py",
            "fires": "scripts/derive_open_fires.py",
            "static_context": "scripts/derive_static_context.py",
            "case_attachment": "scripts/attach_open_evidence.py",
        }.items()
    }
    contract_provenance = {
        "case_manifests": [manifest_identity] * 3,
        "audit_implementation": file_identity(
            REPO_ROOT / "scripts/audit_expert_evidence_contract.py"),
        "process_evidence_implementation": file_identity(
            REPO_ROOT / "src/sitian/process_evidence.py"),
        "open_evidence_implementation": file_identity(
            REPO_ROOT / "src/sitian/open_evidence.py"),
        "guidance_bias_implementation": file_identity(
            REPO_ROOT / "src/sitian/guidance_bias.py"),
    }
    rows = _resolved_coverage_audit(
        {
            "ready_for_full_attachment": True,
            "contract_version": EVIDENCE_VERSION,
            "nwp_temporal_profile": {"version": NWP_TEMPORAL_PROFILE_VERSION},
            "source_manifest": manifest_identity,
            "provenance": {
                "audit_implementation": file_identity(
                    REPO_ROOT / "scripts/audit_open_evidence_coverage.py"),
                "open_evidence_contract_implementation": file_identity(
                    REPO_ROOT / "src/sitian/open_evidence.py"),
                "derivation_implementations": coverage_scripts,
            },
        },
        probe,
        {"preflight_contract_passed": True, "provenance": contract_provenance},
    )

    assert _formal_probe(probe)
    assert rows[1]["raw_data"] == "强（逐 case 合同通过）"
    assert rows[1]["policy_use"] == "已观测使用（最高 100.0%）"
    assert rows[1]["reward_test"] == "弱"
    assert rows[9]["policy_use"] == "完整 grounding 75.0%"


def test_tiny_probe_never_promotes_policy_use_to_formal_evidence():
    rows = _resolved_coverage_audit(
        {"ready_for_full_attachment": False},
        {"n_cases": 2, "group_size": 2, "n_rollouts": 4, "rows": []},
        {},
    )

    assert rows[1]["raw_data"].startswith("阻塞")
    assert rows[1]["policy_use"] == "待正式 50×8 探针"
    assert rows[9]["policy_use"] == "待正式 50×8 探针"


def test_large_probe_with_stale_reward_is_not_formal_evidence():
    stale = {
        "n_cases": 50, "group_size": 8, "n_rollouts": 400,
        "provenance": {"reward": {"version": "old", "config_sha256": "old"}},
    }
    assert _formal_probe(stale) is False


def test_large_probe_without_scoring_hash_is_not_formal_evidence():
    probe = {
        "n_cases": 50, "group_size": 8, "n_rollouts": 400,
        "temperature": 1.0,
        "provenance": {"reward": reward_spec()},
    }
    assert _formal_probe(probe) is False


def test_variance_probe_must_match_train_sampling_temperature():
    probe = {
        "n_cases": 50, "group_size": 8, "n_rollouts": 400,
        "temperature": 0.6,
        "provenance": {"reward": reward_spec()},
    }
    assert _formal_probe(probe) is False


def test_stale_preflight_cannot_promote_current_report_gates():
    old = {
        "reward": {"version": "0.4.0", "config_sha256": "old"},
        "preflight": {
            "data_and_reward_integrity": {"pass": True},
            "format_and_trajectory": {"pass": True},
            "trainable_stack_and_hardware_smoke": {"pass": True},
        },
    }
    rows, freshness = _effective_gates(
        old, coverage_pass=True, contract_pass=True, reward_contract_pass=True,
        formal_probe=True,
        training_smoke_pass=True,
    )

    assert freshness["reward_identity_current"] is False
    assert all(status == "BLOCKED" for _, status in rows)


def test_current_preflight_still_requires_current_upstream_contracts():
    current = reward_spec()
    preflight = {
        "reward": current,
        "preflight": {
            "data_and_reward_integrity": {"pass": True},
            "format_and_trajectory": {"pass": True},
        },
    }
    rows, freshness = _effective_gates(
        preflight, coverage_pass=False, contract_pass=False,
        reward_contract_pass=True, formal_probe=False,
        training_smoke_pass=False,
    )

    assert freshness["reward_identity_current"] is True
    assert dict(rows)["数据、Reward 与专家证据合同"] == "BLOCKED"
    assert dict(rows)["格式与轨迹"] == "BLOCKED"


def test_verl_training_uses_joint_winter_and_spatial_purged_manifest():
    assert _manifest_path("train").name == "train_without_winter_or_spatial_holdout.json"
    assert _manifest_path("val").name == "valid_cases_val.json"
    assert _manifest_path("challenge_winter").name == "challenge_winter.json"


def test_evidence_ablation_is_copy_on_write_and_channel_specific():
    bundle = CaseBundle(
        "case", "2025-12-19", "target", 5,
        evidence={"synoptic": {"sources": {"gfs": [1]}}, "pollution": {
            "composition": {"available": True},
            "fires": {"available": True},
            "source_context": {"target": {}},
        }},
    )
    masked = ablate_bundle(bundle, ["synoptic", "composition"])

    assert bundle.evidence["synoptic"]["sources"]["gfs"] == [1]
    assert bundle.evidence["pollution"]["composition"]["available"] is True
    assert masked.evidence["synoptic"] == {}
    assert masked.evidence["pollution"]["composition"] == {}
    assert masked.evidence["pollution"]["fires"]["available"] is True


def test_evidence_ablation_comparison_is_case_first_and_seed_paired():
    base = {
        "model": "m", "policy_id": "m", "policy_stage": "frozen_baseline",
        "split": "val", "n_cases": 2, "group_size": 2, "temperature": 0.6,
        "case_sampling_seed": 7, "rollout_seed_base": 7,
        "rollout_seed_strategy": "stable", "thinking_enabled": True,
        "stratum_filter": None, "stratified": True,
        "provenance": {
            "reward": reward_spec(),
            "scoring_implementation": _scoring_identity(),
            "schema_implementation": _schema_identity(),
        },
    }
    full = {**base, "evidence_ablation": [], "rows": [
        {"case_id": case, "rep": rep, "rollout_seed": rep,
         "issue_date": f"2025-12-{19 + index:02d}", "stratum": "event",
         "outcome_composite": score, "reward": score, "tools": []}
        for index, (case, scores) in enumerate((("a", (0.8, 0.6)), ("b", (0.4, 0.2))))
        for rep, score in enumerate(scores)
    ]}
    masked = {**base, "evidence_ablation": ["synoptic"], "rows": [
        {**row, "outcome_composite": row["outcome_composite"] - 0.1,
         "reward": row["reward"] - 0.1}
        for row in full["rows"]
    ]}

    _validate(full, masked)
    cases = _paired_cases(full, masked, "outcome_composite")
    result = _inference(cases, replicates=100, seed=7)
    assert len(cases) == 2
    assert result["full_minus_masked"] == 0.1


def test_training_input_snapshot_forbids_expert_file(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "expert.json").write_text("{}", encoding="utf-8")

    import pytest
    with pytest.raises(ValueError, match="expert file forbidden"):
        _input_snapshot([case_dir])


def test_verl_case_path_and_hidden_truth_stay_out_of_model_prompt(tmp_path, monkeypatch):
    import json
    from scripts import prepare_verl_dataset
    from sitian.synth import make_case

    relative = "cases/national/sample"
    case_dir = tmp_path / relative
    bundle = make_case("clean")
    bundle.expert = None
    bundle.save(case_dir)
    monkeypatch.setattr(prepare_verl_dataset, "REPO_ROOT", tmp_path)
    row = _row(case_dir, "train", 0, require_evidence=False)
    prompt_text = json.dumps(row["prompt"], ensure_ascii=False)

    assert str(case_dir) not in prompt_text
    assert relative not in prompt_text
    assert row["reward_model"]["ground_truth"] == "FORECAST_ENV_HIDDEN_TRUTH"
    assert all(
        values["create_kwargs"]["case_dir"] == relative
        for values in row["extra_info"]["tools_kwargs"].values()
    )
