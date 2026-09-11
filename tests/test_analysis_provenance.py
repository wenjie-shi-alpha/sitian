from pathlib import Path

import pytest

from scripts.analyze_thinking_ablation import _validate_pair
from scripts.compare_probe_clustered import _validate_score_artifact
from sitian.provenance import (
    case_bundle_snapshot, case_bundle_snapshot_matches, file_identity,
)
from sitian.scoring import reward_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


def _provenance() -> dict:
    return {
        "reward": reward_spec(),
        "scoring_implementation": file_identity(
            REPO_ROOT / "src/sitian/scoring.py", relative_to=REPO_ROOT
        ),
        "schema_implementation": file_identity(
            REPO_ROOT / "src/sitian/schema.py", relative_to=REPO_ROOT
        ),
    }


def test_composite_comparison_requires_current_scoring_contents():
    artifact = {"provenance": _provenance()}
    _validate_score_artifact(artifact, "probe")

    artifact["provenance"]["scoring_implementation"]["sha256"] = "stale"
    with pytest.raises(ValueError, match="scoring/schema implementation"):
        _validate_score_artifact(artifact, "probe")


def test_thinking_ablation_rejects_mixed_scoring_implementations():
    base = {
        "model": "m", "split": "val", "n_cases": 1, "group_size": 2,
        "temperature": 1.0, "case_sampling_seed": 7, "rollout_seed_base": 7,
        "rollout_seed_strategy": "stable", "provenance": _provenance(),
        "rows": [{"case_id": "a", "rep": 0, "rollout_seed": 11,
                  "truth_has_event": False}],
    }
    off = {**base, "thinking_enabled": False}
    on = {**base, "thinking_enabled": True, "provenance": _provenance()}
    _validate_pair(off, on)

    on["provenance"]["scoring_implementation"]["bytes"] += 1
    with pytest.raises(ValueError, match="on scoring/schema implementation"):
        _validate_pair(off, on)


def test_composite_comparison_rejects_stale_schema_contents():
    artifact = {"provenance": _provenance()}
    artifact["provenance"]["schema_implementation"]["sha256"] = "stale"
    with pytest.raises(ValueError, match="scoring/schema implementation"):
        _validate_score_artifact(artifact, "probe")


def test_case_snapshot_detects_visible_input_or_truth_changes(tmp_path):
    case_dir = tmp_path / "case_a"
    case_dir.mkdir()
    for name, value in {
        "case.json": '{"case_id":"case_a"}',
        "observations.json": '{"PM2.5":1}',
        "evidence.json": '{"synoptic":{}}',
        "truth.json": '{"daily":{}}',
    }.items():
        (case_dir / name).write_text(value, encoding="utf-8")
    snapshot = case_bundle_snapshot(
        [case_dir], relative_to=tmp_path, include_truth=True
    )
    assert case_bundle_snapshot_matches(snapshot, tmp_path)
    (case_dir / "evidence.json").write_text('{"synoptic":{"changed":true}}',
                                             encoding="utf-8")
    assert not case_bundle_snapshot_matches(snapshot, tmp_path)


def test_case_snapshot_rejects_expert_file_in_national_replay(tmp_path):
    case_dir = tmp_path / "case_a"
    case_dir.mkdir()
    (case_dir / "case.json").write_text("{}", encoding="utf-8")
    (case_dir / "expert.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="expert file forbidden"):
        case_bundle_snapshot([case_dir], relative_to=tmp_path)
