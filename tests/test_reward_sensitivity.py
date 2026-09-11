from pathlib import Path

import pytest

from scripts.audit_reward_weight_sensitivity import analyze, recombine, scenario_grid
from sitian.provenance import file_identity
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


def test_scenario_grid_is_normalized_and_recombine_respects_abstention():
    scenarios = scenario_grid()
    assert len(scenarios) == 12
    assert all(abs(sum(weights.values()) - 1.0) < 1e-12
               for weights in scenarios.values())
    assert recombine(
        {"level": 1.0, "event": None, "interval": 0.0},
        scenarios["default"],
    ) == 0.7


def test_weight_sensitivity_is_case_first_and_reports_direction_stability():
    identity = reward_spec()
    probe = {
        "policy_stage": "trained", "policy_id": "adapter", "split": "val",
        "provenance": {
            "reward": identity,
            "scoring_implementation": _scoring_identity(),
            "schema_implementation": _schema_identity(),
        },
        "rows": [
            {"case_id": case, "issue_date": issue, "stratum": "event", "rep": rep,
             "submitted": True,
             "components": {"level": 0.9, "event": 0.8, "interval": 0.8,
                            "turning": 0.8, "primary": 0.8}}
            for case, issue in (("a", "2026-01-01"), ("b", "2026-01-02"))
            for rep in range(2)
        ],
    }
    tabular = {
        "provenance": {
            "reward": identity,
            "scoring_implementation": _scoring_identity(),
            "schema_implementation": _schema_identity(),
        },
        "results": {"val": {"rows": [
            {"case_id": case,
             "components": {"level": 0.6, "event": 0.6, "interval": 0.6,
                            "turning": 0.6, "primary": 0.6}}
            for case in ("a", "b")
        ]}},
    }
    report = analyze(probe, tabular, tabular_split="val", bootstrap=100, seed=3)

    assert report["n_cases"] == 2
    assert report["n_issue_date_clusters"] == 2
    assert report["direction_stable_across_grid"] is True
    assert report["directions_observed"] == ["policy_above"]
    assert all(row["paired_difference"] > 0 for row in report["scenarios"])


def test_weight_sensitivity_rejects_missing_scoring_implementation():
    identity = reward_spec()
    probe = {"provenance": {"reward": identity}, "rows": []}
    tabular = {
        "provenance": {
            "reward": identity,
            "scoring_implementation": _scoring_identity(),
            "schema_implementation": _schema_identity(),
        }
    }
    with pytest.raises(ValueError, match="probe scoring/schema implementation"):
        analyze(probe, tabular, tabular_split="val", bootstrap=10, seed=3)
