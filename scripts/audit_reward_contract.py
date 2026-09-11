#!/usr/bin/env python3
"""Executable red-team contract for the rule-based forecast reward.

This is deliberately independent of model quality.  It checks monotonicity,
abstention, comparison fairness and the main grounding attack surfaces every
time the data/training preflight is rebuilt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.schema import (  # noqa: E402
    AQI_STANDARD_2012,
    AQI_STANDARD_2026,
    daily_aqi,
    pm25_to_level,
)
from sitian.scoring import (  # noqa: E402
    RewardConfig,
    extract_event,
    reward_spec,
    score_forecast,
)
from sitian.provenance import file_identity  # noqa: E402


ISSUE_DATE = "2026-06-01"
REGION = "reward-audit-city"
TRUTH = {
    "2026-06-02": {"PM2.5": 30.0, "PM10": 45.0, "O3": 80.0,
                   "SO2": 8.0, "NO2": 25.0, "CO": 0.8},
    "2026-06-03": {"PM2.5": 140.0, "PM10": 260.0, "O3": 170.0,
                   "SO2": 20.0, "NO2": 70.0, "CO": 2.0},
    "2026-06-04": {"PM2.5": 180.0, "PM10": 380.0, "O3": 210.0,
                   "SO2": 30.0, "NO2": 90.0, "CO": 3.0},
    "2026-06-05": {"PM2.5": 50.0, "PM10": 80.0, "O3": 110.0,
                   "SO2": 10.0, "NO2": 35.0, "CO": 1.0},
}


def _perfect_forecast(*, evidence: list[dict] | None = None) -> dict:
    levels = {
        day: daily_aqi(values, standard=AQI_STANDARD_2026)
        for day, values in TRUTH.items()
    }
    process = extract_event(
        {day: values["aqi"] for day, values in levels.items()},
        RewardConfig().process_level,
        levels={day: values["level"] for day, values in levels.items()},
    )
    return {
        "issue_date": ISSUE_DATE,
        "region": REGION,
        "daily": [
            {
                "date": day,
                "aqi_level": levels[day]["level"],
                "primary_pollutant": levels[day]["primary"][0]
                    if levels[day]["primary"] else None,
                "pm25_range": [values["PM2.5"] - 5, values["PM2.5"] + 5],
                "pm10_range": [values["PM10"] - 5, values["PM10"] + 5],
                "o3_range": [values["O3"] - 5, values["O3"] + 5],
            }
            for day, values in TRUTH.items()
        ],
        "process": {
            "has_event": process is not None,
            "start": process[0] if process else None,
            "peak": process[1] if process else None,
            "end": process[2] if process else None,
        },
        "evidence": evidence or [],
        "confidence": "medium",
    }


def _score(forecast: dict, **kwargs):
    return score_forecast(
        forecast, TRUTH, issue_date=ISSUE_DATE, horizon=len(TRUTH),
        region=REGION, aqi_standard=AQI_STANDARD_2026, **kwargs,
    )


def run_audit() -> dict:
    checks: dict[str, dict] = {}

    perfect = _score(_perfect_forecast())
    checks["perfect_oracle_reaches_one"] = {
        "pass": perfect.valid and perfect.outcome_composite == 1.0,
        "observed": perfect.outcome_composite,
    }

    # schema-v0.6.0: levels are derived from interval midpoints, so a level
    # error can only be expressed through the intervals themselves.
    degraded_fc = _perfect_forecast()
    degraded_fc["daily"][1]["pm25_range"] = [60, 80]    # truth 140 -> level 4; forecast level 2/3
    degraded_fc["daily"][1]["pm10_range"] = [100, 140]
    degraded_fc["daily"][1]["o3_range"] = [100, 140]
    degraded = _score(degraded_fc)
    checks["level_error_strictly_reduces_outcome"] = {
        "pass": degraded.valid and degraded.outcome_composite < perfect.outcome_composite,
        "perfect": perfect.outcome_composite,
        "degraded": degraded.outcome_composite,
    }

    wide_fc = _perfect_forecast()
    for daily in wide_fc["daily"]:
        daily["pm25_range"] = [0, 500]
        daily["pm10_range"] = [0, 800]
        daily["o3_range"] = [0, 800]
    wide = _score(wide_fc)
    checks["wider_intervals_are_penalized"] = {
        "pass": wide.components["interval"] < perfect.components["interval"],
        "narrow": perfect.components["interval"],
        "wide": wide.components["interval"],
    }

    maximum_width_fc = _perfect_forecast()
    for daily in maximum_width_fc["daily"]:
        daily["pm25_range"] = [0, 1000]
        daily["pm10_range"] = [0, 2000]
        daily["o3_range"] = [0, 1200]
    maximum_width = _score(maximum_width_fc)
    checks["maximum_width_intervals_have_no_reward_floor"] = {
        "pass": (
            maximum_width.valid
            and maximum_width.components["interval"] < 0.01
            and maximum_width.outcome_composite < wide.outcome_composite
        ),
        "maximum_width_interval_component": maximum_width.components["interval"],
        "maximum_width_outcome": maximum_width.outcome_composite,
        "ordinary_wide_outcome": wide.outcome_composite,
    }

    clean_truth = {
        (date.fromisoformat(ISSUE_DATE) + timedelta(days=index)).isoformat(): 20.0
        for index in range(1, 5)
    }
    clean_fc = {
        "issue_date": ISSUE_DATE, "region": REGION,
        "daily": [
            {"date": day, "aqi_level": 1, "pm25_range": [10, 30]}
            for day in clean_truth
        ],
        "process": {"has_event": False}, "evidence": [], "confidence": "medium",
    }
    clean = score_forecast(
        clean_fc, clean_truth, issue_date=ISSUE_DATE, horizon=4, region=REGION,
        aqi_standard=AQI_STANDARD_2026,
    )
    checks["clean_correct_negative_abstains_event_and_turning"] = {
        "pass": clean.components["event"] is None and clean.components["turning"] is None,
        "event": clean.components["event"], "turning": clean.components["turning"],
    }
    false_alarm_fc = deepcopy(clean_fc)
    false_alarm_fc["daily"][0]["pm25_range"] = [120, 140]  # midpoint 130 -> level 4 (HJ 633-2026)
    false_alarm = score_forecast(
        false_alarm_fc, clean_truth, issue_date=ISSUE_DATE, horizon=4, region=REGION,
        aqi_standard=AQI_STANDARD_2026,
    )
    checks["clean_false_alarm_is_penalized"] = {
        "pass": (
            false_alarm.valid and false_alarm.components["event"] == 0.0
            and false_alarm.outcome_composite < clean.outcome_composite
        ),
        "event": false_alarm.components["event"],
        "clean_outcome": clean.outcome_composite,
        "false_alarm_outcome": false_alarm.outcome_composite,
    }

    invalid = _score({"garbage": True})
    checks["invalid_schema_is_zero"] = {
        "pass": not invalid.valid and invalid.composite == 0 and invalid.outcome_composite == 0,
        "composite": invalid.composite, "outcome_composite": invalid.outcome_composite,
    }

    # schema-v0.6.0 derivation invariants: the categorical heads are functions
    # of the submitted intervals, so a policy cannot collect reward from heads
    # that contradict its own numbers, nor lose format credit for arithmetic.
    derived = _score(_perfect_forecast()).details["normalized_forecast"]
    checks["derived_heads_match_truth_for_oracle_intervals"] = {
        "pass": (
            [item["aqi_level"] for item in derived["daily"]]
            == [daily_aqi(values, standard=AQI_STANDARD_2026)["level"]
                for values in TRUTH.values()]
            and derived["process"]["has_event"] is True
            and derived["process"]["start"] == "2026-06-03"
            and derived["process"]["peak"] == "2026-06-04"
            and derived["process"]["end"] == "2026-06-04"
        ),
        "daily": [[item["date"], item["aqi_level"], item["primary_pollutant"]]
                  for item in derived["daily"]],
        "process": derived["process"],
    }

    contradictory_fc = _perfect_forecast()
    contradictory_fc["daily"][0].update(aqi_level=5, primary_pollutant="NO2")
    contradictory_fc["process"] = {"has_event": False}
    contradictory = _score(contradictory_fc)
    checks["supplied_categorical_heads_cannot_override_intervals"] = {
        "pass": (
            contradictory.valid
            and contradictory.outcome_composite == perfect.outcome_composite
            and contradictory.details["normalized_forecast"]["daily"][0]["aqi_level"] == 1
            and contradictory.details["normalized_forecast"]["daily"][0]["primary_pollutant"] is None
            and contradictory.details["normalized_forecast"]["process"]["has_event"] is True
        ),
        "outcome": contradictory.outcome_composite,
        "perfect": perfect.outcome_composite,
    }

    interval_only_fc = _perfect_forecast()
    for item in interval_only_fc["daily"]:
        item.pop("aqi_level", None)
        item.pop("primary_pollutant", None)
    interval_only_fc.pop("process", None)
    interval_only = _score(interval_only_fc)
    checks["interval_only_submission_is_complete"] = {
        "pass": interval_only.valid and interval_only.outcome_composite == perfect.outcome_composite,
        "outcome": interval_only.outcome_composite,
    }

    dominated_fc = _perfect_forecast()
    dominated_fc["daily"][0].update(
        pm25_range=[36, 45], pm10_range=[115, 119], o3_range=[20, 40],
    )
    dominated = _score(dominated_fc)
    checks["derived_primary_is_the_highest_iaqi_midpoint"] = {
        "pass": (
            dominated.valid
            and dominated.details["normalized_forecast"]["daily"][0]["primary_pollutant"] == "PM10"
            and dominated.details["normalized_forecast"]["daily"][0]["aqi_level"] == 2
        ),
        "daily0": dominated.details["normalized_forecast"]["daily"][0],
    }

    registry = {
        "e1": {"tool": "get_observations", "content": {"latest": 42.0}},
        "e2": {"tool": "get_diagnostics", "content": {"wind_ms": 2.5}},
    }
    diverse_evidence = [
        {"type": "observation", "claim": "起报实况", "ref": "e1",
         "field": "/latest", "value": 42.0},
        {"type": "diagnostic", "claim": "近地风", "ref": "e2",
         "field": "/wind_ms", "value": 2.5},
    ]
    grounded = _score(
        _perfect_forecast(evidence=diverse_evidence),
        tools_called={"get_observations", "get_diagnostics"}, evidence_registry=registry,
    )
    checks["two_facts_two_types_reach_full_grounding"] = {
        "pass": grounded.components["grounding"] == 1.0,
        "observed": grounded.components["grounding"],
    }
    checks["grounding_cannot_change_fair_outcome_score"] = {
        "pass": grounded.outcome_composite == perfect.outcome_composite,
        "without_tools": perfect.outcome_composite,
        "with_tools": grounded.outcome_composite,
    }

    tool_spam = _score(
        _perfect_forecast(evidence=[]),
        tools_called={"get_observations", "get_diagnostics", "get_model_guidance"},
        evidence_registry=registry,
    )
    checks["tool_call_spam_without_citations_gets_zero_grounding"] = {
        "pass": tool_spam.components["grounding"] == 0.0,
        "grounding": tool_spam.components["grounding"],
        "tools_called": 3,
    }

    poisoned_evidence = [
        *diverse_evidence,
        {"type": "observation", "claim": "追加伪造值", "ref": "e1",
         "field": "/latest", "value": 999.0},
    ]
    poisoned = _score(
        _perfect_forecast(evidence=poisoned_evidence),
        tools_called={"get_observations", "get_diagnostics"},
        evidence_registry=registry,
    )
    checks["extra_invalid_citation_cannot_preserve_full_grounding"] = {
        "pass": (
            poisoned.components["grounding"] < grounded.components["grounding"]
            and poisoned.details["grounding"]["structured_invalid_count"] == 1
            and poisoned.details["grounding"]["unique_assertion_count"] == 2
            and poisoned.details["grounding"]["scored_evidence_item_count"] == 3
        ),
        "clean_grounding": grounded.components["grounding"],
        "poisoned_grounding": poisoned.components["grounding"],
        "structured_invalid": poisoned.details["grounding"]["structured_invalid_count"],
        "unique_assertions": poisoned.details["grounding"]["unique_assertion_count"],
        "scored_items": poisoned.details["grounding"]["scored_evidence_item_count"],
    }

    duplicate = [diverse_evidence[0], {**diverse_evidence[0], "claim": "重复包装"}]
    duplicate_score = _score(
        _perfect_forecast(evidence=duplicate), tools_called={"get_observations"},
        evidence_registry=registry,
    )
    checks["duplicate_fact_cannot_reach_full_grounding"] = {
        "pass": duplicate_score.components["grounding"] < 1.0,
        "observed": duplicate_score.components["grounding"],
    }

    replay_content = {"daily_signals": {"2026-06-02": {"wind_speed_ms": 2}}}
    replay_registry = {
        "e1": {"tool": "get_assessment", "content": replay_content},
        "e2": {"tool": "get_assessment", "content": deepcopy(replay_content)},
    }
    replay_field = "/daily_signals/2026-06-02/wind_speed_ms"
    replay_evidence = [
        {"type": "diagnostic", "claim": "弱风", "ref": "e1",
         "field": replay_field, "value": 2},
        {"type": "synoptic", "claim": "重复包装弱风", "ref": "e2",
         "field": replay_field, "value": 2.0},
    ]
    replay_score = _score(
        _perfect_forecast(evidence=replay_evidence),
        tools_called={"get_assessment"}, evidence_registry=replay_registry,
    )
    checks["replayed_ref_or_numeric_spelling_cannot_duplicate_fact"] = {
        "pass": (
            replay_score.components["grounding"] < 1.0
            and replay_score.details["grounding"]["unique_assertion_count"] == 1
        ),
        "grounding": replay_score.components["grounding"],
        "unique_assertions": replay_score.details["grounding"]["unique_assertion_count"],
    }

    missing_registry = {
        "e1": {"tool": "get_observations", "content": {"latest": None}},
        "e2": {"tool": "get_diagnostics", "content": {"wind_ms": math.nan}},
    }
    missing_evidence = [
        {"type": "observation", "claim": "缺测", "ref": "e1",
         "field": "/latest", "value": None},
        {"type": "diagnostic", "claim": "非有限值", "ref": "e2",
         "field": "/wind_ms", "value": math.nan},
    ]
    missing_score = _score(
        _perfect_forecast(evidence=missing_evidence),
        tools_called={"get_observations", "get_diagnostics"},
        evidence_registry=missing_registry,
    )
    checks["missing_values_cannot_be_semantic_facts"] = {
        "pass": (
            missing_score.details["grounding"]["semantic_verified_count"] == 0
            and missing_score.components["grounding"] == 0.0
        ),
        "grounding": missing_score.components["grounding"],
        "semantic_verified": missing_score.details["grounding"][
            "semantic_verified_count"
        ],
    }

    forged_evidence = [
        {"type": "observation", "claim": "伪造引用", "ref": "not-real",
         "field": "/latest", "value": 42.0},
        {"type": "diagnostic", "claim": "不完整引用", "ref": "e2"},
    ]
    forged_score = _score(
        _perfect_forecast(evidence=forged_evidence),
        tools_called={"get_observations", "get_diagnostics"},
        evidence_registry=registry,
    )
    checks["invalid_structured_citations_get_no_type_fallback"] = {
        "pass": (
            forged_score.components["grounding"] == 0.0
            and forged_score.details["grounding"]["structured_invalid_count"] == 2
        ),
        "grounding": forged_score.components["grounding"],
        "structured_invalid": forged_score.details["grounding"][
            "structured_invalid_count"
        ],
    }

    type_only_score = _score(
        _perfect_forecast(evidence=[
            {"type": "observation", "claim": "已查询实况但未给结构化字段"}
        ]),
        tools_called={"get_observations"}, evidence_registry=registry,
    )
    checks["unstructured_citation_retains_bounded_type_credit"] = {
        "pass": (
            type_only_score.components["grounding"]
                == RewardConfig().grounding_type_only_credit / 2
            and type_only_score.details["grounding"]["type_only_item_count"] == 1
        ),
        "grounding": type_only_score.components["grounding"],
        "type_only_items": type_only_score.details["grounding"]["type_only_item_count"],
    }

    checks["standard_change_is_explicit"] = {
        "pass": (
            pm25_to_level(70, standard=AQI_STANDARD_2012) == 2
            and pm25_to_level(70, standard=AQI_STANDARD_2026) == 3
        ),
        "pm25_70_level_2012": pm25_to_level(70, standard=AQI_STANDARD_2012),
        "pm25_70_level_2026": pm25_to_level(70, standard=AQI_STANDARD_2026),
    }

    metadata_registry = {
        "e1": {"tool": "get_diagnostics", "content": {"daily": {
            "2026-06-02": {"native_coverage": {"u10_ms": {"samples": 8}}}}}},
        "e2": {"tool": "get_model_guidance", "content": {"sources": {"cams": {
            "day_coverage": {"daily_pm25": {"2026-06-02": {"samples": 8}}}}}}},
    }
    metadata = _score(_perfect_forecast(evidence=[
        {"type": "diagnostic", "claim": "样本计数", "ref": "e1",
         "field": "/daily/2026-06-02/native_coverage/u10_ms/samples", "value": 8},
        {"type": "model_guidance", "claim": "样本计数", "ref": "e2",
         "field": "/sources/cams/day_coverage/daily_pm25/2026-06-02/samples", "value": 8},
    ]), tools_called={"get_diagnostics", "get_model_guidance"}, evidence_registry=metadata_registry)
    checks["metadata_only_citations_receive_zero_grounding"] = {
        "pass": metadata.components["grounding"] == 0.0,
        "grounding": metadata.components["grounding"],
    }

    same_level_truth = {"2026-06-02": 120.0, "2026-06-03": 140.0}
    same_level = score_forecast({"issue_date": ISSUE_DATE, "daily": [
        {"date": day, "pm25_range": [value, value]} for day, value in same_level_truth.items()]},
        same_level_truth, issue_date=ISSUE_DATE, horizon=2)
    checks["same_level_peak_uses_numeric_aqi_on_both_sides"] = {
        "pass": same_level.outcome_composite == 1.0,
        "turning": same_level.details["turning"],
    }
    gas_scores = {}
    for pollutant, value, field in (("SO2", 600, "so2_range"), ("NO2", 200, "no2_range"), ("CO", 18, "co_range")):
        truth = {"2026-06-02": {"PM2.5": 20, "PM10": 30, "O3": 40, pollutant: value}}
        result = score_forecast({"issue_date": ISSUE_DATE, "daily": [{"date": "2026-06-02",
            "pm25_range": [20, 20], "pm10_range": [30, 30], "o3_range": [40, 40], field: [value, value]}]},
            truth, issue_date=ISSUE_DATE, horizon=1)
        gas_scores[pollutant] = result.outcome_composite
    checks["each_optional_gas_can_receive_full_forecast_outcome_credit"] = {
        "pass": all(value == 1.0 for value in gas_scores.values()), "scores": gas_scores,
    }

    return {
        "artifact_type": "reward_contract_audit",
        "reward": reward_spec(),
        "scoring_implementation": file_identity(
            REPO_ROOT / "src/sitian/scoring.py", relative_to=REPO_ROOT
        ),
        "schema_implementation": file_identity(
            REPO_ROOT / "src/sitian/schema.py", relative_to=REPO_ROOT
        ),
        "checks": checks,
        "passed": all(row["pass"] for row in checks.values()),
        "scientific_boundary": (
            "These executable invariants prevent known scoring hacks; they do not validate "
            "the empirical choice of component weights or prove forecast skill."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "data/interim/reward_contract_audit.json")
    args = parser.parse_args()
    report = run_audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"-> {args.out}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
