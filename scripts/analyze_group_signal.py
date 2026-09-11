#!/usr/bin/env python3
"""诊断单个 grouped-rollout probe 的组内 reward 区分度。"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.provenance import file_identity, score_implementation_matches  # noqa: E402
from sitian.scoring import reward_spec  # noqa: E402

COMPONENTS = ("level", "event", "interval", "turning", "primary", "grounding")
DECISION_COMPONENTS = ("level", "event", "turning", "primary")
ZERO_TOLERANCE = 1e-12


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    probe_path = REPO_ROOT / args.probe
    artifact = json.loads(probe_path.read_text(encoding="utf-8"))
    provenance = artifact.get("provenance") or {}
    observed_reward = provenance.get("reward") or {}
    current_reward = reward_spec()
    if (
        observed_reward.get("version") != current_reward.get("version")
        or observed_reward.get("config_sha256") != current_reward.get("config_sha256")
    ):
        raise ValueError("probe reward identity is stale")
    if not score_implementation_matches(provenance, REPO_ROOT):
        raise ValueError("probe scoring/schema implementation identity is missing or stale")
    if not artifact.get("rows") or not all(
        isinstance(row.get("truth_has_event"), bool) for row in artifact["rows"]
    ):
        raise ValueError("probe lacks the complete truth-derived event audit; rebuild it")
    rows = [row for row in artifact["rows"] if row.get("submitted")]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)

    cases = []
    pair_gaps = []
    for case_id, items in grouped.items():
        rewards = [float(row["reward"]) for row in items]
        pair_gaps.extend(abs(a - b) for a, b in itertools.combinations(rewards, 2))
        component_sigmas = {}
        for component in COMPONENTS:
            values = [float(row["components"][component]) for row in items
                      if (row.get("components") or {}).get(component) is not None]
            component_sigmas[component] = (
                statistics.pstdev(values) if len(values) >= 2 else 0.0
            )
        cases.append({
            "case_id": case_id,
            "stratum": items[0].get("stratum"),
            "truth_has_event": items[0]["truth_has_event"],
            "valid_rollouts": len(items),
            "reward_sigma": statistics.pstdev(rewards),
            "reward_range": max(rewards) - min(rewards),
            "unique_rewards": len(set(rewards)),
            "component_sigmas": component_sigmas,
        })

    sigmas = [case["reward_sigma"] for case in cases]
    ranges = [case["reward_range"] for case in cases]
    thresholds = []
    for threshold in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        qualifying = sum(value > threshold for value in ranges)
        pair_qualifying = sum(value > threshold for value in pair_gaps)
        thresholds.append({
            "threshold": threshold,
            "threshold_label": f"> {threshold:g}",
            "qualifying_groups": qualifying,
            "group_rate": qualifying / len(cases),
            "qualifying_pairs": pair_qualifying,
            "pair_rate": pair_qualifying / len(pair_gaps),
        })

    strata = []
    for stratum in sorted({case["stratum"] for case in cases}):
        subset = [case for case in cases if case["stratum"] == stratum]
        variable = sum(case["reward_sigma"] > ZERO_TOLERANCE for case in subset)
        decision = sum(any(case["component_sigmas"][name] > ZERO_TOLERANCE
                           for name in DECISION_COMPONENTS) for case in subset)
        event = sum(case["component_sigmas"]["event"] > ZERO_TOLERANCE for case in subset)
        interval_only = sum(
            case["component_sigmas"]["interval"] > ZERO_TOLERANCE
            and not any(case["component_sigmas"][name] > ZERO_TOLERANCE
                        for name in DECISION_COMPONENTS)
            for case in subset
        )
        strata.append({
            "stratum": stratum,
            "cases": len(subset),
            "variable_composite_groups": variable,
            "variable_composite_rate": variable / len(subset),
            "variable_decision_component_groups": decision,
            "variable_decision_component_rate": decision / len(subset),
            "variable_event_component_groups": event,
            "variable_event_component_rate": event / len(subset),
            "interval_only_groups": interval_only,
            "mean_reward_sigma": _mean([case["reward_sigma"] for case in subset]),
            "median_reward_sigma": statistics.median(
                [case["reward_sigma"] for case in subset]
            ),
        })

    component_rows = []
    for component in COMPONENTS:
        values = [case["component_sigmas"][component] for case in cases]
        nonzero = [value for value in values if value > ZERO_TOLERANCE]
        component_rows.append({
            "component": component,
            "variable_groups": len(nonzero),
            "variable_rate": len(nonzero) / len(cases),
            "mean_sigma_all_groups": _mean(values),
            "median_sigma_variable_groups": statistics.median(nonzero) if nonzero else None,
        })

    event_rows = [row for row in rows if row["truth_has_event"]]
    event_cases = [case for case in cases if case["truth_has_event"]]
    event_component_values = [float(row["components"]["event"]) for row in event_rows]
    ordered = sorted(cases, key=lambda case: case["reward_sigma"], reverse=True)
    sigma_total = sum(sigmas)
    unique_counts = Counter(case["unique_rewards"] for case in cases)
    report = {
        "artifact_type": "group_reward_signal_diagnostic",
        "analysis_version": "1.1.0",
        "analysis_implementation": file_identity(Path(__file__), relative_to=REPO_ROOT),
        "probe": file_identity(probe_path, relative_to=REPO_ROOT),
        "reward": (artifact.get("provenance") or {}).get("reward"),
        "scoring_implementation": (artifact.get("provenance") or {}).get(
            "scoring_implementation"
        ),
        "schema_implementation": (artifact.get("provenance") or {}).get(
            "schema_implementation"
        ),
        "definitions": {
            "population": "thinking-on Round-2 final valid submissions",
            "grain": "50 cases x up to 4 valid rollouts",
            "population_sigma": True,
            "zero_tolerance": ZERO_TOLERANCE,
            "decision_components": list(DECISION_COMPONENTS),
            "pair_denominator": len(pair_gaps),
            "event_population": (
                "complete hidden truth has at least one daily AQI level >=4; "
                "independent of mutually exclusive sampling stratum"
            ),
        },
        "overall": {
            "cases": len(cases),
            "valid_rollouts": len(rows),
            "zero_variance_groups": sum(value <= ZERO_TOLERANCE for value in sigmas),
            "nonzero_group_rate": sum(value > ZERO_TOLERANCE for value in sigmas) / len(cases),
            "range_gt_0_01_group_rate": sum(value > 0.01 for value in ranges) / len(cases),
            "range_gt_0_05_group_rate": sum(value > 0.05 for value in ranges) / len(cases),
            "mean_sigma": _mean(sigmas),
            "median_sigma": statistics.median(sigmas),
            "sigma_quantiles": {
                f"p{int(q * 100):02d}": _quantile(sigmas, q)
                for q in (0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
            },
            "range_quantiles": {
                f"p{int(q * 100):02d}": _quantile(ranges, q)
                for q in (0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
            },
            "unique_reward_groups": {
                str(unique): unique_counts.get(unique, 0) for unique in range(1, 5)
            },
            "groups_with_3_or_4_unique_rewards": sum(
                case["unique_rewards"] >= 3 for case in cases
            ),
            "decision_component_variable_groups": sum(
                any(case["component_sigmas"][name] > ZERO_TOLERANCE
                    for name in DECISION_COMPONENTS) for case in cases
            ),
            "interval_only_variable_groups": sum(
                case["component_sigmas"]["interval"] > ZERO_TOLERANCE
                and not any(case["component_sigmas"][name] > ZERO_TOLERANCE
                            for name in DECISION_COMPONENTS)
                for case in cases
            ),
            "top_2_share_of_summed_sigma": (
                sum(case["reward_sigma"] for case in ordered[:2]) / sigma_total
            ),
            "top_10_share_of_summed_sigma": (
                sum(case["reward_sigma"] for case in ordered[:10]) / sigma_total
            ),
            "mean_sigma_excluding_top_2": _mean(
                [case["reward_sigma"] for case in ordered[2:]]
            ),
        },
        "event_core_signal": {
            "event_cases": len({row["case_id"] for row in event_rows}),
            "valid_event_rollouts": len(event_rows),
            "event_component_mean": (
                _mean(event_component_values) if event_component_values else None
            ),
            "event_component_sigma": (
                statistics.pstdev(event_component_values)
                if event_component_values else None
            ),
            "event_component_positive_rollouts": sum(value > 0 for value in event_component_values),
            "event_composite_variable_groups": sum(
                case["reward_sigma"] > ZERO_TOLERANCE for case in event_cases
            ),
            "event_decision_component_variable_groups": sum(
                any(case["component_sigmas"][name] > ZERO_TOLERANCE
                    for name in DECISION_COMPONENTS) for case in event_cases
            ),
            "event_event_component_variable_groups": sum(
                case["component_sigmas"]["event"] > ZERO_TOLERANCE
                for case in event_cases
            ),
        },
        "threshold_sensitivity": thresholds,
        "by_stratum": strata,
        "by_component": component_rows,
        "cases": cases,
    }
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "definitions", "overall", "event_core_signal", "threshold_sensitivity",
        "by_stratum", "by_component"
    )}, ensure_ascii=False, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
