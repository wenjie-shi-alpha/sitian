#!/usr/bin/env python3
"""Pre-registered outcome-weight sensitivity for agent vs strong tabular.

The analysis never changes component definitions and never treats rollout
replicates as independent cases.  It recombines already audited outcome
components under a fixed scenario grid, averages policy rollouts within case,
then block-bootstraps issue dates.  Hard metrics remain the graduation target;
this artifact only reveals whether an outcome-composite conclusion is a
single-weight artefact.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.provenance import file_identity, score_implementation_matches  # noqa: E402
from sitian.scoring import OUTCOME_COMPONENTS, reward_spec  # noqa: E402

AUDIT_VERSION = "1.0.0"


def _validate_score_artifact(artifact: dict, label: str) -> None:
    current = reward_spec()
    provenance = artifact.get("provenance") or {}
    identity = provenance.get("reward") or {}
    if (
        identity.get("version") != current["version"]
        or identity.get("config_sha256") != current["config_sha256"]
    ):
        raise ValueError(f"{label} reward identity is stale")
    if not score_implementation_matches(provenance, REPO_ROOT):
        raise ValueError(f"{label} scoring/schema implementation identity is missing or stale")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = q * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return values[low] * (1 - fraction) + values[high] * fraction


def scenario_grid() -> dict[str, dict[str, float]]:
    default = {
        key: float(reward_spec()["config"]["weights"][key])
        for key in OUTCOME_COMPONENTS
    }

    def normalized(values: dict[str, float]) -> dict[str, float]:
        total = sum(values.values())
        return {key: value / total for key, value in values.items()}

    scenarios = {"default": normalized(default)}
    scenarios["equal_components"] = normalized({key: 1.0 for key in OUTCOME_COMPONENTS})
    for key in OUTCOME_COMPONENTS:
        for suffix, multiplier in (("minus25pct", 0.75), ("plus25pct", 1.25)):
            varied = dict(default)
            varied[key] *= multiplier
            scenarios[f"{key}_{suffix}"] = normalized(varied)
    return scenarios


def recombine(components: dict, weights: dict[str, float]) -> float:
    active = {
        key: float(components[key]) for key in OUTCOME_COMPONENTS
        if components.get(key) is not None
    }
    if not active:
        return 0.0
    denominator = sum(weights[key] for key in active)
    return sum(active[key] * weights[key] for key in active) / denominator


def analyze(
    probe: dict,
    tabular: dict,
    *,
    tabular_split: str,
    bootstrap: int,
    seed: int,
) -> dict:
    current = reward_spec()
    _validate_score_artifact(probe, "probe")
    _validate_score_artifact(tabular, "tabular")

    tabular_rows = (tabular.get("results") or {}).get(tabular_split, {}).get("rows", [])
    tabular_by_case = {row["case_id"]: row for row in tabular_rows}
    probe_by_case: dict[str, list[dict]] = defaultdict(list)
    for row in probe.get("rows", []):
        probe_by_case[row["case_id"]].append(row)
    common = sorted(set(probe_by_case) & set(tabular_by_case))
    if not common:
        raise ValueError("probe and tabular have no common cases")

    scenarios = scenario_grid()
    case_rows = []
    for case_id in common:
        policy_rows = probe_by_case[case_id]
        issue_date = next((row.get("issue_date") for row in policy_rows
                           if row.get("issue_date")), None)
        if issue_date is None:
            raise ValueError(f"missing issue_date for {case_id}")
        tabular_row = tabular_by_case[case_id]
        scores = {}
        for name, weights in scenarios.items():
            policy = _mean([
                recombine(row.get("components") or {}, weights)
                if row.get("submitted") else 0.0
                for row in policy_rows
            ])
            baseline = recombine(tabular_row.get("components") or {}, weights)
            scores[name] = {
                "policy": policy,
                "tabular": baseline,
                "difference": policy - baseline,
            }
        case_rows.append({
            "case_id": case_id,
            "issue_date": issue_date,
            "stratum": policy_rows[0].get("stratum"),
            "rollouts": len(policy_rows),
            "scores": scores,
        })

    blocks: dict[str, list[dict]] = defaultdict(list)
    for row in case_rows:
        blocks[row["issue_date"]].append(row)
    block_ids = sorted(blocks)
    rng = random.Random(seed)
    scenario_rows = []
    for name, weights in scenarios.items():
        differences = [row["scores"][name]["difference"] for row in case_rows]
        boot = []
        for _ in range(bootstrap):
            sampled = [rng.choice(block_ids) for _ in block_ids]
            boot.append(_mean([
                row["scores"][name]["difference"]
                for block in sampled for row in blocks[block]
            ]))
        difference = _mean(differences)
        ci = [_percentile(boot, 0.025), _percentile(boot, 0.975)]
        scenario_rows.append({
            "scenario": name,
            "weights": {key: round(value, 6) for key, value in weights.items()},
            "policy_mean": round(_mean([
                row["scores"][name]["policy"] for row in case_rows
            ]), 6),
            "tabular_mean": round(_mean([
                row["scores"][name]["tabular"] for row in case_rows
            ]), 6),
            "paired_difference": round(difference, 6),
            "cluster_bootstrap_95pct": [round(value, 6) for value in ci],
            "direction": "policy_above" if difference > 0 else (
                "policy_below" if difference < 0 else "tie"
            ),
        })

    directions = {row["direction"] for row in scenario_rows}
    enough_clusters = len(block_ids) >= 30
    return {
        "artifact_type": "reward_weight_sensitivity",
        "audit_version": AUDIT_VERSION,
        "audit_implementation": file_identity(Path(__file__), relative_to=REPO_ROOT),
        "reward": current,
        "scoring_implementation": (probe.get("provenance") or {}).get(
            "scoring_implementation"
        ),
        "schema_implementation": (probe.get("provenance") or {}).get(
            "schema_implementation"
        ),
        "policy_stage": probe.get("policy_stage"),
        "policy_id": probe.get("policy_id") or probe.get("model"),
        "split": probe.get("split"),
        "tabular_split": tabular_split,
        "n_cases": len(case_rows),
        "n_issue_date_clusters": len(block_ids),
        "inference_status": (
            "estimable" if enough_clusters else
            "descriptive_only_too_few_independent_clusters"
        ),
        "method": {
            "grid": "default, equal, and one-component-at-a-time +/-25%",
            "case_rollouts_averaged_first": True,
            "failed_rollouts_retained_as_zero": True,
            "cluster": "issue_date",
            "bootstrap_replicates": bootstrap,
            "seed": seed,
            "interactive_auxiliary_components_excluded": True,
        },
        "scenarios": scenario_rows,
        "direction_stable_across_grid": len(directions) == 1,
        "directions_observed": sorted(directions),
        "scientific_boundary": (
            "This grid detects conclusions driven by one reasonable outcome-weight choice. "
            "It does not replace pre-registered hard metrics or justify the component definitions."
        ),
        "cases": case_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="data/interim/probe_open_evidence_v079_n50_g8.json")
    parser.add_argument("--tabular", default="data/interim/eval_tabular_open_evidence_v079.json")
    parser.add_argument("--tabular-split", default="val")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/interim/reward_weight_sensitivity_v079.json")
    args = parser.parse_args()
    probe_path = REPO_ROOT / args.probe
    tabular_path = REPO_ROOT / args.tabular
    report = analyze(
        json.loads(probe_path.read_text(encoding="utf-8")),
        json.loads(tabular_path.read_text(encoding="utf-8")),
        tabular_split=args.tabular_split,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    report["probe"] = file_identity(probe_path, relative_to=REPO_ROOT)
    report["tabular"] = file_identity(tabular_path, relative_to=REPO_ROOT)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"},
                     ensure_ascii=False, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
