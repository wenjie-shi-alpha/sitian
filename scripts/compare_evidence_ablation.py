#!/usr/bin/env python3
"""Compare full vs channel-masked policy probes with clustered paired inference.

Rollouts are paired by case/rep/seed, averaged within case, then issue dates are
resampled as blocks.  The primary estimand uses outcome_composite so losing a
grounding opportunity cannot masquerade as a loss of forecast skill.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.provenance import file_identity, score_implementation_matches
from sitian.scoring import reward_spec

COMPARISON_VERSION = "1.0.0"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(full: dict, masked: dict) -> None:
    if full.get("evidence_ablation") not in (None, []):
        raise ValueError("--full artifact must have no evidence ablation")
    if not masked.get("evidence_ablation"):
        raise ValueError("--masked artifact must declare evidence_ablation")
    for field in (
        "model", "policy_id", "policy_stage", "split", "n_cases", "group_size",
        "temperature", "case_sampling_seed", "rollout_seed_base", "rollout_seed_strategy",
        "thinking_enabled", "stratum_filter", "truth_event_filter", "stratified",
    ):
        if full.get(field) != masked.get(field):
            raise ValueError(f"paired artifact mismatch in {field}")
    full_reward = (full.get("provenance") or {}).get("reward") or {}
    masked_reward = (masked.get("provenance") or {}).get("reward") or {}
    if (full_reward.get("version"), full_reward.get("config_sha256")) != (
        masked_reward.get("version"), masked_reward.get("config_sha256")
    ):
        raise ValueError("paired reward identity mismatch")
    current = reward_spec()
    if (
        full_reward.get("version") != current.get("version")
        or full_reward.get("config_sha256") != current.get("config_sha256")
    ):
        raise ValueError("paired reward identity is stale")
    for label, artifact in (("full", full), ("masked", masked)):
        if not score_implementation_matches(
            artifact.get("provenance") or {}, REPO_ROOT
        ):
            raise ValueError(
                f"{label} scoring/schema implementation identity is missing or stale"
            )


def _rows(artifact: dict) -> dict[tuple[str, int], dict]:
    return {(row["case_id"], int(row["rep"])): row for row in artifact["rows"]}


def _paired_cases(full: dict, masked: dict, score_field: str) -> list[dict]:
    full_rows, masked_rows = _rows(full), _rows(masked)
    if set(full_rows) != set(masked_rows):
        raise ValueError("paired artifacts do not contain identical case/rep keys")
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for key in sorted(full_rows):
        left, right = full_rows[key], masked_rows[key]
        if left.get("rollout_seed") != right.get("rollout_seed"):
            raise ValueError(f"rollout seed mismatch at {key}")
        grouped[key[0]].append((left, right))
    cases = []
    for case_id, pairs in grouped.items():
        full_values = [float(left.get(score_field) or 0.0) for left, _ in pairs]
        masked_values = [float(right.get(score_field) or 0.0) for _, right in pairs]
        cases.append({
            "case_id": case_id,
            "issue_date": pairs[0][0]["issue_date"],
            "stratum": pairs[0][0].get("stratum"),
            "rollouts": len(pairs),
            "full": _mean(full_values),
            "masked": _mean(masked_values),
            "full_minus_masked": _mean(full_values) - _mean(masked_values),
        })
    return cases


def _inference(cases: list[dict], replicates: int, seed: int) -> dict:
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in cases:
        by_date[row["issue_date"]].append(row)
    blocks = sorted(by_date)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(replicates):
        sampled = [rng.choice(blocks) for _ in blocks]
        bootstrap.append(_mean([
            row["full_minus_masked"] for block in sampled for row in by_date[block]
        ]))
    difference = _mean([row["full_minus_masked"] for row in cases])
    interval = [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)]
    estimable = len(blocks) >= 30
    return {
        "n_cases": len(cases),
        "issue_date_clusters": len(blocks),
        "full_mean": round(_mean([row["full"] for row in cases]), 6),
        "masked_mean": round(_mean([row["masked"] for row in cases]), 6),
        "full_minus_masked": round(difference, 6),
        "cluster_bootstrap_95pct": [round(value, 6) for value in interval],
        "inference_status": (
            "estimable" if estimable else "descriptive_only_too_few_independent_clusters"
        ),
        "supports_positive_evidence_dose_response": bool(estimable and interval[0] > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--masked", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    full_path, masked_path = REPO_ROOT / args.full, REPO_ROOT / args.masked
    full, masked = _load(full_path), _load(masked_path)
    _validate(full, masked)
    outcome_cases = _paired_cases(full, masked, "outcome_composite")
    reward_cases = _paired_cases(full, masked, "reward")
    strata = sorted({row.get("stratum") for row in outcome_cases if row.get("stratum")})
    full_tools = Counter(tool for row in full["rows"] for tool in set(row.get("tools", [])))
    masked_tools = Counter(tool for row in masked["rows"] for tool in set(row.get("tools", [])))
    n_rollouts = len(full["rows"])
    report = {
        "artifact_type": "evidence_dose_response_comparison",
        "comparison_version": COMPARISON_VERSION,
        "comparison_implementation": file_identity(Path(__file__), relative_to=REPO_ROOT),
        "policy_stage": full.get("policy_stage"),
        "policy_id": full.get("policy_id") or full.get("model"),
        "split": full.get("split"),
        "full": file_identity(full_path, relative_to=REPO_ROOT),
        "masked": file_identity(masked_path, relative_to=REPO_ROOT),
        "masked_channels": sorted(masked["evidence_ablation"]),
        "reward": (full.get("provenance") or {}).get("reward"),
        "scoring_implementation": (full.get("provenance") or {}).get(
            "scoring_implementation"
        ),
        "schema_implementation": (full.get("provenance") or {}).get(
            "schema_implementation"
        ),
        "method": {
            "paired_by": ["case_id", "rep", "rollout_seed"],
            "case_rollouts_averaged_first": True,
            "failed_rollouts_retained_as_zero": True,
            "cluster": "issue_date",
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "primary_estimand": "full minus masked outcome_composite",
            "grounding_excluded_from_primary_estimand": True,
        },
        "outcome_composite": _inference(outcome_cases, args.bootstrap, args.seed),
        "training_composite_diagnostic": _inference(
            reward_cases, args.bootstrap, args.seed + 1
        ),
        "outcome_by_stratum": {
            stratum: _inference(
                [row for row in outcome_cases if row.get("stratum") == stratum],
                args.bootstrap,
                args.seed + 10 + index,
            )
            for index, stratum in enumerate(strata)
            if any(row.get("stratum") == stratum for row in outcome_cases)
        },
        "tool_use_rate": {
            tool: {
                "full": round(full_tools[tool] / n_rollouts, 6),
                "masked": round(masked_tools[tool] / n_rollouts, 6),
            }
            for tool in sorted(set(full_tools) | set(masked_tools))
        },
        "cases": outcome_cases,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"},
                     ensure_ascii=False, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
