#!/usr/bin/env python3
"""配对分析 thinking off/on probe，并按 Round-2 预注册规则裁决训练臂。

同一 case 的 rollout 先聚合；bootstrap 按 issue_date 整块重采样。格式失败在
均分比较中保留为 0，但组内决策方差同时报告全部 rollout 与仅最终有效提交两套口径。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.provenance import file_identity, score_implementation_matches  # noqa: E402
from sitian.scoring import reward_spec  # noqa: E402

ANALYSIS_VERSION = "1.1.0"
ZERO_TOLERANCE = 1e-12


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def _sigma(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) >= 2 else None


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    return dict(grouped)


def _arm_summary(artifact: dict) -> tuple[dict, dict[str, dict]]:
    rows = artifact["rows"]
    grouped = _group_rows(rows)
    cases = {}
    for case_id, items in grouped.items():
        all_rewards = [float(row.get("reward", 0.0)) for row in items]
        valid_rewards = [float(row["reward"]) for row in items if row.get("submitted")]
        cases[case_id] = {
            "issue_date": items[0]["issue_date"],
            "stratum": items[0].get("stratum"),
            "truth_has_event": items[0]["truth_has_event"],
            "reward_mean_all": _mean(all_rewards),
            "sigma_all": _sigma(all_rewards),
            "sigma_valid": _sigma(valid_rewards),
            "valid_rollouts": len(valid_rewards),
        }

    all_sigmas = [case["sigma_all"] for case in cases.values()
                  if case["sigma_all"] is not None]
    valid_sigmas = [case["sigma_valid"] for case in cases.values()
                    if case["sigma_valid"] is not None]
    event_sigmas = [case["sigma_valid"] for case in cases.values()
                    if case["truth_has_event"] and case["sigma_valid"] is not None]
    submitted = [row for row in rows if row.get("submitted")]
    first_valid = [row for row in rows
                   if row.get("submitted") and int(row.get("invalid_submits", 0)) == 0]
    usage = [row.get("usage") or {} for row in rows]
    summary = {
        "rollouts": len(rows),
        "cases": len(cases),
        "event_cases": sum(case["truth_has_event"] for case in cases.values()),
        "eventual_submit_rate": len(submitted) / len(rows),
        "first_submit_valid_rate": len(first_valid) / len(rows),
        "mean_reward_all": _mean([float(row.get("reward", 0.0)) for row in rows]),
        "mean_reward_submitted": _mean([float(row["reward"]) for row in submitted]),
        "zero_variance_groups_all": sum(
            sigma <= ZERO_TOLERANCE for sigma in all_sigmas
        ),
        "usable_group_rate_all": sum(
            sigma > ZERO_TOLERANCE for sigma in all_sigmas
        ) / len(all_sigmas),
        "zero_variance_groups_valid_only": sum(
            sigma <= ZERO_TOLERANCE for sigma in valid_sigmas
        ),
        "usable_group_rate_valid_only": sum(
            sigma > ZERO_TOLERANCE for sigma in valid_sigmas
        ) / len(valid_sigmas),
        "mean_sigma_valid_only": _mean(valid_sigmas),
        "event_mean_sigma_valid_only": _mean(event_sigmas),
        "invalid_submit_attempts": sum(int(row.get("invalid_submits", 0)) for row in rows),
        "completion_tokens_mean": _mean(
            [float(item["completion_tokens"]) for item in usage if item.get("completion_tokens") is not None]
        ),
        "completion_tokens_mean_submitted": _mean([
            float((row.get("usage") or {})["completion_tokens"])
            for row in submitted
            if (row.get("usage") or {}).get("completion_tokens") is not None
        ]),
    }
    return summary, cases


def _bootstrap_paired(
    off_cases: dict[str, dict],
    on_cases: dict[str, dict],
    metric: Callable[[dict, dict], float | None],
    *,
    replicates: int,
    seed: int,
    truth_event_only: bool = False,
) -> dict:
    records = []
    for case_id in sorted(off_cases):
        off = off_cases[case_id]
        on = on_cases[case_id]
        if truth_event_only and not off["truth_has_event"]:
            continue
        value = metric(off, on)
        if value is not None:
            records.append((off["issue_date"], value))
    if not records:
        raise ValueError("no paired cases for bootstrap metric")
    by_block: dict[str, list[float]] = defaultdict(list)
    for block, value in records:
        by_block[block].append(value)
    blocks = sorted(by_block)
    rng = random.Random(seed)
    sampled_means = []
    for _ in range(replicates):
        sampled = [rng.choice(blocks) for _ in blocks]
        sampled_means.append(_mean([value for block in sampled for value in by_block[block]]))
    estimate = _mean([value for _, value in records])
    return {
        "estimate": round(estimate, 6),
        "cluster_bootstrap_95pct": [
            round(_percentile(sampled_means, 0.025), 6),
            round(_percentile(sampled_means, 0.975), 6),
        ],
        "cases": len(records),
        "issue_date_clusters": len(blocks),
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_pair(off: dict, on: dict) -> None:
    if off.get("thinking_enabled") is not False or on.get("thinking_enabled") is not True:
        raise ValueError("expected --off thinking_enabled=false and --on thinking_enabled=true")
    for field in ("model", "split", "n_cases", "group_size", "temperature",
                  "case_sampling_seed", "rollout_seed_base", "rollout_seed_strategy"):
        if off.get(field) != on.get(field):
            raise ValueError(f"paired artifact mismatch in {field}: {off.get(field)!r} != {on.get(field)!r}")
    off_reward = (off.get("provenance") or {}).get("reward") or {}
    on_reward = (on.get("provenance") or {}).get("reward") or {}
    for field in ("version", "config_sha256"):
        if off_reward.get(field) != on_reward.get(field):
            raise ValueError(f"paired reward mismatch in {field}")
    current = reward_spec()
    if (
        off_reward.get("version") != current.get("version")
        or off_reward.get("config_sha256") != current.get("config_sha256")
    ):
        raise ValueError("paired reward identity is stale")
    for label, artifact in (("off", off), ("on", on)):
        if not artifact.get("rows") or not all(
            isinstance(row.get("truth_has_event"), bool) for row in artifact["rows"]
        ):
            raise ValueError(f"{label} probe lacks complete truth-derived event flags")
        if not score_implementation_matches(
            artifact.get("provenance") or {}, REPO_ROOT
        ):
            raise ValueError(
                f"{label} scoring/schema implementation identity is missing or stale"
            )
    off_rows = {(row["case_id"], row["rep"]): row for row in off["rows"]}
    on_rows = {(row["case_id"], row["rep"]): row for row in on["rows"]}
    if set(off_rows) != set(on_rows):
        raise ValueError("paired artifacts do not contain identical case_id/rep keys")
    for key in off_rows:
        if off_rows[key].get("rollout_seed") != on_rows[key].get("rollout_seed"):
            raise ValueError(f"rollout seed mismatch at {key}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--format-threshold", type=float, default=0.95)
    ap.add_argument("--mean-noninferiority-margin", type=float, default=-0.03)
    ap.add_argument("--expected-reward", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    off_path, on_path = REPO_ROOT / args.off, REPO_ROOT / args.on
    off, on = _load(off_path), _load(on_path)
    _validate_pair(off, on)
    reward = (off.get("provenance") or {}).get("reward") or {}
    if args.expected_reward and reward.get("version") != args.expected_reward:
        raise ValueError(
            f"expected reward {args.expected_reward}, got {reward.get('version')}"
        )

    off_summary, off_cases = _arm_summary(off)
    on_summary, on_cases = _arm_summary(on)
    if set(off_cases) != set(on_cases):
        raise ValueError("case sets differ after grouping")

    reward_delta = _bootstrap_paired(
        off_cases, on_cases,
        lambda a, b: b["reward_mean_all"] - a["reward_mean_all"],
        replicates=args.bootstrap, seed=args.seed,
    )
    sigma_delta = _bootstrap_paired(
        off_cases, on_cases,
        lambda a, b: (None if a["sigma_valid"] is None or b["sigma_valid"] is None
                      else b["sigma_valid"] - a["sigma_valid"]),
        replicates=args.bootstrap, seed=args.seed + 1,
    )
    event_sigma_delta = _bootstrap_paired(
        off_cases, on_cases,
        lambda a, b: (None if a["sigma_valid"] is None or b["sigma_valid"] is None
                      else b["sigma_valid"] - a["sigma_valid"]),
        replicates=args.bootstrap, seed=args.seed + 2, truth_event_only=True,
    )
    usable_delta = _bootstrap_paired(
        off_cases, on_cases,
        lambda a, b: (None if a["sigma_valid"] is None or b["sigma_valid"] is None else
                      float(b["sigma_valid"] > ZERO_TOLERANCE)
                      - float(a["sigma_valid"] > ZERO_TOLERANCE)),
        replicates=args.bootstrap, seed=args.seed + 3,
    )

    format_gate = (
        off_summary["first_submit_valid_rate"] >= args.format_threshold
        and on_summary["first_submit_valid_rate"] >= args.format_threshold
    )
    mean_noninferior = (
        reward_delta["cluster_bootstrap_95pct"][0] > args.mean_noninferiority_margin
    )
    usable_advantage = (
        on_summary["usable_group_rate_valid_only"]
        > off_summary["usable_group_rate_valid_only"]
    )
    variance_advantage = (
        on_summary["mean_sigma_valid_only"] > off_summary["mean_sigma_valid_only"]
        and on_summary["event_mean_sigma_valid_only"]
        > off_summary["event_mean_sigma_valid_only"]
    )
    choose_on = format_gate and mean_noninferior and usable_advantage and variance_advantage
    selected = (
        "pending_format_remediation" if not format_gate else
        ("on" if choose_on else "off")
    )
    report = {
        "artifact_type": "thinking_ablation_comparison",
        "analysis_version": ANALYSIS_VERSION,
        "analysis_implementation": file_identity(Path(__file__), relative_to=REPO_ROOT),
        "off": file_identity(off_path, relative_to=REPO_ROOT),
        "on": file_identity(on_path, relative_to=REPO_ROOT),
        "reward": reward,
        "scoring_implementation": (off.get("provenance") or {}).get(
            "scoring_implementation"
        ),
        "schema_implementation": (off.get("provenance") or {}).get(
            "schema_implementation"
        ),
        "method": {
            "paired_by": ["case_id", "rep", "rollout_seed"],
            "case_rollouts_averaged_before_mean_inference": True,
            "failed_rollouts_retained_as_zero_for_mean": True,
            "decision_sigma_uses_final_valid_submissions_only": True,
            "population_sigma": True,
            "zero_tolerance": ZERO_TOLERANCE,
            "bootstrap_cluster": "issue_date",
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "event_population": (
                "complete hidden truth has at least one daily AQI level >=4; "
                "independent of mutually exclusive sampling stratum"
            ),
        },
        "arms": {"off": off_summary, "on": on_summary},
        "paired_deltas_on_minus_off": {
            "mean_reward_all": reward_delta,
            "mean_sigma_valid_only": sigma_delta,
            "event_mean_sigma_valid_only": event_sigma_delta,
            "usable_group_rate_valid_only": usable_delta,
        },
        "cost": {
            "completion_token_ratio_on_over_off": round(
                on_summary["completion_tokens_mean"] / off_summary["completion_tokens_mean"], 4
            ),
            "completion_token_ratio_on_over_off_submitted": round(
                on_summary["completion_tokens_mean_submitted"]
                / off_summary["completion_tokens_mean_submitted"], 4
            ),
            "note": (
                "API prompt usage is cumulative across multi-turn prefixes; these telemetry totals "
                "do not identify training FLOPs."
            ),
        },
        "round2_preregistered_decision": {
            "format_threshold_each_arm": args.format_threshold,
            "mean_delta_ci_lower_bound_threshold": args.mean_noninferiority_margin,
            "format_gate_pass": format_gate,
            "mean_noninferiority_pass": mean_noninferior,
            "usable_group_advantage_on": usable_advantage,
            "valid_and_event_sigma_advantage_on": variance_advantage,
            "skip_format_sft": format_gate,
            "selected_training_thinking": selected,
            "decision_pass_for_thinking_on": choose_on,
        },
    }
    out_path = REPO_ROOT / args.out
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({
        "reward": reward.get("version"),
        "arms": report["arms"],
        "paired_deltas_on_minus_off": report["paired_deltas_on_minus_off"],
        "cost": report["cost"],
        "decision": report["round2_preregistered_decision"],
        "out": str(out_path),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
