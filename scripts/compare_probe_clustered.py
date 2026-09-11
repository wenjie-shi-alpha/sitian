#!/usr/bin/env python3
"""对 agent probe 与基线做配对、分块 bootstrap。

先在 case 内平均多条 rollout，再按起报日或固定天数的时间块
整块重采样；绝不把同 case 的 G 条 rollout 当作 G 个独立样本。
格式失败的 rollout 保留 reward=0，避免按成功轨迹条件化后高估策略。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.provenance import file_identity, score_implementation_matches  # noqa: E402
from sitian.scoring import reward_spec  # noqa: E402

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})$")
COMPARISON_VERSION = "1.2.0"


def _validate_score_artifact(artifact: dict, label: str) -> None:
    current = reward_spec()
    provenance = artifact.get("provenance") or {}
    observed = provenance.get("reward") or {}
    if (
        observed.get("version") != current.get("version")
        or observed.get("config_sha256") != current.get("config_sha256")
    ):
        raise ValueError(f"{label} reward identity is stale")
    if not score_implementation_matches(provenance, REPO_ROOT):
        raise ValueError(f"{label} scoring/schema implementation identity is missing or stale")


def _mean(values) -> float:
    return sum(values) / len(values)


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--baseline", default="guidance",
                    help="probe row.baselines 中的 key，或 tabular")
    ap.add_argument("--tabular", default="data/interim/eval_tabular.json")
    ap.add_argument("--tabular-split", default="val")
    population = ap.add_mutually_exclusive_group()
    population.add_argument("--stratum", default=None,
                            help="可选，只比较指定互斥采样层")
    population.add_argument("--truth-event-only", action="store_true",
                            help="只比较隐藏真值中至少一天 AQI>=4 的完整事件层")
    ap.add_argument("--block", choices=("issue_date", "calendar"), default="issue_date")
    ap.add_argument("--block-days", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    probe_path = REPO_ROOT / args.probe
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    _validate_score_artifact(probe, "probe")
    if args.truth_event_only and not all(
        isinstance(row.get("truth_has_event"), bool) for row in probe.get("rows", [])
    ):
        raise ValueError("probe lacks complete truth-derived event flags")
    tabular_by_case = {}
    tabular_path = REPO_ROOT / args.tabular
    if args.baseline == "tabular":
        tabular = json.loads(tabular_path.read_text(encoding="utf-8"))
        _validate_score_artifact(tabular, "tabular")
        if any("outcome_composite" not in row
               for row in tabular["results"][args.tabular_split]["rows"]):
            raise RuntimeError(
                "tabular artifact predates outcome_composite; rebuild under reward v0.7+"
            )
        tabular_by_case = {
            r["case_id"]: r["outcome_composite"]
            for r in tabular["results"][args.tabular_split]["rows"]
        }

    grouped = defaultdict(list)
    metadata = {}
    for row in probe["rows"]:
        if args.truth_event_only and row.get("truth_has_event") is not True:
            continue
        if args.stratum and row.get("stratum") != args.stratum:
            continue
        if "outcome_composite" not in row:
            raise RuntimeError(
                "probe predates outcome_composite; rebuild under reward v0.7+ instead of "
                "mixing interactive training reward with baseline evaluation scores"
            )
        case_id = row["case_id"]
        # Unsubmitted / invalid rollouts carry no outcome score; the pre-registered
        # convention keeps them in the case mean as 0 (format failure is a zero,
        # not a missing value).
        grouped[case_id].append(float(row.get("outcome_composite") or 0.0))
        match = DATE_RE.search(case_id)
        issue_date = row.get("issue_date") or (match.group(1) if match else None)
        baseline = (tabular_by_case.get(case_id) if args.baseline == "tabular"
                    else row.get("baselines", {}).get(args.baseline))
        meta = metadata.setdefault(case_id, {
            "issue_date": issue_date, "baseline": baseline,
            "stratum": row.get("stratum"), "submitted": 0,
        })
        if meta["baseline"] is None and baseline is not None:
            meta["baseline"] = baseline
        meta["submitted"] += int(bool(row.get("submitted")))

    cases = []
    for case_id, values in grouped.items():
        meta = metadata[case_id]
        if meta["issue_date"] is None or meta["baseline"] is None:
            continue
        day = date.fromisoformat(meta["issue_date"])
        block = (meta["issue_date"] if args.block == "issue_date"
                 else f"calendar-{day.toordinal() // args.block_days}")
        model = _mean(values)
        cases.append({"case_id": case_id, "issue_date": meta["issue_date"],
                      "stratum": meta["stratum"], "block": block,
                      "rollouts": len(values), "model": model,
                      "submitted_rollouts": meta["submitted"],
                      "baseline": float(meta["baseline"]),
                      "difference": model - float(meta["baseline"])})
    if not cases:
        raise RuntimeError("no comparable submitted cases")

    by_block = defaultdict(list)
    for rec in cases:
        by_block[rec["block"]].append(rec)
    blocks = sorted(by_block)
    rng = random.Random(args.seed)
    boot = []
    for _ in range(args.bootstrap):
        sampled = [rng.choice(blocks) for _ in blocks]
        diffs = [r["difference"] for b in sampled for r in by_block[b]]
        boot.append(_mean(diffs))
    diff = _mean([r["difference"] for r in cases])
    ci = [_percentile(boot, 0.025), _percentile(boot, 0.975)]
    enough_clusters = len(blocks) >= 30
    report = {
        "artifact_type": "paired_policy_comparison",
        "comparison_version": COMPARISON_VERSION,
        "comparison_implementation": file_identity(Path(__file__), relative_to=REPO_ROOT),
        "probe": file_identity(probe_path, relative_to=REPO_ROOT),
        "policy_stage": probe.get("policy_stage"),
        "policy_id": probe.get("policy_id") or (
            probe.get("model") if probe.get("policy_stage") == "frozen_baseline" else None),
        "split": probe.get("split"),
        "reward": (probe.get("provenance") or {}).get("reward"),
        "scoring_implementation": (probe.get("provenance") or {}).get(
            "scoring_implementation"
        ),
        "schema_implementation": (probe.get("provenance") or {}).get(
            "schema_implementation"
        ),
        "baseline": args.baseline,
        "stratum_filter": args.stratum,
        "truth_event_filter": args.truth_event_only,
        "baseline_artifact": (file_identity(tabular_path, relative_to=REPO_ROOT)
                              if args.baseline == "tabular" else None),
        "method": {
            "case_rollouts_averaged_first": True,
            "failed_rollouts_retained_as_zero": True,
            "cluster": args.block,
            "block_days": args.block_days if args.block == "calendar" else None,
            "bootstrap_replicates": args.bootstrap,
            "seed": args.seed,
            "estimand": "case-weighted paired mean over the fixed probe sample",
            "score_field": "outcome_composite",
            "interactive_auxiliary_reward_excluded": True,
        },
        "n_cases": len(cases),
        "n_clusters": len(blocks),
        "model_mean": round(_mean([r["model"] for r in cases]), 6),
        "baseline_mean": round(_mean([r["baseline"] for r in cases]), 6),
        "paired_difference": round(diff, 6),
        "cluster_bootstrap_95pct": [round(x, 6) for x in ci],
        "model_case_win_rate": round(sum(r["difference"] > 0 for r in cases) / len(cases), 6),
        "inference_status": (
            "estimable" if enough_clusters else
            "descriptive_only_too_few_independent_clusters"
        ),
        "lower_bound_above_zero": bool(enough_clusters and ci[0] > 0),
        "cases": cases,
    }
    suffix = f"_{args.stratum}" if args.stratum else ""
    out = (REPO_ROOT / args.out if args.out else
           REPO_ROOT / "data" / "interim" /
           f"compare_{Path(args.probe).stem}_{args.baseline}{suffix}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"},
                     ensure_ascii=False, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
