#!/usr/bin/env python3
"""Compare agent and baseline with non-shaped metrics and clustered uncertainty."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.case import CaseBundle  # noqa: E402
from sitian.hard_metrics import (  # noqa: E402
    failed_forecast_hard_counts, forecast_hard_counts, metric_improvements, scale_counts,
    sum_counts, summarize_hard_counts,
)
from sitian.provenance import (  # noqa: E402
    file_identity, identity_matches_file, score_implementation_matches,
)
from sitian.scoring import reward_spec  # noqa: E402

COMPARISON_VERSION = "1.2.0"


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _case_paths(probe: dict) -> dict[str, Path]:
    identity = (probe.get("provenance") or {}).get("case_manifest") or {}
    path = REPO_ROOT / identity.get("path", "")
    if not path.is_file() or not identity_matches_file(
        identity, path, relative_to=REPO_ROOT
    ):
        raise RuntimeError("probe provenance has no readable case manifest")
    return {candidate.name: candidate for candidate in
            (REPO_ROOT / value for value in json.loads(path.read_text(encoding="utf-8")))}


def _average_counts(items: list[dict]) -> dict:
    return sum_counts([scale_counts(item, 1.0 / len(items)) for item in items])


def _comparison_counts(cases: list[dict]) -> tuple[dict, dict, dict, dict]:
    model = summarize_hard_counts(sum_counts([item["model_counts"] for item in cases]))
    baseline = summarize_hard_counts(sum_counts([item["baseline_counts"] for item in cases]))
    return model, baseline, metric_improvements(model, baseline), {
        "model_counts": sum_counts([item["model_counts"] for item in cases]),
        "baseline_counts": sum_counts([item["baseline_counts"] for item in cases]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--baseline", default="guidance",
                        choices=("persistence", "guidance", "climatology", "tabular"))
    parser.add_argument("--tabular", default="data/interim/eval_tabular_open_evidence_v079.json")
    parser.add_argument("--tabular-split", default="val")
    population = parser.add_mutually_exclusive_group()
    population.add_argument("--stratum")
    population.add_argument("--truth-event-only", action="store_true",
                            help="只比较隐藏真值中至少一天 AQI>=4 的完整事件层")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out")
    args = parser.parse_args()

    probe_path = REPO_ROOT / args.probe
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if args.truth_event_only and not all(
        isinstance(row.get("truth_has_event"), bool) for row in probe.get("rows", [])
    ):
        raise ValueError("probe lacks complete truth-derived event flags")
    current_reward = reward_spec()
    probe_reward = (probe.get("provenance") or {}).get("reward") or {}
    if (
        probe_reward.get("version") != current_reward.get("version")
        or probe_reward.get("config_sha256") != current_reward.get("config_sha256")
        or not score_implementation_matches(probe.get("provenance") or {}, REPO_ROOT)
    ):
        raise ValueError("probe reward/scoring/schema identity is stale")
    paths = _case_paths(probe)
    tabular_path = REPO_ROOT / args.tabular
    tabular_forecasts = {}
    if args.baseline == "tabular":
        tabular = json.loads(tabular_path.read_text(encoding="utf-8"))
        tabular_reward = (tabular.get("provenance") or {}).get("reward") or {}
        if (
            tabular_reward.get("version") != current_reward.get("version")
            or tabular_reward.get("config_sha256") != current_reward.get("config_sha256")
            or not score_implementation_matches(
                tabular.get("provenance") or {}, REPO_ROOT
            )
        ):
            raise ValueError("tabular reward/scoring/schema identity is stale")
        tabular_forecasts = {
            item["case_id"]: item.get("forecast")
            for item in tabular["results"][args.tabular_split]["rows"]
        }
        if any(value is None for value in tabular_forecasts.values()):
            raise RuntimeError("tabular artifact predates stored forecasts; rebuild it with current evaluator")

    rows_by_case = defaultdict(list)
    for row in probe.get("rows", []):
        if args.truth_event_only and row.get("truth_has_event") is not True:
            continue
        if args.stratum and row.get("stratum") != args.stratum:
            continue
        rows_by_case[row["case_id"]].append(row)
    cases = []
    failed_rollouts = 0
    for case_id, rows in rows_by_case.items():
        if case_id not in paths:
            continue
        bundle = CaseBundle.load(paths[case_id])
        truth = bundle.truth_daily_full()
        model_forecasts = [row.get("submitted_forecast") for row in rows
                           if row.get("submitted") and row.get("submitted_forecast")]
        failed_rollouts += len(rows) - len(model_forecasts)
        baseline_forecast = (
            tabular_forecasts.get(case_id) if args.baseline == "tabular"
            else next((row.get("baseline_forecasts", {}).get(args.baseline) for row in rows
                       if row.get("baseline_forecasts", {}).get(args.baseline)), None)
        )
        if baseline_forecast is None:
            continue
        standard = (bundle.meta or {}).get("aqi_standard")
        valid_counts = [
            forecast_hard_counts(forecast, truth, aqi_standard=standard)
            for forecast in model_forecasts
        ]
        failed_counts = [
            failed_forecast_hard_counts(truth, aqi_standard=standard)
            for _ in range(len(rows) - len(model_forecasts))
        ]
        model_counts = _average_counts(valid_counts + failed_counts)
        baseline_counts = forecast_hard_counts(
            baseline_forecast, truth, aqi_standard=standard
        )
        cases.append({
            "case_id": case_id,
            "issue_date": rows[0].get("issue_date") or bundle.issue_date,
            "stratum": rows[0].get("stratum"),
            "rollouts": len(rows),
            "valid_rollouts": len(model_forecasts),
            "model_counts": model_counts,
            "baseline_counts": baseline_counts,
        })
    if not cases:
        raise RuntimeError("no cases have both model rollout rows and a baseline forecast")

    model, baseline, improvements, _ = _comparison_counts(cases)
    by_block = defaultdict(list)
    for item in cases:
        by_block[item["issue_date"]].append(item)
    blocks = sorted(by_block)
    randomizer = random.Random(args.seed)
    boot = defaultdict(list)
    for _ in range(args.bootstrap):
        sampled_blocks = [randomizer.choice(blocks) for _ in blocks]
        sampled_cases = [item for block in sampled_blocks for item in by_block[block]]
        _, _, deltas, _ = _comparison_counts(sampled_cases)
        for name, value in deltas.items():
            if value is not None:
                boot[name].append(value)
    intervals = {
        name: ([round(_percentile(values, 0.025), 6),
                round(_percentile(values, 0.975), 6)] if values else None)
        for name, values in boot.items()
    }
    enough_clusters = len(blocks) >= 30
    report = {
        "artifact_type": "paired_hard_metric_comparison",
        "comparison_version": COMPARISON_VERSION,
        "comparison_implementation": file_identity(Path(__file__), relative_to=REPO_ROOT),
        "hard_metric_implementation": file_identity(
            REPO_ROOT / "src/sitian/hard_metrics.py", relative_to=REPO_ROOT
        ),
        "probe": file_identity(probe_path, relative_to=REPO_ROOT),
        "policy_stage": probe.get("policy_stage"),
        "policy_id": probe.get("policy_id") or probe.get("model"),
        "split": probe.get("split"),
        "reward": (probe.get("provenance") or {}).get("reward"),
        "scoring_implementation": (probe.get("provenance") or {}).get(
            "scoring_implementation"
        ),
        "schema_implementation": (probe.get("provenance") or {}).get(
            "schema_implementation"
        ),
        "baseline": args.baseline,
        "baseline_artifact": (file_identity(tabular_path, relative_to=REPO_ROOT)
                              if args.baseline == "tabular" else None),
        "stratum_filter": args.stratum,
        "truth_event_filter": args.truth_event_only,
        "method": {
            "rollout_counts_averaged_within_case_first": True,
            "cluster": "issue_date",
            "bootstrap_replicates": args.bootstrap,
            "seed": args.seed,
            "hard_metrics_are_separate_from_rl_reward": True,
            "failed_rollouts": (
                "intention-to-treat worst-case counts before within-case averaging; "
                "never condition hard metrics on successful submission"
            ),
            "concentration_point_estimate": "midpoint of each submitted 80% interval",
            "turning_missing_process_penalty": "full forecast horizon per start/peak/end phase",
        },
        "n_cases": len(cases),
        "n_issue_date_clusters": len(blocks),
        "failed_rollouts": failed_rollouts,
        "format_submit_rate_in_compared_rows": round(
            sum(item["valid_rollouts"] for item in cases) /
            sum(item["rollouts"] for item in cases), 6),
        "inference_status": ("estimable" if enough_clusters else
                             "descriptive_only_too_few_independent_clusters"),
        "model": model,
        "baseline_metrics": baseline,
        "positive_is_better_improvements": {
            name: (round(value, 6) if value is not None else None)
            for name, value in improvements.items()
        },
        "cluster_bootstrap_95pct": intervals,
        "cases": [{key: value for key, value in item.items()
                   if key not in ("model_counts", "baseline_counts")} for item in cases],
    }
    suffix = f"_{args.stratum}" if args.stratum else ""
    output = (REPO_ROOT / args.out if args.out else REPO_ROOT / "data" / "interim" /
              f"hard_{probe_path.stem}_{args.baseline}{suffix}.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"},
                     ensure_ascii=False, indent=1))
    print(f"-> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
