#!/usr/bin/env python3
"""裸模型能力探针：决定是否需要 SFT 冷启动，并给出 RL 起点基线。

回答四个问题（docs/TRAINING_DESIGN.md §1）：
    1. 格式通过率——能否产出通过 schema 校验的 submit（<95% 则需先诊断）
    2. 提交前是否会查证据（工具调用分布、grounding 分量）
    3. 裸模型分数 vs 脚本基线 / oracle-switch（RL 的起跑线在哪）
    4. 步数与 token 消耗（rollout 成本估算）

用法：
    python3 scripts/probe_model.py --split val --n 40 [--stratified] [--out ...]
环境变量：FH_BASE_URL / FH_MODEL（对接 vLLM 等 OpenAI 兼容端点）
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.agents.llm_openai import OpenAICompatAgent, SYSTEM_PROMPT  # noqa: E402
from sitian.agents.scripted import run_baselines  # noqa: E402
from sitian.case import CaseBundle  # noqa: E402
from sitian.env import ForecastEnv  # noqa: E402
from sitian.provenance import case_bundle_snapshot, file_identity  # noqa: E402
from sitian.schema import aqi_standard_for_date, daily_aqi  # noqa: E402
from sitian.scoring import RewardConfig, reward_spec  # noqa: E402

PROBE_VERSION = "1.14.0"
FORMAL_FORMAT_MIN_CASES = 50
ROLLOUT_SEED_STRATEGY = "sha256(base_seed\\0case_id\\0rep)-int63-v1"
ZERO_VARIANCE_TOLERANCE = 1e-12
ABLATION_CHANNELS = ("synoptic", "composition", "fires", "source_context")


def truth_has_event(bundle: CaseBundle) -> bool:
    """Truth-derived AQI>=4 indicator, independent of mutually exclusive strata."""
    truth = bundle.truth_daily_full()
    if truth is None:
        raise ValueError(f"{bundle.case_id}: complete truth required for probe")
    standard = (bundle.meta or {}).get("aqi_standard")
    return any(
        daily_aqi(
            values,
            standard=(standard.get(day) if isinstance(standard, dict) else standard)
                or aqi_standard_for_date(day),
        )["level"] >= RewardConfig().event_level
        for day, values in truth.items()
    )


def rollout_seed(base_seed: int, case_id: str, rep: int) -> int:
    """Stable per-rollout seed shared by paired experiment arms."""
    raw = f"{base_seed}\0{case_id}\0{rep}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def ablate_bundle(bundle: CaseBundle, channels: list[str]) -> CaseBundle:
    """Copy a case and hide selected evidence channels for paired ablation."""
    output = deepcopy(bundle)
    evidence = output.evidence
    if not isinstance(evidence, dict):
        raise ValueError("bundle evidence must be a dictionary")
    pollution = evidence.setdefault("pollution", {})
    for channel in channels:
        if channel == "synoptic":
            evidence["synoptic"] = {}
        elif channel in {"composition", "fires", "source_context"}:
            pollution[channel] = {}
        else:
            raise ValueError(f"unknown evidence ablation channel: {channel}")
    return output


def _event_exposure(rows: list[dict], truth_events: dict) -> dict:
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    exposed_groups = 0
    exposed_rollouts = 0
    event_rollouts = 0
    for case_id, group in by_case.items():
        if not truth_events.get(case_id):
            continue
        hits = 0
        for row in group:
            if not row.get("submitted"):
                continue
            event_rollouts += 1
            daily = (row.get("submitted_forecast") or {}).get("daily", [])
            if any(int(item.get("aqi_level") or 0) >= 4 for item in daily):
                hits += 1
        exposed_rollouts += hits
        exposed_groups += int(hits > 0)
    event_groups = sum(bool(value) for value in truth_events.values())
    return {
        "definition": "truth-event groups with >=1 submitted rollout forecasting any daily AQI level >=4",
        "event_groups": event_groups,
        "groups_with_exposure": exposed_groups,
        "group_exposure_rate": (exposed_groups / event_groups) if event_groups else None,
        "submitted_event_rollouts": event_rollouts,
        "rollouts_with_exposure": exposed_rollouts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--manifest", default=None,
                    help="可选 manifest 覆盖，用于 winter challenge 等独立集")
    ap.add_argument("--n", type=int, default=40)
    population = ap.add_mutually_exclusive_group()
    population.add_argument("--stratum", default=None,
                            help="只抽指定互斥采样层；不得代替隐藏真值事件层")
    population.add_argument(
        "--truth-event-only", action="store_true",
        help="只抽隐藏完整真值中至少一天 AQI>=4 的 case；用于事件层评测",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rollout-seed", type=int, default=7,
                    help="模型采样 base seed；按 case_id+rep 稳定派生，供实验臂配对")
    ap.add_argument("--temperature", type=float, default=0.6)
    thinking = ap.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="enable_thinking", action="store_true",
                          help="显式开启 Qwen3 等混合模型的 thinking mode")
    thinking.add_argument("--no-thinking", dest="enable_thinking", action="store_false",
                          help="显式关闭 thinking mode")
    ap.set_defaults(enable_thinking=None)
    ap.add_argument("--policy-stage", choices=("frozen_baseline", "trained"),
                    default="frozen_baseline")
    ap.add_argument("--policy-id", default=None,
                    help="训后评测必填的 checkpoint/adapter 不变身份")
    sampling = ap.add_mutually_exclusive_group()
    sampling.add_argument("--stratified", dest="stratified", action="store_true")
    sampling.add_argument("--random", dest="stratified", action="store_false")
    ap.set_defaults(stratified=True)
    ap.add_argument("--group-size", type=int, default=1,
                    help="每个 case 重复采样次数；>1 时审计 GRPO 组内奖励方差")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="独立 rollout 的并发数；本地冻结 vLLM 默认 max-num-seqs=2")
    ap.add_argument("--ablate-evidence", action="append", default=[],
                    choices=ABLATION_CHANNELS,
                    help="可重复：遮蔽一个开放证据通道，其他 case/seed/模型/评分保持不变")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.concurrency < 1:
        ap.error("--concurrency must be >= 1")
    a.ablate_evidence = sorted(set(a.ablate_evidence))

    root = REPO_ROOT / "cases" / "national" / a.split
    valid_manifest = (REPO_ROOT / a.manifest if a.manifest else
                      REPO_ROOT / "data" / "interim" / f"valid_cases_{a.split}.json")
    if valid_manifest.exists():
        dirs = [REPO_ROOT / p for p in json.loads(valid_manifest.read_text(encoding="utf-8"))]
    else:
        dirs = sorted(d for d in root.iterdir() if (d / "case.json").exists())
    rng = random.Random(a.seed)
    if a.truth_event_only:
        # Event is an outcome-defined evaluation population, not the mutually
        # exclusive sampling stratum.  Compute this from hidden complete truth
        # before sampling so PM10-primary/switch cases cannot be silently lost.
        dirs = [d for d in dirs if truth_has_event(CaseBundle.load(d))]
    if a.stratum:
        dirs = [d for d in dirs if json.loads(
            (d / "case.json").read_text(encoding="utf-8"))["meta"].get("stratum") == a.stratum]
        dirs = rng.sample(dirs, min(a.n, len(dirs)))
    elif a.stratified:
        by: dict[str, list] = defaultdict(list)
        for d in dirs:
            meta = json.loads((d / "case.json").read_text(encoding="utf-8"))["meta"]
            by[meta.get("stratum")].append(d)
        picked, strata = [], sorted(by)
        base, remainder = divmod(a.n, len(strata))
        for i, s in enumerate(strata):
            take = min(base + (i < remainder), len(by[s]))
            picked += rng.sample(by[s], take)
        if len(picked) < a.n:
            used = set(picked)
            rest = [d for d in dirs if d not in used]
            picked += rng.sample(rest, min(a.n - len(picked), len(rest)))
        dirs = picked[:a.n]
    else:
        dirs = rng.sample(dirs, min(a.n, len(dirs)))

    agent = OpenAICompatAgent(temperature=a.temperature, enable_thinking=a.enable_thinking)
    policy_id = a.policy_id or (agent.model if a.policy_stage == "frozen_baseline" else None)
    if a.policy_stage == "trained" and not policy_id:
        ap.error("--policy-id is required when --policy-stage=trained")
    print(f"probe: model={agent.model} base={agent.base_url} cases={len(dirs)} "
          f"group_size={a.group_size} thinking={agent.enable_thinking} "
          f"case_seed={a.seed} rollout_seed={a.rollout_seed} "
          f"concurrency={a.concurrency}\n")

    rows = []
    t0 = time.time()
    tasks = [(d, rep) for d in dirs for rep in range(a.group_size)]
    bundle_cache = {d: CaseBundle.load(d) for d in dirs}
    truth_event_cache = {
        bundle.case_id: truth_has_event(bundle) for bundle in bundle_cache.values()
    }
    baseline_cache = {}
    for full_bundle in bundle_cache.values():
        base = run_baselines(full_bundle, ["persistence", "guidance", "climatology"])
        baseline_cache[full_bundle.case_id] = {
            "scores": {
                k: v["composite"] for k, v in base.items() if v["composite"] is not None
            },
            "forecasts": {
                k: v.get("forecast") for k, v in base.items() if v.get("forecast") is not None
            },
        }

    def execute(task):
        i, d, rep = task
        full_bundle = bundle_cache[d]
        request_seed = rollout_seed(a.rollout_seed, full_bundle.case_id, rep)
        base_scores = baseline_cache[full_bundle.case_id]["scores"]
        base_forecasts = baseline_cache[full_bundle.case_id]["forecasts"]
        b = ablate_bundle(full_bundle, a.ablate_evidence)
        has_truth_event = truth_event_cache[full_bundle.case_id]
        env = ForecastEnv(b)
        worker_agent = OpenAICompatAgent(
            base_url=agent.base_url,
            model=agent.model,
            api_key=agent.api_key,
            temperature=agent.temperature,
            max_llm_calls=agent.max_llm_calls,
            timeout=agent.timeout,
            enable_thinking=agent.enable_thinking,
            seed=request_seed,
        )
        try:
            out = worker_agent.run(env)
        except Exception as e:
            return i, {"case_id": b.case_id, "stratum": b.meta.get("stratum"),
                       "truth_has_event": has_truth_event,
                       "issue_date": b.issue_date, "region": b.region,
                       "forecast_dates": b.forecast_dates(),
                       "rep": rep, "rollout_seed": request_seed,
                       "error": type(e).__name__, "submitted": False,
                       "reward": 0.0, "outcome_composite": 0.0,
                       "baselines": base_scores,
                       "baseline_forecasts": base_forecasts,
                       "oracle_switch": max(base_scores.values()) if base_scores else None}, str(e)
        score = out["info"].get("score") or {}
        invalid_attempts = [
            e for e in out["transcript"]
            if e["event"] == "step" and e["action"].get("name") == "submit_forecast"
            and not e["obs"].get("content", {}).get("accepted", False)
        ]
        invalid_submit_errors = [
            str(error)[:500]
            for event in invalid_attempts
            for error in event["obs"].get("content", {}).get("errors", [])
        ]
        invalid_submits = len(invalid_attempts)
        accepted_submit = next((
            event for event in reversed(out["transcript"])
            if event["event"] == "step"
            and event["action"].get("name") == "submit_forecast"
            and event["obs"].get("content", {}).get("accepted", False)
        ), None)
        # schema-v0.6.0: the harness derives aqi_level/primary/process from the
        # submitted intervals; downstream hard metrics must read the normalized
        # object, not the raw policy text.
        submitted_forecast = (
            ((accepted_submit or {}).get("obs", {}).get("content", {})
             .get("normalized_forecast"))
            or ((accepted_submit or {}).get("action", {}).get("args", {})
                .get("forecast"))
        )
        assistant_messages = [
            message for message in out.get("messages", [])
            if message.get("role") == "assistant"
        ]
        # A text/reasoning-only assistant response is a genuine void turn in a
        # tool-only environment.  Count it explicitly instead of inferring it
        # from terminal status or from whether the rollout called any tool at
        # some other point.
        void_assistant_turns = sum(
            not (message.get("tool_calls") or []) for message in assistant_messages
        )
        successful_data_tools = [
            event["action"]["name"] for event in out["transcript"]
            if event["event"] == "step"
            and event["action"].get("name") != "submit_forecast"
            and event.get("obs", {}).get("ok") is True
        ]
        row = {
            "case_id": b.case_id, "stratum": b.meta.get("stratum"),
            "truth_has_event": has_truth_event,
            "issue_date": b.issue_date, "region": b.region,
            "forecast_dates": b.forecast_dates(),
            "rep": rep, "rollout_seed": request_seed,
            "submitted": out["info"].get("reason") == "submitted",
            "termination_reason": out["info"].get("reason"),
            "invalid_submits": invalid_submits,
            "invalid_submit_errors": invalid_submit_errors,
            "reward": out["reward"], "steps": out["steps"],
            "outcome_composite": score.get("outcome_composite"),
            "components": score.get("components", {}),
            "grounding_diagnostics": (score.get("details", {}) or {}).get("grounding", {}),
            "submitted_forecast": submitted_forecast,
            "submitted_evidence": ((submitted_forecast or {}).get("evidence", [])),
            "tools": [e["action"]["name"] for e in out["transcript"] if e["event"] == "step"],
            "successful_data_tools": successful_data_tools,
            "assistant_messages": len(assistant_messages),
            "void_assistant_turns": void_assistant_turns,
            "usage": out["usage"],
            "thinking_activity": out["thinking_activity"],
            "baselines": base_scores,
            "baseline_forecasts": base_forecasts,
            "oracle_switch": max(base_scores.values()) if base_scores else None,
        }
        return i, row, None

    indexed_tasks = [(i, d, rep) for i, (d, rep) in enumerate(tasks)]
    completed = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = [pool.submit(execute, task) for task in indexed_tasks]
        for done_count, future in enumerate(as_completed(futures), 1):
            i, r, error = future.result()
            completed.append((i, r, error))
            if error is not None:
                print(f"  [{done_count}/{len(tasks)} done; task={i+1}] {r['case_id']} "
                      f"rep={r['rep']} ERROR {r['error']}: {error[:80]}", flush=True)
                continue
            print(f"  [{done_count}/{len(tasks)} done; task={i+1}] "
                  f"{r['case_id']:<24} {str(r.get('stratum')):<14} rep={r['rep']} "
                  f"submit={'Y' if r['submitted'] else 'N'} reward={r['reward']:.3f} "
                  f"steps={r['steps']} oracle={r['oracle_switch']}", flush=True)
    rows = [row for _, row, _ in sorted(completed)]

    ok = [r for r in rows if r.get("submitted")]
    first_submit_ok = [r for r in rows if r.get("submitted") and r.get("invalid_submits", 0) == 0]
    n = len(rows)
    print(f"\n{'='*64}\n耗时 {time.time()-t0:.0f}s | 个例 {len(dirs)} | rollout {n}")
    print(f"最终提交率: {len(ok)}/{n} = {len(ok)/n:.0%}")
    first_submit_rate = len(first_submit_ok) / n if n else 0.0
    if len(dirs) < FORMAL_FORMAT_MIN_CASES:
        format_note = (f"→ 仅 provisional（<{FORMAL_FORMAT_MIN_CASES} cases）；"
                       "不得据此决定跳过或启动 SFT")
    elif first_submit_rate >= 0.95:
        format_note = "→ 格式冷启动门槛通过；开训还需检查组内方差/训练栈"
    else:
        format_note = "→ 正式样本未过门槛；先做定向格式 SFT"
    print(f"首次提交合法率: {len(first_submit_ok)}/{n} = {first_submit_rate:.0%}   "
          f"{format_note}")
    if ok:
        print(f"平均 reward: {sum(r['reward'] for r in ok)/len(ok):.3f}"
              f" | 平均步数: {sum(r['steps'] for r in ok)/len(ok):.1f}"
              f" | 平均 tokens: prompt {sum(r['usage']['prompt_tokens'] for r in ok)//len(ok)}"
              f" / completion {sum(r['usage']['completion_tokens'] for r in ok)//len(ok)}")
        print("思考活动: "
              f"每 rollout 平均 {sum(r['thinking_activity']['messages'] for r in ok)/len(ok):.1f} 轮 / "
              f"{sum(r['thinking_activity']['characters'] for r in ok)//len(ok)} chars")
        for key in ("persistence", "guidance", "climatology", "oracle_switch"):
            vals = [(r["baselines"].get(key) if key != "oracle_switch" else r["oracle_switch"])
                    for r in ok]
            vals = [v for v in vals if v is not None]
            if vals:
                print(f"  对照 {key:<14} {sum(vals)/len(vals):.3f}")
        comp = defaultdict(list)
        for r in ok:
            for k, v in (r["components"] or {}).items():
                if v is not None:
                    comp[k].append(v)
        print("  分量均值:", {k: round(sum(v)/len(v), 3) for k, v in sorted(comp.items())})
        by_s = defaultdict(list)
        for r in ok:
            by_s[r["stratum"]].append(r["reward"])
        print("  分层 reward:", {k: round(sum(v)/len(v), 3) for k, v in sorted(by_s.items())})
    print("  工具调用分布:", dict(Counter(t for r in rows for t in r.get("tools", []))))
    errs = Counter(r["error"] for r in rows if "error" in r)
    if errs:
        print("  错误:", dict(errs))
    format_errors = Counter(
        error for r in rows for error in r.get("invalid_submit_errors", [])
    )
    if format_errors:
        print("  首次/重试提交校验错误:", dict(format_errors))
    grouped = defaultdict(list)
    valid_grouped = defaultdict(list)
    valid_outcome_grouped = defaultdict(list)
    valid_component_grouped = defaultdict(lambda: defaultdict(list))
    strata = {}
    truth_events = {}
    for r in rows:
        grouped[r["case_id"]].append(r.get("reward", 0.0))
        strata[r["case_id"]] = r.get("stratum")
        truth_events[r["case_id"]] = r.get("truth_has_event") is True
        if r.get("submitted"):
            valid_grouped[r["case_id"]].append(r.get("reward", 0.0))
            if r.get("outcome_composite") is not None:
                valid_outcome_grouped[r["case_id"]].append(
                    float(r["outcome_composite"])
                )
            for component in ("level", "event", "turning"):
                value = (r.get("components") or {}).get(component)
                if value is not None:
                    valid_component_grouped[r["case_id"]][component].append(
                        float(value)
                    )
    all_sigmas = {
        case_id: statistics.pstdev(values)
        for case_id, values in grouped.items() if len(values) >= 2
    }
    valid_sigmas = {
        case_id: statistics.pstdev(values)
        for case_id, values in valid_grouped.items() if len(values) >= 2
    }
    event_valid_sigmas = [
        sigma for case_id, sigma in valid_sigmas.items() if truth_events[case_id]
    ]
    outcome_sigmas = {
        case_id: statistics.pstdev(values)
        for case_id, values in valid_outcome_grouped.items() if len(values) >= 2
    }
    event_decision_variable = {
        case_id: any(
            len(values) >= 2
            and statistics.pstdev(values) > ZERO_VARIANCE_TOLERANCE
            for values in valid_component_grouped[case_id].values()
        )
        for case_id in grouped if truth_events[case_id]
    }
    zero_var = sum(v <= ZERO_VARIANCE_TOLERANCE for v in all_sigmas.values())
    usable_groups = len(grouped) - zero_var
    zero_var_valid = sum(v <= ZERO_VARIANCE_TOLERANCE for v in valid_sigmas.values())
    usable_groups_valid = len(valid_sigmas) - zero_var_valid
    if a.group_size > 1:
        print(f"  GRPO 零方差组: {zero_var}/{len(grouped)} = {zero_var/len(grouped):.1%}")
        print(f"  零方差过滤后有效 group 率: {usable_groups}/{len(grouped)} = "
              f"{usable_groups/len(grouped):.1%}")
        if valid_sigmas:
            print(f"  仅有效提交零方差组: {zero_var_valid}/{len(valid_sigmas)} = "
                  f"{zero_var_valid/len(valid_sigmas):.1%}")
            event_sigma_text = (
                f"{sum(event_valid_sigmas)/len(event_valid_sigmas):.4f}"
                if event_valid_sigmas else "n/a"
            )
            print(f"  仅有效提交组内 σ 均值: {sum(valid_sigmas.values())/len(valid_sigmas):.4f}; "
                  f"event: {event_sigma_text}")
        print("  组内 reward 标准差:", {k: round(v, 4) for k, v in all_sigmas.items()})
        print(
            "  有效提交的 outcome 可学 group 率: "
            f"{sum(v > ZERO_VARIANCE_TOLERANCE for v in outcome_sigmas.values())}/"
            f"{len(grouped)}"
        )
        if event_decision_variable:
            print(
                "  真值事件组 level/event/turning 决策可学率: "
                f"{sum(event_decision_variable.values())}/{len(event_decision_variable)}"
            )

    ablation_suffix = ("_ablate_" + "_".join(sorted(a.ablate_evidence))
                       if a.ablate_evidence else "")
    out_path = (Path(a.out) if a.out else REPO_ROOT / "data" / "interim" /
                f"probe_{a.split}{ablation_suffix}.json")
    summary = {
        "submitted_rollouts": len(ok),
        "submit_rate": len(ok) / n if n else 0.0,
        "first_submit_valid_rollouts": len(first_submit_ok),
        "first_submit_valid_rate": len(first_submit_ok) / n if n else 0.0,
        "format_evidence_grade": (
            "confirmed" if len(dirs) >= FORMAL_FORMAT_MIN_CASES else "provisional_small_sample"
        ),
        "invalid_submit_attempts": sum(int(r.get("invalid_submits", 0)) for r in rows),
        "invalid_submit_error_counts": dict(format_errors),
        "void_turn_audit_complete": all(
            isinstance(r.get("void_assistant_turns"), int) for r in rows
        ),
        "void_assistant_turns": sum(
            int(r.get("void_assistant_turns", 0)) for r in rows
        ),
        "rollouts_with_void_turn": sum(
            int(r.get("void_assistant_turns", 0) > 0) for r in rows
        ),
        "rollout_void_turn_rate": (
            sum(int(r.get("void_assistant_turns", 0) > 0) for r in rows) / n
            if n else 0.0
        ),
        "mean_reward_submitted": (sum(r["reward"] for r in ok) / len(ok)) if ok else None,
        "mean_outcome_composite_submitted": (
            sum(float(r["outcome_composite"]) for r in ok) / len(ok)
            if ok and all(r.get("outcome_composite") is not None for r in ok) else None
        ),
        "semantic_grounding": {
            "verified": sum(int((r.get("grounding_diagnostics") or {}).get(
                "semantic_verified_count", 0)) for r in ok),
            "verifiable": sum(int((r.get("grounding_diagnostics") or {}).get(
                "verifiable_evidence_count", 0)) for r in ok),
            "rollouts_meeting_minimum": sum(
                int((r.get("grounding_diagnostics") or {}).get(
                    "semantic_verified_count", 0) >=
                    (r.get("grounding_diagnostics") or {}).get(
                        "minimum_assertions_for_full_credit", 2)
                    and (r.get("grounding_diagnostics") or {}).get(
                        "semantic_verified_type_count", 0) >=
                    (r.get("grounding_diagnostics") or {}).get(
                        "minimum_types_for_full_credit", 2))
                for r in ok
            ),
            "full_credit_rollouts": sum(
                int(abs(float((r.get("components") or {}).get("grounding", 0.0)) - 1.0)
                    <= 1e-12)
                for r in ok
            ),
        },
        "process_tool_use_rate": (
            sum("get_process_evidence" in r.get("tools", []) for r in rows) / n if n else 0.0
        ),
        "groups": len(grouped),
        "truth_event_audit_complete": all(
            isinstance(r.get("truth_has_event"), bool) for r in rows
        ),
        "truth_event_groups": sum(truth_events.values()),
        # G2 readout: does the policy ever expose an AQI>=4 day on a truth-event
        # case?  Without exposure the event component has no relative advantage
        # for GRPO to learn from, whatever the rest of the reward does.
        "event_exposure": _event_exposure(rows, truth_events),
        "truth_event_group_definition": (
            "complete hidden truth has at least one daily AQI level >=4; "
            "independent of mutually exclusive sampling stratum"
        ),
        "zero_variance_tolerance": ZERO_VARIANCE_TOLERANCE,
        "zero_variance_groups": zero_var,
        "usable_groups_after_zero_variance_filter": usable_groups,
        "usable_group_rate": usable_groups / len(grouped) if grouped else 0.0,
        "valid_only_groups_with_at_least_2_submissions": len(valid_sigmas),
        "zero_variance_groups_valid_only": zero_var_valid,
        "usable_groups_valid_only": usable_groups_valid,
        "usable_group_rate_valid_only": (
            usable_groups_valid / len(valid_sigmas) if valid_sigmas else 0.0
        ),
        "mean_group_sigma_valid_only": (
            sum(valid_sigmas.values()) / len(valid_sigmas) if valid_sigmas else None
        ),
        "event_mean_group_sigma_valid_only": (
            sum(event_valid_sigmas) / len(event_valid_sigmas) if event_valid_sigmas else None
        ),
        "effective_rollouts_after_zero_variance_filter": sum(
            len(grouped[case_id]) for case_id, sigma in all_sigmas.items()
            if sigma > ZERO_VARIANCE_TOLERANCE),
        "effective_rollout_rate": (
            sum(len(grouped[case_id]) for case_id, sigma in all_sigmas.items()
                if sigma > ZERO_VARIANCE_TOLERANCE) / n
            if n else 0.0
        ),
        "outcome_variable_groups_valid_only": sum(
            sigma > ZERO_VARIANCE_TOLERANCE for sigma in outcome_sigmas.values()
        ),
        "outcome_usable_group_rate_valid_only": (
            sum(sigma > ZERO_VARIANCE_TOLERANCE for sigma in outcome_sigmas.values())
            / len(grouped) if grouped else 0.0
        ),
        "event_decision_groups": len(event_decision_variable),
        "event_decision_variable_groups_valid_only": sum(
            event_decision_variable.values()
        ),
        "event_decision_usable_group_rate_valid_only": (
            sum(event_decision_variable.values()) / len(event_decision_variable)
            if event_decision_variable else 0.0
        ),
    }
    grounding_summary = summary["semantic_grounding"]
    grounding_summary["rate"] = (
        grounding_summary["verified"] / grounding_summary["verifiable"]
        if grounding_summary["verifiable"] else 0.0
    )
    grounding_summary["full_credit_rollout_rate"] = (
        grounding_summary["full_credit_rollouts"] / len(ok) if ok else 0.0
    )
    out_path.write_text(json.dumps({"artifact_type": "policy_probe",
                                    "probe_version": PROBE_VERSION,
                                    "model": agent.model, "policy_id": policy_id,
                                    "split": a.split,
                                    "n_cases": len(dirs),
                                    "group_size": a.group_size, "n_rollouts": n,
                                    "rollout_concurrency": a.concurrency,
                                    "temperature": a.temperature, "seed": a.seed,
                                    "case_sampling_seed": a.seed,
                                    "rollout_seed_base": a.rollout_seed,
                                    "rollout_seed_strategy": ROLLOUT_SEED_STRATEGY,
                                    "thinking_enabled": agent.enable_thinking,
                                    "policy_stage": a.policy_stage,
                                    "evidence_ablation": sorted(a.ablate_evidence),
                                    "evidence_dose": (
                                        "full" if not a.ablate_evidence else "channel_masked"
                                    ),
                                    "stratum_filter": a.stratum,
                                    "truth_event_filter": a.truth_event_only,
                                    "stratified": a.stratified,
                                    "provenance": {
                                        "reward": reward_spec(),
                                        "system_prompt_sha256": hashlib.sha256(
                                            SYSTEM_PROMPT.encode("utf-8")
                                        ).hexdigest(),
                                        "probe_script": file_identity(
                                            Path(__file__), relative_to=REPO_ROOT),
                                        "agent_driver": file_identity(
                                            REPO_ROOT / "src" / "sitian" / "agents" / "llm_openai.py",
                                            relative_to=REPO_ROOT),
                                        "environment_implementation": file_identity(
                                            REPO_ROOT / "src" / "sitian" / "env.py",
                                            relative_to=REPO_ROOT),
                                        "process_evidence_implementation": file_identity(
                                            REPO_ROOT / "src" / "sitian" / "process_evidence.py",
                                            relative_to=REPO_ROOT),
                                        "guidance_bias_implementation": file_identity(
                                            REPO_ROOT / "src" / "sitian" / "guidance_bias.py",
                                            relative_to=REPO_ROOT),
                                        "schema_implementation": file_identity(
                                            REPO_ROOT / "src" / "sitian" / "schema.py",
                                            relative_to=REPO_ROOT),
                                        "scoring_implementation": file_identity(
                                            REPO_ROOT / "src" / "sitian" / "scoring.py",
                                            relative_to=REPO_ROOT),
                                        "standards_manifest": file_identity(
                                            REPO_ROOT / "references" / "standards" / "manifest.json",
                                            relative_to=REPO_ROOT),
                                        "case_manifest": (
                                            file_identity(valid_manifest, relative_to=REPO_ROOT)
                                            if valid_manifest.exists() else None),
                                        "case_input_snapshot": case_bundle_snapshot(
                                            dirs, relative_to=REPO_ROOT, include_truth=True
                                        ),
                                    },
                                    "summary": summary, "rows": rows},
                                   ensure_ascii=False, indent=1, allow_nan=False),
                        encoding="utf-8")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
