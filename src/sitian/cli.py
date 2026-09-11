"""命令行入口。

    python -m sitian.cli gen --out cases/synthetic
    python -m sitian.cli demo --case cases/synthetic/<case_id> [--agent guidance]
    python -m sitian.cli baselines --cases 'cases/synthetic/*'
    python -m sitian.cli llm --case <dir> [--model ... --base-url ...]
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import glob
import json
import sys
from pathlib import Path

from .agents.scripted import BASELINES, run_baselines, run_episode
from .case import CaseBundle
from .env import EnvConfig, ForecastEnv
from .provenance import file_identity
from .scoring import reward_spec
from .synth import make_default_suite


def _fmt(v) -> str:
    if v is None:
        return "   -  "
    return f"{v:6.3f}"


def cmd_gen(args) -> int:
    bundles = make_default_suite(args.out, seed=args.seed)
    for b in bundles:
        print(f"generated {args.out}/{b.case_id}  (kind={b.meta['kind']}, issue={b.issue_date})")
    return 0


def cmd_demo(args) -> int:
    bundle = CaseBundle.load(args.case)
    agent = BASELINES[args.agent](bundle)
    env = ForecastEnv(bundle, EnvConfig())
    out = run_episode(env, agent)
    print(f"case={bundle.case_id}  agent={args.agent}  steps={out['steps']}")
    for entry in out["transcript"]:
        if entry["event"] != "step":
            continue
        a = entry["action"]
        print(f"  -> {a['name']}({json.dumps(a.get('args', {}), ensure_ascii=False)[:100]})")
    print("score:", json.dumps(out["info"].get("score", {}), ensure_ascii=False, indent=1))
    return 0


def cmd_baselines(args) -> int:
    case_dirs = sorted(d for d in glob.glob(args.cases) if Path(d, "case.json").exists())
    if not case_dirs:
        print(f"no cases match {args.cases!r}", file=sys.stderr)
        return 1
    comp_names = ["level", "event", "interval", "turning", "primary", "evidence"]
    header = f"{'case':<40} {'agent':<12} {'comp':>6} " + " ".join(f"{c:>6}" for c in comp_names)
    print(header)
    print("-" * len(header))
    for case_dir in case_dirs:
        bundle = CaseBundle.load(case_dir)
        results = run_baselines(bundle)
        for name, res in results.items():
            comps = res["components"]
            row = f"{bundle.case_id:<40} {name:<12} {_fmt(res['composite'])} "
            row += " ".join(_fmt(comps.get(c)) for c in comp_names)
            print(row)
    return 0


def cmd_sft(args) -> int:
    from .agents.sft import generate_sft_records
    if args.manifest:
        base = Path.cwd()
        case_dirs = [str(base / p) for p in json.loads(Path(args.manifest).read_text(encoding="utf-8"))]
    else:
        case_dirs = sorted(d for d in glob.glob(args.cases) if Path(d, "case.json").exists())
    if not case_dirs:
        print("no valid case directories found", file=sys.stderr)
        return 1
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + ".part")
    n = candidates = 0
    selected_cases = set()
    selected_target_days = set()
    agent_counts = Counter()
    stratum_counts = Counter()
    with temporary.open("w", encoding="utf-8") as f:
        for case_dir in case_dirs:
            bundle = CaseBundle.load(case_dir)
            for rec in generate_sft_records(bundle):
                candidates += 1
                if rec["meta"]["reward"] < args.min_reward:
                    continue  # 差数值轨迹不进 SFT（只教格式，别教坏预报）
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                selected_cases.add(bundle.case_id)
                selected_target_days.update((bundle.region, day)
                                            for day in bundle.forecast_dates())
                agent_counts[rec["meta"]["agent"]] += 1
                stratum_counts[(bundle.meta or {}).get("stratum", "unknown")] += 1

    challenge_path = Path(args.challenge_manifest) if args.challenge_manifest else None
    challenge_case_ids = set()
    challenge_target_days = set()
    if challenge_path and challenge_path.is_file():
        for path in json.loads(challenge_path.read_text(encoding="utf-8")):
            bundle = CaseBundle.load(path)
            challenge_case_ids.add(bundle.case_id)
            challenge_target_days.update((bundle.region, day)
                                         for day in bundle.forecast_dates())
    case_overlap = selected_cases & challenge_case_ids
    target_overlap = selected_target_days & challenge_target_days
    if case_overlap or target_overlap:
        temporary.unlink(missing_ok=True)
        print(f"SFT/challenge leakage: case_ids={len(case_overlap)} "
              f"forecast_city_days={len(target_overlap)}", file=sys.stderr)
        return 1

    temporary.replace(target)
    source_identity = (file_identity(args.manifest, relative_to=Path.cwd())
                       if args.manifest else {"case_glob": args.cases})
    audit = {
        "artifact_type": "format_sft_dataset",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": source_identity,
        "challenge_manifest": (
            file_identity(challenge_path, relative_to=Path.cwd())
            if challenge_path and challenge_path.is_file() else None
        ),
        "reward": reward_spec(),
        "min_reward": args.min_reward,
        "source_cases": len(case_dirs),
        "candidate_records": candidates,
        "records": n,
        "filtered_records": candidates - n,
        "unique_cases": len(selected_cases),
        "agent_counts": dict(sorted(agent_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "challenge_integrity": {
            "case_id_overlap": len(case_overlap),
            "forecast_city_day_overlap": len(target_overlap),
        },
        "output": file_identity(target, relative_to=Path.cwd()),
    }
    sidecar = Path(str(target) + ".manifest.json")
    sidecar_temporary = Path(str(sidecar) + ".part")
    sidecar_temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
    sidecar_temporary.replace(sidecar)
    print(f"{n} SFT records ({len(selected_cases)}/{len(case_dirs)} cases; "
          f"filtered={candidates - n}) -> {target}")
    print(f"audit -> {sidecar}")
    return 0


def cmd_llm(args) -> int:
    from .agents.llm_openai import OpenAICompatAgent

    bundle = CaseBundle.load(args.case)
    agent = OpenAICompatAgent(base_url=args.base_url, model=args.model, temperature=args.temperature)
    env = ForecastEnv(bundle, EnvConfig())
    out = agent.run(env)
    print(f"case={bundle.case_id}  model={agent.model}  steps={out['steps']}  "
          f"tokens={out['usage']}")
    print("score:", json.dumps(out["info"].get("score", {}), ensure_ascii=False, indent=1))
    if args.save_transcript:
        Path(args.save_transcript).write_text(
            json.dumps(out["messages"], ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"transcript saved to {args.save_transcript}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fh", description="forecast harness CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="生成合成个例套件")
    g.add_argument("--out", default="cases/synthetic")
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(fn=cmd_gen)

    d = sub.add_parser("demo", help="用脚本化 agent 跑一个 episode")
    d.add_argument("--case", required=True)
    d.add_argument("--agent", default="guidance", choices=sorted(BASELINES))
    d.set_defaults(fn=cmd_demo)

    b = sub.add_parser("baselines", help="对一批个例跑全部基线并打表")
    b.add_argument("--cases", required=True, help="glob，如 'cases/synthetic/*'")
    b.set_defaults(fn=cmd_baselines)

    s = sub.add_parser("sft", help="脚本基线轨迹 → SFT 冷启动数据（jsonl）")
    source = s.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases", help="case 目录 glob")
    source.add_argument("--manifest", help="case 目录列表 JSON（全国池建议用 valid manifest）")
    s.add_argument("--out", required=True)
    s.add_argument("--min-reward", type=float, default=0.3)
    s.add_argument("--challenge-manifest", default="data/interim/challenge_winter.json",
                   help="若存在则强制 SFT 与 challenge 的 case/city-day 零重叠")
    s.set_defaults(fn=cmd_sft)

    l = sub.add_parser("llm", help="用 OpenAI 兼容服务上的 LLM 跑一个 episode")
    l.add_argument("--case", required=True)
    l.add_argument("--model", default=None, help="缺省读 FH_MODEL")
    l.add_argument("--base-url", default=None, help="缺省读 FH_BASE_URL")
    l.add_argument("--temperature", type=float, default=0.6)
    l.add_argument("--save-transcript", default=None)
    l.set_defaults(fn=cmd_llm)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
