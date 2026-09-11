#!/usr/bin/env python3
"""全池基线评测：对 cases/real 的可打分个例跑 persistence / cmaq直译 / naqp直译。

产出四年尺度的量化答案：模式基线多强、哪些月份最难、事件个例占比——
这些数字直接用于 reward 事件加权的校准和课程设计。

用法：python3 sitian/scripts/eval_pool.py
输出：控制台汇总表 + data/interim/pool_baseline_eval.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.agents.scripted import GuidanceFollowAgent, PersistenceAgent, run_episode  # noqa: E402
from sitian.case import CaseBundle  # noqa: E402
from sitian.env import ForecastEnv  # noqa: E402
from sitian.schema import pm25_to_level  # noqa: E402

CASES_ROOT = REPO_ROOT / "cases" / "real"
HEATING = {11, 12, 1, 2, 3}

AGENTS = {
    "persistence": lambda: PersistenceAgent(),
    "cmaq": lambda: GuidanceFollowAgent(prefer=("cmaq",)),
    "naqp": lambda: GuidanceFollowAgent(prefer=("naqp",)),
}


def main() -> int:
    manifest = json.loads((CASES_ROOT / "manifest.json").read_text(encoding="utf-8"))
    days = manifest["built"]
    print(f"可打分个例: {len(days)}")
    records = []
    t0 = time.time()
    for i, day in enumerate(days, 1):
        bundle = CaseBundle.load(CASES_ROOT / f"beijing_{day}")
        truth = bundle.truth_daily()
        truth_levels = [pm25_to_level(v) for v in truth.values()]
        rec = {
            "day": day,
            "month": day[:7],
            "heating": int(day[5:7]) in HEATING,
            "has_event": max(truth_levels) >= 4,
            "max_level": max(truth_levels),
            "scores": {},
        }
        for name, make in AGENTS.items():
            out = run_episode(ForecastEnv(bundle), make())
            score = out["info"].get("score", {})
            rec["scores"][name] = {
                "composite": score.get("outcome_composite"),
                "training_composite": score.get("composite"),
                "level": score.get("components", {}).get("level"),
                "event": score.get("components", {}).get("event"),
            }
        records.append(rec)
        if i % 300 == 0:
            print(f"  {i}/{len(days)} ({time.time() - t0:.0f}s)", flush=True)

    def agg(subset, key="composite"):
        out = {}
        for name in AGENTS:
            vals = [r["scores"][name][key] for r in subset if r["scores"][name][key] is not None]
            out[name] = round(sum(vals) / len(vals), 3) if vals else None
        return out

    n_event = sum(r["has_event"] for r in records)
    print(f"\n=== 全池（{len(records)} 个例，事件个例 {n_event} 个 / {n_event/len(records):.0%}）===")
    print(f"{'切片':<16}{'n':>5}  " + "  ".join(f"{a:>11}" for a in AGENTS))
    rows = [
        ("全部", records),
        ("采暖期", [r for r in records if r["heating"]]),
        ("非采暖期", [r for r in records if not r["heating"]]),
        ("事件个例", [r for r in records if r["has_event"]]),
        ("无事件个例", [r for r in records if not r["has_event"]]),
    ]
    for yr in ("2022", "2023", "2024", "2025"):
        rows.append((yr, [r for r in records if r["day"].startswith(yr)]))
    for label, subset in rows:
        a = agg(subset)
        print(f"{label:<16}{len(subset):>5}  " + "  ".join(f"{a[n]:>11}" for n in AGENTS))

    print("\n=== 事件个例上的等级分（预报难点所在）===")
    ev = [r for r in records if r["has_event"]]
    a = agg(ev, key="level")
    print("  " + "  ".join(f"{n}={a[n]}" for n in AGENTS))

    monthly = defaultdict(list)
    for r in records:
        monthly[r["month"]].append(r)
    hardest = sorted(monthly, key=lambda m: agg(monthly[m])["cmaq"] or 1)[:6]
    print("\n=== CMAQ 直译最难的 6 个月（RL 难例课程候选）===")
    for m in hardest:
        a = agg(monthly[m])
        n_ev = sum(r["has_event"] for r in monthly[m])
        print(f"  {m}: cmaq={a['cmaq']} naqp={a['naqp']} 事件日个例={n_ev}/{len(monthly[m])}")

    out_path = REPO_ROOT / "data" / "interim" / "pool_baseline_eval.json"
    out_path.write_text(json.dumps(
        {"n_cases": len(records), "records": records}, ensure_ascii=False), encoding="utf-8")
    print(f"\n明细已写入 {out_path}（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
