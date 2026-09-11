#!/usr/bin/env python3
"""抽样框与分层抽样：全国池 episode 清单（确定性，seed=7）。

抽样单元 = (城市, 起报日)。真值侧标签（观测日值库可算，不依赖任何指导数据）：
    pm10_primary  预报窗内出现 PM10 首要污染物日（仅是颗粒物类型代理，不等同沙尘）
    switch     窗内首要污染物同时出现颗粒物与 O3（换季切换型）
    event      窗内出现 AQI ≥4 级日（污染过程，与 RewardConfig.event_level 一致）
    o3         窗内出现 O3 首要污染物的 ≥3 级日（夏季光化学污染，未达 4 级）
    turning    窗内含过程边界（≥3 级连续段的起/末日）但未达 4 级
    clean      其余（平稳/清洁）
标签按上述优先级取首个命中（互斥，简洁）。指导 bust 属于事后分析标签，不参与抽样。

切分（起报日期，边界各留 6 天缓冲防真值窗跨集）：
    train 2025-04-01..2026-04-30 | val 2026-05-07..2026-06-30 | test 2026-07-07..2026-08-21
    按 Qwen3 首发日设计时间留出；但官方未披露可审计的预训练 cutoff，
    因此“发布后日期”只是污染风险控制，不能单独证明无记忆泄漏。
    前瞻测试：2026-27 冬季数据到位后滚动扩充 test。

配额（不足时先补 event 再补 clean）：pm10_primary 10% / switch 15% / event 30% /
o3 15% / turning 15% / clean 15%。

用法：python3 scripts/build_frame.py [--train 6000 --val 800 --test 1500]
产出：data/interim/episodes_{split}.json + frame_stats.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from sitian.schema import aqi_standard_for_date, daily_aqi  # noqa: E402

DB = REPO_ROOT / "data" / "aq_daily.sqlite"
CITIES = REPO_ROOT / "data" / "interim" / "cities_selected.json"
OUT_DIR = REPO_ROOT / "data" / "interim"

HORIZON = 5
SEED = 7
SPLITS = {"train": ("2025-04-01", "2026-04-30"),
          "val": ("2026-05-07", "2026-06-30"),
          "test": ("2026-07-07", "2026-08-21")}
QUOTA = {"pm10_primary": 0.10, "switch": 0.15, "event": 0.30,
         "o3": 0.15, "turning": 0.15, "clean": 0.15}
PARTICLE = {"PM2.5", "PM10"}


def day_info(rec: dict) -> dict:
    concs = {}
    for k, pol in (("pm25", "PM2.5"), ("pm10", "PM10"), ("o3_8h", "O3"),
                   ("so2", "SO2"), ("no2", "NO2"), ("co", "CO")):
        if rec.get(k) is not None:
            concs[pol] = rec[k]
    r = daily_aqi(concs, standard=aqi_standard_for_date(rec["date"]))
    # build_daily_all 已将不满足各污染物有效性的日值置空；抽样框
    # 必须要求六项都在，与 build_national_case.read_truth 保持一致。
    return {"level": r["level"], "primary": r["primary"],
            "valid": len(concs) == 6 and (rec.get("n_hours") or 0) >= 20}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--test", type=int, default=1500)
    args = ap.parse_args()
    targets = {"train": args.train, "val": args.val, "test": args.test}

    cities_doc = json.loads(CITIES.read_text(encoding="utf-8"))
    cities = sorted(cities_doc["cities"])
    cluster = {c: v["cluster"] for c, v in cities_doc["cities"].items()}

    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    # 逐城逐日标签
    info: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in db.execute(
            "SELECT * FROM daily WHERE city IN (%s) AND date >= '2025-03-20'"
            % ",".join("?" * len(cities)), cities):
        if row["pm25"] is None:
            continue
        info[row["city"]][row["date"]] = day_info(dict(row))

    # 过程边界：level>=3 连续段的起/末日
    boundaries: dict[str, set[str]] = defaultdict(set)
    for c, days in info.items():
        ds = sorted(days)
        prev = False
        for i, d in enumerate(ds):
            cur = days[d]["level"] >= 3
            if cur and not prev:
                boundaries[c].add(d)
            if prev and not cur:
                boundaries[c].add(ds[i - 1])
            prev = cur

    def label(city: str, issue: str):
        base = date.fromisoformat(issue)
        window = [(base + timedelta(days=i + 1)).isoformat() for i in range(HORIZON)]
        days = info[city]
        if any(w not in days or not days[w]["valid"] for w in window):
            return None  # 真值缺失或日均无效（<20 有效小时），不可用
        primaries = [set(days[w]["primary"]) for w in window]
        levels = [days[w]["level"] for w in window]
        if any("PM10" in p for p in primaries):
            return "pm10_primary"
        has_particle = any(p & PARTICLE for p in primaries)
        has_o3 = any("O3" in p for p in primaries)
        if has_particle and has_o3:
            return "switch"
        if max(levels) >= 4:
            return "event"
        if any("O3" in p and lv >= 3 for p, lv in zip(primaries, levels)):
            return "o3"
        if any(w in boundaries[city] for w in window):
            return "turning"
        return "clean"

    rng = np.random.RandomState(SEED)
    stats = {}
    for split, (lo, hi) in SPLITS.items():
        pool: dict[str, list] = defaultdict(list)
        d = date.fromisoformat(lo)
        end = date.fromisoformat(hi)
        while d <= end:
            issue = d.isoformat()
            for c in cities:
                s = label(c, issue)
                if s:
                    pool[s].append({"city": c, "issue_date": issue,
                                    "stratum": s, "cluster": cluster[c]})
            d += timedelta(days=1)
        target = targets[split]
        chosen = []
        counts = {s: len(v) for s, v in pool.items()}
        # 一轮配额 + 缺口顺次补给 event → clean
        want = {s: int(round(target * q)) for s, q in QUOTA.items()}
        deficit = 0
        for s in QUOTA:
            have = pool.get(s, [])
            take = min(len(have), want[s])
            idx = rng.choice(len(have), take, replace=False) if have else []
            chosen += [have[i] for i in sorted(idx)]
            deficit += want[s] - take
        for s in ("event", "o3", "clean"):
            if deficit <= 0:
                break
            already = {(e["city"], e["issue_date"]) for e in chosen}
            extra_pool = [e for e in pool.get(s, []) if (e["city"], e["issue_date"]) not in already]
            take = min(len(extra_pool), deficit)
            idx = rng.choice(len(extra_pool), take, replace=False) if extra_pool else []
            chosen += [extra_pool[i] for i in sorted(idx)]
            deficit -= take
        chosen.sort(key=lambda e: (e["issue_date"], e["city"]))
        (OUT_DIR / f"episodes_{split}.json").write_text(
            json.dumps(chosen, ensure_ascii=False, indent=0), encoding="utf-8")
        stats[split] = {"target": target, "selected": len(chosen),
                        "pool": counts,
                        "selected_by_stratum": dict(Counter(e["stratum"] for e in chosen)),
                        "selected_by_cluster": dict(Counter(e["cluster"] for e in chosen))}
        print(split, stats[split]["selected"], stats[split]["selected_by_stratum"])
    stats["quota"] = QUOTA
    stats["splits"] = SPLITS
    stats["seed"] = SEED
    (OUT_DIR / "frame_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
