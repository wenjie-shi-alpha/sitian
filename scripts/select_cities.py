#!/usr/bin/env python3
"""全国城市选择：数据完备性门槛 + 省会强制 + 污染区制聚类分层配额。

方法学（可对审稿人陈述，详见 docs/CITY_SELECTION.md）：
    1. 候选 = 档案全部城市（~375）。
    2. 硬门槛：评测窗口内 PM2.5 日值覆盖率 >= 0.95 且 O3_8h 覆盖率 >= 0.90。
    3. 强制入选：31 个省级行政区首府（行政/人口代表性；未过门槛者记录为例外）。
    4. 区制聚类：气候态特征（逐月 PM2.5 均值 12 + 逐月 O3_8h 均值 12 +
       log(PM10/PM2.5) + PM2.5 事件日频率 + O3 超标日频率）z-score 后 k-means(k=8)。
       特征用全档案期计算——气候态是公开背景知识，不构成泄漏。
    5. 剩余名额按簇规模比例分配，簇内按到簇心距离取最近（区制代表性优先）。
       全程确定性（seed=7），选择清单落盘可复现。

用法：python3 scripts/select_cities.py [--target 120]
产出：data/interim/cities_selected.json（含每市的簇号/特征/入选理由）
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "data" / "aq_daily.sqlite"
OUT = REPO_ROOT / "data" / "interim" / "cities_selected.json"

CLEAN_START, CLEAN_END = "2025-04-01", "2026-08-26"  # Qwen3 首发(2025-04-29)前完成预训练
SEED = 7
K = 8

CAPITALS = ["北京", "天津", "上海", "重庆", "石家庄", "太原", "呼和浩特", "沈阳", "长春",
            "哈尔滨", "南京", "杭州", "合肥", "福州", "南昌", "济南", "郑州", "武汉",
            "长沙", "广州", "南宁", "海口", "成都", "贵阳", "昆明", "拉萨", "西安",
            "兰州", "西宁", "银川", "乌鲁木齐"]

PM25_EVENT = 115.0  # 4 级（中度污染）日均下界
O3_EXCEED = 160.0   # O3_8h 超 2 级标准


def load_features(db: sqlite3.Connection):
    rows = db.execute("SELECT city, date, pm25, pm10, o3_8h FROM daily").fetchall()
    n_days_clean = db.execute(
        "SELECT COUNT(DISTINCT date) FROM daily WHERE date BETWEEN ? AND ?",
        (CLEAN_START, CLEAN_END)).fetchone()[0]
    by_city: dict[str, dict] = {}
    for city, date, pm25, pm10, o3 in rows:
        c = by_city.setdefault(city, {"pm25_m": [[] for _ in range(12)],
                                      "o3_m": [[] for _ in range(12)],
                                      "pm25": [], "pm10": [], "events": 0, "o3_ex": 0,
                                      "clean_pm25": 0, "clean_o3": 0})
        m = int(date[5:7]) - 1
        if pm25 is not None:
            c["pm25_m"][m].append(pm25)
            c["pm25"].append(pm25)
            if pm25 > PM25_EVENT:
                c["events"] += 1
            if CLEAN_START <= date <= CLEAN_END:
                c["clean_pm25"] += 1
        if pm10 is not None:
            c["pm10"].append(pm10)
        if o3 is not None:
            c["o3_m"][m].append(o3)
            if o3 > O3_EXCEED:
                c["o3_ex"] += 1
            if CLEAN_START <= date <= CLEAN_END:
                c["clean_o3"] += 1
    feats, meta = {}, {}
    for city, c in by_city.items():
        n = len(c["pm25"])
        if n < 600 or not c["pm10"] or any(not m for m in c["pm25_m"]) or any(not m for m in c["o3_m"]):
            continue
        vec = ([float(np.mean(m)) for m in c["pm25_m"]]
               + [float(np.mean(m)) for m in c["o3_m"]]
               + [float(np.log(np.mean(c["pm10"]) / max(np.mean(c["pm25"]), 1e-6))),
                  c["events"] / n, c["o3_ex"] / n])
        feats[city] = vec
        meta[city] = {"coverage_pm25": round(c["clean_pm25"] / n_days_clean, 3),
                      "coverage_o3": round(c["clean_o3"] / n_days_clean, 3),
                      "event_freq": round(c["events"] / n, 4),
                      "o3_exceed_freq": round(c["o3_ex"] / n, 4)}
    return feats, meta


def kmeans(X: np.ndarray, k: int, seed: int, iters: int = 100):
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = np.array([X[lab == j].mean(0) if (lab == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return lab, centers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=120)
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    feats, meta = load_features(db)
    eligible = {c for c, m in meta.items()
                if m["coverage_pm25"] >= 0.95 and m["coverage_o3"] >= 0.90}
    print(f"候选 {len(feats)}，过完备性门槛 {len(eligible)}")

    cities = sorted(eligible)
    X = np.array([feats[c] for c in cities])
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    lab, centers = kmeans(Xz, K, SEED)
    dist = np.linalg.norm(Xz - centers[lab], axis=1)
    cluster = dict(zip(cities, lab.tolist()))
    d_center = dict(zip(cities, dist.tolist()))

    selected: dict[str, str] = {}
    missing_caps = []
    for cap in CAPITALS:
        if cap in eligible:
            selected[cap] = "capital"
        else:
            missing_caps.append(cap)

    # 剩余名额：按簇规模比例，簇内取离簇心最近
    remain = args.target - len(selected)
    sizes = {j: sum(1 for c in cities if cluster[c] == j and c not in selected) for j in range(K)}
    total = sum(sizes.values())
    quota = {j: round(remain * sizes[j] / total) for j in range(K)}
    for j in range(K):
        pool = sorted((c for c in cities if cluster[c] == j and c not in selected),
                      key=lambda c: d_center[c])
        for c in pool[:quota[j]]:
            selected[c] = f"cluster_{j}"

    out = {
        "method": "completeness>=0.95/0.90 + capitals + kmeans(k=8,seed=7) proportional quota",
        "clean_window": [CLEAN_START, CLEAN_END],
        "target": args.target, "n_selected": len(selected),
        "capitals_excluded_by_completeness": missing_caps,
        "cities": {c: {"reason": r, "cluster": cluster.get(c), **meta[c]}
                   for c, r in sorted(selected.items())},
        "cluster_sizes_eligible": sizes,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"选定 {len(selected)} 城 → {OUT}")
    if missing_caps:
        print(f"⚠ 未过门槛的省会: {missing_caps}")
    from collections import Counter
    print("簇分布:", Counter(v["cluster"] for v in out["cities"].values()))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
