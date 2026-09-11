"""自我会商 / 集合预报层：k 份采样预报 → 规则化合成 + 离散度诊断。

对应关系：LLM 采样集合 = 政策集合预报；合成 = 首席综合；离散度 = ensemble spread。
用于推断期评测（单人 vs 自我会商）与 spread-skill 分析（RL 是否坍缩了集合多样性）。
全规则化，不引入任何训练期机制。
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Optional


def _mid(rng) -> Optional[float]:
    if not rng or rng[0] is None:
        return None
    return (float(rng[0]) + float(rng[1])) / 2


def _median_range(ranges: list) -> Optional[list]:
    ranges = [r for r in ranges if r and r[0] is not None]
    if not ranges:
        return None
    return [round(statistics.median(r[0] for r in ranges), 1),
            round(statistics.median(r[1] for r in ranges), 1)]


def _mode(vals: list):
    vals = [v for v in vals if v is not None]
    return Counter(vals).most_common(1)[0][0] if vals else None


def aggregate_forecasts(forecasts: list[dict]) -> dict:
    """k 份合法预报对象 → 合成预报（首席综合）。

    非逐日字段取 medoid 成员（与其余成员逐日中点 MAE 之和最小者）为底，
    逐日字段用逐元素统计量覆盖：等级=多数票、区间=逐端点中位、首要污染物=众数。
    """
    if not forecasts:
        raise ValueError("empty ensemble")
    if len(forecasts) == 1:
        return forecasts[0]

    def dist(a: dict, b: dict) -> float:
        pairs = [( _mid(x.get("pm25_range")), _mid(y.get("pm25_range")))
                 for x, y in zip(a["daily"], b["daily"])]
        vals = [abs(p - q) for p, q in pairs if p is not None and q is not None]
        return sum(vals) / len(vals) if vals else 0.0

    medoid = min(forecasts, key=lambda f: sum(dist(f, g) for g in forecasts))
    out = json_copy = {k: v for k, v in medoid.items()}
    out = dict(json_copy)

    daily = []
    for i, base in enumerate(medoid["daily"]):
        members = [f["daily"][i] for f in forecasts if i < len(f["daily"])]
        item = dict(base)
        item["aqi_level"] = _mode([m.get("aqi_level") for m in members])
        item["primary_pollutant"] = _mode([m.get("primary_pollutant") for m in members])
        pm = _median_range([m.get("pm25_range") for m in members])
        if pm:
            item["pm25_range"] = pm
        pm10 = _median_range([m.get("pm10_range") for m in members])
        if pm10:
            item["pm10_range"] = pm10
        o3 = _median_range([m.get("o3_range") for m in members])
        if o3:
            item["o3_range"] = o3
        daily.append(item)
    out["daily"] = daily

    has_event = [bool((f.get("process") or {}).get("has_event")) for f in forecasts]
    if sum(has_event) * 2 < len(forecasts):
        out["process"] = {"has_event": False, "start": None, "peak": None, "end": None}
    else:
        procs = [f["process"] for f, he in zip(forecasts, has_event) if he and f.get("process")]
        def med_date(key):
            ds = sorted(p[key] for p in procs if p.get(key))
            return ds[len(ds) // 2] if ds else None
        out["process"] = {"has_event": True, "start": med_date("start"),
                          "peak": med_date("peak"), "end": med_date("end")}
    return out


def ensemble_spread(forecasts: list[dict]) -> dict:
    """离散度诊断：逐日中点标准差均值、等级分歧率、过程有无分歧率。

    spread-skill 分析用：与事后真实误差求相关，检验离散度是否为可信的不确定性信号。
    """
    if len(forecasts) < 2:
        return {"n": len(forecasts), "pm25_mid_std": 0.0, "level_disagreement": 0.0,
                "event_disagreement": 0.0}
    n_days = min(len(f["daily"]) for f in forecasts)
    stds, level_dis = [], []
    for i in range(n_days):
        mids = [m for m in (_mid(f["daily"][i].get("pm25_range")) for f in forecasts)
                if m is not None]
        if len(mids) >= 2:
            stds.append(statistics.pstdev(mids))
        levels = [f["daily"][i].get("aqi_level") for f in forecasts]
        level_dis.append(1.0 - Counter(levels).most_common(1)[0][1] / len(levels))
    ev = [bool((f.get("process") or {}).get("has_event")) for f in forecasts]
    return {
        "n": len(forecasts),
        "pm25_mid_std": round(sum(stds) / len(stds), 2) if stds else 0.0,
        "level_disagreement": round(sum(level_dis) / len(level_dis), 3),
        "event_disagreement": round(1.0 - Counter(ev).most_common(1)[0][1] / len(ev), 3),
    }
