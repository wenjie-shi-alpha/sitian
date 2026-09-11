"""行为归因诊断：一份提交的预报"最像哪个简单策略"——防假学的监控面。

全规则可算：提交的逐日 PM2.5 区间中点，与各参照序列（各指导源直译 / 持续性 /
气候态 p50）的平均绝对距离（MAE），距离最小者为 nearest。训练全程监控归因分布
迁移：健康的学习轨迹是早期贴 guidance、学成后在 guidance_bust 类个例上脱离 guidance。
不读 truth——这是行为画像，不是评分。
"""
from __future__ import annotations

from typing import Optional

from .case import CaseBundle


def _midpoints(forecast_obj: dict) -> tuple[list[str], list[float]]:
    dates, mids = [], []
    for d in forecast_obj.get("daily", []):
        rng = d.get("pm25_range") or [None, None]
        if rng[0] is None:
            continue
        dates.append(d["date"])
        mids.append((float(rng[0]) + float(rng[1])) / 2)
    return dates, mids


def _mae(a: list[float], b: list[Optional[float]]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if y is not None]
    if not pairs:
        return None
    return round(sum(abs(x - y) for x, y in pairs) / len(pairs), 1)


def strategy_attribution(forecast_obj: dict, bundle: CaseBundle) -> dict:
    """→ {"mae": {参照: 距离}, "nearest": 最近参照}。参照缺失则不入表。"""
    dates, mids = _midpoints(forecast_obj)
    if not mids:
        return {"mae": {}, "nearest": None}
    refs: dict[str, list[Optional[float]]] = {}

    for name, src in (bundle.guidance or {}).get("sources", {}).items():
        daily = src.get("daily_pm25") or {}
        refs[f"guidance:{name}"] = [daily.get(d) for d in dates]

    obs = (bundle.observations or {}).get("PM2.5", {})
    series = obs.get("series", {}).get(bundle.region) or []
    recent = [v for v in series[-24:] if v is not None]
    if recent:
        cur = sum(recent) / len(recent)
        refs["persistence"] = [cur] * len(dates)

    clim = (bundle.meta or {}).get("climatology") or {}
    month = f"{int(bundle.issue_date[5:7]):02d}"
    p50 = ((clim.get(month) or {}).get("pm25") or {}).get("p50")
    if p50 is not None:
        refs["climatology"] = [p50] * len(dates)

    mae = {k: v for k, v in ((k, _mae(mids, ref)) for k, ref in refs.items()) if v is not None}
    nearest = min(mae, key=mae.get) if mae else None
    return {"mae": mae, "nearest": nearest}
