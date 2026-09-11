"""确定性评估层：把预报员的分析套路固化为规则可复算的"信号"，供 get_assessment 工具输出。

设计边界：**只算信号（客观事实），不给结论（预报决策）**——趋势斜率、静稳/冷空气标志、
指导多源分歧、气候态分位都是从 agent 本就可见的数据（obs/diagnostics/guidance/气候背景）
确定性推导；预报结论留给策略模型学，否则 RL 学到的是抄摘要。
不触碰 truth/expert（防泄漏哨兵测试覆盖）。

信号判据（业务口径的简化，均可在此集中调整）：
    cold_air    风向进入北向扇区(270°..45°) 且 风速较前日跳升 >= 2 m/s
    stagnation  风速 < 2 m/s 且 湿度 >= 60%（有边界层数据时再要求 blh < 600 m）
    rain        日降水 >= 1 mm
    o3_potential 最高温 >= 28°C 且 云量 < 50%（字段缺失则不评）
"""
from __future__ import annotations

from typing import Optional
from datetime import timedelta

from .data_contract import finite_number, valid_concentration, observation_time, issue_time


def _mean(vals):
    vals = [v for v in vals if valid_concentration(v)]
    return round(sum(vals) / len(vals), 1) if vals else None


def obs_trend(observations: dict, region: str, issue_date: str | None = None) -> dict:
    """各污染物：近 24h 均值 vs 前 24h 均值及变化量。"""
    out = {}
    for pol, block in observations.items():
        series = block.get("series", {}).get(region)
        if not series:
            continue
        times = block.get("times", [])
        if not times or len(times) != len(series):
            continue
        parsed = [observation_time(t) for t in times]
        if len(set(parsed)) != len(parsed) or parsed != sorted(parsed):
            continue
        cutoff = issue_time(issue_date) if issue_date else max(parsed) + timedelta(hours=1)
        recent = [v for t, v in zip(parsed, series) if cutoff-timedelta(hours=24) <= t < cutoff]
        previous = [v for t, v in zip(parsed, series)
                    if cutoff-timedelta(hours=48) <= t < cutoff-timedelta(hours=24)]
        last24, prev24 = _mean(recent), _mean(previous)
        if last24 is None:
            continue
        rec = {"last24h_mean": last24, "last24h_valid_hours": sum(valid_concentration(v) for v in recent),
               "prev24h_valid_hours": sum(valid_concentration(v) for v in previous)}
        if prev24 is not None:
            rec["prev24h_mean"] = prev24
            rec["change_24h"] = round(last24 - prev24, 1)
        out[pol] = rec
    return out


def _in_north_sector(deg: Optional[float]) -> bool:
    return deg is not None and (deg >= 270 or deg <= 45)


def daily_signals(diagnostics: dict) -> dict:
    """逐日信号旗标（判据见模块 docstring）。"""
    daily = diagnostics.get("daily", {})
    days = sorted(daily)
    out = {}
    prev_speed = None
    for d in days:
        f = daily[d]
        speed = f.get("wind_speed_ms")
        flags = []
        if (_in_north_sector(f.get("wind_dir_deg")) and speed is not None
                and prev_speed is not None and speed - prev_speed >= 2):
            flags.append("cold_air")
        blh = f.get("blh_max_m")
        if blh is None:
            blh = f.get("blh_m")
        if (speed is not None and speed < 2 and (f.get("rh_pct") or 0) >= 60
                and (blh is None or blh < 600)):
            flags.append("stagnation")
        if (f.get("rain_mm") or 0) >= 1:
            flags.append("rain")
        tmax = f.get("tmax_c")
        if finite_number(tmax) and tmax >= 28 and finite_number(f.get("cloud_pct")) and f["cloud_pct"] < 50:
            flags.append("o3_potential")
        out[d] = {"flags": flags, "wind_speed_ms": speed,
                  "wind_change_ms": round(speed - prev_speed, 1)
                  if speed is not None and prev_speed is not None else None}
        prev_speed = speed
    return out


def guidance_meta(guidance: dict) -> dict:
    """多源逐日分歧：spread = max-min；源少于 2 个时只报均值。"""
    sources = guidance.get("sources", {})
    by_day: dict[str, list[float]] = {}
    for src in sources.values():
        for d, v in (src.get("daily_pm25") or {}).items():
            if valid_concentration(v):
                by_day.setdefault(d, []).append(float(v))
    out = {}
    for d, vals in sorted(by_day.items()):
        rec = {"n_sources": len(vals), "mean": round(sum(vals) / len(vals), 1)}
        if len(vals) >= 2:
            rec["spread"] = round(max(vals) - min(vals), 1)
        out[d] = rec
    return out


def climatology_context(climatology: Optional[dict], month: int, current: dict) -> Optional[dict]:
    """当前观测水平在该城该月气候分位表中的位置。climatology: {"MM": {"pm25": {p50..}}}"""
    if not climatology:
        return None
    table = climatology.get(f"{month:02d}")
    if not table:
        return None
    out = {"month_percentiles": table}
    pm = (current.get("PM2.5") or {}).get("last24h_mean")
    pcts = table.get("pm25")
    if pm is not None and pcts:
        pos = "below_p50"
        for name in ("p50", "p75", "p90"):
            if pcts.get(name) is not None and pm > pcts[name]:
                pos = f"above_{name}"
        out["pm25_position"] = pos
    return out


def compute_assessment(observations: dict, diagnostics: dict, guidance: dict,
                       region: str, issue_date: str,
                       climatology: Optional[dict] = None) -> dict:
    trend = obs_trend(observations, region, issue_date)
    result = {
        "note": "以下为确定性推导的客观信号（判据固定、可复算），不含预报结论；结论须由你综合判断。",
        "obs_trend": trend,
        "daily_signals": daily_signals(diagnostics),
        "guidance_meta": guidance_meta(guidance),
    }
    clim = climatology_context(climatology, int(issue_date[5:7]), trend)
    if clim:
        result["climatology"] = clim
    return result
