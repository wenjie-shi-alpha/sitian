"""合成个例生成器：让 harness 全链路（环境→agent→评分）无真实数据也可跑通。

三种剧本（对应 2025-12 月真实语料中的典型情景）：
    accumulation        积累—峰值—清除的完整污染过程
    clean               扩散条件持续有利的清洁时段
    guidance_misleading EC 指导把清除时间报早一天（"低压推迟"型 bust），
                        考察 agent 是否盲从模式指导

生成完全由 seed 决定，可复现。数值为拟真而非真实观测。
"""
from __future__ import annotations

import math
import random
import hashlib
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from .case import CaseBundle
from .schema import pm25_to_level
from .scoring import RewardConfig, extract_event

REGIONS = {
    "beijing": 1.0,
    "tianjin": 1.05,
    "baoding": 1.25,
    "langfang": 1.15,
    "shijiazhuang": 1.2,
    "tangshan": 0.95,
}

# 每种剧本：过去 3 日均值（issue-2, issue-1, issue晨）与未来 6 日真值均值（目标区域基准）
_KINDS = {
    "accumulation": {
        "past": [45, 62, 78],
        "future": [95, 135, 168, 150, 52, 38],
        "ec_override": None,
    },
    "clean": {
        "past": [30, 28, 35],
        "future": [32, 40, 36, 30, 44, 38],
        "ec_override": None,
    },
    "guidance_misleading": {
        "past": [50, 70, 88],
        "future": [120, 155, 175, 160, 60, 40],
        # EC 把清除提前一天：第 4 天即报好转，真值第 5 天才清除
        "ec_override": [115, 150, 170, 70, 45, 38],
    },
}


def _diagnostics_for(mean_today: float, mean_next: float) -> dict:
    rising = mean_next > mean_today * 1.05
    falling = mean_next < mean_today * 0.7
    polluted = mean_today >= 115
    if falling and mean_today >= 75:
        synoptic, wind_dir, wind_speed = "冷空气过境，转西北风清除", "NW", 5.0
    elif polluted:
        synoptic, wind_dir, wind_speed = "低压前部，静稳高湿", "S", 1.2
    elif rising and mean_today >= 50:
        synoptic, wind_dir, wind_speed = "高压后部转均压场，偏南风山前辐合", "S", 1.8
    else:
        synoptic, wind_dir, wind_speed = "高压控制，扩散条件有利", "NW", 4.0
    return {
        "synoptic": synoptic,
        "wind_dir": wind_dir,
        "wind_speed_ms": wind_speed,
        "inversion": "strong" if polluted else ("weak" if rising else "none"),
        "blh_m": 300 if polluted else (700 if rising else 1300),
        "rh_pct": 80 if polluted else (65 if rising else 45),
        "precip_mm": 1.5 if falling and mean_today >= 115 else 0.0,
        "transport": "southwest_channel" if rising and mean_today >= 50 else "none",
    }


def make_case(
    kind: str,
    seed: int = 0,
    issue_date: str = "2025-12-18",
    case_id: Optional[str] = None,
    out_dir: Optional[str | Path] = None,
) -> CaseBundle:
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}")
    spec = _KINDS[kind]
    kind_seed = int.from_bytes(hashlib.sha256(kind.encode("utf-8")).digest()[:2], "big")
    rng = random.Random(seed * 1000003 + kind_seed)
    base = date.fromisoformat(issue_date)
    horizon = len(spec["future"])
    case_id = case_id or f"synth_{kind}_{issue_date}_s{seed}"

    region_mult = {r: m * (1 + rng.uniform(-0.05, 0.05)) for r, m in REGIONS.items()}

    # ---- truth：目标区域（beijing）未来逐日均值，带小数避免与整数指导混淆 ----
    truth_daily = {}
    truth_values = []
    for i, m in enumerate(spec["future"]):
        v = round(m * (1 + rng.uniform(-0.03, 0.03)) + 0.13, 1)
        truth_values.append(v)
        truth_daily[(base + timedelta(days=i + 1)).isoformat()] = {"pm25_avg": v}

    # ---- observations：过去 55 小时逐小时（截至起报日 07:00），锚点线性插值+日变化+噪声 ----
    anchors = [(12.0, spec["past"][0]), (36.0, spec["past"][1]), (55.0, spec["past"][2])]
    times = []
    series = {r: [] for r in REGIONS}
    start_dt = base - timedelta(days=2)
    for h in range(56):  # 0..55 → (issue-2)T00:00 .. issueT07:00
        day = start_dt + timedelta(days=h // 24)
        times.append(f"{day.isoformat()}T{h % 24:02d}:00")
        # 分段线性
        if h <= anchors[0][0]:
            v = anchors[0][1]
        elif h >= anchors[-1][0]:
            v = anchors[-1][1]
        else:
            for (h0, v0), (h1, v1) in zip(anchors, anchors[1:]):
                if h0 <= h <= h1:
                    v = v0 + (v1 - v0) * (h - h0) / (h1 - h0)
                    break
        diurnal = 1 + 0.10 * math.sin(2 * math.pi * ((h % 24) - 15) / 24)
        for r in REGIONS:
            noise = 1 + rng.gauss(0, 0.04)
            series[r].append(round(max(3.0, v * region_mult[r] * diurnal * noise), 1))
    observations = {"PM2.5": {"times": times, "series": series}}

    # ---- diagnostics：issue-2 .. issue+horizon 逐日特征 ----
    traj = spec["past"] + spec["future"]  # 对应 issue-2 .. issue+horizon
    diag_daily = {}
    for i, m in enumerate(traj):
        d = (base + timedelta(days=i - 2)).isoformat()
        nxt = traj[i + 1] if i + 1 < len(traj) else m
        diag_daily[d] = _diagnostics_for(m, nxt)
    diagnostics = {"daily": diag_daily}

    # ---- guidance：EC（偏差小/或误导剧本）与 CMAQ（偏差大）----
    ec_base = spec["ec_override"] or spec["future"]
    ec = {}
    cmaq = {}
    for i in range(horizon):
        d = (base + timedelta(days=i + 1)).isoformat()
        ec[d] = int(round(ec_base[i] * (1 + rng.gauss(0, 0.06))))
        cmaq[d] = int(round(spec["future"][i] * (1 + rng.gauss(0, 0.20))))
    guidance = {"sources": {
        "ec": {"daily_pm25": ec, "note": "EC 派生 PM2.5 指导"},
        "cmaq": {"daily_pm25": cmaq, "note": "CMAQ 空气质量数值预报"},
    }}

    # ---- previous_forecast：昨日发布的预报（issue-1 起报，覆盖 issue..issue+horizon-1）----
    prev_base = [spec["past"][2]] + spec["future"][: horizon - 1]
    prev_daily = []
    for i, m in enumerate(prev_base):
        d = (base + timedelta(days=i)).isoformat()
        v = int(round(m * (1 + rng.gauss(0, 0.12))))
        prev_daily.append({
            "date": d,
            "aqi_level": pm25_to_level(v),
            "pm25_range": [max(0, int(v * 0.8) - 5), int(v * 1.2) + 5],
            "primary_pollutant": "PM2.5",
        })
    previous_forecast = {
        "issue_date": (base - timedelta(days=1)).isoformat(),
        "region": "beijing",
        "daily": prev_daily,
        "process": None,
        "evidence": [],
        "confidence": "medium",
    }

    # ---- expert：接近真值的专家预报 + 证据类型（评分参照与基线，不进工具）----
    truth_map = {(base + timedelta(days=i + 1)).isoformat(): truth_values[i] for i in range(horizon)}
    expert_daily = []
    for i in range(horizon):
        d = (base + timedelta(days=i + 1)).isoformat()
        v = truth_values[i] * (1 + rng.gauss(0, 0.06))
        expert_daily.append({
            "date": d,
            "aqi_level": pm25_to_level(v),
            "pm25_range": [int(max(0, v * 0.85 - 8)), int(v * 1.15 + 8)],
            "primary_pollutant": "PM2.5",
        })
    ev = extract_event(truth_map, RewardConfig().process_level)
    expert_process = (
        {"has_event": True, "start": ev[0], "peak": ev[1], "end": ev[2]}
        if ev else {"has_event": False, "start": None, "peak": None, "end": None}
    )
    expert = {
        "forecast": {
            "issue_date": issue_date,
            "region": "beijing",
            "daily": expert_daily,
            "process": expert_process,
            "evidence": [
                {"type": "observation", "claim": "近 48 小时区域浓度演变"},
                {"type": "synoptic", "claim": "地面与高空形势配置"},
                {"type": "diagnostic", "claim": "逆温、边界层与湿度条件"},
                {"type": "model_guidance", "claim": "EC 与 CMAQ 指导对比"},
            ],
            "confidence": "medium" if kind == "guidance_misleading" else "high",
        },
        "evidence_types": ["observation", "synoptic", "diagnostic", "model_guidance"],
        "notes": f"synthetic expert reference for kind={kind}",
    }

    bundle = CaseBundle(
        case_id=case_id,
        issue_date=issue_date,
        region="beijing",
        horizon=horizon,
        regions=sorted(REGIONS),
        meta={"kind": kind, "seed": seed, "synthetic": True},
        observations=observations,
        diagnostics=diagnostics,
        guidance=guidance,
        previous_forecast=previous_forecast,
        truth={"daily": truth_daily},
        expert=expert,
    )
    if out_dir is not None:
        bundle.save(Path(out_dir) / case_id)
    return bundle


def make_default_suite(out_dir: str | Path, seed: int = 7) -> list[CaseBundle]:
    """生成三剧本标准套件，返回 bundle 列表。"""
    issue_dates = {"accumulation": "2025-12-16", "clean": "2025-12-04", "guidance_misleading": "2025-12-19"}
    return [
        make_case(kind, seed=seed, issue_date=issue_dates[kind], out_dir=out_dir)
        for kind in ("accumulation", "clean", "guidance_misleading")
    ]
