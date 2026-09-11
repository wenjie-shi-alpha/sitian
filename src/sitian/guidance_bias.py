"""历史模式指导偏差的时间安全聚合。

内部索引可包含逐样本误差，但工具只返回聚合统计。所有查询严格要求
``verification_available_at < issue_time``，避免把尚未完成/发布的真值用于订正。
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from .case import CaseBundle
from .schema import aqi_standard_for_date, aqi_to_level, iaqi

GUIDANCE_BIAS_VERSION = "1.0.0"
CHINA_TZ = timezone(timedelta(hours=8))

POLLUTANT_FIELDS = {
    "PM2.5": ("daily_pm25", "pm25_avg"),
    "PM10": ("daily_pm10", "pm10_avg"),
    "O3": ("daily_o3max", "o3_8h"),
}

DEFAULT_FALLBACK_TIERS = (
    ("city_season", 12),
    ("city_all_seasons", 20),
    ("national_season", 50),
    ("national_all_seasons", 50),
)


def season(day: str | date) -> str:
    d = date.fromisoformat(day) if isinstance(day, str) else day
    if d.month in (12, 1, 2):
        return "DJF"
    if d.month in (3, 4, 5):
        return "MAM"
    if d.month in (6, 7, 8):
        return "JJA"
    return "SON"


def issue_time(issue_date: str) -> datetime:
    """业务起报时刻：起报日 08:00 BJT。"""
    return datetime.combine(date.fromisoformat(issue_date), time(8), tzinfo=CHINA_TZ)


def verification_available_at(target_date: str, *, publication_hour: int = 12) -> datetime:
    """无逐记录发布时间时采用的保守可验证时刻：目标日次日 12:00 BJT。

    因查询使用严格小于号，D 日真值不会进入 D+1 08:00 的起报，只会从更晚起报可用。
    """
    target = date.fromisoformat(target_date)
    return datetime.combine(target + timedelta(days=1), time(publication_hour), tzinfo=CHINA_TZ)


def _event(pollutant: str, value: float, target_date: str) -> bool:
    standard = aqi_standard_for_date(target_date)
    return aqi_to_level(iaqi(pollutant, value, standard=standard)) >= 4


def records_from_bundle(bundle: CaseBundle, *, publication_hour: int = 12) -> list[dict]:
    """从一个 case 生成内部 guidance-truth 误差记录。"""
    truth = bundle.truth or {}
    truth_daily = truth.get("daily") or {}
    out = []
    for source, source_data in (bundle.guidance.get("sources") or {}).items():
        for pollutant, (guidance_key, truth_key) in POLLUTANT_FIELDS.items():
            guidance_daily = source_data.get(guidance_key) or {}
            for lead, target_date in enumerate(bundle.forecast_dates(), start=1):
                truth_record = truth_daily.get(target_date) or {}
                if target_date not in guidance_daily or truth_key not in truth_record:
                    continue
                guidance_value = float(guidance_daily[target_date])
                truth_value = float(truth_record[truth_key])
                available = verification_available_at(
                    target_date, publication_hour=publication_hour
                )
                out.append({
                    "case_id": bundle.case_id,
                    "region": bundle.region,
                    "source": source,
                    "pollutant": pollutant,
                    "lead_days": lead,
                    "target_date": target_date,
                    "target_season": season(target_date),
                    "verification_available_at": available.isoformat(),
                    "error_guidance_minus_truth": round(guidance_value - truth_value, 6),
                    "absolute_error": round(abs(guidance_value - truth_value), 6),
                    "guidance_event": _event(pollutant, guidance_value, target_date),
                    "truth_event": _event(pollutant, truth_value, target_date),
                })
    return out


def build_records(case_dirs: Iterable[str | Path], *, publication_hour: int = 12) -> list[dict]:
    records = []
    seen = set()
    for case_dir in case_dirs:
        bundle = CaseBundle.load(case_dir)
        for record in records_from_bundle(bundle, publication_hour=publication_hour):
            key = (record["case_id"], record["source"], record["pollutant"],
                   record["lead_days"], record["target_date"])
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def _aggregate(records: list[dict], *, scope: str, cutoff: datetime,
               window_start: datetime) -> dict:
    errors = [float(record["error_guidance_minus_truth"]) for record in records]
    truth_events = sum(bool(record["truth_event"]) for record in records)
    guidance_events = sum(bool(record["guidance_event"]) for record in records)
    misses = sum(bool(record["truth_event"]) and not bool(record["guidance_event"])
                 for record in records)
    false_alarms = sum(not bool(record["truth_event"]) and bool(record["guidance_event"])
                       for record in records)
    non_events = len(records) - truth_events
    available = sorted(record["verification_available_at"] for record in records)
    return {
        "available": True,
        "scope": scope,
        "n": len(records),
        "error_definition": "guidance_minus_truth",
        "mean_error": round(_mean(errors), 3),
        "median_error": round(statistics.median(errors), 3),
        "mae": round(_mean([abs(value) for value in errors]), 3),
        "error_quantiles": {
            f"p{int(q * 100):02d}": round(_percentile(errors, q), 3)
            for q in (0.10, 0.25, 0.50, 0.75, 0.90)
        },
        "event": {
            "truth_event_count": truth_events,
            "guidance_event_count": guidance_events,
            "miss_count": misses,
            "false_alarm_count": false_alarms,
            "miss_rate_given_truth_event": (
                round(misses / truth_events, 4) if truth_events else None
            ),
            "false_alarm_rate_given_truth_non_event": (
                round(false_alarms / non_events, 4) if non_events else None
            ),
        },
        "eligible_window": {
            "requested_start": window_start.isoformat(),
            "strict_cutoff": cutoff.isoformat(),
            "first_verification": available[0],
            "last_verification": available[-1],
        },
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


class GuidanceBiasIndex:
    """进程内只读索引；按 source/pollutant/lead 预分桶。"""

    def __init__(self, artifact: dict):
        if artifact.get("artifact_type") != "guidance_bias_history":
            raise ValueError("not a guidance_bias_history artifact")
        self.artifact = artifact
        self._by_key: dict[tuple[str, str, int], list[tuple[datetime, dict]]] = defaultdict(list)
        for raw in artifact.get("records", []):
            record = dict(raw)
            available = datetime.fromisoformat(record["verification_available_at"])
            self._by_key[(record["source"], record["pollutant"],
                          int(record["lead_days"]))].append((available, record))

    @classmethod
    def load(cls, path: str | Path) -> "GuidanceBiasIndex":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def query(
        self,
        *,
        region: str,
        issue_date: str,
        horizon: int,
        source: Optional[str] = None,
        pollutant: Optional[str] = None,
        window_days: int = 365,
        fallback_tiers: tuple[tuple[str, int], ...] = DEFAULT_FALLBACK_TIERS,
    ) -> dict:
        cutoff = issue_time(issue_date)
        start = cutoff - timedelta(days=window_days)
        requested_keys = [key for key in sorted(self._by_key)
                          if key[2] <= horizon
                          and (source is None or key[0] == source)
                          and (pollutant is None or key[1] == pollutant)]
        series = []
        for key in requested_keys:
            src, pol, lead = key
            target_date = (date.fromisoformat(issue_date) + timedelta(days=lead)).isoformat()
            target_season = season(target_date)
            eligible = [record for available, record in self._by_key[key]
                        if start <= available < cutoff]
            candidates = {
                "city_season": [record for record in eligible
                                if record["region"] == region
                                and record["target_season"] == target_season],
                "city_all_seasons": [record for record in eligible
                                      if record["region"] == region],
                "national_season": [record for record in eligible
                                    if record["target_season"] == target_season],
                "national_all_seasons": eligible,
            }
            selected = None
            chain = []
            for scope, minimum in fallback_tiers:
                count = len(candidates[scope])
                chain.append({"scope": scope, "n": count, "minimum": minimum})
                if count >= minimum:
                    selected = _aggregate(
                        candidates[scope], scope=scope, cutoff=cutoff, window_start=start
                    )
                    break
            if selected is None:
                selected = {
                    "available": False,
                    "reason": "insufficient_history_after_time_gate",
                }
            series.append({
                "source": src,
                "pollutant": pol,
                "lead_days": lead,
                "target_date": target_date,
                "target_season": target_season,
                "fallback_chain": chain,
                **selected,
            })
        return {
            "available": bool(series) and any(item["available"] for item in series),
            "signal_only": True,
            "note": (
                "仅报告起报前已可验证的历史指导误差信号，不给订正方向或最终预报结论；"
                "error=guidance-truth。"
            ),
            "issue_time": cutoff.isoformat(),
            "strict_time_gate": "verification_available_at < issue_time",
            "window_days": window_days,
            "verification_policy": self.artifact.get("verification_policy"),
            "index_provenance": self.artifact.get("provenance"),
            "series": series,
        }


@lru_cache(maxsize=4)
def load_guidance_bias_index(path: str) -> GuidanceBiasIndex:
    """Load and cache the immutable history index across environment episodes."""
    return GuidanceBiasIndex.load(Path(path).resolve())
