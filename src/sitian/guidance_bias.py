"""历史模式指导偏差的时间安全聚合。

内部索引可包含逐样本误差，但工具只返回聚合统计。所有查询严格要求
``verification_available_at < issue_time``，避免把尚未完成/发布的真值用于订正。
"""
from __future__ import annotations

import json
import hashlib
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from .case import CaseBundle
from .data_contract import (
    CHINA_TZ as CHINA_TZ, POLLUTANT_FIELDS, TARGET_STATISTICS, finite_number, guidance_statistic, issue_time,
    outcome_available_at, timestamp, valid_concentration, verification_available_at,
)
from .schema import aqi_standard_for_date, aqi_to_level, iaqi

GUIDANCE_BIAS_VERSION = "2.0.0"

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
            statistic, basis = guidance_statistic(source, source_data, guidance_key)
            if statistic != TARGET_STATISTICS[pollutant]:
                continue
            guidance_daily = source_data.get(guidance_key) or {}
            for lead, target_date in enumerate(bundle.forecast_dates(), start=1):
                truth_record = truth_daily.get(target_date) or {}
                if target_date not in guidance_daily or truth_key not in truth_record:
                    continue
                if not valid_concentration(guidance_daily[target_date]) or not valid_concentration(truth_record[truth_key]):
                    continue
                guidance_value = float(guidance_daily[target_date])
                truth_value = float(truth_record[truth_key])
                available = outcome_available_at(
                    truth, target_date, publication_hour=publication_hour
                )
                out.append({
                    "case_id": bundle.case_id,
                    "issue_date": bundle.issue_date,
                    "statistic": statistic,
                    "statistic_basis": basis,
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
        if bundle.meta.get("split") != "train":
            raise ValueError(f"guidance bias history requires explicit train cases: {bundle.case_id}")
        if bundle.audit_time_gate():
            raise ValueError(f"guidance bias input time gate failed: {bundle.case_id}")
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
        self.identity = hashlib.sha256(json.dumps(artifact, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.excluded_statistics = set()
        seen = set()
        self._by_key: dict[tuple[str, str, int], list[tuple[datetime, dict]]] = defaultdict(list)
        for raw in artifact.get("records", []):
            record = dict(raw)
            pol = record["pollutant"]
            statistic = record.get("statistic") or ("daily_mean" if pol in {"PM2.5", "PM10"} else "unknown")
            if statistic != TARGET_STATISTICS[pol]:
                self.excluded_statistics.add((record["source"], pol, statistic))
                continue
            available = timestamp(record["verification_available_at"])
            if available < verification_available_at(record["target_date"], publication_hour=0):
                raise ValueError("guidance bias verification precedes completed target day")
            if not valid_concentration(record["absolute_error"]):
                raise ValueError("invalid historical absolute error")
            error = record["error_guidance_minus_truth"]
            if not finite_number(error) or abs(abs(error) - record["absolute_error"]) > 1e-5:
                raise ValueError("invalid or inconsistent historical error")
            key = (record["case_id"], record["source"], pol, int(record["lead_days"]), record["target_date"])
            if key in seen:
                continue
            seen.add(key)
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
        if type(window_days) is not int or not 30 <= window_days <= 1095:
            raise ValueError("window_days must be an integer in 30..1095")
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
                        if start <= available < cutoff and record["target_date"] < issue_date]
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
            "excluded_incomparable_statistics": [list(item) for item in sorted(self.excluded_statistics)
                                                  if (source is None or item[0] == source)
                                                  and (pollutant is None or item[1] == pollutant)],
            "index_provenance": self.artifact.get("provenance"),
            "series": series,
        }


@lru_cache(maxsize=4)
def _load_index(path: str, mtime_ns: int, size: int) -> GuidanceBiasIndex:
    return GuidanceBiasIndex.load(path)


def load_guidance_bias_index(path: str) -> GuidanceBiasIndex:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return _load_index(str(path), stat.st_mtime_ns, stat.st_size)
