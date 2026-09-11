"""Shared input semantics and temporal boundaries for tools and builders."""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone

CHINA_TZ = timezone(timedelta(hours=8))
POLLUTANT_FIELDS = {
    "PM2.5": ("daily_pm25", "pm25_avg"),
    "PM10": ("daily_pm10", "pm10_avg"),
    "O3": ("daily_o3max", "o3_8h"),
}
TARGET_STATISTICS = {"PM2.5": "daily_mean", "PM10": "daily_mean", "O3": "daily_max_8h_mean"}


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_concentration(value) -> bool:
    return finite_number(value) and value >= 0


def timestamp(value: str, *, allow_naive: bool = False) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string")
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        if not allow_naive:
            raise ValueError("timestamp requires timezone")
        stamp = stamp.replace(tzinfo=CHINA_TZ)
    return stamp


def observation_time(value: str) -> datetime:
    return timestamp(value, allow_naive=True)


def issue_time(issue_date: str) -> datetime:
    return datetime.combine(date.fromisoformat(issue_date), time(8), tzinfo=CHINA_TZ)


def verification_available_at(target_date: str, *, publication_hour: int = 12) -> datetime:
    if type(publication_hour) is not int or not 0 <= publication_hour <= 23:
        raise ValueError("publication_hour must be in 0..23")
    return datetime.combine(date.fromisoformat(target_date) + timedelta(days=1),
                            time(publication_hour), tzinfo=CHINA_TZ)


def outcome_available_at(truth: dict, target_date: str, *, publication_hour: int = 12) -> datetime:
    available = verification_available_at(target_date, publication_hour=publication_hour)
    for record in (truth, truth.get("daily", {}).get(target_date, {})):
        for key in ("verification_available_at", "available_at"):
            if record.get(key):
                available = max(available, timestamp(record[key]))
    return available


def guidance_statistic(source: str, block: dict, field: str) -> tuple[str, str]:
    declared = (block.get("measurement_contracts") or {}).get(field) or {}
    if declared.get("statistic"):
        return declared["statistic"], "declared"
    if field in {"daily_pm25", "daily_pm10"}:
        return "daily_mean", "legacy_field_contract"
    if source.lower() == "cams" and "3h 瞬时日最大" in block.get("note", ""):
        return "daily_max_of_3hourly_instantaneous", "legacy_source_note"
    return "unknown", "not_recorded"


def visible_time_violations(bundle) -> list[str]:
    cutoff = issue_time(bundle.issue_date)
    violations = []
    for pollutant, block in bundle.observations.items():
        availability = block.get("sample_available_at")
        if availability is not None:
            if len(availability) != len(block.get("times", [])):
                violations.append(f"observations[{pollutant}] availability axis mismatch")
            for raw_time, raw_available in zip(block.get("times", []), availability):
                try:
                    available = timestamp(raw_available)
                    if available >= cutoff or available < observation_time(raw_time):
                        violations.append(f"observations[{pollutant}] invalid sample availability boundary")
                except (TypeError, ValueError):
                    violations.append(f"observations[{pollutant}] invalid sample availability timestamp")
        for raw in block.get("times", []):
            try:
                stamp = observation_time(raw)
                if stamp >= cutoff:
                    violations.append(f"observations[{pollutant}] time {raw} not before cutoff {cutoff.isoformat()}")
            except (TypeError, ValueError):
                violations.append(f"observations[{pollutant}] invalid timestamp {raw!r}")

    def visit(value, path):
        if isinstance(value, dict):
            for key in ("available_at", "cycle", "forecast_reference_time"):
                if value.get(key) is None:
                    continue
                try:
                    stamp = timestamp(value[key])
                    if stamp >= cutoff:
                        violations.append(f"{path}.{key} {value[key]} not before cutoff {cutoff.isoformat()}")
                    if key == "available_at" and value.get("cycle") and timestamp(value["cycle"]) > stamp:
                        violations.append(f"{path}.cycle after available_at")
                except (TypeError, ValueError):
                    violations.append(f"{path}.{key} invalid or timezone missing")
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    for name in ("observations", "diagnostics", "guidance", "evidence", "previous_forecast"):
        visit(getattr(bundle, name), name)
    if bundle.meta.get("climatology"):
        visit(bundle.meta.get("climatology_provenance", {}), "climatology_provenance")
    previous = bundle.previous_forecast or {}
    if previous.get("issue_date") and previous["issue_date"] >= bundle.issue_date:
        violations.append("previous_forecast.issue_date must precede current issue_date")
    return violations
