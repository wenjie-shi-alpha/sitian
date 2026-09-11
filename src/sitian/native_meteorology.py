"""Bounded native weather queries and reproducible dispersion calculations."""
from __future__ import annotations

import math
from datetime import timedelta

from .data_contract import CHINA_TZ, finite_number, issue_time, timestamp
from .native_data import paired

WEATHER_FIELDS = ("u10_ms", "v10_ms", "blh_m", "temperature_2m_c", "dewpoint_2m_c",
                  "cloud_pct", "precipitation_depth_m")


def _window(bundle, start_hour, end_hour):
    if type(start_hour) is not int or type(end_hour) is not int or not 0 <= start_hour < end_hour <= 144:
        raise ValueError("issue-relative window must be integer hours: 0 <= start < end <= 144")
    cutoff = issue_time(bundle.issue_date)
    return cutoff+timedelta(hours=start_hour), cutoff+timedelta(hours=end_hour)


def native_weather(bundle, *, fields=None, start_hour=0, end_hour=120):
    start, end = _window(bundle, start_hour, end_hour)
    names = list(fields) if isinstance(fields, list) else (["blh_m", "u10_ms", "v10_ms"] if fields is None else [])
    if not names or len(names) > len(WEATHER_FIELDS) or len(set(names)) != len(names) or any(n not in WEATHER_FIELDS for n in names):
        raise ValueError(f"fields must be unique names from {WEATHER_FIELDS}")
    source = bundle.diagnostics.get("native", {}).get("cams", {}).get("fields", {})
    series, provenance = {}, {}
    for name in names:
        block = source.get(name)
        if block is None:
            series[name] = {"available": False, "samples": []}
            continue
        samples = [{"valid_time": timestamp(t).astimezone(CHINA_TZ).isoformat(), "value": v}
                   for t, v in zip(block["times"], block["values"]) if start <= timestamp(t) < end]
        series[name] = {"unit": block["unit"], "available": any(s["value"] is not None for s in samples), "samples": samples}
        provenance[name] = {k: block.get(k) for k in ("source_sha256", "cycle", "available_at", "availability_basis",
            "actual_publication_verified", "native_variable", "native_unit", "conversion", "selected_grid", "grib_step_type")}
        provenance[name]["last_native_valid_time"] = block["times"][-1]
    return {"available": any(row["available"] for row in series.values()), "series": series,
            "window": {"start_inclusive": start.isoformat(), "end_exclusive": end.isoformat()},
            "provenance": provenance, "submission_evidence_type": "diagnostic",
            "limitations": ["保留原生采样间隔和缺测；不插值、不外推。", "tp累计区间未知，原值不能直接解释为逐时或日降水。",
                            "这些是起报前可用时次的模式预报，未来valid_time不是未来实况。"]}


def diffusion_conditions(bundle, *, start_hour=0, end_hour=120, weak_wind_ms=2.0, shallow_blh_m=300.0):
    start, end = _window(bundle, start_hour, end_hour)
    if not finite_number(weak_wind_ms) or not 0 < weak_wind_ms <= 20:
        raise ValueError("weak_wind_ms must be finite in (0,20]")
    if not finite_number(shallow_blh_m) or not 0 < shallow_blh_m <= 5000:
        raise ValueError("shallow_blh_m must be finite in (0,5000]")
    fields = bundle.diagnostics.get("native", {}).get("cams", {}).get("fields", {})
    if not all(name in fields for name in ("u10_ms", "v10_ms", "blh_m")):
        return {"available": False, "reason": "native_wind_or_blh_unavailable"}
    all_times = sorted({timestamp(t) for name in ("u10_ms", "v10_ms", "blh_m") for t in fields[name]["times"]})
    step = min((b-a).total_seconds()/3600 for a, b in zip(all_times, all_times[1:]))
    samples, spans, run = [], [], []
    for t, (u, v, blh) in paired(fields, ["u10_ms", "v10_ms", "blh_m"]):
        stamp = timestamp(t).astimezone(CHINA_TZ)
        if not start <= stamp < end:
            continue
        speed = math.hypot(u, v)
        match = speed < weak_wind_ms and blh < shallow_blh_m
        samples.append({"valid_time": stamp.isoformat(), "wind_speed_ms": round(speed, 4), "blh_m": blh,
                        "ventilation_proxy_m2_s": round(speed*blh, 2), "thresholds_met": match})
        if run and (not match or (stamp-run[-1]).total_seconds()/3600 > step+1e-6):
            spans.append(_span(run))
            run = []
        if match:
            run.append(stamp)
    if run:
        spans.append(_span(run))
    return {"available": bool(samples), "samples": samples, "matched_sample_spans": spans,
            "parameters": {"weak_wind_ms": weak_wind_ms, "shallow_blh_m": shallow_blh_m},
            "calculation": "sqrt(u10²+v10²) × BLH; thresholds use strict <; null samples excluded without bridging gaps",
            "submission_evidence_type": "diagnostic", "source": "cams_native",
            "limitations": ["10米风×BLH只是通风代理，不是混合层平均风的通风系数。",
                            "相邻达阈值采样的跨度不证明期间连续静稳，更不等于污染持续时间。",
                            "不判定污染结束；逆温底高、厚度仍需更密垂直廓线。"],
            "provenance": {name: {k: fields[name].get(k) for k in ("source_sha256", "available_at", "availability_basis")}
                           for name in ("u10_ms", "v10_ms", "blh_m")}}


def _span(run):
    return {"first_sample": run[0].isoformat(), "last_sample": run[-1].isoformat(),
            "sample_count": len(run), "sample_span_hours": (run[-1]-run[0]).total_seconds()/3600}
