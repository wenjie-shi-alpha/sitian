"""Agent-visible quality metadata computed only from visible case inputs."""
from __future__ import annotations

from datetime import timedelta

from .case import CaseBundle
from .data_contract import (
    POLLUTANT_FIELDS, guidance_statistic, issue_time, observation_time,
    valid_concentration,
)


def build_asset_quality(bundle: CaseBundle) -> dict:
    cutoff = issue_time(bundle.issue_date)
    dates = bundle.forecast_dates()
    warnings = []
    observations = {}
    for pollutant, block in bundle.observations.items():
        times = block.get("times", [])
        parsed = []
        invalid_times = 0
        for raw in times:
            try:
                parsed.append(observation_time(raw))
            except (ValueError, TypeError):
                parsed.append(None)
                invalid_times += 1
        valid_times = [t for t in parsed if t is not None]
        recent = [i for i, t in enumerate(parsed)
                  if t is not None and cutoff - timedelta(hours=24) <= t < cutoff]
        per_region = {}
        for region, values in block.get("series", {}).items():
            valid = [i for i in range(min(len(times), len(values))) if valid_concentration(values[i])]
            recent_valid = [i for i in recent if i < len(values) and valid_concentration(values[i])]
            per_region[region] = {
                "aligned": len(values) == len(times), "values": len(values),
                "valid_values": len(valid), "missing_or_invalid_values": len(times) - len(valid),
                "recent_24h_valid_hours": len({parsed[i] for i in recent_valid}),
                "recent_24h_expected_hours": 24,
            }
        observations[pollutant] = {
            "unit": block.get("unit", "mg/m³" if pollutant == "CO" else "µg/m³"),
            "statistic": block.get("statistic", "hourly_concentration"),
            "source": bundle.meta.get("obs_source", "not_recorded"),
            "available_at": block.get("available_at"),
            "publication_status": "declared" if block.get("available_at") else "not_recorded",
            "invalid_timestamps": invalid_times,
            "duplicate_timestamps": len(valid_times) - len(set(valid_times)),
            "ordered": valid_times == sorted(valid_times),
            "at_or_after_issue": sum(t >= cutoff for t in valid_times),
            "regions": per_region,
        }
        target = per_region.get(bundle.region, {})
        if target.get("recent_24h_valid_hours", 0) < 24:
            warnings.append({"code": "incomplete_recent_observations", "pollutant": pollutant})
        if invalid_times or len(valid_times) != len(set(valid_times)) or any(
            not item["aligned"] for item in per_region.values()
        ):
            warnings.append({"code": "observation_axis_problem", "pollutant": pollutant})

    guidance = {}
    for source, block in bundle.guidance.get("sources", {}).items():
        fields = {}
        for pollutant, (key, _) in POLLUTANT_FIELDS.items():
            values = block.get(key) or {}
            present = [day for day in dates if valid_concentration(values.get(day))]
            statistic, basis = guidance_statistic(source, block, key)
            target_statistic = "daily_max_8h_mean" if pollutant == "O3" else "daily_mean"
            fields[pollutant] = {
                "field": key, "available_dates": present,
                "missing_dates": [day for day in dates if day not in present],
                "unit": (block.get("measurement_contracts", {}).get(key) or {}).get("unit", "µg/m³"),
                "statistic": statistic, "statistic_basis": basis,
                "target_statistic": target_statistic,
                "comparable_to_target": None if statistic == "unknown" else statistic == target_statistic,
            }
            if len(present) < len(dates):
                warnings.append({"code": "guidance_missing_dates", "source": source,
                                 "pollutant": pollutant, "dates": fields[pollutant]["missing_dates"]})
            if statistic != target_statistic and present:
                warnings.append({"code": "statistic_unknown" if statistic == "unknown" else "statistic_mismatch",
                                 "source": source, "pollutant": pollutant})
        guidance[source] = {
            "cycle": block.get("cycle"), "available_at": block.get("available_at"),
            "publication_status": "declared" if block.get("available_at") else "not_recorded",
            "dependency_group": block.get("dependency_group", source), "pollutants": fields,
        }
    diagnostic_daily = bundle.diagnostics.get("daily", {})
    dependencies = {}
    for day, row in diagnostic_daily.items():
        source = row.get("source", "not_recorded")
        # The legacy producer names its CAMS-derived diagnostics explicitly.
        group = "cams" if source.startswith("cams_") else source
        dependencies.setdefault(group, []).append(day)
    return {
        "contract_version": "asset-quality-v1", "issue_time": cutoff.isoformat(),
        "target_dates": dates, "observations": observations, "guidance": guidance,
        "diagnostics": {
            "missing_dates": [day for day in dates if not diagnostic_daily.get(day)],
            "dependency_groups": dependencies,
            "interpretation": "blh_max_m是日最大边界层高度，不能代表夜间混合层；少数气压层温差不能确定近地逆温底高和厚度。",
        },
        "derived_views": {
            "get_assessment": ["observations", "diagnostics", "guidance"],
            "get_process_evidence": ["observations", "diagnostics", "guidance", "evidence"],
        },
        "warnings": warnings,
        "interpretation": "缺测不等于零；摘要不增加独立来源。未记录发布时间不等于已验证可用；质量视图不替代构建审计。",
    }
