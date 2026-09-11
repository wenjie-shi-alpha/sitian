"""Time-gated numerical analog retrieval over an explicitly purged train pool.

Only issue-visible features are used for ranking. Outcomes live in a separate
record section and are exposed only after the whole historical horizon is
verified. Fixed scales are design heuristics, not a fitted Analog Ensemble.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict, defaultdict
from copy import deepcopy
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

from .analog_synoptic import SYNOPTIC_SCALES, synoptic_features
from .case import CaseBundle
from .data_contract import (
    POLLUTANT_FIELDS, finite_number, guidance_statistic, observation_time,
    issue_time, outcome_available_at, verification_available_at,
)

ANALOG_VERSION = "historical-analogs-v3"
POLLUTANT_SCALES = {"PM2.5": 50.0, "PM10": 80.0, "O3": 60.0}
MET_SCALES = {"wind_speed_ms": 3.0, "blh_max_m": 800.0, "rh_pct": 30.0,
              "blh_min_m": 400.0, "blh_night_min_m": 400.0,
              "rain_mm": 10.0, "tmax_c": 15.0, "cloud_pct": 40.0,
              "wind_sin": 1.0, "wind_cos": 1.0}


def _assert_visible_time_gate(bundle: CaseBundle) -> None:
    violations = bundle.audit_time_gate()
    if violations:
        raise ValueError(f"{bundle.case_id}: visible input time gate failed: {violations[0]}")


def issue_features(bundle: CaseBundle) -> dict[str, float]:
    """No truth/expert, strata, climatology, or later fitted transformations."""
    _assert_visible_time_gate(bundle)
    features = {}
    cutoff = issue_time(bundle.issue_date)
    for pollutant in POLLUTANT_SCALES:
        block = bundle.observations.get(pollutant, {})
        times = block.get("times", [])
        values = block.get("series", {}).get(bundle.region, [])
        if len(times) != len(values):
            raise ValueError("observation values must align with timestamps")
        samples = [(observation_time(t), v) for t, v in zip(times, values)
                   if finite_number(v) and v >= 0]
        if len({t for t, _ in samples}) != len(samples):
            raise ValueError("duplicate observation timestamps")
        recent = [(t, v) for t, v in samples if cutoff - timedelta(hours=24) <= t < cutoff]
        if len(recent) >= 12:
            features[f"observation/{pollutant}/mean24h"] = sum(v for _, v in recent) / len(recent)
            first = [v for t, v in recent if t < cutoff - timedelta(hours=12)]
            last = [v for t, v in recent if t >= cutoff - timedelta(hours=12)]
            if len(first) >= 6 and len(last) >= 6:
                features[f"observation/{pollutant}/change12h"] = sum(last) / len(last) - sum(first) / len(first)
    for lead, day in enumerate(bundle.forecast_dates(), 1):
        row = bundle.diagnostics.get("daily", {}).get(day, {})
        for key in MET_SCALES:
            value = row.get(key)
            if key in {"wind_sin", "wind_cos"} and finite_number(row.get("wind_dir_deg")):
                angle = math.radians(row["wind_dir_deg"])
                value = math.sin(angle) if key == "wind_sin" else math.cos(angle)
            if finite_number(value):
                features[f"meteorology/{key}/d{lead}"] = value
        for source, block in bundle.guidance.get("sources", {}).items():
            for pollutant, (key, _) in POLLUTANT_FIELDS.items():
                statistic, _ = guidance_statistic(source, block, key)
                value = (block.get(key) or {}).get(day)
                if finite_number(value) and value >= 0:
                    features[f"guidance/{source}/{pollutant}/{statistic}/d{lead}"] = value
    day = date.fromisoformat(bundle.issue_date)
    angle = 2 * math.pi * (day.timetuple().tm_yday - 1) / 365.25
    features.update({"season/sin": math.sin(angle), "season/cos": math.cos(angle)})
    terrain = (bundle.evidence.get("pollution", {}).get("source_context", {})
               .get("static", {}).get("target", {}).get("terrain", {}))
    for key in ("elevation_m", "basin_index_m"):
        if finite_number(terrain.get(key)):
            features[f"terrain/{key}"] = terrain[key]
    features.update(synoptic_features(bundle))
    return {key: round(value, 6) for key, value in sorted(features.items())}


def _scale(key: str) -> float:
    parts = key.split("/")
    if parts[0] == "observation":
        return POLLUTANT_SCALES[parts[1]]
    if parts[0] == "guidance":
        return POLLUTANT_SCALES[parts[2]]
    if parts[0] == "meteorology":
        return MET_SCALES[parts[1]]
    if parts[0] == "synoptic":
        return SYNOPTIC_SCALES[parts[3]]
    if key in {"terrain/elevation_m", "terrain/basin_index_m"}:
        return 500.0
    if key in {"season/sin", "season/cos"}:
        return 1.0
    raise ValueError(f"unknown analog feature: {key}")


def record_from_bundle(bundle: CaseBundle) -> dict:
    if bundle.meta.get("split") != "train":
        raise ValueError(f"history must be explicitly train: {bundle.case_id}")
    features = issue_features(bundle)
    outcomes = {}
    for day in bundle.forecast_dates():
        raw = (bundle.truth or {}).get("daily", {}).get(day, {})
        row = {pol: raw[key] for pol, (_, key) in POLLUTANT_FIELDS.items()
               if finite_number(raw.get(key)) and raw[key] >= 0}
        required = set(POLLUTANT_FIELDS) if bundle.meta.get("multi_pollutant") else {"PM2.5"}
        if not required <= row.keys():
            raise ValueError(f"incomplete historical verification: {bundle.case_id}/{day}")
        outcomes[day] = row
    available = max(outcome_available_at(bundle.truth or {}, day) for day in bundle.forecast_dates())
    guidance = {}
    errors = {}
    for source, block in bundle.guidance.get("sources", {}).items():
        guidance[source] = {}
        errors[source] = {}
        for pollutant, (key, _) in POLLUTANT_FIELDS.items():
            statistic, basis = guidance_statistic(source, block, key)
            values = {day: (block.get(key) or {})[day] for day in outcomes
                      if finite_number((block.get(key) or {}).get(day))}
            guidance[source][pollutant] = {"statistic": statistic, "statistic_basis": basis, "daily": values}
            # An instantaneous O3 maximum is not an error against an 8h target.
            expected = "daily_max_8h_mean" if pollutant == "O3" else "daily_mean"
            errors[source][pollutant] = {
                "available": statistic == expected,
                "daily": {day: round(value - outcomes[day][pollutant], 3)
                          for day, value in values.items()
                          if pollutant in outcomes[day] and statistic == expected},
                "definition": "guidance_minus_observed" if statistic == expected else "incomparable_or_unknown_statistic",
            }
    return {
        "case_id": bundle.case_id, "region": bundle.region, "split": "train",
        "issue_date": bundle.issue_date, "target_dates": bundle.forecast_dates(),
        "verification_available_at": available.isoformat(),
        "event_id": bundle.meta.get("historical_event_id"), "features": features,
        "issue_inputs": {
            "guidance": guidance,
            "publication_metadata": "legacy inputs may lack availability; explicit timestamps were audited",
        },
        "published_forecast": {"available": False, "reason": "not_recorded_in_case_contract"},
        "observed_outcomes": outcomes, "guidance_errors": errors,
    }


def make_artifact(records: list[dict], *, excluded_city_days: set[tuple[str, str]],
                  provenance: dict) -> dict:
    """Purge by city/target day, independently of case ids or directory names."""
    kept, excluded, seen = [], [], set()
    for record in records:
        if record["case_id"] in seen:
            raise ValueError("duplicate historical case_id")
        seen.add(record["case_id"])
        if any((record["region"], day) in excluded_city_days for day in record["target_dates"]):
            excluded.append(record["case_id"])
        else:
            kept.append(record)
    return {
        "artifact_type": "historical_analogs", "contract_version": ANALOG_VERSION,
        "split_policy": "train_only_city_day_purged", "internal_only": True,
        "provenance": provenance,
        "summary": {"included_cases": len(kept), "purged_cases": len(excluded)},
        "records": kept,
    }


class HistoricalCaseIndex:
    def __init__(self, artifact: dict):
        if (artifact.get("contract_version") != ANALOG_VERSION
                or artifact.get("artifact_type") != "historical_analogs"
                or artifact.get("split_policy") != "train_only_city_day_purged"):
            raise ValueError("unsupported or unpurged historical index")
        self.records = {}
        self._query_cache = OrderedDict()
        self.identity = hashlib.sha256(json.dumps(artifact, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        for row in deepcopy(artifact["records"]):
            if row.get("split") != "train" or row["case_id"] in self.records:
                raise ValueError("historical index requires unique train cases")
            stamp = datetime.fromisoformat(row["verification_available_at"])
            if stamp.tzinfo is None or stamp < verification_available_at(max(row["target_dates"])):
                raise ValueError("invalid historical verification availability")
            if min(row["target_dates"]) <= row["issue_date"]:
                raise ValueError("historical targets must follow issue")
            for key, value in row["features"].items():
                if not finite_number(value):
                    raise ValueError("non-finite historical feature")
                _scale(key)
            self.records[row["case_id"]] = row

    def eligible(self, bundle: CaseBundle) -> list[dict]:
        cutoff = issue_time(bundle.issue_date)
        return [row for row in self.records.values()
                if row["case_id"] != bundle.case_id and row["issue_date"] < bundle.issue_date
                and datetime.fromisoformat(row["verification_available_at"]) < cutoff
                and max(row["target_dates"]) < bundle.issue_date]

    def query(self, bundle: CaseBundle, *, top_k: int = 3, pollutant: str | None = None,
              same_region: bool = False) -> dict:
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise ValueError("top_k must be an integer in 1..5")
        if type(same_region) is not bool:
            raise ValueError("same_region must be boolean")
        if pollutant is not None and pollutant not in POLLUTANT_FIELDS:
            raise ValueError("unsupported pollutant")
        current = issue_features(bundle)
        if pollutant:
            current = {k: v for k, v in current.items()
                       if not k.startswith(("guidance/", "observation/")) or f"/{pollutant}/" in k}
        cache_key = (bundle.case_id, bundle.issue_date, bundle.region, top_k, pollutant,
                     same_region, tuple(sorted(current.items())))
        if cache_key in self._query_cache:
            self._query_cache.move_to_end(cache_key)
            return deepcopy(self._query_cache[cache_key])
        ranked = []
        for row in self.eligible(bundle):
            if same_region and row["region"] != bundle.region:
                continue
            shared = sorted(current.keys() & row["features"].keys())
            groups = {key.split("/")[0] for key in shared}
            coverage = len(shared) / max(1, len(current))
            if len(shared) < 6 or coverage < 0.5 or "observation" not in groups or not groups & {"meteorology", "synoptic", "guidance"}:
                continue
            distances = defaultdict(list)
            for key in shared:
                value = row["features"][key]
                delta = abs(current[key] - value) / _scale(key)
                distances[key.split("/")[0]].append(min(delta, 3.0))
            group_distances = {group: sum(values) / len(values) for group, values in distances.items()}
            distance = sum(group_distances.values()) / len(group_distances) + (1 - coverage)
            ranked.append((distance, row, {
                "case_id": row["case_id"], "region": row["region"], "issue_date": row["issue_date"],
                "verification_available_at": row["verification_available_at"],
                "distance": round(distance, 4), "feature_coverage": round(coverage, 4),
                "shared_features": len(shared), "group_distances": group_distances,
            }))
        ranked.sort(key=lambda item: (item[0], item[1]["case_id"]))
        selected = []
        records = []
        for _, row, summary in ranked:
            duplicate = any(
                (row.get("event_id") and row["event_id"] == previous.get("event_id"))
                or (row["region"] == previous["region"] and
                    abs((date.fromisoformat(row["issue_date"]) - date.fromisoformat(previous["issue_date"])).days)
                    <= max(len(row["target_dates"]), len(previous["target_dates"])))
                for previous in records
            )
            if duplicate:
                continue
            dimensions = [{"feature": key, "current": current[key], "historical": row["features"][key],
                           "normalized_difference": round(abs(current[key]-row["features"][key])/_scale(key), 4)}
                          for key in sorted(current.keys() & row["features"].keys())]
            dimensions.sort(key=lambda d: (d["normalized_difference"], d["feature"]))
            summary.update(similar_dimensions=dimensions[:4], different_dimensions=list(reversed(dimensions[-4:])),
                           missing_current_dimensions_in_history=sorted(current.keys()-row["features"].keys())[:12])
            selected.append(summary)
            records.append(row)
            if len(selected) == top_k:
                break
        result = {
            "available": bool(selected), "analogs": selected, "contract_version": ANALOG_VERSION,
            "reason": None if selected else "no_eligible_history_with_sufficient_feature_overlap",
            "ranking": "mean of group mean clipped normalized differences + missing-feature fraction; lower is closer",
            "deduplication": "explicit event_id, otherwise same-city issue dates within one forecast horizon",
            "limitation": "固定尺度尚未验证；距离不是概率；时间窗去重不等于已识别完整污染过程。先检查关键差异再决定是否借鉴。",
            "submission_evidence_type": "analog",
        }
        self._query_cache[cache_key] = deepcopy(result)
        if len(self._query_cache) > 64:
            self._query_cache.popitem(last=False)
        return result

    def get_case(self, bundle: CaseBundle, case_id: str, *, detail: str = "summary",
                 feature_prefix: str = "", feature_offset: int = 0, feature_limit: int = 64) -> dict:
        # The same eligibility rule applies to direct id lookup. No path access,
        # no existence oracle for future or held-out records.
        if detail not in {"summary", "full"}:
            raise ValueError("detail must be summary or full")
        if (not isinstance(feature_prefix, str) or len(feature_prefix) > 200
                or type(feature_offset) is not int or feature_offset < 0
                or type(feature_limit) is not int or not 1 <= feature_limit <= 64):
            raise ValueError("invalid historical feature prefix/page")
        if detail != "full" and (feature_prefix or feature_offset or feature_limit != 64):
            raise ValueError("feature filtering/pagination requires detail=full")
        row = next((r for r in self.eligible(bundle) if r["case_id"] == case_id), None)
        if row is None:
            return {"available": False, "reason": "historical_case_unavailable"}
        historical = {key: deepcopy(row[key]) for key in (
                    "case_id", "region", "issue_date", "target_dates", "verification_available_at",
                    "issue_inputs", "published_forecast", "observed_outcomes", "guidance_errors")}
        counts = defaultdict(int)
        features = {}
        matching = [(k, v) for k, v in row["features"].items() if k.startswith(feature_prefix)]
        selected = matching[feature_offset:feature_offset+feature_limit] if detail == "full" else matching
        for key, value in selected:
            group = key.split("/")[0]
            if detail == "full" or counts[group] < 8:
                features[key] = value
                counts[group] += 1
        historical["issue_inputs"]["feature_summary"] = features
        historical["issue_inputs"]["feature_count"] = len(row["features"])
        if detail == "full":
            historical["issue_inputs"]["feature_page"] = {
                "prefix": feature_prefix, "offset": feature_offset, "matching_features": len(matching),
                "next_offset": feature_offset+len(selected) if feature_offset+len(selected) < len(matching) else None}
        return {"available": True, "submission_evidence_type": "analog", "detail": detail,
                "historical_case": historical,
                "summary_note": "summary每组最多8项；full按feature_prefix筛选并分页（最多64项），next_offset继续。排序使用全部特征。",
                "interpretation": "已验证历史结果，仅供类比；不是本次起报的观测或答案。"}


@lru_cache(maxsize=4)
def _load_index(path: str, mtime_ns: int, size: int) -> HistoricalCaseIndex:
    return HistoricalCaseIndex(json.loads(Path(path).read_text(encoding="utf-8")))


def load_historical_case_index(path: str) -> HistoricalCaseIndex:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _load_index(str(resolved), stat.st_mtime_ns, stat.st_size)
