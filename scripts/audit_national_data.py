#!/usr/bin/env python3
"""全国 case 池的数据质量与评测可用性审计。

检查粒度：case（城市, 起报日）与其 5 个 forecast city-day。默认回查原始城市小时
CSV，验证六项真值的小时覆盖门槛；输出只含汇总和少量示例，不复制原始数据。

用法：
    python3 scripts/audit_national_data.py
    python3 scripts/audit_national_data.py --skip-raw --out data/interim/data_quality.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from sitian.provenance import file_identity  # noqa: E402
from sitian.schema import aqi_standard_for_date, daily_aqi  # noqa: E402
from sitian.scoring import RewardConfig  # noqa: E402
CASE_ROOT = REPO_ROOT / "cases" / "national"
RAW_OBS = REPO_ROOT / "data" / "raw" / "aq_obs" / "cities"

TRUTH_KEYS = {"pm25_avg", "pm10_avg", "o3_8h", "so2_avg", "no2_avg", "co_avg"}
RAW_TYPES = {
    "PM2.5": 20,
    "PM10": 20,
    "SO2": 20,
    "NO2": 20,
    "CO": 20,
    "O3_8h": 14,
}
O3_DAILY_LIMIT = 160.0
OBS_POLLUTANTS = {"PM2.5", "PM10", "O3", "SO2", "NO2", "CO"}
OBS_UNITS = {name: ("mg/m³" if name == "CO" else "µg/m³") for name in OBS_POLLUTANTS}
TRUTH_POLLUTANT_MAP = {
    "pm25_avg": "PM2.5", "pm10_avg": "PM10", "o3_8h": "O3",
    "so2_avg": "SO2", "no2_avg": "NO2", "co_avg": "CO",
}


def load_cases() -> tuple[list[dict], dict[tuple[str, str], set[str]]]:
    rows = []
    city_day_cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split in ("train", "val", "test"):
        for case_dir in sorted((CASE_ROOT / split).iterdir()):
            if not case_dir.is_dir() or not (case_dir / "case.json").exists():
                continue
            meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
            truth = json.loads((case_dir / "truth.json").read_text(encoding="utf-8"))["daily"]
            obs = json.loads((case_dir / "observations.json").read_text(encoding="utf-8"))
            guidance = json.loads((case_dir / "guidance.json").read_text(encoding="utf-8"))["sources"]
            diagnostics = json.loads((case_dir / "diagnostics.json").read_text(encoding="utf-8"))["daily"]
            missing_truth = {
                day: sorted(TRUTH_KEYS - set(rec)) for day, rec in truth.items()
                if TRUTH_KEYS - set(rec)
            }
            cutoff = f"{meta['issue_date']}T08:00"
            late_obs = sum(t >= cutoff for block in obs.values() for t in block.get("times", []))
            missing_obs = sorted(OBS_POLLUTANTS - set(obs))
            bad_obs_units = {
                pollutant: (obs.get(pollutant) or {}).get("unit")
                for pollutant in OBS_POLLUTANTS
                if pollutant in obs and (obs[pollutant].get("unit") != OBS_UNITS[pollutant])
            }
            bad_obs_shapes = []
            for pollutant, block in obs.items():
                times = block.get("times", [])
                values = block.get("series", {}).get(meta["region"], [])
                if len(times) != len(values):
                    bad_obs_shapes.append(pollutant)
            cams = guidance.get("cams", {})
            truth_has_event = any(
                daily_aqi(
                    {TRUTH_POLLUTANT_MAP[key]: float(value)
                     for key, value in record.items() if key in TRUTH_POLLUTANT_MAP},
                    standard=aqi_standard_for_date(day),
                )["level"] >= RewardConfig().event_level
                for day, record in truth.items()
                if not (TRUTH_KEYS - set(record))
            )
            row = {
                "case_id": meta["case_id"],
                "split": split,
                "city": meta["region"],
                "issue_date": meta["issue_date"],
                "stratum": meta.get("meta", {}).get("stratum"),
                "truth_has_event": truth_has_event,
                "truth_days": len(truth),
                "missing_truth": missing_truth,
                "late_observations": late_obs,
                "missing_observation_pollutants": missing_obs,
                "bad_observation_units": bad_obs_units,
                "bad_observation_shapes": bad_obs_shapes,
                "cams_pm25_days": len(cams.get("daily_pm25", {})),
                "diagnostic_days": len(diagnostics),
            }
            rows.append(row)
            for day in truth:
                city_day_cases[(meta["region"], day)].add(meta["case_id"])
    return rows, city_day_cases


def audit_raw(city_day_cases: dict[tuple[str, str], set[str]], case_meta: dict[str, dict]) -> dict:
    by_day: dict[str, set[str]] = defaultdict(set)
    for city, day in city_day_cases:
        by_day[day].add(city)

    violations = Counter()
    affected_city_days: set[tuple[str, str]] = set()
    examples = []
    missing_files = []
    for day, cities in sorted(by_day.items()):
        path = RAW_OBS / day[:4] / f"china_cities_{day.replace('-', '')}.csv"
        if not path.exists():
            missing_files.append(day)
            continue
        with open(path, encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            indices = {city: header.index(city) for city in cities if city in header}
            counts = {city: Counter() for city in cities}
            maxima: dict[str, dict[str, float]] = {city: {} for city in cities}
            for rec in reader:
                pollutant = rec[2]
                if pollutant not in RAW_TYPES:
                    continue
                for city, idx in indices.items():
                    if idx < len(rec) and rec[idx].strip():
                        try:
                            value = float(rec[idx])
                        except ValueError:
                            continue
                        counts[city][pollutant] += 1
                        maxima[city][pollutant] = max(
                            value, maxima[city].get(pollutant, float("-inf")))
        for city, pol_counts in counts.items():
            for pollutant, minimum in RAW_TYPES.items():
                actual = pol_counts.get(pollutant, 0)
                # HJ 663—2013/2026 6.1.2：O3日最大8h少于14个时，若结果已超
                # GB 3095 二级日限值（160）仍视为有效；该规则跨 2026 切换点一致。
                o3_exception = (
                    pollutant == "O3_8h"
                    and maxima[city].get(pollutant, float("-inf")) > O3_DAILY_LIMIT
                )
                if actual >= minimum or o3_exception:
                    continue
                violations[pollutant] += 1
                affected_city_days.add((city, day))
                if len(examples) < 30:
                    examples.append({"city": city, "date": day, "pollutant": pollutant,
                                     "valid_values": actual, "required": minimum})

    affected_cases = set()
    for city_day in affected_city_days:
        affected_cases.update(city_day_cases[city_day])
    affected_by_split = Counter(case_meta[c]["split"] for c in affected_cases)
    affected_by_stratum = Counter(case_meta[c]["stratum"] for c in affected_cases)
    denominator = len(city_day_cases)
    return {
        "checked_unique_city_days": denominator,
        "missing_raw_files": missing_files,
        "invalid_city_days_by_pollutant": dict(violations),
        "affected_unique_city_days": len(affected_city_days),
        "affected_unique_city_day_rate": round(len(affected_city_days) / denominator, 6),
        "affected_cases": len(affected_cases),
        "affected_case_rate": round(len(affected_cases) / len({c for cs in city_day_cases.values() for c in cs}), 6),
        "affected_cases_by_split": dict(affected_by_split),
        "affected_cases_by_stratum": dict(affected_by_stratum),
        "examples": examples,
        "_affected_case_ids": sorted(affected_cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--out", default="data/interim/data_quality_audit.json")
    args = parser.parse_args()

    rows, city_day_cases = load_cases()
    by_split = {}
    split_cities = {}
    for split in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == split]
        split_cities[split] = {r["city"] for r in subset}
        by_split[split] = {
            "cases": len(subset),
            "cities": len(split_cities[split]),
            "issue_range": [min(r["issue_date"] for r in subset), max(r["issue_date"] for r in subset)],
            "strata": dict(Counter(r["stratum"] for r in subset)),
            "truth_event_cases": sum(r["truth_has_event"] for r in subset),
            "truth_event_cases_by_sampling_stratum": dict(Counter(
                r["stratum"] for r in subset if r["truth_has_event"]
            )),
            "sampling_event_stratum_recall": round(
                sum(r["truth_has_event"] and r["stratum"] == "event" for r in subset)
                / sum(r["truth_has_event"] for r in subset), 6
            ) if any(r["truth_has_event"] for r in subset) else None,
            "truth_day_failures": sum(r["truth_days"] != 5 for r in subset),
            "truth_field_failures": sum(bool(r["missing_truth"]) for r in subset),
            "late_observation_cases": sum(r["late_observations"] > 0 for r in subset),
            "six_pollutant_observation_failures": sum(
                bool(r["missing_observation_pollutants"]) for r in subset),
            "observation_unit_failures": sum(bool(r["bad_observation_units"]) for r in subset),
            "observation_shape_failures": sum(bool(r["bad_observation_shapes"]) for r in subset),
            "cams_guidance_day_counts": dict(Counter(r["cams_pm25_days"] for r in subset)),
            "diagnostic_day_counts": dict(Counter(r["diagnostic_days"] for r in subset)),
        }

    duplicate_ids = [case_id for case_id, n in Counter(r["case_id"] for r in rows).items() if n > 1]
    report = {
        "audit_version": "1.4.0",
        "audit_implementation": file_identity(Path(__file__), relative_to=REPO_ROOT),
        "standards_manifest": file_identity(
            REPO_ROOT / "references" / "standards" / "manifest.json", relative_to=REPO_ROOT),
        "dataset": "cases/national/{train,val,test}",
        "grain": "one case per (city, issue_date), five forecast city-days per case",
        "cases": len(rows),
        "unique_case_ids": len({r["case_id"] for r in rows}),
        "duplicate_case_ids": duplicate_ids[:30],
        "splits": by_split,
        "ood_city_holdout": {
            "val_cities_unseen_in_train": len(split_cities["val"] - split_cities["train"]),
            "test_cities_unseen_in_train": len(split_cities["test"] - split_cities["train"]),
        },
        "evaluation_risks": {
            "event_population_definition": (
                "complete hidden truth has at least one daily AQI level >=4; "
                "independent of mutually exclusive sampling stratum"
            ),
            "test_event_cases": by_split["test"]["truth_event_cases"],
            "test_event_cases_missed_by_event_sampling_stratum": (
                by_split["test"]["truth_event_cases"]
                - by_split["test"]["truth_event_cases_by_sampling_stratum"].get(
                    "event", 0
                )
            ),
            "test_turning_cases": by_split["test"]["strata"].get("turning", 0),
            "has_true_city_ood": bool(split_cities["test"] - split_cities["train"]),
        },
    }
    if not args.skip_raw:
        raw_result = audit_raw(
            city_day_cases,
            {r["case_id"]: {"split": r["split"], "stratum": r["stratum"]} for r in rows},
        )
        affected = set(raw_result.pop("_affected_case_ids"))
        manifests = {}
        for split in ("train", "val", "test"):
            valid_rows = [r for r in rows
                          if r["split"] == split and r["case_id"] not in affected]
            valid = [f"cases/national/{split}/{r['case_id']}" for r in valid_rows]
            manifest_path = REPO_ROOT / "data" / "interim" / f"valid_cases_{split}.json"
            manifest_path.write_text(json.dumps(valid, ensure_ascii=False, indent=1), encoding="utf-8")
            manifests[split] = {
                **file_identity(manifest_path, relative_to=REPO_ROOT),
                "cases": len(valid),
            }
            valid_events = [r for r in valid_rows if r["truth_has_event"]]
            by_split[split]["valid_truth_event_cases"] = len(valid_events)
            by_split[split]["valid_truth_event_cases_by_sampling_stratum"] = dict(
                Counter(r["stratum"] for r in valid_events)
            )
            by_split[split]["valid_sampling_event_stratum_recall"] = round(
                sum(r["stratum"] == "event" for r in valid_events) / len(valid_events), 6
            ) if valid_events else None
        raw_result["valid_case_manifests"] = manifests
        report["raw_validity"] = raw_result
        report["evaluation_risks"]["test_event_cases"] = by_split["test"][
            "valid_truth_event_cases"
        ]
        report["evaluation_risks"]["test_event_cases_missed_by_event_sampling_stratum"] = (
            by_split["test"]["valid_truth_event_cases"]
            - by_split["test"]["valid_truth_event_cases_by_sampling_stratum"].get(
                "event", 0
            )
        )

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
