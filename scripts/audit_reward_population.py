#!/usr/bin/env python3
"""Check that every admitted truth is expressible by the forecast action space.

Oracle forecasts exist only inside this audit; they never enter prompts,
Parquet, historical retrieval or training trajectories. This proves score
attainability, not forecast skill or interval calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sitian.case import CaseBundle
from sitian.provenance import case_bundle_snapshot, file_identity
from sitian.schema import aqi_standard_for_date, daily_aqi
from sitian.scoring import reward_spec, score_forecast

FIELDS = {"PM2.5": "pm25", "PM10": "pm10", "O3": "o3", "SO2": "so2", "NO2": "no2", "CO": "co"}


def audit(snapshot: Path) -> dict:
    manifests = {p.stem: json.loads(p.read_text()) for p in (snapshot / "manifests").glob("*.json")}
    paths = manifests["all"]
    before = case_bundle_snapshot(paths, relative_to=ROOT)
    implementation = [file_identity(ROOT / "src/sitian" / name) for name in ("scoring.py", "schema.py", "case.py")]
    implementation.append(file_identity(Path(__file__)))
    per_case, failures = {}, []
    for path in paths:
        bundle = CaseBundle.load(path)
        truth = bundle.truth_daily_full()
        standard = bundle.meta.get("aqi_standard")
        forecast = {"issue_date": bundle.issue_date, "region": bundle.region}
        for pollutant, field in FIELDS.items():
            forecast[field + "_lo"] = forecast[field + "_hi"] = [row[pollutant] for row in truth.values()]
        score = score_forecast(forecast, truth, issue_date=bundle.issue_date, horizon=bundle.horizon,
                               region=bundle.region, aqi_standard=standard)
        if not score.valid or score.outcome_composite != 1.0:
            failures.append({"case_id": bundle.case_id, "score": score.outcome_composite, "errors": score.errors,
                             "components": score.components})
        counts = Counter(days=len(truth))
        for day, values in truth.items():
            daily_standard = (standard.get(day) if isinstance(standard, dict) else standard) or aqi_standard_for_date(day)
            full = daily_aqi(values, standard=daily_standard)
            major = daily_aqi({p: v for p, v in values.items() if p in {"PM2.5", "PM10", "O3"}}, standard=daily_standard)
            counts["other_primary_only_days"] += bool(full["primary"]) and not set(full["primary"]) & {"PM2.5", "PM10", "O3"}
            counts["three_pollutant_level_mismatch_days"] += full["level"] != major["level"]
        pm = bundle.observations["PM2.5"]["series"][bundle.region]
        valid_hours = sum(isinstance(v, (float, int)) and not isinstance(v, bool) and v >= 0 for v in pm)
        if valid_hours < 48:
            failures.append({"case_id": bundle.case_id, "valid_pm25_input_hours": valid_hours})
        per_case[path] = (counts, valid_hours)
    splits = {}
    for name, members in sorted(manifests.items()):
        counts = Counter()
        for path in members:
            counts.update(per_case[path][0])
        splits[name] = {"cases": len(members), **counts,
                        "minimum_valid_pm25_input_hours": min(per_case[p][1] for p in members)}
    if before != case_bundle_snapshot(paths, relative_to=ROOT):
        raise ValueError("cases changed during population reward audit")
    if any(file_identity(item["path"]) != item for item in implementation):
        raise ValueError("implementation changed during population reward audit")
    return {"passed": not failures, "cases": len(paths), "reward": reward_spec(),
            "case_snapshot": before, "implementation": implementation, "splits": splits,
            "oracle_outcome_one_cases": len(paths) - sum("score" in row for row in failures),
            "failures": failures, "scope": __doc__.strip()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("use a new audit output")
    report = audit(args.snapshot.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "cases", "oracle_outcome_one_cases", "splits", "failures")}, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 2)
