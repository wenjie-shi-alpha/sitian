"""Read-only readiness checks; writes only this audit's aggregate evidence.json.

Run: PYTHONPATH=src python3 docs/reports/rl_readiness_20260907/audit.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from sitian.provenance import case_bundle_snapshot
from sitian.schema import aqi_standard_for_date, daily_aqi
from sitian.scoring import reward_spec


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = {
        "valid_train": "valid_cases_train.json",
        "val": "valid_cases_val.json",
        "test": "valid_cases_test.json",
        "rl_train": "train_without_winter_or_spatial_holdout.json",
        "winter_selection": "challenge_winter.json",
        "ood_val": "spatial_ood_val.json",
        "ood_test": "spatial_ood_test.json",
    }
    profiles, city_days, cities, ids = {}, {}, {}, {}
    metadata_cache = {}
    key_map = {"pm25_avg": "PM2.5", "pm10_avg": "PM10", "o3_8h": "O3",
               "so2_avg": "SO2", "no2_avg": "NO2", "co_avg": "CO"}
    for label, name in names.items():
        paths = read("data/interim/" + name)
        strata, primary, issues = Counter(), Counter(), Counter()
        city_days[label], cities[label], ids[label] = set(), set(), set()
        events = missing = 0
        for rel in paths:
            if rel not in metadata_cache:
                meta = read(rel + "/case.json")
                truth = read(rel + "/truth.json")["daily"]
                aqi = {day: daily_aqi({key_map[k]: v for k, v in record.items() if k in key_map},
                                     standard=aqi_standard_for_date(day))
                       for day, record in truth.items()}
                metadata_cache[rel] = meta, aqi
            meta, aqi = metadata_cache[rel]
            cities[label].add(meta["region"])
            ids[label].add(meta["case_id"])
            strata[meta["meta"]["stratum"]] += 1
            issues[meta["issue_date"]] += 1
            events += any(day["level"] >= 4 for day in aqi.values())
            for day, record in aqi.items():
                city_days[label].add((meta["region"], day))
                primary.update(record["primary"])
            missing += not (ROOT / rel / "evidence.json").is_file()
        profiles[label] = {"cases": len(paths), "unique_ids": len(ids[label]),
                           "cities": len(cities[label]), "issue_dates": len(issues),
                           "issue_min": min(issues), "issue_max": max(issues),
                           "strata": dict(strata), "truth_event_cases": events,
                           "primary_memberships_episode_days": dict(primary),
                           "unique_forecast_city_days": len(city_days[label]),
                           "missing_evidence_files": missing}
    overlaps = {f"rl_train_vs_{label}": {
        "case_ids": len(ids["rl_train"] & ids[label]),
        "forecast_city_days": len(city_days["rl_train"] & city_days[label]),
        "cities": len(cities["rl_train"] & cities[label])}
        for label in ("winter_selection", "val", "test", "ood_val", "ood_test")}
    manifest = read("data/verl/manifest.json")
    snapshots = {}
    for label, previous in manifest["input_snapshots"].items():
        current = case_bundle_snapshot([ROOT / rel for rel in previous["case_paths"]],
                                       relative_to=ROOT, include_truth=True)
        snapshots[label] = {"matches": current["sha256"] == previous["sha256"],
                            "cases": current["cases"], "files": current["files"],
                            "bytes": current["bytes"], "sha256": current["sha256"]}
    parquet = {}
    for filename, expected in manifest["outputs"].items():
        p = ROOT / "data/verl" / filename
        parquet[filename] = {"matches": sha(p) == expected["sha256"],
                             "rows": expected["rows"]}
    provenance = {}
    for filename in ("probe_open_evidence_v079_n50_g8.json", "eval_tabular_open_evidence_v079.json",
                     "expert_evidence_contract_audit.json", "verl_runtime_contract.json"):
        artifact = read("data/interim/" + filename)
        checks = []
        def walk(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    path = ROOT / value["path"]
                    checks.append({"path": value["path"], "matches": path.is_file() and sha(path) == value["sha256"]})
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(artifact.get("provenance", artifact.get("inputs", {})))
        provenance[filename] = checks
    probe = read("data/interim/probe_open_evidence_v079_n50_g8.json")
    rows = probe["rows"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"]].append(row)
    prompt_maxima = sorted(row["usage"]["max_prompt_tokens"] for row in rows if row.get("usage"))
    submitted = sum(bool(row.get("submitted")) for row in rows)
    void_submitted = sum(bool(row.get("submitted")) and (row.get("void_assistant_turns") or 0) > 0 for row in rows)
    log = (ROOT / "data/interim/verl_training_smoke.log").read_text()
    preflight = read("data/interim/rl_preflight_audit_20260907.json")
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "reward": reward_spec(),
        "grain": "one target city, one issue date, five forecast days; event means any truth AQI level >= 4",
        "profiles": profiles, "overlaps": overlaps,
        "dataset_input_snapshots": snapshots, "parquet_identity": parquet, "provenance_identity": provenance,
        "raw_case_counts": {s: sum(1 for _ in (ROOT / "cases/national" / s).glob("*/case.json"))
                            for s in ("train", "val", "test")},
        "probe": {"cases": len(groups), "rollouts": len(rows), "submitted": submitted,
                  "submitted_with_void": void_submitted,
                  "submitted_without_void": submitted - void_submitted,
                  "no_void_submitted_fraction": (submitted - void_submitted) / len(rows),
                  "counterfactual_boundary": "Counts trajectories that used a frozen-driver retry; not a measured veRL submission rate.",
                  "usage_present": len(prompt_maxima), "max_prompt": max(prompt_maxima),
                  "p95_prompt": prompt_maxima[int(.95 * (len(prompt_maxima) - 1))],
                  "prompt_over_24000": sum(n > 24000 for n in prompt_maxima),
                  "prompt_over_28000": sum(n > 28000 for n in prompt_maxima),
                  "llm_calls_ge12": sum((r.get("usage") or {}).get("llm_calls", 0) >= 12 for r in rows),
                  "mean_outcome_all_rollouts": sum(r.get("outcome_composite") or 0 for r in rows) / len(rows),
                  "summary": probe["summary"]},
        "training_evidence": {"smoke_report_exists": (ROOT / "data/interim/training_smoke.json").is_file(),
                              "checkpoint_file_count": sum(p.is_file() for p in (ROOT / "data/checkpoints").rglob("*")),
                              "pg_loss_log_entries": len(re.findall("actor/pg_loss", log)),
                              "sync_log_entries": len(re.findall("update_weights done|timing_s/update_weights", log)),
                              "decode_error_entries": len(re.findall("Failed to decode tool call", log)),
                              "requested_run": re.findall(r"phase=train[^\n]*", log)[0]},
        "fresh_preflight": {"passed": preflight["preflight_passed"],
                            "gates": {k: v["pass"] for k, v in preflight["preflight"].items()}},
    }
    output = Path(__file__).with_name("evidence.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("profiles", "overlaps", "dataset_input_snapshots", "training_evidence", "fresh_preflight")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
