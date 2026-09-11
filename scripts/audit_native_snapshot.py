#!/usr/bin/env python3
"""Independent receiver comparisons: saved cases versus delivered raw values."""
import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sitian.case import CaseBundle
from sitian.data_contract import CHINA_TZ, issue_time, timestamp
from sitian.native_data import jsonl, parse_concentration
from sitian.provenance import file_identity, case_bundle_snapshot


def audit(snapshot, raw, admission):
    native = {(r["city"], r["issue_date"], r["canonical_variable"], r["source_sha256"]): r
              for r in jsonl(raw / "batch2/cams_native.jsonl.gz")}
    observations = {(datetime.strptime(r["date"], "%Y%m%d").date().isoformat(), int(r["hour"]), r["type"]): r
                    for r in jsonl(raw / "batch2/observations_native.jsonl.gz")}
    paths = json.loads((snapshot / "manifests/all.json").read_text())
    snapshot_before = case_bundle_snapshot(paths, relative_to=ROOT)
    candidates = json.loads((admission / "candidate_pool.json").read_text())
    if {Path(p).name for p in candidates} != {Path(p).name for p in paths} or len(paths) != len(candidates):
        raise ValueError("candidate population changed")
    counts, failures = Counter(), []
    for i, path in enumerate(paths, 1):
        try:
            b = CaseBundle.load(path)
            json.dumps([b.observations, b.diagnostics, b.guidance, b.evidence, b.truth], allow_nan=False)
            if b.audit_time_gate(): raise ValueError("visible timestamp gate")
            cutoff = issue_time(b.issue_date)
            for name, block in b.diagnostics["native"]["cams"]["fields"].items():
                r = native[(b.region, b.issue_date, block["native_variable"], block["source_sha256"])]
                values = [v[0] for v in r["values"]] if r["canonical_variable"] == "go3" else r["values"]
                if block["times"] != r["valid_times"]: raise ValueError("native time mismatch")
                factor = 1e9 if r["canonical_variable"] in {"pm2p5", "pm10"} else 100 if r["canonical_variable"] == "tcc" else 1
                offset = -273.15 if r["canonical_variable"] in {"t2m", "d2m"} else 0
                expected = [None if v is None else v*factor+offset for v in values]
                if block["values"] != expected: raise ValueError("native numeric mismatch")
                if timestamp(block["available_at"]) != timestamp(r["cycle"])+timedelta(hours=10):
                    raise ValueError("CAMS latency mismatch")
                counts["native_samples_compared"] += len(values)
            for pol, block in b.observations.items():
                expected_times, expected_values = [], []
                for offset in range(72, 1, -1):
                    t = cutoff-timedelta(hours=offset)
                    r = observations.get((t.date().isoformat(), t.hour, pol))
                    if r:
                        expected_times.append(t.isoformat())
                        expected_values.append(parse_concentration(r["city_values"].get(b.region)))
                if block["times"] != expected_times or block["series"][b.region] != expected_values:
                    raise ValueError("observation raw/window mismatch")
                counts["input_observation_samples_compared"] += len(expected_times)
            for day in b.forecast_dates():
                for pol, key in {"PM2.5":"pm25_avg", "PM10":"pm10_avg", "O3_8h":"o3_8h", "NO2":"no2_avg", "SO2":"so2_avg", "CO":"co_avg"}.items():
                    values = [parse_concentration(observations.get((day,h,pol), {}).get("city_values", {}).get(b.region)) for h in range(24)]
                    values = [v for v in values if v is not None]
                    exception = pol == "O3_8h" and values and max(values)>160
                    if len(values)<(14 if pol=="O3_8h" else 20) and not exception: raise ValueError("invalid truth hours")
                    value = round(max(values) if pol == "O3_8h" else math.fsum(values)/len(values), 1)
                    # Half-way decimal rounding can differ by 0.1 when fsum and
                    # the preserved legacy sequential sum straddle the tie.
                    sequential = round(sum(values)/len(values), 1) if pol != "O3_8h" else value
                    if b.truth["daily"][day][key] != sequential: raise ValueError("truth aggregate mismatch")
                    counts["floating_point_rounding_ties"] += int(value != sequential)
                    counts["truth_values_compared"] += 1
                cams = b.guidance["sources"]["cams"]
                for name, field, factor in [("pm25_ug_m3","daily_pm25",1), ("pm10_ug_m3","daily_pm10",1), ("o3_mass_mixing_ratio","daily_o3max",1.2e9)]:
                    block = b.diagnostics["native"]["cams"]["fields"][name]
                    start = datetime.fromisoformat(day).replace(tzinfo=CHINA_TZ)
                    vals = [v for t,v in zip(block["times"],block["values"]) if start <= timestamp(t) < start+timedelta(days=1) and v is not None]
                    if not vals:
                        if day in cams[field] or day in cams["partial_"+field]: raise ValueError("invented daily guidance")
                        continue
                    value = round((max(vals) if field=="daily_o3max" else sum(vals)/len(vals))*factor,1)
                    coverage = cams["day_coverage"][field][day]
                    full = timestamp(block["times"][-1]) >= start+timedelta(days=1)
                    if coverage["complete"] != full: raise ValueError("incorrect daily coverage")
                    if value != cams[field if full else "partial_"+field].get(day): raise ValueError("daily native aggregate mismatch")
                    counts["daily_guidance_values_compared"] += 1
            for fires in [b.evidence["pollution"]["fires"]]:
                if fires.get("available"):
                    if fires.get("standard_type0_detections", 0): raise ValueError("retrospective fire leak")
                    if timestamp(fires["window"]["newest_acquisition"])+timedelta(hours=6) >= cutoff:
                        raise ValueError("fire boundary leak")
            for row in b.diagnostics["daily"].values():
                if row["rain_mm"] is not None: raise ValueError("unsupported precipitation accumulation")
            counts["cases"] += 1
        except Exception as exc:
            failures.append({"path": path, "error": str(exc)})
        if i % 500 == 0: print(f"Compared {i}/{len(paths)}, failures={len(failures)}", flush=True)
    split_counts, targets, cities = {}, {}, {}
    for p in sorted((admission / "candidate_splits").glob("*.json")):
        saved = json.loads((snapshot / "manifests" / p.name).read_text())
        original = json.loads(p.read_text())
        if [Path(x).name for x in saved] != [Path(x).name for x in original]:
            failures.append({"split":p.stem,"error":"frozen membership changed"})
        split_counts[p.stem], targets[p.stem], cities[p.stem] = len(saved), set(), set()
        for path in saved:
            b = CaseBundle.load(path)
            targets[p.stem].update((b.region,d) for d in b.forecast_dates())
            cities[p.stem].add(b.region)
    overlaps = {s: len(targets["train"]&days) for s,days in targets.items() if s!="train"}
    ood_overlap = cities["train"] & (cities["spatial_ood_val"]|cities["spatial_ood_test"])
    if any(overlaps.values()) or ood_overlap: failures.append({"error":"split leakage"})
    if case_bundle_snapshot(paths, relative_to=ROOT) != snapshot_before:
        failures.append({"error":"case files changed during audit"})
    return {"passed": not failures, "counts":dict(counts), "split_counts":split_counts,
            "case_snapshot": snapshot_before,
            "source_inputs": [file_identity(raw/"batch2"/name) for name in ("manifest.json", "cams_native.jsonl.gz", "observations_native.jsonl.gz")],
            "train_target_city_day_overlap":overlaps,"spatial_ood_cities_in_train":sorted(ood_overlap),
            "failures":failures,"audit_implementation":file_identity(Path(__file__)),
            "scope":"Raw-export values, timestamps, aggregation, source identity, frozen splits; original large files remain on 9800X3D"}


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--snapshot",type=Path,required=True);p.add_argument("--raw",type=Path,required=True);p.add_argument("--admission",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    report=audit(a.snapshot,a.raw,a.admission)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=1)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"failures","audit_implementation"}},ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)
