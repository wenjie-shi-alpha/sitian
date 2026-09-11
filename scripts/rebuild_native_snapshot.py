#!/usr/bin/env python3
"""Reconstruct an immutable offline snapshot from the 9800X3D city exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sitian.case import CaseBundle
from sitian.data_contract import issue_time, timestamp
from sitian.native_data import NativeCams, NativeObservations, jsonl, parse_concentration
from sitian.provenance import file_identity
import attach_open_evidence as attach


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")


def verify_delivery(root):
    identities = []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (root / name.lstrip("*")).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("delivery path escapes root")
        actual = file_identity(path)
        if actual["sha256"] != expected:
            raise ValueError(f"delivery checksum mismatch: {path}")
        identities.append(actual)
    if not identities:
        raise ValueError("empty delivery")
    return identities


def regional_rows(path, sources):
    acc, refs, seen = defaultdict(list), defaultdict(set), set()
    for record in jsonl(path):
        if record["frequency"] != "day" or record["domain"] != "d02" or record["model"] not in {"cmaq", "naqp"}:
            continue
        if record["source_sha256"] not in sources:
            raise ValueError("unknown regional source")
        row = record["row"]
        cycle = issue_time(record["issue_date"]) - timedelta(hours=12)
        if timestamp(record["cycle_bjt"]) != cycle:
            raise ValueError("regional cycle mismatch")
        key = record["issue_date"], record["model"], row["cityname"], row["datadate"]
        unique = (*key, row["stationcode"])
        if unique in seen:
            raise ValueError("duplicate regional station/day")
        seen.add(unique)
        value = parse_concentration(row.get("pm25_24h"))
        if value is not None:
            acc[key].append(value)
            refs[key].add(record["source_sha256"])
    return acc, refs


def annotate_estimates(value):
    if isinstance(value, dict):
        if value.get("available_at"):
            value["availability_basis"] = "legacy_fixed_latency"
            value["actual_publication_verified"] = False
        for child in list(value.values()):
            annotate_estimates(child)
    elif isinstance(value, list):
        for child in value:
            annotate_estimates(child)


def run(raw, admission, out):
    receipts = verify_delivery(raw) + verify_delivery(admission)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "delivery_verification.json", receipts)
    manifest = json.loads((raw / "batch2/manifest.json").read_text())
    if manifest["failures"]:
        raise ValueError("exporter failures")
    sources = {r["sha256"]: r for r in manifest["sources"]}
    cams = NativeCams(raw / "batch2/cams_native.jsonl.gz", sources)
    print(f"Native CAMS: {len(cams.rows)} cases, {len(cams.overlaps)} checked overlaps", flush=True)
    obs = NativeObservations(raw / "batch2/observations_native.jsonl.gz", sources)
    print(f"Native observations: {len(obs.rows)} rows", flush=True)
    regional_manifest = json.loads((raw / "regional/manifest.json").read_text())
    regional_sources = {r["sha256"]: r for r in regional_manifest["sources"]}
    regional, regional_refs = regional_rows(raw / "regional/regional_city_rows.jsonl.gz", regional_sources)
    print(f"Regional daily city aggregates: {len(regional)}", flush=True)
    coords = json.loads((ROOT / "data/interim/city_coords.json").read_text())
    documents, document_refs = {}, {}
    for wrapper in jsonl(raw / "evidence/existing_city_evidence.jsonl.gz"):
        d = wrapper["data"]
        if "static_context" in d:
            key = (None, "static_context")
        else:
            kind = ("synoptic" if "synoptic" in d else next(iter(d["pollution"])))
            kind = "spatial_obs" if kind == "spatial_observations" else kind
            key = wrapper["issue_date"], kind
        if key in documents:
            raise ValueError("duplicate derived evidence document")
        annotate_estimates(d)
        documents[key], document_refs[key] = d, wrapper["source_sha256"]
    # The existing attachment functions consume immutable shared documents and
    # copy the city data before terrain masking. No temporary 20GB unpack needed.
    original_read = attach._read
    def read_document(path):
        if path.name == "static_context.json":
            return documents[(None, "static_context")]
        day, kind, _ = path.name.split(".")
        return documents.get((day, kind))
    attach._read = read_document
    paths = json.loads((admission / "candidate_pool.json").read_text())
    mapping, changes, counts = {}, [], Counter()
    spatial_cache = set()
    try:
        for i, path in enumerate(paths, 1):
            old = CaseBundle.load(ROOT / path)
            b = CaseBundle(old.case_id, old.issue_date, old.region, old.horizon,
                           [old.region], deepcopy(old.meta))
            # The legacy climatology includes 2025-03-31. Under the same
            # conservative daily-verification latency it is not yet available
            # for the very first 2025-04-01 08:00 issue.
            if issue_time(b.issue_date) <= timestamp("2025-04-01T12:00:00+08:00"):
                b.meta.pop("climatology", None)
                counts["climatology_before_assumed_availability_masked"] += 1
            days = b.forecast_dates()
            b.observations = obs.inputs(b.region, b.issue_date)
            b.truth = obs.truth(b.region, days)
            cams_source, b.diagnostics = cams.build(b.region, b.issue_date, days)
            b.guidance = {"sources": {"cams": cams_source}}
            for model in ("cmaq", "naqp"):
                if model not in old.guidance.get("sources", {}):
                    continue
                cn = b.region + ("" if b.region.endswith(("市", "州", "区")) else "市")
                daily, source_refs = {}, set()
                for day in days:
                    key = b.issue_date, model, cn, day
                    values = regional.get(key)
                    if values:
                        daily[day] = round(sum(values)/len(values), 1)
                        source_refs.update(regional_refs[key])
                if not daily:
                    raise ValueError(f"cannot reconstruct existing regional source: {b.case_id}/{model}")
                if any(value != old.guidance["sources"][model]["daily_pm25"].get(day) for day, value in daily.items()):
                    raise ValueError("regional aggregate differs from original contract")
                cycle = issue_time(b.issue_date)-timedelta(hours=12)
                b.guidance["sources"][model] = {
                    "daily_pm25": daily, "cycle": cycle.isoformat(),
                    "available_at": (cycle+timedelta(hours=10)).isoformat(),
                    "availability_basis": "configured_latency", "actual_publication_verified": False,
                    "source_sha256": sorted(source_refs), "dependency_group": model,
                    "measurement_contracts": {"daily_pm25": {"statistic": "daily_mean", "unit": "µg/m³",
                        "unit_basis": "inherited_project_contract_not_independently_documented_by_provider"}},
                    "note": "d02前日20时起报；原生CSV pm25_24h站点城市均值。沿用原项目µg/m³单位契约，原README未注明单位；假设10h供数延迟。"}
            if b.issue_date not in spatial_cache:
                documents[(b.issue_date, "spatial_obs")] = obs.spatial(coords, b.issue_date)
                spatial_cache.add(b.issue_date)
            b.evidence = attach.build_case_evidence(Path("by_issue"), b.issue_date, b.region, coords)
            if not b.evidence.get("synoptic") or not b.evidence.get("pollution", {}).get("composition"):
                raise ValueError(f"required evidence missing: {b.case_id}")
            context = b.evidence["pollution"]["source_context"]
            spatial = documents[(b.issue_date, "spatial_obs")]["pollution"]["spatial_observations"]
            context.update({k: spatial[k] for k in ("source_sha256", "available_at", "availability_basis", "actual_publication_verified")})
            fires = b.evidence["pollution"].get("fires", {})
            if fires.get("standard_type0_detections", 0):
                # Aggregated NRT and retrospectively reprocessed locations cannot
                # be separated here. Publication-lag approval is NOT approval to
                # treat a retrospective product as an as-issued observation.
                b.evidence["pollution"]["fires"] = {
                    "available": False, "reason": "mixed_retrospective_and_nrt_aggregates_cannot_be_separated",
                    "note": "已接收历史火点混合汇总；本离线协议不将事后科学处理产品当时效内实况。"}
                counts["mixed_fire_aggregate_masked"] += 1
            elif fires.get("available"):
                latest = timestamp(fires["window"]["newest_acquisition"])
                available = latest + timedelta(hours=6)
                if available >= issue_time(b.issue_date):
                    b.evidence["pollution"]["fires"] = {
                        "available": False, "reason": "aggregate_window_not_strictly_before_issue_after_6h_latency",
                        "note": "需原生NRT火点按采集时间+6h<起报时间重新过滤后才能提供。"}
                    counts["nrt_boundary_aggregate_masked"] += 1
                else:
                    fires.update(available_at=available.isoformat(), availability_basis="configured_latency",
                                 actual_publication_verified=False)
                    counts["nrt_fire_cases"] += 1
            b.evidence["provenance"] = {
                "representation": "existing_derived_city_evidence_with_native_spatial_observations",
                "actual_publication_verified": False,
                "source_documents": {kind: document_refs[(b.issue_date, kind)] for kind in ("synoptic", "composition", "fires")},
                "static_document": document_refs[(None, "static_context")],
                "raw_grib_full_hash_reverified": False,
            }
            b.meta.update(data_snapshot=out.name, publication_policy="offline-assumed-latency-v1",
                          previous_forecast_policy="not_imported_without_source_and_issue_lineage",
                          expert_policy="no_verified_expert_cards_available",
                          climatology_provenance={"file": file_identity(ROOT / "data/interim/climatology.json"),
                              "latest_training_date": "2025-03-31", "representation": "legacy_derived_background",
                              "available_at": "2025-04-01T12:00:00+08:00", "availability_basis": "configured_latency",
                              "actual_publication_verified": False})
            if b.audit_time_gate():
                raise ValueError(f"{b.case_id}: {b.audit_time_gate()[:2]}")
            destination = out / "cases" / old.meta["split"] / b.case_id
            b.save(destination)
            mapping[path] = str(destination.resolve())
            differing = [f"{day}/{field}" for day in days for field in b._TRUTH_KEYS
                         if b.truth["daily"][day][field] != (old.truth or {}).get("daily", {}).get(day, {}).get(field)]
            if differing:
                changes.append({"case_id": b.case_id, "truth_fields_recomputed_differently": differing})
            counts["cases"] += 1
            counts["regional_cases"] += int(len(b.guidance["sources"]) > 1)
            if i % 250 == 0:
                print(f"Rebuilt {i}/{len(paths)}", flush=True)
    finally:
        attach._read = original_read
    write(out / "manifests/all.json", [mapping[p] for p in paths])
    for manifest_path in sorted((admission / "candidate_splits").glob("*.json")):
        members = json.loads(manifest_path.read_text())
        write(out / "manifests" / manifest_path.name, [mapping[p] for p in members])
    write(out / "source_registry.json", {**sources, **regional_sources})
    write(out / "source_overlap_audit.json", cams.overlaps)
    write(out / "truth_reconstruction_changes.json", changes)
    write(out / "rebuild_summary.json", {**counts, "truth_changed_cases": len(changes),
        "offline_publication_assumptions_accepted": True, "actual_publication_verified": False,
        "candidate_pool_identity": file_identity(admission / "candidate_pool.json"),
        "original_case_paths": paths, "builder": file_identity(Path(__file__)),
        "native_implementation": file_identity(ROOT / "src/sitian/native_data.py"), "training_started": False})
    print(json.dumps(dict(counts)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.raw.resolve(), args.admission.resolve(), args.out.resolve())
