#!/usr/bin/env python3
"""Audit open-evidence download, derivation, and case-attachment coverage.

The audit is intentionally cheap: it validates expected filenames, sidecar
byte counts, and per-issue derived artifacts without re-hashing every large
GRIB file. Downloaders already record SHA256; a full hash audit can be run
separately before publication.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sitian.open_evidence import (
    CAMS_AEROSOL_LEAD_HOURS,
    CAMS_TRACE_GAS_LEAD_HOURS,
    DEFAULT_EVIDENCE_ROOT,
    NWP_SNAPSHOTS_PER_ISSUE,
    NWP_TEMPORAL_PROFILE_VERSION,
)
from sitian.provenance import file_identity


REPO_ROOT = Path(__file__).resolve().parents[1]


def _complete(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return path.stat().st_size > 0 and int(meta.get("bytes", -1)) == path.stat().st_size


def _ratio(found: int, expected: int) -> dict:
    return {"found": found, "expected": expected,
            "fraction": round(found / expected, 6) if expected else 1.0}


def _valid_derived(document: dict, suffix: str) -> bool:
    if suffix == "synoptic":
        sources = document.get("synoptic", {}).get("sources", {})
        return all(len(rows) == NWP_SNAPSHOTS_PER_ISSUE and
                   all(row.get("available") for row in rows) and
                   all(int(row.get("step_hour", 0)) == 0 or row.get("radiation_available")
                       for row in rows)
                   for rows in (sources.get("gfs", []), sources.get("ifs", [])))
    pollution = document.get("pollution", {})
    if suffix == "composition":
        composition = pollution.get("composition", {})
        return all(composition.get(name, {}).get("available")
                   for name in ("aerosol", "column_gases"))
    if suffix == "spatial_obs":
        return bool(pollution.get("spatial_observations", {}).get("available"))
    if suffix == "fires":
        return bool(pollution.get("fires", {}).get("available"))
    return False


def _derived(root: Path, issue_dates: list[str], suffix: str) -> dict:
    base = root / "derived" / "by_issue"
    files, valid = 0, 0
    for day in issue_dates:
        path = base / f"{day}.{suffix}.json"
        if not path.is_file():
            continue
        files += 1
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            valid += _valid_derived(document, suffix)
        except (OSError, json.JSONDecodeError):
            pass
    result = _ratio(valid, len(issue_dates))
    result["files"] = files
    result["invalid_or_incomplete"] = files - valid
    return result


def _composition_component(root: Path, issue_dates: list[str], name: str) -> dict:
    found = 0
    for day in issue_dates:
        path = root / "derived" / "by_issue" / f"{day}.composition.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            found += bool(document.get("pollution", {}).get("composition", {})
                          .get(name, {}).get("available"))
        except (OSError, json.JSONDecodeError):
            pass
    return _ratio(found, len(issue_dates))


def _trace_gas_contract(root: Path, issue_dates: list[str]) -> dict:
    """Validate the full issue-legal model-level-137 gas trajectory contract."""
    expected_leads = list(range(0, 121, 6))
    value_keys = tuple(
        name + "_ppbv_approx"
        for name in ("carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide")
    )
    available = valid = 0
    invalid_issue_dates = []
    for day in issue_dates:
        reasons = set()
        path = root / "derived" / "by_issue" / f"{day}.composition.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            gases = document["pollution"]["composition"]["trace_gases"]
            if not gases.get("available"):
                reasons.add("unavailable")
            else:
                available += 1
                issue = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
                cycle = issue - timedelta(hours=12)
                expected_cycle = cycle.isoformat().replace("+00:00", "Z")
                if gases.get("cycle") != expected_cycle:
                    reasons.add("cycle")
                records = gases.get("records", [])
                if [row.get("lead_hour") for row in records] != expected_leads:
                    reasons.add("lead_hours")
                for row in records:
                    lead = row.get("lead_hour")
                    if not isinstance(lead, int):
                        reasons.add("lead_hour_type")
                        continue
                    if row.get("issue_relative_hour") != lead - 12:
                        reasons.add("issue_relative_hour")
                    expected_valid = (cycle + timedelta(hours=lead)).isoformat().replace(
                        "+00:00", "Z"
                    )
                    if row.get("valid_time") != expected_valid:
                        reasons.add("valid_time")
                    try:
                        visible = datetime.fromisoformat(
                            row["available_at"].replace("Z", "+00:00")
                        )
                        if visible > issue:
                            reasons.add("available_after_issue")
                    except (KeyError, TypeError, ValueError):
                        reasons.add("available_at")
                    cities = row.get("cities")
                    if not isinstance(cities, dict) or not cities:
                        reasons.add("cities")
                        continue
                    for values in cities.values():
                        for key in value_keys:
                            value = values.get(key) if isinstance(values, dict) else None
                            if (not isinstance(value, (int, float))
                                    or not math.isfinite(value) or value < 0):
                                reasons.add("gas_values")
                relative = gases.get("raw", {}).get("relative_path")
                if not isinstance(relative, str) or not _complete(root / relative):
                    reasons.add("raw_provenance")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            reasons.add("document")
        if reasons:
            invalid_issue_dates.append({"issue_date": day, "reasons": sorted(reasons)})
        else:
            valid += 1
    result = _ratio(valid, len(issue_dates))
    result.update({
        "available": available,
        "invalid_contract": len(invalid_issue_dates),
        "expected_lead_hours": expected_leads,
        "invalid_samples": invalid_issue_dates[:20],
    })
    return result


def _nested_value(document: dict, path: tuple[str, ...]):
    value = document
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _composition_contract(
    root: Path,
    issue_dates: list[str],
    *,
    name: str,
    expected_leads: tuple[int, ...],
    value_paths: tuple[tuple[str, ...], ...],
    required_metadata: dict[str, str] | None = None,
) -> dict:
    """Validate timing, scalar fields and raw provenance for one CAMS layer."""
    available = valid = 0
    invalid_issue_dates = []
    for day in issue_dates:
        reasons = set()
        path = root / "derived" / "by_issue" / f"{day}.composition.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            block = document["pollution"]["composition"][name]
            if not block.get("available"):
                reasons.add("unavailable")
            else:
                available += 1
                issue = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
                cycle = issue - timedelta(hours=12)
                expected_cycle = cycle.isoformat().replace("+00:00", "Z")
                if block.get("cycle") != expected_cycle:
                    reasons.add("cycle")
                for key, expected in (required_metadata or {}).items():
                    if block.get(key) != expected:
                        reasons.add(f"metadata_{key}")
                records = block.get("records", [])
                if [row.get("lead_hour") for row in records] != list(expected_leads):
                    reasons.add("lead_hours")
                for row in records:
                    lead = row.get("lead_hour")
                    if not isinstance(lead, int):
                        reasons.add("lead_hour_type")
                        continue
                    if row.get("issue_relative_hour") != lead - 12:
                        reasons.add("issue_relative_hour")
                    expected_valid = (cycle + timedelta(hours=lead)).isoformat().replace(
                        "+00:00", "Z"
                    )
                    if row.get("valid_time") != expected_valid:
                        reasons.add("valid_time")
                    try:
                        visible = datetime.fromisoformat(
                            row["available_at"].replace("Z", "+00:00")
                        )
                        if visible > issue:
                            reasons.add("available_after_issue")
                    except (KeyError, TypeError, ValueError):
                        reasons.add("available_at")
                    cities = row.get("cities")
                    if not isinstance(cities, dict) or not cities:
                        reasons.add("cities")
                        continue
                    for values in cities.values():
                        for value_path in value_paths:
                            value = _nested_value(values, value_path)
                            if (
                                not isinstance(value, (int, float))
                                or isinstance(value, bool)
                                or not math.isfinite(value)
                                or value < 0
                            ):
                                reasons.add("scientific_values")
                relative = block.get("raw", {}).get("relative_path")
                if not isinstance(relative, str) or not _complete(root / relative):
                    reasons.add("raw_provenance")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            reasons.add("document")
        if reasons:
            invalid_issue_dates.append({"issue_date": day, "reasons": sorted(reasons)})
        else:
            valid += 1
    result = _ratio(valid, len(issue_dates))
    result.update({
        "available": available,
        "invalid_contract": len(invalid_issue_dates),
        "expected_lead_hours": list(expected_leads),
        "invalid_samples": invalid_issue_dates[:20],
    })
    return result


def _attachment_complete(case_path: Path) -> tuple[bool, list[str]]:
    reasons = []
    try:
        case = json.loads(case_path.read_text(encoding="utf-8"))
        evidence = json.loads(case_path.with_name("evidence.json").read_text(encoding="utf-8"))
        city, issue_date = case["region"], case["issue_date"]
        if evidence.get("target_city") != city or evidence.get("issue_date") != issue_date:
            reasons.append("case_identity")
        location = evidence.get("target_location", {})
        if not all(isinstance(location.get(key), (int, float)) for key in ("lat", "lon")):
            reasons.append("target_location")
        sources = evidence.get("synoptic", {}).get("sources", {})
        if not all(len(sources.get(source, [])) == NWP_SNAPSHOTS_PER_ISSUE
                   and all(row.get("available") for row in sources[source])
                   and all(int(row.get("step_hour", 0)) == 0 or row.get("radiation_available")
                           for row in sources[source])
                   for source in ("gfs", "ifs")):
            reasons.append("synoptic_contract")
        pollution = evidence.get("pollution", {})
        composition = pollution.get("composition", {})
        if not all(composition.get(name, {}).get("available")
                   for name in ("aerosol", "trace_gases", "column_gases")):
            reasons.append("composition")
        if not pollution.get("fires", {}).get("available"):
            reasons.append("fires")
        context = pollution.get("source_context", {})
        target = context.get("target", {}).get(city, {})
        if not all(name in target for name in ("PM2.5", "PM10", "O3", "SO2", "NO2", "CO")):
            reasons.append("six_pollutant_source_context")
        if not context.get("static", {}).get("target"):
            reasons.append("static_context")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        reasons.append("document")
    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=DEFAULT_EVIDENCE_ROOT / "manifests" / "open_evidence_v1.json")
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--cases", type=Path, default=Path("cases/national"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    issue_dates = manifest["issue_dates"]

    nwp = {}
    nwp_radiation = {}
    for source in ("gfs", "ifs"):
        expected = set()
        expected_radiation = set()
        for row in manifest["nwp"][source]:
            cycle = datetime.fromisoformat(row["cycle"].replace("Z", "+00:00"))
            stamp = cycle.strftime("%Y%m%d%H")
            expected.add(args.root / "raw" / "nwp" / source / stamp /
                         f"{source}_{stamp}_f{int(row['step_hour']):03d}.grib2")
        for row in manifest["nwp_radiation"]["records"][source]:
            cycle = datetime.fromisoformat(row["cycle"].replace("Z", "+00:00"))
            stamp = cycle.strftime("%Y%m%d%H")
            expected_radiation.add(
                args.root / "raw" / "nwp_radiation" / source / stamp /
                f"{source}_{stamp}_f{int(row['step_hour']):03d}.radiation.grib2"
            )
        nwp[source] = _ratio(sum(_complete(path) for path in expected), len(expected))
        nwp[source]["partial_files"] = sum(
            1 for _ in (args.root / "raw" / "nwp" / source).glob("**/*.part")
        )
        nwp_radiation[source] = _ratio(
            sum(_complete(path) for path in expected_radiation), len(expected_radiation)
        )
        nwp_radiation[source]["partial_files"] = sum(
            1 for _ in (args.root / "raw" / "nwp_radiation" / source).glob("**/*.part")
        )

    raw_firms = {}
    for processing in ("standard_monthly", "nrt_daily"):
        base = args.root / "raw" / "firms" / processing
        data = [path for path in base.glob("**/*")
                if path.is_file() and path.suffix in {".gz", ".txt"}]
        raw_firms[processing] = {
            "complete_files": sum(_complete(path) for path in data),
            "data_files": len(data),
            "incomplete_data_files": sum(not _complete(path) for path in data),
            "bytes": sum(path.stat().st_size for path in data),
            "partial_files": sum(1 for _ in base.glob("**/*.part")),
        }

    raw_cams_base = args.root / "raw" / "cams"
    raw_cams = [path for path in raw_cams_base.glob("*.nc") if path.is_file()]
    raw_cams_earthengine_base = args.root / "raw" / "cams_earthengine"
    raw_cams_earthengine = [path for path in raw_cams_earthengine_base.glob("*.tif")
                            if path.is_file()]
    static_base = args.root / "raw" / "static"
    raw_static = [path for path in static_base.glob("**/*")
                  if path.is_file() and path.suffix in {".zip", ".grib2"}]
    extracted_static = [path for path in static_base.glob("**/*.nc") if path.is_file()]
    case_roots = [args.cases / split for split in ("train", "val", "test")]
    case_paths = [path for root in case_roots if root.exists() for path in root.glob("*/case.json")]
    attached = 0
    invalid_attachments = []
    for path in case_paths:
        complete, reasons = _attachment_complete(path)
        attached += complete
        if not complete and len(invalid_attachments) < 30:
            invalid_attachments.append({"case": str(path.parent), "reasons": reasons})
    total_cases = len(case_paths)

    required_aerosol = _composition_contract(
        args.root,
        issue_dates,
        name="aerosol",
        expected_leads=CAMS_AEROSOL_LEAD_HOURS,
        value_paths=tuple(
            ("aod550", name)
            for name in (
                "total", "fine", "dust", "organic_matter", "black_carbon",
                "sulphate", "nitrate",
            )
        ),
    )
    required_trace_gases = _trace_gas_contract(args.root, issue_dates)
    required_column_gases = _composition_contract(
        args.root,
        issue_dates,
        name="column_gases",
        expected_leads=CAMS_TRACE_GAS_LEAD_HOURS,
        value_paths=tuple(
            (name + "_kg_m2",)
            for name in ("carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide")
        ),
        required_metadata={"semantics": "total_column_mass", "unit": "kg m-2"},
    )
    report = {
        "artifact_type": "open_evidence_coverage",
        "audit_version": "1.2.0",
        "contract_version": manifest["contract_version"],
        "source_manifest": file_identity(args.manifest),
        "nwp_temporal_profile": manifest.get("nwp_temporal_profile"),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provenance": {
            "audit_implementation": file_identity(Path(__file__), relative_to=REPO_ROOT),
            "derivation_implementations": {
                name: file_identity(REPO_ROOT / path, relative_to=REPO_ROOT)
                for name, path in {
                    "synoptic": "scripts/derive_open_synoptic.py",
                    "composition": "scripts/derive_open_composition.py",
                    "spatial_observations": "scripts/derive_spatial_observations.py",
                    "fires": "scripts/derive_open_fires.py",
                    "static_context": "scripts/derive_static_context.py",
                    "case_attachment": "scripts/attach_open_evidence.py",
                }.items()
            },
            "open_evidence_contract_implementation": file_identity(
                REPO_ROOT / "src/sitian/open_evidence.py", relative_to=REPO_ROOT
            ),
        },
        "issue_dates": len(issue_dates),
        "raw": {
            "nwp": nwp,
            "nwp_radiation": nwp_radiation,
            "cams": {"complete_files": sum(_complete(path) for path in raw_cams),
                     "data_files": len(raw_cams),
                     "bytes": sum(path.stat().st_size for path in raw_cams)},
            "cams_earthengine_columns": {
                "complete_files": sum(_complete(path) for path in raw_cams_earthengine),
                "data_files": len(raw_cams_earthengine),
                "bytes": sum(path.stat().st_size for path in raw_cams_earthengine),
            },
            "firms": raw_firms,
            "static": {"complete_files": sum(_complete(path) for path in raw_static),
                       "downloaded_files": len(raw_static),
                       "extracted_files": len(extracted_static),
                       "bytes": sum(path.stat().st_size for path in raw_static + extracted_static)},
        },
        "derived_by_issue": {
            "synoptic": _derived(args.root, issue_dates, "synoptic"),
            "composition": _derived(args.root, issue_dates, "composition"),
            "spatial_observations": _derived(args.root, issue_dates, "spatial_obs"),
            "fires": _derived(args.root, issue_dates, "fires"),
        },
        "required_by_issue": {
            "cams_aerosol_speciation": required_aerosol,
            "cams_model_level_137_trace_gases":
                required_trace_gases,
            "cams_total_column_trace_gases": required_column_gases,
        },
        "static_context": {
            "available": (args.root / "derived" / "static_context.json").is_file()
        },
        "case_attachments": {
            **_ratio(attached, total_cases),
            "invalid_samples": invalid_attachments,
        },
    }
    report["ready_for_full_attachment"] = (
        (manifest.get("nwp_temporal_profile") or {}).get("version")
            == NWP_TEMPORAL_PROFILE_VERSION
        and all(value["fraction"] == 1.0 for value in nwp.values())
        and all(value["fraction"] == 1.0 for value in nwp_radiation.values())
        and all(value["fraction"] == 1.0 for value in report["derived_by_issue"].values())
        and all(value["fraction"] == 1.0 for value in report["required_by_issue"].values())
        and report["static_context"]["available"]
    )
    out = args.out or args.root / "manifests" / "open_evidence_v1_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
