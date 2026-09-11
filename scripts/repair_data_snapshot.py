#!/usr/bin/env python3
"""Build an isolated corrected dataset; never mutate frozen cases or manifests.

The snapshot is a data preparation artifact, not authorization to resume an old
checkpoint with new inputs or to regard a newly held-out city as unseen by it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sitian.meteorology import wind_direction_label  # noqa: E402
from sitian.schema import aqi_standard_for_date, daily_aqi  # noqa: E402

MANIFESTS = (
    "valid_cases_train", "valid_cases_val", "valid_cases_test",
    "train_without_winter_challenge", "challenge_winter",
    "train_without_winter_or_spatial_holdout", "spatial_ood_val", "spatial_ood_test",
)
TRUTH_KEYS = {"pm25_avg": "PM2.5", "pm10_avg": "PM10", "o3_8h": "O3",
              "so2_avg": "SO2", "no2_avg": "NO2", "co_avg": "CO"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def repair_diagnostics(document: dict) -> list[dict]:
    changes = []
    for day, row in document.get("daily", {}).items():
        if row.get("wind_dir_deg") is None:
            continue
        expected = wind_direction_label(row["wind_dir_deg"])
        if row.get("wind_dir") != expected:
            changes.append({"date": day, "degrees": row["wind_dir_deg"],
                            "before": row.get("wind_dir"), "after": expected})
            row["wind_dir"] = expected
    return changes


def profile(paths: list[Path]) -> dict:
    """Count overlapping forecast windows separately from unique event days."""
    cities, issues, events, event_cities, event_dates, targets = set(), set(), set(), set(), set(), set()
    strata, months, sources, coverage = Counter(), Counter(), Counter(), Counter()
    event_cases = mismatch = direction_mismatch = target_occurrences = 0
    for path in paths:
        meta = read(path / "case.json")
        city, issue = meta["region"], meta["issue_date"]
        if (city, issue) in issues:
            raise ValueError(f"Duplicate city/issue: {city}, {issue}")
        cities.add(city)
        issues.add((city, issue))
        months[issue[:7]] += 1
        strata[meta.get("meta", {}).get("stratum", "unknown")] += 1
        truth = read(path / "truth.json")["daily"]
        has_event = False
        # Case truth is expected to cover exactly issue+1 through issue+horizon.
        days = [(date.fromisoformat(issue) + timedelta(days=i + 1)).isoformat()
                for i in range(meta["horizon"])]
        for day in days:
            row = truth[day]
            concs = {pol: row[key] for key, pol in TRUTH_KEYS.items()}
            full = daily_aqi(concs, standard=aqi_standard_for_date(day))
            three = daily_aqi({k: v for k, v in concs.items() if k in {"PM2.5", "PM10", "O3"}},
                              standard=aqi_standard_for_date(day))
            mismatch += full["level"] != three["level"] or set(full["primary"]) != set(three["primary"])
            targets.add((city, day))
            target_occurrences += 1
            if full["level"] >= 4:
                has_event = True
                events.add((city, day))
                event_cities.add(city)
                event_dates.add(day)
        event_cases += has_event
        guidance = read(path / "guidance.json")["sources"]
        sources.update(guidance.keys())
        for key in ("daily_pm25", "daily_pm10", "daily_o3max"):
            for lead, day in enumerate(days, 1):
                coverage[f"{key}:D{lead}"] += guidance.get("cams", {}).get(key, {}).get(day) is not None
        diag = read(path / "diagnostics.json")
        direction_mismatch += len(repair_diagnostics(diag))
    return {"cases": len(paths), "cities": len(cities),
            "issue_dates": len({issue for _, issue in issues}),
            "months": dict(sorted(months.items())), "strata": dict(strata),
            "event_cases": event_cases, "event_unique_city_days": len(events),
            "event_cities": len(event_cities), "event_valid_dates": len(event_dates),
            "unique_target_city_days": len(targets), "target_day_occurrences": target_occurrences,
            "three_vs_six_mismatch_day_occurrences": mismatch,
            "guidance_source_case_counts": dict(sources), "cams_lead_coverage": dict(coverage),
            "wind_label_mismatches": direction_mismatch}


def copy_corrected(source: Path, out: Path) -> dict:
    source, out = source.resolve(), out.resolve()
    if not source.is_dir():
        raise ValueError(f"Missing case source: {source}")
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("snapshot must be separate from the source tree")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {out}")
    # Copy-on-disk, not hardlinks: future edits cannot mutate the frozen source.
    shutil.copytree(source, out)
    changes, identities, split_counts = [], [], Counter()
    for original in sorted(source.glob("*/*/diagnostics.json")):
        rel = original.relative_to(source)
        split_counts[rel.parts[0]] += 1
        original_hash = sha(original)
        repaired = out / rel
        if sha(repaired) != original_hash:
            raise ValueError(f"Source changed during snapshot copy: {rel}")
        document = read(repaired)
        corrected = repair_diagnostics(document)
        if corrected:
            write(repaired, document)
            changes.append({"file": rel.as_posix(), "changes": corrected})
        if sha(original) != original_hash:
            raise ValueError(f"Source changed during repair: {rel}")
        identities.append({"file": rel.as_posix(), "source_sha256": original_hash,
                           "snapshot_sha256": sha(repaired)})
    return {"case_files_checked": len(identities), "changed_cases": len(changes),
            "cases_by_source_directory": dict(split_counts),
            "changed_day_labels": sum(len(row["changes"]) for row in changes),
            "diagnostic_identities": identities, "changes": changes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "cases/national")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if out.exists() or out == source or source in out.parents or out in source.parents:
        parser.error("--out must be a new directory separate from the source")
    source_manifests = {name: ROOT / "data/interim" / f"{name}.json" for name in MANIFESTS}
    manifest_rows = {name: read(path) for name, path in source_manifests.items()}
    manifest_hashes = {name: sha(path) for name, path in source_manifests.items()}
    for rows in manifest_rows.values():
        for row in rows:
            (ROOT / row).resolve().relative_to(source)
    out.mkdir(parents=True)
    receipt = copy_corrected(source, out / "cases")
    # Retain original split membership for input-only comparisons. Corrected
    # OOD membership goes to a separate directory and requires fresh training.
    for name, rows in manifest_rows.items():
        write(out / "manifests" / f"{name}.json", [
            str(out / "cases" / (ROOT / row).resolve().relative_to(source)) for row in rows
        ])
    corrected = out / "corrected_ood"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_spatial_ood.py"),
                    "--train-source", str(out / "manifests/train_without_winter_challenge.json"),
                    "--regime-train-source", str(out / "manifests/valid_cases_train.json"),
                    "--val-source", str(out / "manifests/valid_cases_val.json"),
                    "--test-source", str(out / "manifests/valid_cases_test.json"),
                    "--winter-challenge", str(out / "manifests/challenge_winter.json"),
                    "--train-out", str(corrected / "train.json"),
                    "--val-out", str(corrected / "val.json"),
                    "--test-out", str(corrected / "test.json"),
                    "--audit-out", str(corrected / "audit.json")], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_guidance_bias.py"),
                    "--manifests", str(corrected / "train.json"),
                    "--out", str(corrected / "guidance_bias_history.json")], check=True)
    profiles = {name: profile([Path(p) for p in read(out / "manifests" / f"{name}.json")])
                for name in ("train_without_winter_or_spatial_holdout", "valid_cases_val",
                             "challenge_winter", "valid_cases_test", "spatial_ood_test")}
    profiles["corrected_ood_train"] = profile([Path(p) for p in read(corrected / "train.json")])
    gap_rows = []
    for split in ("train", "val", "test"):
        for ep in read(ROOT / "data/interim" / f"episodes_{split}.json"):
            if not (source / split / f"{ep['city']}_{ep['issue_date']}" / "case.json").exists():
                gap_rows.append({"split": split, **ep, "reason": "not_built; per-case original failure log unavailable"})
    if any(sha(path) != manifest_hashes[name] for name, path in source_manifests.items()):
        raise ValueError("Frozen source manifest changed during repair")
    write(out / "repair_receipt.json", {"version": "data-quality-v2", "source": str(source),
          "created_utc": datetime.now(UTC).isoformat(),
          "source_manifests_sha256": manifest_hashes, **receipt})
    write(out / "composition_audit.json", profiles)
    write(out / "unbuilt_episodes.json", gap_rows)
    old_holdout = read(ROOT / "data/interim/spatial_ood_audit.json")["heldout_cities"]
    new_holdout = read(corrected / "audit.json")["heldout_cities"]
    write(out / "status.json", {"data_repair_complete": True, "training_ready": False,
          "cases": receipt["case_files_checked"], "changed_day_labels": receipt["changed_day_labels"],
          "cases_by_source_directory": receipt["cases_by_source_directory"],
          "heldout_city_membership_changed": set(old_holdout) != set(new_holdout),
          "unbuilt_episodes": len(gap_rows),
          "raw_cams_available": (ROOT / "data/raw/cams").is_dir(),
          "raw_observations_available": (ROOT / "data/raw/aq_obs/cities").is_dir(),
          "remaining": ["Rebuild upstream missing months from original data; never synthesize truth.",
                        "Freeze a new protocol and regenerate audited training Parquet for the selected snapshot.",
                        "Use corrected_ood/guidance_bias_history.json with corrected_ood/train.json, never the v1 index.",
                        "If OOD membership changes, newly held-out cities require fresh training; never mix old and new input versions in comparisons.",
                        "O3 proxy, D5 absence, winter imbalance and limited independent events remain documented limitations."]})
    print(json.dumps(read(out / "status.json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
