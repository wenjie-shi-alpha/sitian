#!/usr/bin/env python3
"""Audit and atomically enrich national cases with all six issue-time observations.

The first pass is read-only and requires the existing PM2.5/PM10/O3 values to
match their public raw archive exactly. Writes happen only after every selected
case passes that audit, preventing a field addition from silently changing the
established inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_national_case import OBS_TYPES, read_obs_hours  # noqa: E402

EXISTING_POLLUTANTS = ("PM2.5", "PM10", "O3")


def _matches_audited_prefix(current: dict, desired: dict, cutoff: str) -> bool:
    """Accept the exact safe slice plus, before migration, only a trailing 08:00 row."""
    current_times = current.get("times", [])
    desired_times = desired.get("times", [])
    if current_times[:len(desired_times)] != desired_times:
        return False
    if any(timestamp != cutoff for timestamp in current_times[len(desired_times):]):
        return False
    current_series = current.get("series", {})
    desired_series = desired.get("series", {})
    if set(current_series) != set(desired_series):
        return False
    return all(
        current_series[region][:len(values)] == values
        for region, values in desired_series.items()
    )


def _case_paths(case_root: Path) -> list[Path]:
    paths = []
    for split in ("train", "val", "test"):
        paths.extend(path for path in sorted((case_root / split).iterdir())
                     if path.is_dir() and (path / "case.json").is_file())
    return paths


def _candidate(path: Path) -> dict:
    meta = json.loads((path / "case.json").read_text(encoding="utf-8"))
    current = json.loads((path / "observations.json").read_text(encoding="utf-8"))
    city, issue_date = meta["region"], meta["issue_date"]
    desired = read_obs_hours(city, date.fromisoformat(issue_date))
    missing = sorted(set(OBS_TYPES) - set(desired))
    if missing:
        raise ValueError(f"{path.name}: raw archive has no observations for {missing}")
    cutoff = f"{issue_date}T08:00"
    for pollutant in EXISTING_POLLUTANTS:
        if not _matches_audited_prefix(
            current.get(pollutant, {}), desired[pollutant], cutoff
        ):
            raise ValueError(f"{path.name}: existing {pollutant} differs from raw archive")
    if any(timestamp >= cutoff for block in desired.values()
           for timestamp in block.get("times", [])):
        raise ValueError(f"{path.name}: issue-time gate violation")
    for pollutant, block in desired.items():
        values = block["series"].get(city, [])
        if len(values) != len(block["times"]):
            raise ValueError(f"{path.name}: {pollutant} time/value length mismatch")
    return desired


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, default=REPO_ROOT / "cases" / "national")
    parser.add_argument("--apply", action="store_true",
                        help="write after the complete read-only audit succeeds")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "data" / "interim" /
                        "national_observation_enrichment_v1.json")
    args = parser.parse_args()
    paths = _case_paths(args.case_root)
    if args.limit is not None:
        paths = paths[:args.limit]

    counts = Counter()
    audited: dict[Path, dict] = {}
    for number, path in enumerate(paths, 1):
        desired = _candidate(path)
        audited[path] = desired
        for pollutant, block in desired.items():
            counts[f"{pollutant}_cases"] += 1
            counts[f"{pollutant}_values"] += sum(
                value is not None for values in block["series"].values() for value in values)
        if number % 500 == 0:
            print(f"audit {number}/{len(paths)}", flush=True)

    if args.apply:
        for number, (path, desired) in enumerate(audited.items(), 1):
            target = path / "observations.json"
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            temporary.write_text(json.dumps(desired, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
            temporary.replace(target)
            if number % 500 == 0:
                print(f"write {number}/{len(paths)}", flush=True)

    report = {
        "artifact_type": "national_observation_enrichment",
        "version": "1.1.0",
        "case_count": len(paths),
        "pollutants": list(OBS_TYPES),
        "audit": (
            "existing PM2.5/PM10/O3 safe prefix exactly matches raw archive; "
            "all six present; observations strictly before 08:00 issue time"
        ),
        "applied": args.apply,
        "counts": dict(sorted(counts.items())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
