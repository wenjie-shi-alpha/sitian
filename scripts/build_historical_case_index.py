#!/usr/bin/env python3
"""Build a separate, auditable train-only analog index; never start training."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.analogs import (
    HistoricalCaseIndex,
    make_artifact,
    record_from_bundle,
)
from sitian.case import CaseBundle
from sitian.paths import resolve_case_dir
from sitian.provenance import case_bundle_snapshot, file_identity


def manifest_paths(path: Path) -> list[Path]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
        raise ValueError(f"manifest must be a list of case paths: {path}")
    return [resolve_case_dir(row) for row in rows]


def build_index(train_manifest: Path, exclude_manifests: list[Path]) -> dict:
    if not exclude_manifests:
        raise ValueError("explicit validation/test/selection exclusion manifests are required")
    train = manifest_paths(train_manifest)
    if not train:
        raise ValueError("training manifest is empty")
    exclusions = set()
    excluded_ids = set()
    # Read metadata only: held-out outcomes never enter this builder.
    for manifest in exclude_manifests:
        for path in manifest_paths(manifest):
            meta = json.loads((path / "case.json").read_text(encoding="utf-8"))
            excluded_ids.add(meta["case_id"])
            start = date.fromisoformat(meta["issue_date"])
            exclusions.update((meta["region"], (start + timedelta(days=i)).isoformat())
                              for i in range(1, int(meta["horizon"]) + 1))
    records = []
    for path in train:
        bundle = CaseBundle.load(path)
        if bundle.case_id in excluded_ids:
            raise ValueError(f"case appears in train and excluded manifests: {bundle.case_id}")
        records.append(record_from_bundle(bundle))
    artifact = make_artifact(records, excluded_city_days=exclusions, provenance={
        "train_manifest": file_identity(train_manifest),
        "excluded_manifests": [file_identity(path) for path in exclude_manifests],
        "input_snapshot": case_bundle_snapshot(train, relative_to=REPO_ROOT),
        "builder": file_identity(Path(__file__)),
        "implementation": file_identity(REPO_ROOT / "src/sitian/analogs.py"),
        "quality_implementation": file_identity(REPO_ROOT / "src/sitian/asset_quality.py"),
        "synoptic_implementation": file_identity(REPO_ROOT / "src/sitian/analog_synoptic.py"),
        "data_contract_implementation": file_identity(REPO_ROOT / "src/sitian/data_contract.py"),
    })
    HistoricalCaseIndex(artifact)  # Validate the exact object that will be saved.
    if not artifact["records"]:
        raise ValueError("no historical cases remain after purging")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifests", type=Path, nargs="+", required=True,
                        help="all frozen validation/test/selection/holdout case manifests")
    parser.add_argument("--out", type=Path, required=True, help="new artifact path; existing files are refused")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; choose a new artifact path")
    artifact = build_index(args.train_manifest, args.exclude_manifests)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    print(json.dumps({**artifact["summary"], "out": str(args.out.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
