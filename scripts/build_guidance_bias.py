#!/usr/bin/env python3
"""构建 get_guidance_bias 使用的内部历史误差索引。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.guidance_bias import GUIDANCE_BIAS_VERSION, build_records  # noqa: E402
from sitian.provenance import case_bundle_snapshot, file_identity  # noqa: E402

DEFAULT_MANIFESTS = ["data/interim/train_without_winter_or_spatial_holdout.json"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifests", nargs="+",
        default=DEFAULT_MANIFESTS,
        help=(
            "history manifests; the main offline experiment defaults to the "
            "city-day-purged training pool only so earlier validation/test truth "
            "cannot update later evaluation episodes"
        ),
    )
    ap.add_argument("--publication-hour", type=int, default=12)
    ap.add_argument("--out", required=True, help="new artifact path; existing files are refused")
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    if out.exists():
        ap.error("output exists; choose a new artifact path")

    manifests = [REPO_ROOT / value for value in args.manifests]
    case_dirs = []
    for manifest in manifests:
        case_dirs.extend(json.loads(manifest.read_text(encoding="utf-8")))
    resolved = [REPO_ROOT / value for value in case_dirs]
    records = build_records(resolved, publication_hour=args.publication_hour)
    artifact = {
        "artifact_type": "guidance_bias_history",
        "guidance_bias_version": GUIDANCE_BIAS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "internal_only": True,
        "split_policy": {
            "main_experiment": "train_only_city_day_purged",
            "evaluation_truth_updates_index": False,
            "online_rolling_update_is_separate_experiment": True,
        },
        "verification_policy": {
            "source": "derived_conservative_default_no_per-record_publication_timestamp",
            "rule": f"target_date + 1 day at {args.publication_hour:02d}:00 Asia/Shanghai",
            "query_gate": "verification_available_at < issue_time (08:00 Asia/Shanghai)",
        },
        "provenance": {
            "manifests": [file_identity(path, relative_to=REPO_ROOT) for path in manifests],
            "input_snapshot": case_bundle_snapshot(resolved, relative_to=REPO_ROOT),
            "builder": file_identity(Path(__file__), relative_to=REPO_ROOT),
            "data_contract_implementation": file_identity(
                REPO_ROOT / "src/sitian/data_contract.py", relative_to=REPO_ROOT
            ),
            "implementation": file_identity(
                REPO_ROOT / "src/sitian/guidance_bias.py", relative_to=REPO_ROOT
            ),
        },
        "summary": {
            "case_paths": len(case_dirs),
            "records": len(records),
            "regions": len({record["region"] for record in records}),
            "sources": sorted({record["source"] for record in records}),
            "pollutants": sorted({record["pollutant"] for record in records}),
        },
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(json.dumps({**artifact["summary"], "out": str(out)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
