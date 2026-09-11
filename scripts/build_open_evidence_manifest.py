#!/usr/bin/env python3
"""Freeze the 401-date open-evidence download and time-gate manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sitian.open_evidence import DEFAULT_EVIDENCE_ROOT, build_download_manifest, issue_dates_from_cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("cases/national"))
    parser.add_argument("--out", type=Path,
                        default=DEFAULT_EVIDENCE_ROOT / "manifests" / "open_evidence_v1.json")
    args = parser.parse_args()
    dates = issue_dates_from_cases(args.cases)
    if not dates:
        raise SystemExit(f"no issue dates found under {args.cases}")
    manifest = build_download_manifest(dates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "issue_dates": len(dates),
        "first": dates[0],
        "last": dates[-1],
        "nwp_records": {name: len(rows) for name, rows in manifest["nwp"].items()},
        "nwp_radiation_records": {
            name: len(rows)
            for name, rows in manifest["nwp_radiation"]["records"].items()
        },
        "cams_cycles": len(manifest["cams"]["cycles"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
