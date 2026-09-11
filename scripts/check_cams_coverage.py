#!/usr/bin/env python3
"""CAMS 分块覆盖率校验：逐日检查 sfc/o3 是否齐备，列出缺口起报日。

个例需要的是"起报日前一日"的 CAMS 场，故按 ref_day 口径检查。
用法：python3 scripts/check_cams_coverage.py [--start 2025-03-31 --end 2026-08-20]
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

CAMS = Path(__file__).resolve().parents[1] / "data" / "raw" / "cams"


def covered(kind: str) -> set[str]:
    days = set()
    for p in CAMS.glob(f"{kind}_*_*.nc"):
        parts = p.stem.split("_")
        if len(parts) != 3:
            continue
        try:
            d0, d1 = date.fromisoformat(parts[1]), date.fromisoformat(parts[2])
        except ValueError:
            continue
        d = d0
        while d <= d1:
            days.add(d.isoformat())
            d += timedelta(days=1)
    return days


def runs(days: list[str]) -> list[str]:
    """连续日期段压缩显示。"""
    out, start, prev = [], None, None
    for d in days:
        cur = date.fromisoformat(d)
        if start is None:
            start = prev = cur
        elif (cur - prev).days == 1:
            prev = cur
        else:
            out.append(f"{start}..{prev}" if start != prev else f"{start}")
            start = prev = cur
    if start is not None:
        out.append(f"{start}..{prev}" if start != prev else f"{start}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-03-31")
    ap.add_argument("--end", default="2026-08-20")
    a = ap.parse_args()
    want = []
    d, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    while d <= end:
        want.append(d.isoformat())
        d += timedelta(days=1)
    for kind in ("sfc", "o3"):
        have = covered(kind)
        miss = [x for x in want if x not in have]
        print(f"{kind}: {len(want)-len(miss)}/{len(want)} ref-days covered"
              f"{'' if not miss else '  MISSING: ' + ', '.join(runs(miss))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
