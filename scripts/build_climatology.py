#!/usr/bin/env python3
"""城市逐月气候态分位表：get_assessment 的气候背景层。

只用评测窗口之前的年份（2022-01-01 .. 2025-03-31）——气候态是公开背景知识，
且严格先于全部训练/评测起报日，无泄漏疑虑。
产出：data/interim/climatology.json  {city: {"MM": {"pm25": {p50,p75,p90}, "o3_8h": {p50,p90}}}}
用法：python3 scripts/build_climatology.py
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "data" / "aq_daily.sqlite"
OUT = REPO_ROOT / "data" / "interim" / "climatology.json"
END = "2025-03-31"  # 评测窗口起点之前


def main() -> int:
    db = sqlite3.connect(DB)
    acc: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {"pm25": [], "o3_8h": []}))
    for city, d, pm25, o3 in db.execute(
            "SELECT city, date, pm25, o3_8h FROM daily WHERE date <= ?", (END,)):
        m = d[5:7]
        if pm25 is not None:
            acc[city][m]["pm25"].append(pm25)
        if o3 is not None:
            acc[city][m]["o3_8h"].append(o3)
    out = {}
    for city, months in acc.items():
        table = {}
        for m, series in months.items():
            rec = {}
            if len(series["pm25"]) >= 30:
                q = np.percentile(series["pm25"], [50, 75, 90])
                rec["pm25"] = {"p50": round(q[0], 1), "p75": round(q[1], 1), "p90": round(q[2], 1)}
            if len(series["o3_8h"]) >= 30:
                q = np.percentile(series["o3_8h"], [50, 90])
                rec["o3_8h"] = {"p50": round(q[0], 1), "p90": round(q[1], 1)}
            if rec:
                table[m] = rec
        if len(table) == 12:
            out[city] = table
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} cities x 12 months -> {OUT}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
