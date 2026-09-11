#!/usr/bin/env python3
"""全国城市日值库：raw china_cities CSV → data/aq_daily.sqlite。

日尺度口径（GB 3095 / HJ 633）：
    pm25/pm10/so2/no2  当日小时值算术均值（µg/m³）
    co                 当日小时值算术均值（mg/m³）
    o3_8h              当日 O3_8h 滑动均值的最大值（取档案 O3_8h 类型的日最大）
    n_hours            当日 PM2.5 有效小时数（数据完备性审计用；日均有效性要求 ≥20h）

用途：城市选择的气候态特征、抽样框真值标签、全国个例 truth 回填。纯标准库。
用法：python3 scripts/build_daily_all.py [--rebuild]
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "aq_obs" / "cities"
DB_PATH = REPO_ROOT / "data" / "aq_daily.sqlite"

# 档案 type → (输出列, 聚合方式)
TYPES = {
    "PM2.5": ("pm25", "mean"),
    "PM10": ("pm10", "mean"),
    "SO2": ("so2", "mean"),
    "NO2": ("no2", "mean"),
    "CO": ("co", "mean"),
    "O3_8h": ("o3_8h", "max"),
}
# 日均项目严格要求至少 20 个小时值；O3 日最大 8 小时滑动平均至少需要
# 14 个有效 8 小时平均值。旧实现统一用 6，会把缺测日当作完整六项 AQI 真值。
MIN_VALID_COUNTS = {
    "pm25": 20,
    "pm10": 20,
    "so2": 20,
    "no2": 20,
    "co": 20,
    "o3_8h": 14,
}
O3_DAILY_LIMIT = 160.0  # GB 3095 二级日最大8h限值；超限时不足14个仍有效


def process_file(path: Path) -> list[tuple]:
    day = path.stem.split("_")[-1]
    date = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        cities = header[3:]
        for row in reader:
            t = row[2]
            if t not in TYPES:
                continue
            col, _ = TYPES[t]
            for city, v in zip(cities, row[3:]):
                if v.strip():
                    try:
                        acc[city][col].append(float(v))
                    except ValueError:
                        pass
    out = []
    for city, series in acc.items():
        rec = {}
        for col, agg in TYPES.values():
            vals = series.get(col, [])
            valid = len(vals) >= MIN_VALID_COUNTS[col]
            if col == "o3_8h" and vals and max(vals) > O3_DAILY_LIMIT:
                valid = True  # HJ 663—2026 6.1.2 的超限例外
            if valid:
                rec[col] = round(max(vals) if agg == "max" else sum(vals) / len(vals), 1)
        if "pm25" in rec:
            out.append((city, date, rec.get("pm25"), rec.get("pm10"), rec.get("o3_8h"),
                        rec.get("so2"), rec.get("no2"), rec.get("co"),
                        len(series.get("pm25", []))))
    return out


def main() -> int:
    rebuild = "--rebuild" in sys.argv
    if rebuild and DB_PATH.exists():
        DB_PATH.unlink()
    db = sqlite3.connect(DB_PATH)
    db.execute("""CREATE TABLE IF NOT EXISTS daily (
        city TEXT, date TEXT, pm25 REAL, pm10 REAL, o3_8h REAL,
        so2 REAL, no2 REAL, co REAL, n_hours INTEGER,
        PRIMARY KEY (city, date))""")
    done = {r[0] for r in db.execute("SELECT DISTINCT date FROM daily")}
    files = sorted(RAW_DIR.glob("*/china_cities_*.csv"))
    todo = [p for p in files
            if f"{p.stem[-8:-4]}-{p.stem[-4:-2]}-{p.stem[-2:]}" not in done]
    print(f"files={len(files)} todo={len(todo)}")
    for i, p in enumerate(todo):
        rows = process_file(p)
        db.executemany("INSERT OR REPLACE INTO daily VALUES (?,?,?,?,?,?,?,?,?)", rows)
        if i % 100 == 0:
            db.commit()
            print(f"{i}/{len(todo)} {p.name} rows={len(rows)}")
    db.commit()
    n, cities, days = db.execute(
        "SELECT COUNT(*), COUNT(DISTINCT city), COUNT(DISTINCT date) FROM daily").fetchone()
    print(f"done: rows={n} cities={cities} days={days}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
