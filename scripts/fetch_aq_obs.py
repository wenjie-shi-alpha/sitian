#!/usr/bin/env python3
"""公开空气质量观测档案：下载 + 入库（obs/truth 的数据源，纯标准库）。

数据源：quotsoft.net/air 镜像的全国城市级小时值（中国环境监测总站发布数据的存档），
文件按天打包：china_cities_YYYYMMDD.csv，列 = date,hour,type,北京,天津,...
type 含 AQI/PM2.5/PM2.5_24h/PM10/SO2/NO2/O3/O3_8h/CO 等。

用法（在仓库根目录）：
    # 1) 下载（断点续传，重复运行只补缺）
    python3 sitian/scripts/fetch_aq_obs.py download \
        --start 2021-12-01 --end 2026-01-31
    # 2) 入库到 SQLite（筛选京津冀及周边城市）
    python3 sitian/scripts/fetch_aq_obs.py ingest
    # 3) 快速查询验证
    python3 sitian/scripts/fetch_aq_obs.py query --city 北京 --date 2025-12-18

目录约定（相对仓库根）：
    data/raw/aq_obs/cities/YYYY/china_cities_YYYYMMDD.csv   原始文件（全国）
    data/aq_obs.sqlite                            筛选后的库
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE_URL = "https://quotsoft.net/air/data"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "aq_obs" / "cities"
DB_PATH = REPO_ROOT / "data" / "aq_obs.sqlite"

# 京津冀 13 市 + 大同 + 山东半岛西北部 + 豫北南部通道（与 CMAQ d03 域对齐并略放宽）
DEFAULT_CITIES = [
    "北京", "天津", "石家庄", "唐山", "秦皇岛", "邯郸", "保定", "张家口",
    "承德", "廊坊", "沧州", "衡水", "邢台",
    "大同", "济南", "德州", "聊城", "淄博", "泰安", "滨州", "东营", "潍坊",
    "安阳", "新乡", "鹤壁", "焦作", "濮阳",
]
KEEP_TYPES = {"AQI", "PM2.5", "PM2.5_24h", "PM10", "PM10_24h",
              "SO2", "NO2", "O3", "O3_8h", "CO"}


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def raw_path(d: date) -> Path:
    return RAW_DIR / f"{d.year}" / f"china_cities_{d:%Y%m%d}.csv"


def looks_valid(path: Path) -> bool:
    try:
        if path.stat().st_size < 10_000:
            return False
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readline().startswith("date,hour,type")
    except OSError:
        return False


def fetch_one(d: date, timeout: float = 120.0, retries: int = 3) -> str:
    """下载一天。返回 'ok' | 'skip' | 'missing' | 'fail'。"""
    dst = raw_path(d)
    if looks_valid(dst):
        return "skip"
    url = f"{BASE_URL}/china_cities_{d:%Y%m%d}.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "jjj-atmo-harness/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(dst)
            if looks_valid(dst):
                return "ok"
            dst.unlink(missing_ok=True)
            return "fail"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return "missing"
            time.sleep(2 * attempt)
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2 * attempt)
    return "fail"


def cmd_download(args) -> int:
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    days = list(daterange(start, end))
    print(f"目标 {len(days)} 天：{start} → {end}，并发 {args.workers}")
    counts = {"ok": 0, "skip": 0, "missing": 0, "fail": 0}
    failures: list[str] = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, d): d for d in days}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            d = futs[fut]
            status = fut.result()
            counts[status] += 1
            if status in ("fail", "missing"):
                failures.append(f"{d} {status}")
            if i % 100 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(days)}  ok={counts['ok']} skip={counts['skip']} "
                      f"missing={counts['missing']} fail={counts['fail']}  ({el:.0f}s)", flush=True)
    print(f"完成：{counts}（{time.time() - t0:.0f}s）")
    if failures:
        flog = RAW_DIR / "failures.txt"
        flog.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"{len(failures)} 个未成功，清单：{flog}（重跑本命令即断点续传）")
    return 0 if counts["fail"] == 0 else 1


def cmd_ingest(args) -> int:
    cities = args.cities.split(",") if args.cities else DEFAULT_CITIES
    files = sorted(RAW_DIR.glob("*/china_cities_*.csv"))
    if not files:
        print("没有原始文件，先运行 download", file=sys.stderr)
        return 1
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS hourly(
        date TEXT NOT NULL, hour INTEGER NOT NULL, type TEXT NOT NULL,
        city TEXT NOT NULL, value REAL,
        PRIMARY KEY (date, hour, type, city)) WITHOUT ROWID""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_city_type ON hourly(city, type, date)")
    done = {r[0] for r in con.execute(
        "SELECT DISTINCT date FROM hourly")} if not args.rebuild else set()
    if args.rebuild:
        con.execute("DELETE FROM hourly")
    n_files = n_rows = 0
    t0 = time.time()
    for path in files:
        day = path.stem.split("_")[-1]
        iso = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
        if iso in done:
            continue
        rows = []
        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            city_idx = [(i, c) for i, c in enumerate(header[3:], start=3) if c in cities]
            for rec in reader:
                if len(rec) < 4 or rec[2] not in KEEP_TYPES:
                    continue
                for i, c in city_idx:
                    v = rec[i].strip() if i < len(rec) else ""
                    if v:
                        try:
                            rows.append((iso, int(rec[1]), rec[2], c, float(v)))
                        except ValueError:
                            pass
        con.executemany("INSERT OR REPLACE INTO hourly VALUES (?,?,?,?,?)", rows)
        con.commit()
        n_files += 1
        n_rows += len(rows)
        if n_files % 100 == 0:
            print(f"  已入库 {n_files} 天 / {n_rows} 行（{time.time() - t0:.0f}s）", flush=True)
    total = con.execute("SELECT COUNT(*), COUNT(DISTINCT date) FROM hourly").fetchone()
    print(f"入库完成：本次 {n_files} 天 / {n_rows} 行；库中共 {total[0]} 行、{total[1]} 天 → {DB_PATH}")
    con.close()
    return 0


def get_daily_pm25(con: sqlite3.Connection, city: str, day: str) -> float | None:
    """由小时值计算日均 PM2.5（≥20 个有效小时才计）。个例构建器复用此函数。"""
    row = con.execute(
        "SELECT AVG(value), COUNT(*) FROM hourly WHERE city=? AND type='PM2.5' AND date=?",
        (city, day)).fetchone()
    if row and row[1] and row[1] >= 20:
        return round(row[0], 1)
    return None


def cmd_query(args) -> int:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT hour, value FROM hourly WHERE city=? AND type=? AND date=? ORDER BY hour",
        (args.city, args.type, args.date)).fetchall()
    print(f"{args.city} {args.date} {args.type} 小时值（{len(rows)} 条）:")
    print("  " + " ".join(f"{h}:{v:g}" for h, v in rows))
    print(f"  日均PM2.5: {get_daily_pm25(con, args.city, args.date)}")
    con.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="公开空气质量观测档案下载与入库")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--start", default="2021-12-01")
    d.add_argument("--end", default="2026-01-31")
    d.add_argument("--workers", type=int, default=3)
    d.set_defaults(fn=cmd_download)
    i = sub.add_parser("ingest")
    i.add_argument("--cities", default=None, help="逗号分隔，默认京津冀及周边26市")
    i.add_argument("--rebuild", action="store_true")
    i.set_defaults(fn=cmd_ingest)
    q = sub.add_parser("query")
    q.add_argument("--city", default="北京")
    q.add_argument("--date", required=True)
    q.add_argument("--type", default="PM2.5")
    q.set_defaults(fn=cmd_query)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
