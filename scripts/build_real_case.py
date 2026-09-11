#!/usr/bin/env python3
"""真实个例构建器：观测库 + eaget keti_data → cases/real/ 的 case bundle。

数据源：
    observations  data/aq_obs.sqlite（公开档案入库，fetch_aq_obs.py）
    guidance      /mnt/eaget/keti_data/{cmaq,naqp}/day/*_d03_{前一日}20.csv（站点→城市均值）
    diagnostics   /mnt/eaget/keti_data/wrf/day/wrf_d03_{前一日}20.csv（气象站→城市日值）
    truth         观测库逐日均值回填（历史个例的"未来"已发生）
    expert        不在本脚本范围（12 月基准个例由语料蒸馏后手工/半自动写入）

用法（仓库根目录，硬盘已挂载）：
    python3 sitian/scripts/build_real_case.py --issue-date 2025-12-19 --horizon 5
"""
from __future__ import annotations

import os

import argparse
import csv
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.case import CaseBundle  # noqa: E402

DB_PATH = REPO_ROOT / "data" / "aq_obs.sqlite"
EAGET_ROOT = Path(os.environ.get("SITIAN_EAGET_ROOT", "/mnt/eaget"))
KETI = EAGET_ROOT / "keti_data"
OBS_CITIES = ["北京", "天津", "石家庄", "唐山", "保定", "廊坊", "沧州", "衡水", "邢台", "邯郸"]
CITY_KEY = {"北京": "beijing", "天津": "tianjin", "石家庄": "shijiazhuang", "唐山": "tangshan",
            "保定": "baoding", "廊坊": "langfang", "沧州": "cangzhou", "衡水": "hengshui",
            "邢台": "xingtai", "邯郸": "handan"}
GUIDANCE_CITY = "北京市"   # keti CSV 里的城市名带"市"


def read_keti_csv(model: str, issue: date) -> list[dict]:
    """读取 前一日20时 起报的 d03 日文件（会商晨 8 时能拿到的最新一报）。"""
    pretime = f"{issue - timedelta(days=1):%Y%m%d}20"
    path = KETI / model / "day" / f"{model}_d03_{pretime}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def city_daily_mean(rows: list[dict], field: str) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("cityname") == GUIDANCE_CITY and r.get(field, "").strip():
            try:
                acc[r["datadate"]].append(float(r[field]))
            except ValueError:
                pass
    return {d: round(sum(v) / len(v), 1) for d, v in sorted(acc.items())}


def wind_dir_mean(rows_by_date: dict[str, list[tuple[float, float]]]) -> dict[str, float]:
    out = {}
    for d, pairs in rows_by_date.items():
        x = sum(s * math.sin(math.radians(deg)) for s, deg in pairs)
        y = sum(s * math.cos(math.radians(deg)) for s, deg in pairs)
        out[d] = round(math.degrees(math.atan2(x, y)) % 360, 0)
    return out


def dir_label(deg: float) -> str:
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int(((deg + 22.5) % 360) // 45)]


def build(issue_date: str, horizon: int, region: str = "beijing",
          out_root: Path | None = None) -> Path:
    issue = date.fromisoformat(issue_date)
    con = sqlite3.connect(DB_PATH)

    # ---- observations：过去 48+7h 小时值（issue-2 00时 → issue 07时）----
    start_day = (issue - timedelta(days=2)).isoformat()
    observations: dict = {}
    for pollutant in ("PM2.5", "PM10"):
        times: list[str] = []
        series: dict[str, list] = {CITY_KEY[c]: [] for c in OBS_CITIES}
        grid: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
        for row in con.execute(
            "SELECT date, hour, city, value FROM hourly WHERE type=? AND date>=? AND date<=?",
            (pollutant, start_day, issue.isoformat())):
            d, h, c, v = row
            if d == issue.isoformat() and h > 7:
                continue
            grid[(d, h)][c] = v
        for (d, h) in sorted(grid):
            times.append(f"{d}T{h:02d}:00")
            for c in OBS_CITIES:
                series[CITY_KEY[c]].append(grid[(d, h)].get(c))
        observations[pollutant] = {"times": times, "series": series}
    n_hours = len(observations["PM2.5"]["times"])
    if n_hours < 40:
        raise RuntimeError(f"观测小时数不足: {n_hours}（该时段可能未入库）")

    # ---- guidance：CMAQ 与 NAQP（北京站点均值 → 逐日城市指导）----
    sources = {}
    for model in ("cmaq", "naqp"):
        rows = read_keti_csv(model, issue)
        sources[model] = {
            "daily_pm25": city_daily_mean(rows, "pm25_24h"),
            "daily_aqi": city_daily_mean(rows, "aqi"),
            "note": f"{model} d03 北京国控站均值，{issue - timedelta(days=1)} 20时起报",
        }

    # ---- diagnostics：WRF 城市日值（风速/风向/温/湿/降水）----
    wrf = read_keti_csv("wrf", issue)
    ws = city_daily_mean(wrf, "windspeed_24h")
    tem = city_daily_mean(wrf, "tem_24h")
    hum = city_daily_mean(wrf, "hum_24h")
    rain = city_daily_mean(wrf, "rain_24h")
    wd_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in wrf:
        if r.get("cityname") == GUIDANCE_CITY:
            try:
                wd_pairs[r["datadate"]].append(
                    (float(r["windspeed_24h"]), float(r["winddirect_24h"])))
            except (ValueError, KeyError):
                pass
    wd = wind_dir_mean(wd_pairs)
    diag_daily = {}
    for d in sorted(ws):
        deg = wd.get(d, 0.0)
        diag_daily[d] = {
            "wind_speed_ms": ws[d],
            "wind_dir_deg": deg,
            "wind_dir": dir_label(deg),
            "tem_c": tem.get(d),
            "rh_pct": hum.get(d),
            "rain_mm": rain.get(d),
            "source": "wrf_d03",
        }

    # ---- truth：观测库逐日均值回填 ----
    truth_daily = {}
    missing = []
    for i in range(horizon):
        d = (issue + timedelta(days=i + 1)).isoformat()
        row = con.execute(
            "SELECT AVG(value), COUNT(*) FROM hourly WHERE city='北京' AND type='PM2.5' AND date=?",
            (d,)).fetchone()
        if row and row[1] and row[1] >= 20:
            truth_daily[d] = {"pm25_avg": round(row[0], 1)}
        else:
            missing.append(d)
    con.close()

    # 图像时次目录：优先前一日12时（会商晨必然可用），其次当日00时
    image_cycle_dir = None
    for cand in (f"{issue - timedelta(days=1):%Y%m%d}1200", f"{issue:%Y%m%d}0000"):
        p = EAGET_ROOT / cand
        if p.is_dir() and any(p.iterdir()):
            image_cycle_dir = str(p)
            break

    bundle = CaseBundle(
        case_id=f"{region}_{issue_date}",
        issue_date=issue_date,
        region=region,
        horizon=horizon,
        regions=sorted(CITY_KEY.values()),
        meta={
            "real": True,
            "obs_source": "cnemc_public_archive",
            "guidance_source": "eaget_keti_data",
            "obs_hours": n_hours,
            "truth_missing_days": missing,
            "image_cycle_dir": image_cycle_dir,
        },
        observations=observations,
        diagnostics={"daily": diag_daily},
        guidance={"sources": sources},
        truth={"daily": truth_daily} if not missing else None,
    )
    violations = bundle.audit_time_gate()
    if violations:
        raise RuntimeError(f"时间门禁违规: {violations[:3]}")
    out_root = out_root or (REPO_ROOT / "cases" / "real")
    path = bundle.save(out_root / bundle.case_id)
    print(f"已构建 {path}  obs_hours={n_hours} guidance={list(sources)} "
          f"diag_days={len(diag_daily)} truth_days={len(truth_daily)}"
          + (f" 缺真值:{missing}" if missing else ""))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue-date", required=True)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--region", default="beijing")
    args = ap.parse_args(argv)
    build(args.issue_date, args.horizon, args.region)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
