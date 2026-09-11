#!/usr/bin/env python3
"""全国个例构建器：episodes 清单 + 观测档案 + CAMS → cases/national/{split}/。

数据流（与 cases/README.md 契约一致，horizon=5）：
    observations  raw china_cities CSV：起报前小时值（六项基本污染物），≤ 起报日 07:00
    guidance      CAMS 前一日 12UTC（=北京时 20 时）起报：逐日 pm25 均值 / pm10 均值 / o3 日最大
                  （o3 用 3h 瞬时值日最大近似 O3_8h 日最大，note 注明）；北方 68 城另附 cmaq/naqp d02
    diagnostics   CAMS 同一起报的气象场 → 未来逐日（NWP 派生，合法指导）：
                  风速/风向/边界层/湿度/降水/最高温/云量
    truth         data/aq_daily.sqlite 六项日值（评分走全 AQI 口径）
单位换算：CAMS pm2p5/pm10 kg/m³→µg/m³(×1e9)；模式层 o3 质量混合比 kg/kg→µg/m³(×1.2e9，
近地面密度 1.2 kg/m³ 近似——指导允许有偏，评分只对真值负责)。

用法：python3 scripts/build_national_case.py --split train [--limit 20]
"""
from __future__ import annotations

import os

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from sitian.case import CaseBundle  # noqa: E402
from sitian.meteorology import wind_direction_label  # noqa: E402

RAW_OBS = REPO_ROOT / "data" / "raw" / "aq_obs" / "cities"
CAMS_DIR = REPO_ROOT / "data" / "raw" / "cams"
DB = REPO_ROOT / "data" / "aq_daily.sqlite"
COORDS = json.loads((REPO_ROOT / "data" / "interim" / "city_coords.json").read_text(encoding="utf-8"))
_CLIM_PATH = REPO_ROOT / "data" / "interim" / "climatology.json"
CLIMATOLOGY = json.loads(_CLIM_PATH.read_text(encoding="utf-8")) if _CLIM_PATH.exists() else {}
EAGET_ROOT = Path(os.environ.get("SITIAN_EAGET_ROOT", "/mnt/eaget"))
KETI = EAGET_ROOT / "keti_data"
HORIZON = 5
OBS_TYPES = {name: name for name in ("PM2.5", "PM10", "O3", "SO2", "NO2", "CO")}
OBS_UNITS = {name: ("mg/m³" if name == "CO" else "µg/m³") for name in OBS_TYPES}


# ---------------------------------------------------------------- observations
def read_obs_hours(city: str, issue: date) -> dict:
    """起报前 72h（含起报日 ≤07:00）城市小时值，给 08:00 起报留一小时发布余量。"""
    out: dict[str, dict] = {
        key: {"unit": OBS_UNITS[key], "times": [], "series": {city: []}}
        for key in OBS_TYPES
    }
    for offset in (-3, -2, -1, 0):
        d = issue + timedelta(days=offset)
        p = RAW_OBS / f"{d.year}" / f"china_cities_{d:%Y%m%d}.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                ci = header.index(city)
            except ValueError:
                continue
            for row in reader:
                if row[2] not in OBS_TYPES:
                    continue
                hour = int(row[1])
                if offset == 0 and hour > 7:
                    continue
                t = f"{d.isoformat()}T{hour:02d}:00"
                v = row[ci].strip() if ci < len(row) else ""
                key = OBS_TYPES[row[2]]
                out[key]["times"].append(t)
                out[key]["series"][city].append(float(v) if v else None)
    return {k: v for k, v in out.items() if v["times"]}


# ---------------------------------------------------------------- CAMS store
class CamsStore:
    """(issue_date, city) → 逐日 guidance/diagnostics。起报 = issue 前一日 12UTC。

    文件：{kind}_{d0}_{d1}.nc（日期区间分块，fetch_cams.py 产出）；目录扫描建 日期→文件 索引。
    """

    def __init__(self):
        import netCDF4
        self.netCDF4 = netCDF4
        self._cache: dict[Path, object] = {}
        self._index: dict[str, dict[str, Path]] = {"sfc": {}, "o3": {}}
        for p in CAMS_DIR.glob("*_*_*.nc"):
            parts = p.stem.split("_")
            if len(parts) != 3 or parts[0] not in self._index:
                continue
            try:
                d0, d1 = date.fromisoformat(parts[1]), date.fromisoformat(parts[2])
            except ValueError:
                continue
            d = d0
            while d <= d1:
                self._index[parts[0]][d.isoformat()] = p
                d += timedelta(days=1)

    def _open(self, kind: str, ref_day: str):
        p = self._index[kind].get(ref_day)
        if p is None:
            return None
        if p not in self._cache:
            self._cache[p] = self.netCDF4.Dataset(p)
            if len(self._cache) > 8:  # 控制同时打开的文件数
                old = next(iter(self._cache))
                if old != p:
                    self._cache.pop(old).close()
        return self._cache[p]

    @staticmethod
    def _pick(ds, *names):
        for n in names:
            if n in ds.variables:
                return ds.variables[n]
        raise KeyError(f"none of {names} in {list(ds.variables)}")

    def _series(self, ds, var, ref_utc: datetime, lat: float, lon: float) -> dict[int, float]:
        """该起报时次、最近格点的 {lead_hour: value}。"""
        lats = ds.variables["latitude"][:]
        lons = ds.variables["longitude"][:]
        li = int(np.abs(lats - lat).argmin())
        lj = int(np.abs(lons - lon % 360).argmin()) if lons.max() > 180 else int(np.abs(lons - lon).argmin())
        tvar = self._pick(ds, "forecast_reference_time", "time")
        times = self.netCDF4.num2date(tvar[:], tvar.units)
        ti = next((i for i, t in enumerate(times)
                   if (t.year, t.month, t.day, t.hour) ==
                   (ref_utc.year, ref_utc.month, ref_utc.day, ref_utc.hour)), None)
        if ti is None:
            raise KeyError(f"ref {ref_utc} not in file")
        pvar = self._pick(ds, "forecast_period", "leadtime_hour", "step")
        leads = np.asarray(pvar[:], dtype=float)
        if "seconds" in getattr(pvar, "units", ""):
            leads = leads / 3600.0
        # ADS netcdf 维度序为 (forecast_period, forecast_reference_time, [level,] lat, lon)
        arr = var[:, ti] if var.dimensions[0] == "forecast_period" else var[ti]
        if arr.ndim == 4:
            arr = arr[:, 0]
        vals = {}
        for k, lh in enumerate(leads):
            v = float(arr[k, li, lj])
            if not math.isnan(v):
                vals[int(lh)] = v
        return vals

    def day_leads(self, ref_utc: datetime, day: date) -> list[int]:
        """北京时自然日 day 覆盖的 lead hours（3h 步长）。"""
        out = []
        for lh in range(0, 121, 3):
            valid_bjt = ref_utc + timedelta(hours=lh + 8)
            if valid_bjt.date() == day:
                out.append(lh)
        return out

    def extract(self, city: str, issue: date) -> tuple[dict, dict]:
        """→ (guidance_source_dict, diagnostics_daily_dict)；数据缺失抛 KeyError。"""
        lat, lon = COORDS[city]
        ref_day = (issue - timedelta(days=1)).isoformat()
        ref_utc = datetime((issue - timedelta(days=1)).year, (issue - timedelta(days=1)).month,
                           (issue - timedelta(days=1)).day, 12)
        sfc = self._open("sfc", ref_day)
        o3f = self._open("o3", ref_day)
        if sfc is None:
            raise KeyError(f"cams sfc for ref {ref_day} missing")
        get = lambda *names: self._series(sfc, self._pick(sfc, *names), ref_utc, lat, lon)
        pm25 = get("pm2p5", "particulate_matter_2.5um")
        pm10 = get("pm10", "particulate_matter_10um")
        blh = get("blh", "boundary_layer_height")
        u10 = get("u10", "10m_u_component_of_wind")
        v10 = get("v10", "10m_v_component_of_wind")
        t2m = get("t2m", "2m_temperature")
        d2m = get("d2m", "2m_dewpoint_temperature")
        tp = get("tp", "total_precipitation")
        tcc = get("tcc", "total_cloud_cover")
        o3 = None
        if o3f is not None:
            try:
                o3 = self._series(o3f, self._pick(o3f, "go3", "o3", "ozone"), ref_utc, lat, lon)
            except KeyError:
                o3 = None

        days = [issue + timedelta(days=i + 1) for i in range(HORIZON)]
        g_pm25, g_pm10, g_o3, diag = {}, {}, {}, {}
        for di, day in enumerate(days):
            leads = [lh for lh in self.day_leads(ref_utc, day)]
            pv = [pm25[lh] * 1e9 for lh in leads if lh in pm25]
            if len(pv) < 2:
                # 前一日 12UTC 循环的 +120h = 业务起报 +108h，只盖到 issue+4；
                # 第 5 个目标日无 CAMS 指导，属业务现实（agent 结合 NWP 外推）。
                if di < 3:
                    raise KeyError(f"no cams leads for {day}")
                continue
            k = day.isoformat()
            g_pm25[k] = round(float(np.mean(pv)), 1)
            g_pm10[k] = round(float(np.mean([pm10[lh] * 1e9 for lh in leads if lh in pm10])), 1)
            if o3:
                ov = [o3[lh] * 1.2e9 for lh in leads if lh in o3]
                if ov:
                    g_o3[k] = round(float(np.max(ov)), 1)
            us = [u10[lh] for lh in leads if lh in u10]
            vs = [v10[lh] for lh in leads if lh in v10]
            um, vm = float(np.mean(us)), float(np.mean(vs))
            spd = math.hypot(um, vm)
            wdir = (math.degrees(math.atan2(-um, -vm)) + 360) % 360
            ts = [t2m[lh] for lh in leads if lh in t2m]
            dsr = [d2m[lh] for lh in leads if lh in d2m]
            rh = 100 * math.exp(17.625 * (np.mean(dsr) - 273.15) / (243.04 + np.mean(dsr) - 273.15)) \
                / math.exp(17.625 * (np.mean(ts) - 273.15) / (243.04 + np.mean(ts) - 273.15))
            tps = sorted((lh, tp[lh]) for lh in leads if lh in tp)
            rain = max(0.0, (tps[-1][1] - tps[0][1]) * 1000) if len(tps) >= 2 else 0.0
            diag[k] = {
                "wind_speed_ms": round(float(np.hypot(np.array(us), np.array(vs)).mean()), 1),
                "wind_dir_deg": round(wdir, 0),
                "wind_dir": wind_direction_label(round(wdir, 0)),
                "blh_max_m": round(max(blh[lh] for lh in leads if lh in blh), 0),
                "rh_pct": round(min(100.0, rh), 1),
                "rain_mm": round(rain, 1),
                "tmax_c": round(max(ts) - 273.15, 1),
                "cloud_pct": round(float(np.mean([tcc[lh] for lh in leads if lh in tcc])) * 100, 0),
                "source": "cams_12utc_prevday",
            }
        guidance = {"cams": {"daily_pm25": g_pm25, "daily_pm10": g_pm10,
                             **({"daily_o3max": g_o3} if g_o3 else {}),
                             "note": "CAMS 0.4° 最近格点，前一日20时(12UTC)起报；pm 为日均，o3max 为 3h 瞬时日最大"
                                     "（近似 O3_8h 峰值），存在系统偏差需自行订正；cycle lead 0--120h 对应业务相对 -12--+108h，最后一天无指导需自行外推"}}
        return guidance, diag


# ---------------------------------------------------------------- keti (北方 68 城加一路)
def keti_guidance(city: str, issue: date) -> dict:
    out = {}
    cn = city + ("" if city.endswith(("市", "州", "区")) else "市")
    pre = f"{issue - timedelta(days=1):%Y%m%d}20"
    for model in ("cmaq", "naqp"):
        p = KETI / model / "day" / f"{model}_d02_{pre}.csv"
        if not p.exists():
            continue
        acc = defaultdict(list)
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("cityname") == cn and r.get("pm25_24h", "").strip():
                    try:
                        acc[r["datadate"]].append(float(r["pm25_24h"]))
                    except ValueError:
                        pass
        if acc:
            out[model] = {"daily_pm25": {d: round(sum(v) / len(v), 1) for d, v in sorted(acc.items())},
                          "note": f"{model} d02 前一日20时起报，站点城市均值"}
    return out


# ---------------------------------------------------------------- truth
def read_truth(db, city: str, issue: date) -> dict | None:
    daily = {}
    for i in range(HORIZON):
        d = (issue + timedelta(days=i + 1)).isoformat()
        row = db.execute("SELECT pm25, pm10, o3_8h, so2, no2, co, n_hours FROM daily "
                         "WHERE city=? AND date=?", (city, d)).fetchone()
        # 全国多污染物任务要求六项都有效；缺一项就不能声称等级/首要污染物
        # 是“全 AQI 真值”。build_daily_all 已按各污染物的数据有效性门槛置空。
        if row is None or any(v is None for v in row[:6]) or (row[6] or 0) < 20:
            return None
        rec = {"pm25_avg": row[0]}
        for k, v in zip(("pm10_avg", "o3_8h", "so2_avg", "no2_avg", "co_avg"), row[1:6]):
            rec[k] = v
        daily[d] = rec
    return {"daily": daily}


# ---------------------------------------------------------------- main
def build_one(store: CamsStore, db, ep: dict, split: str) -> str:
    city, issue_s = ep["city"], ep["issue_date"]
    if (REPO_ROOT / "cases" / "national" / split / f"{city}_{issue_s}" / "truth.json").exists():
        return "exists"
    issue = date.fromisoformat(issue_s)
    guidance, diag = store.extract(city, issue)
    guidance.update(keti_guidance(city, issue))
    obs = read_obs_hours(city, issue)
    if "PM2.5" not in obs or len(obs["PM2.5"]["times"]) < 48:
        return "obs_insufficient"
    truth = read_truth(db, city, issue)
    if truth is None:
        return "truth_incomplete"
    bundle = CaseBundle(
        case_id=f"{city}_{issue_s}", issue_date=issue_s, region=city,
        horizon=HORIZON, regions=[city],
        observations=obs, diagnostics={"daily": diag},
        guidance={"sources": guidance}, truth=truth,
        meta={"real": True, "national": True, "split": split,
              "stratum": ep["stratum"], "cluster": ep["cluster"],
              "obs_source": "cnemc_public_archive", "guidance_source": "cams+keti",
              "multi_pollutant": True,
              # 气候态分位表（2022..2025-03 窗前数据，公开背景知识，无泄漏）
              **({"climatology": CLIMATOLOGY[city]} if city in CLIMATOLOGY else {})},
    )
    bundle.save(REPO_ROOT / "cases" / "national" / split / f"{city}_{issue_s}")
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    eps = json.loads((REPO_ROOT / "data" / "interim" / f"episodes_{args.split}.json").read_text(encoding="utf-8"))
    if args.limit:
        eps = eps[: args.limit]
    store = CamsStore()
    db = sqlite3.connect(DB)
    stats = defaultdict(int)
    for i, ep in enumerate(eps):
        try:
            stats[build_one(store, db, ep, args.split)] += 1
        except KeyError as e:
            stats[f"cams_missing"] += 1
            if stats["cams_missing"] <= 3:
                print(f"  cams miss: {ep['city']} {ep['issue_date']}: {e}")
        if i % 200 == 199:
            print(i + 1, dict(stats))
    print("done:", dict(stats))
    manifest = {"split": args.split, "stats": dict(stats), "n_episodes": len(eps)}
    out = REPO_ROOT / "cases" / "national" / args.split / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
