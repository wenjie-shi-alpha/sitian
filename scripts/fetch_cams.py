#!/usr/bin/env python3
"""CAMS 全球成分预报下载：中国框、服务端裁剪、guidance+diagnostics 一站式。

数据集 cams-global-atmospheric-composition-forecasts（ADS），12UTC 起报
（= 北京时 20 时，与 keti"前一日 20 时起报"口径对齐），5 天时效，3h 步长。

ADS 有单请求成本上限（整月×10变量×41步会被 403 拒绝），故分块：
    sfc（10 个单层变量）按 SFC_CHUNK=8 天一块；o3（模式层 137 单变量）按月一块。
文件名 {kind}_{d0}_{d1}.nc（日期区间闭区间），下游按目录扫描索引。
断点续传：目标文件存在且非空即跳过；被 403 拒绝时自动对半减小块重试。

用法：
    python3 scripts/fetch_cams.py pilot                      # 试点：1 天，验证 key/变量
    python3 scripts/fetch_cams.py range 2025-04-01 2026-08-26
凭证：~/.cdsapirc（url 指向 ads.atmosphere.copernicus.eu）。输出：data/raw/cams/
"""
from __future__ import annotations

import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import cdsapi

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "cams"
DATASET = "cams-global-atmospheric-composition-forecasts"
AREA = [55, 70, 15, 140]  # N, W, S, E：中国框
# sfc 用 6h 步长（日均聚合足够，磁带调取量减半）；o3 保 3h（日最大需要分辨日循环峰值）
SFC_LEADS = [str(h) for h in range(0, 121, 6)]
O3_LEADS = [str(h) for h in range(0, 121, 3)]
SFC_CHUNK = 8
O3_CHUNK = 16

SFC_VARS = [
    "particulate_matter_2.5um", "particulate_matter_10um",
    "boundary_layer_height", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_temperature", "2m_dewpoint_temperature",
    "total_precipitation", "total_cloud_cover",
]


def _request(client, kind: str, d0: date, d1: date) -> None:
    target = OUT_DIR / f"{kind}_{d0}_{d1}.nc"
    if target.exists() and target.stat().st_size > 0:
        print(f"skip {target.name}")
        return
    req = {
        "date": [f"{d0}/{d1}"],
        "time": ["12:00"],
        "type": ["forecast"],
        "leadtime_hour": SFC_LEADS if kind == "sfc" else O3_LEADS,
        "area": AREA,
        "data_format": "netcdf_zip",
    }
    if kind == "sfc":
        req["variable"] = SFC_VARS
    else:
        req["variable"] = ["ozone"]
        req["model_level"] = ["137"]
    tmp = target.with_suffix(".zip")
    for attempt in range(4):
        try:
            client.retrieve(DATASET, req, str(tmp))
            break
        except Exception as e:
            msg = str(e)
            if "too large" in msg or "cost limits" in msg:
                if d0 == d1:
                    raise
                mid = d0 + (d1 - d0) // 2
                print(f"too large, split {d0}..{d1} at {mid}")
                _request(client, kind, d0, mid)
                _request(client, kind, mid + timedelta(days=1), d1)
                return
            # MARS/磁带归档偶发错误：退避重试（首次失败常已把数据暂存进磁盘缓存）
            if attempt < 3:
                wait = 90 * (attempt + 1)
                print(f"retry {target.name} attempt {attempt + 2} after {wait}s: {msg[:120]}")
                time.sleep(wait)
            else:
                raise
    with zipfile.ZipFile(tmp) as z:
        names = [n for n in z.namelist() if n.endswith(".nc")]
        assert len(names) == 1, names
        target.write_bytes(z.read(names[0]))
    tmp.unlink()
    print(f"done {target.name} ({target.stat().st_size/1e6:.1f} MB)")


def fetch_range(client, start: date, end: date, stride: int = 1, offset: int = 0) -> None:
    """stride/offset 把块清单按取模切给并行工人（ADS 同用户多请求可并发排队）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []
    for kind, chunk in (("sfc", SFC_CHUNK), ("o3", O3_CHUNK)):
        d = start
        while d <= end:
            d1 = min(end, d + timedelta(days=chunk - 1))
            chunks.append((kind, d, d1))
            d = d1 + timedelta(days=1)
    # sfc/o3 交错分配，避免所有 o3 都压在同一批工人尾部
    chunks = chunks[offset::stride]
    failed = []
    for kind, d, d1 in chunks:
        try:
            _request(client, kind, d, d1)
        except Exception as e:
            print(f"FAILED {kind} {d}..{d1}: {str(e)[:150]}")
            failed.append((kind, str(d), str(d1)))
    if failed:
        print(f"WORKER {offset} DONE WITH {len(failed)} FAILED CHUNKS: {failed}")
    else:
        print(f"WORKER {offset} DONE, no failures")


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    client = cdsapi.Client()
    if cmd == "pilot":
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _request(client, "sfc", date(2025, 12, 18), date(2025, 12, 18))
        _request(client, "o3", date(2025, 12, 18), date(2025, 12, 18))
    elif cmd == "range":
        stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        fetch_range(client, date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3]),
                    stride, offset)
    else:
        raise SystemExit(f"unknown cmd {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
