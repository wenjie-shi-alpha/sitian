#!/usr/bin/env python3
"""EAGET 硬盘个例可构建性审计：挂载后一键运行，量出能建多少真实个例。

用法（硬盘挂载到 /mnt/eaget 后）：
    python3 sitian/scripts/audit_eaget.py                # 全量审计
    python3 sitian/scripts/audit_eaget.py --quick        # 只查挂载与非空时次数
    python3 sitian/scripts/audit_eaget.py --sample 200   # 均匀抽样 200 个时次
    python3 sitian/scripts/audit_eaget.py --root /mnt/e/  # 非默认挂载点

输出：
    - 控制台：逐月覆盖表、完整度分类、keti_data 时间范围、可构建个例估算
    - JSON 报告：data/interim/eaget_audit/report_<日期>.json（相对仓库根）

判定标准（可用 --min-leads 调整）：
    complete  9 类核心区域产品目录齐全，且每类 jingjinji 域文件数 >= min_leads(默认30)
    partial   有图但不满足 complete
    empty     无 PNG
仅 complete 时次计入"可构建"。纯标准库，无三方依赖。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

CORE_PRODUCTS = [
    "surface", "10m_wind", "rain", "boundary_layer", "temp_inversion",
    "925hpa", "850hpa", "700hpa", "500hpa",
]
STATION_PRODUCTS = ["MiYSK", "DongS", "DaSW", "YongLD", "DingL"]
CYCLE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})$")
HEATING_MONTHS = {11, 12, 1, 2, 3}  # 采暖期（PM2.5 过程季）


def count_pngs(d: Path) -> int:
    try:
        with os.scandir(d) as it:
            return sum(1 for e in it if e.name.endswith(".png"))
    except OSError:
        return -1


def audit_cycle(cycle_dir: Path, min_leads: int) -> dict:
    products = {}
    total = 0
    for p in CORE_PRODUCTS + STATION_PRODUCTS:
        n = count_pngs(cycle_dir / p)
        if n > 0:
            products[p] = n
            total += n
    if total == 0:
        status = "empty"
    else:
        core_ok = all(products.get(p, 0) >= min_leads for p in CORE_PRODUCTS)
        status = "complete" if core_ok else "partial"
    return {"status": status, "total_pngs": total, "products": products}


def audit_keti(root: Path) -> dict:
    """keti_data（CMAQ/NAQP/kma_obs/wrf）覆盖范围：抽文件名里的日期 min/max 与文件数。"""
    out = {}
    keti = root / "keti_data"
    if not keti.is_dir():
        return {"present": False}
    date_re = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")
    for sub in sorted(p for p in keti.iterdir() if p.is_dir()):
        n_files, dates = 0, []
        for dirpath, _dirnames, filenames in os.walk(sub):
            for fn in filenames:
                n_files += 1
                m = date_re.search(fn)
                if m:
                    y, mo, d = int(m[1]), int(m[2]), int(m[3])
                    if 1 <= mo <= 12 and 1 <= d <= 31:
                        dates.append(f"{y:04d}-{mo:02d}-{d:02d}")
            if n_files > 200000:
                break
        out[sub.name] = {
            "file_count": n_files,
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
        }
    out["present"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/mnt/eaget")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sample", type=int, default=0, help="均匀抽样 N 个时次（0=全量）")
    ap.add_argument("--min-leads", type=int, default=30)
    ap.add_argument("--out", default=None, help="JSON 报告路径（默认 data/interim/eaget_audit/）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[FAIL] {root} 不存在。硬盘未挂载。", file=sys.stderr)
        return 2
    entries = sorted(e.name for e in os.scandir(root) if e.is_dir() and CYCLE_RE.match(e.name))
    if not entries:
        print(f"[FAIL] {root} 可访问但没有 YYYYMMDDHHMM 时次目录——挂载点可能是空目录（硬盘未挂载或挂错盘）。",
              file=sys.stderr)
        return 2
    print(f"[OK] 挂载正常：{len(entries)} 个时次目录，{entries[0]} → {entries[-1]}")

    if args.quick:
        nonempty = sum(1 for c in entries[-500:] if count_pngs(root / c / "surface") > 0)
        print(f"[quick] 最近 500 个时次中 surface 非空的有 {nonempty} 个")
        return 0

    cycles = entries
    if args.sample and args.sample < len(entries):
        step = len(entries) / args.sample
        cycles = [entries[int(i * step)] for i in range(args.sample)]
        print(f"抽样 {len(cycles)} / {len(entries)} 个时次")

    by_status = defaultdict(int)
    monthly = defaultdict(lambda: defaultdict(int))  # {YYYY-MM: {status: n}}
    complete_days = set()
    for i, c in enumerate(cycles):
        rec = audit_cycle(root / c, args.min_leads)
        by_status[rec["status"]] += 1
        ym = f"{c[:4]}-{c[4:6]}"
        monthly[ym][rec["status"]] += 1
        if rec["status"] == "complete":
            complete_days.add(c[:8])
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(cycles)}", file=sys.stderr)

    scale = len(entries) / len(cycles)
    heating_days = {d for d in complete_days if int(d[4:6]) in HEATING_MONTHS}

    print("\n=== 逐月覆盖（complete/partial/empty）===")
    for ym in sorted(monthly):
        m = monthly[ym]
        print(f"  {ym}:  {m.get('complete', 0):3d} / {m.get('partial', 0):3d} / {m.get('empty', 0):3d}")
    print("\n=== 汇总 ===")
    for k in ("complete", "partial", "empty"):
        est = f"（推全量约 {round(by_status[k] * scale)}）" if scale > 1 else ""
        print(f"  {k:9s}: {by_status[k]}{est}")
    print(f"  complete 覆盖天数: {len(complete_days)}"
          + (f"（推全量约 {round(len(complete_days) * scale)}）" if scale > 1 else ""))
    print(f"  其中采暖期天数:   {len(heating_days)}"
          + (f"（推全量约 {round(len(heating_days) * scale)}）" if scale > 1 else ""))
    print("  → 可构建历史个例上限 ≈ complete 天数（每天 1 个 08 时起报个例），")
    print("    实际数还取决于：空气质量观测档案覆盖（obs/truth）与 CMAQ/NAQP 指导覆盖。")

    print("\n=== keti_data（模式指导/气象观测数据）===")
    keti = audit_keti(root)
    if not keti.get("present"):
        print("  未找到 keti_data 目录")
    else:
        for name, rec in keti.items():
            if name == "present":
                continue
            print(f"  {name:12s}: {rec['file_count']:7d} 文件  {rec['date_min']} → {rec['date_max']}")

    out_path = args.out or (Path(__file__).resolve().parents[1]
                            / "data" / "interim" / "eaget_audit" / f"report_{date.today().isoformat()}.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "root": str(root),
        "cycle_dir_count": len(entries),
        "range": [entries[0], entries[-1]],
        "sampled": len(cycles),
        "by_status": dict(by_status),
        "monthly": {k: dict(v) for k, v in sorted(monthly.items())},
        "complete_day_count": len(complete_days),
        "heating_season_day_count": len(heating_days),
        "min_leads": args.min_leads,
        "keti_data": keti,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n报告已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
