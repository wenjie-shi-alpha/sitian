#!/usr/bin/env python3
"""批量构建真实个例：对日期区间逐日调用 build_real_case.build()。

用法（仓库根目录，硬盘已挂载、观测库已入库）：
    python3 sitian/scripts/batch_build_cases.py \
        --start 2022-01-01 --end 2025-12-31

产出：
    sitian/cases/real/beijing_YYYY-MM-DD/   个例目录
    sitian/cases/real/manifest.json         构建清单（成功/跳过/原因）

2025-12 月为盲评基准月，meta.split 标记为 benchmark_dec2025，其余为 train_pool；
按污染过程的精细切分由后续 splitter 基于 manifest 完成。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_real_case import REPO_ROOT, build  # noqa: E402

OUT_ROOT = REPO_ROOT / "cases" / "real"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--horizon", type=int, default=6)
    args = ap.parse_args(argv)

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    results = {"built": [], "unscoreable": [], "failed": {}}
    d, t0 = start, time.time()
    n_total = (end - start).days + 1
    i = 0
    while d <= end:
        i += 1
        iso = d.isoformat()
        try:
            path = build(iso, args.horizon)
            case_meta_path = path / "case.json"
            meta = json.loads(case_meta_path.read_text(encoding="utf-8"))
            meta["meta"]["split"] = ("benchmark_dec2025" if iso.startswith("2025-12")
                                     else "train_pool")
            case_meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            if meta["meta"].get("truth_missing_days"):
                results["unscoreable"].append(iso)
            else:
                results["built"].append(iso)
        except Exception as exc:  # 单日失败不阻塞批量
            results["failed"][iso] = str(exc)[:200]
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{n_total}  built={len(results['built'])} "
                  f"unscoreable={len(results['unscoreable'])} failed={len(results['failed'])} "
                  f"({el:.0f}s)", flush=True)
        d += timedelta(days=1)

    manifest = {
        "range": [args.start, args.end],
        "horizon": args.horizon,
        "counts": {k: len(v) for k, v in results.items()},
        "built": results["built"],
        "unscoreable": results["unscoreable"],
        "failed": results["failed"],
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"完成：built={len(results['built'])} unscoreable={len(results['unscoreable'])} "
          f"failed={len(results['failed'])}（{time.time() - t0:.0f}s）")
    print(f"清单：{OUT_ROOT / 'manifest.json'}")
    if results["failed"]:
        reasons = {}
        for v in results["failed"].values():
            key = v.split("(")[0][:60]
            reasons[key] = reasons.get(key, 0) + 1
        print("失败原因分布:", json.dumps(reasons, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
