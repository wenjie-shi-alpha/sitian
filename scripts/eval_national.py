#!/usr/bin/env python3
"""全国池基线评测：分层区分度表 + oracle-switch 上限。

基线阶梯（docs/TRAINING_DESIGN.md §4.5）：
    persistence   持续性外推
    guidance      模式指导直译（CAMS，北方城市另有 cmaq/naqp）
    climatology   当月气候态 p50
    oracle-switch 逐 case 事后取上述三个完整预报中的最优值。它是不可实现的
                  事后选择参考，不是任意融合/订正策略的理论上限。

用法：python3 scripts/eval_national.py --split val [--limit 0]
产出：控制台分层表 + data/interim/eval_{split}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.agents.scripted import run_baselines  # noqa: E402
from sitian.case import CaseBundle  # noqa: E402
from sitian.provenance import file_identity  # noqa: E402
from sitian.scoring import reward_spec  # noqa: E402

CORE_AGENTS = ["persistence", "guidance", "climatology"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = REPO_ROOT / "cases" / "national" / a.split
    valid_manifest = (REPO_ROOT / a.manifest if a.manifest else
                      REPO_ROOT / "data" / "interim" / f"valid_cases_{a.split}.json")
    if valid_manifest.exists():
        dirs = [REPO_ROOT / p for p in json.loads(valid_manifest.read_text(encoding="utf-8"))]
        print(f"using data-valid manifest: {valid_manifest.relative_to(REPO_ROOT)}")
    else:
        dirs = sorted(d for d in root.iterdir() if (d / "case.json").exists())
    if a.limit:
        dirs = dirs[: a.limit]
    print(f"{a.split}: {len(dirs)} cases")

    rows = []
    for i, d in enumerate(dirs):
        b = CaseBundle.load(d)
        res = run_baselines(b, CORE_AGENTS)
        scores = {k: v["composite"] for k, v in res.items() if v["composite"] is not None}
        if not scores:
            continue
        core_scores = dict(scores)
        # 额外模式源只做诊断，不进入跨 split 的 core oracle；否则 train 有 cmaq/naqp、
        # val/test 只有 CAMS 时，oracle 的候选集合本身发生变化。
        from sitian.agents.scripted import GuidanceFollowAgent, run_episode
        from sitian.env import ForecastEnv
        for source in sorted((b.guidance or {}).get("sources", {})):
            if source == "cams":
                continue
            out = run_episode(ForecastEnv(b), GuidanceFollowAgent(prefer=(source,)))
            score = (out["info"].get("score") or {}).get("outcome_composite")
            if score is not None:
                scores[f"guidance:{source}"] = score
        rows.append({"case_id": b.case_id, "stratum": b.meta.get("stratum"),
                     "cluster": b.meta.get("cluster"), "month": b.issue_date[:7],
                     "scores": scores,
                     "oracle_switch": max(core_scores.values()),
                     "oracle_pick": max(core_scores, key=core_scores.get),
                     "oracle_all_sources": max(scores.values()),
                     "oracle_all_sources_pick": max(scores, key=scores.get),
                     "components": {k: v["components"] for k, v in res.items()}})
        if i % 200 == 199:
            print(f"  {i+1}/{len(dirs)}")

    def agg(subset, key):
        vals = [r["scores"].get(key) if key != "oracle_switch" else r["oracle_switch"]
                for r in subset]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    cols = CORE_AGENTS + ["oracle_switch"]
    hdr = f"{'stratum':<12} {'n':>5} " + " ".join(f"{c:>13}" for c in cols)
    print("\n" + hdr); print("-" * len(hdr))
    by_stratum = defaultdict(list)
    for r in rows:
        by_stratum[r["stratum"]].append(r)
    for s in sorted(by_stratum):
        sub = by_stratum[s]
        print(f"{s:<12} {len(sub):>5} " + " ".join(f"{str(agg(sub, c)):>13}" for c in cols))
    print(f"{'ALL':<12} {len(rows):>5} " + " ".join(f"{str(agg(rows, c)):>13}" for c in cols))

    picks = defaultdict(int)
    for r in rows:
        picks[r["oracle_pick"]] += 1
    print("\noracle 最优策略分布:", dict(picks))

    out = REPO_ROOT / "data" / "interim" / f"eval_{a.out_name or a.split}.json"
    provenance = {
        "reward": reward_spec(),
        "standards_manifest": file_identity(
            REPO_ROOT / "references" / "standards" / "manifest.json", relative_to=REPO_ROOT),
        "case_manifest": (file_identity(valid_manifest, relative_to=REPO_ROOT)
                          if valid_manifest.exists() else None),
    }
    out.write_text(json.dumps({"split": a.split,
                               "evaluation_name": a.out_name or a.split, "n": len(rows),
                               "provenance": provenance, "rows": rows},
                              ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
