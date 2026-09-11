#!/usr/bin/env python3
"""从原 train 池中划出用于 checkpoint selection 的冬季 challenge。

选择单元是完整 issue date，不是单个 city case。被选日期周围的全部
case 从新 train manifest 中剔除，并硬性检查 (city, forecast_valid_date)
零交集，避免 5 天滚动窗口导致的二次污染。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.case import CaseBundle  # noqa: E402
from sitian.provenance import file_identity  # noqa: E402


def _merge_dates(days: set[str]) -> list[list[str]]:
    if not days:
        return []
    ordered = sorted(date.fromisoformat(d) for d in days)
    blocks, current = [], [ordered[0]]
    for d in ordered[1:]:
        if (d - current[-1]).days == 1:
            current.append(d)
        else:
            blocks.append([x.isoformat() for x in current])
            current = [d]
    blocks.append([x.isoformat() for x in current])
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/interim/valid_cases_train.json")
    ap.add_argument("--target-event", type=int, default=120)
    ap.add_argument("--target-turning", type=int, default=120)
    ap.add_argument("--target-switch", type=int, default=60)
    ap.add_argument("--process-buffer-days", type=int, default=5)
    ap.add_argument("--challenge-out", default="data/interim/challenge_winter.json")
    ap.add_argument("--train-out", default="data/interim/train_without_winter_challenge.json")
    ap.add_argument("--audit-out", default="data/interim/challenge_winter_audit.json")
    args = ap.parse_args()

    source_path = REPO_ROOT / args.source
    paths = [REPO_ROOT / p for p in json.loads(source_path.read_text(encoding="utf-8"))]
    records = []
    for path in paths:
        bundle = CaseBundle.load(path)
        records.append({
            "path": path,
            "case_id": bundle.case_id,
            "issue_date": bundle.issue_date,
            "city": bundle.region,
            "stratum": bundle.meta.get("stratum"),
            "forecast_dates": bundle.forecast_dates(),
        })

    wanted = {"event": args.target_event, "turning": args.target_turning,
              "switch": args.target_switch}
    winter_months = {11, 12, 1, 2, 3}
    by_date: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if int(rec["issue_date"][5:7]) in winter_months and rec["stratum"] in wanted:
            by_date[rec["issue_date"]].append(rec)

    selected_dates: set[str] = set()
    counts = Counter()
    # 每轮选择对尚未满足层贡献最大的整个 issue date；并列时
    # 优先覆盖更多目标 case，再优先更早日期，保证 seed-free 可复现。
    while any(counts[s] < target for s, target in wanted.items()):
        best = None
        for day, day_records in by_date.items():
            if day in selected_dates:
                continue
            dc = Counter(r["stratum"] for r in day_records)
            gain = sum(min(dc[s], max(0, wanted[s] - counts[s])) / wanted[s] for s in wanted)
            if gain <= 0:
                continue
            candidate = (gain, sum(dc.values()), day)
            if (best is None
                    or candidate[0] > best[0]
                    or (candidate[0] == best[0] and candidate[1] > best[1])
                    or (candidate[:2] == best[:2] and candidate[2] < best[2])):
                best = candidate
        if best is None:
            break
        selected_dates.add(best[2])
        counts.update(r["stratum"] for r in by_date[best[2]])

    unmet = {s: wanted[s] - counts[s] for s in wanted if counts[s] < wanted[s]}
    if unmet:
        raise RuntimeError(f"winter pool cannot meet challenge targets: {unmet}")

    challenge = [r for r in records
                 if r["issue_date"] in selected_dates and r["stratum"] in wanted]
    challenge_city_days = {(r["city"], d) for r in challenge for d in r["forecast_dates"]}
    blocked_issue_dates = set()
    for day in selected_dates:
        center = date.fromisoformat(day)
        blocked_issue_dates.update(
            (center + timedelta(days=offset)).isoformat()
            for offset in range(-args.process_buffer_days, args.process_buffer_days + 1)
        )
    retained_train = [r for r in records if r["issue_date"] not in blocked_issue_dates]
    train_city_days = {(r["city"], d) for r in retained_train for d in r["forecast_dates"]}
    overlap = train_city_days & challenge_city_days
    if overlap:
        raise AssertionError(f"challenge/train target leakage: {len(overlap)} city-days")

    challenge_paths = [str(r["path"].relative_to(REPO_ROOT)) for r in challenge]
    train_paths = [str(r["path"].relative_to(REPO_ROOT)) for r in retained_train]
    challenge_out, train_out = REPO_ROOT / args.challenge_out, REPO_ROOT / args.train_out
    challenge_out.write_text(json.dumps(challenge_paths, ensure_ascii=False, indent=1), encoding="utf-8")
    train_out.write_text(json.dumps(train_paths, ensure_ascii=False, indent=1), encoding="utf-8")

    audit = {
        "source_manifest": file_identity(source_path, relative_to=REPO_ROOT),
        "selection_unit": "complete issue_date holdout; challenge retains target strata only",
        "winter_months": sorted(winter_months),
        "targets": wanted,
        "process_buffer_days_each_side": args.process_buffer_days,
        "selected_issue_dates": sorted(selected_dates),
        "selected_contiguous_date_blocks": _merge_dates(selected_dates),
        "challenge": {
            "manifest": file_identity(challenge_out, relative_to=REPO_ROOT),
            "cases": len(challenge),
            "issue_dates": len(selected_dates),
            "cities": len({r["city"] for r in challenge}),
            "strata": dict(Counter(r["stratum"] for r in challenge)),
            "forecast_city_days": len(challenge_city_days),
        },
        "train_after_purge": {
            "manifest": file_identity(train_out, relative_to=REPO_ROOT),
            "cases": len(retained_train),
            "removed_cases": len(records) - len(retained_train),
            "blocked_issue_dates_present_in_source": len(
                {r["issue_date"] for r in records} & blocked_issue_dates),
            "forecast_city_days": len(train_city_days),
        },
        "integrity": {
            "challenge_train_case_id_overlap": len(
                {r["case_id"] for r in challenge} & {r["case_id"] for r in retained_train}),
            "challenge_train_forecast_city_day_overlap": len(overlap),
        },
    }
    audit_out = REPO_ROOT / args.audit_out
    audit_out.write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=1))
    print(f"-> {challenge_out}\n-> {train_out}\n-> {audit_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
