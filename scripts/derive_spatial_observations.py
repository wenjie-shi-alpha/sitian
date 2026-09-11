#!/usr/bin/env python3
"""Derive leak-safe nationwide observation summaries available at issue time."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from sitian.open_evidence import DEFAULT_EVIDENCE_ROOT, EVIDENCE_VERSION

POLLUTANTS = ("PM2.5", "PM10", "O3", "SO2", "NO2", "CO")
POLLUTANT_UNITS = {name: ("mg/m³" if name == "CO" else "µg/m³") for name in POLLUTANTS}


def _stats(values: list[tuple[str, int, float]]) -> dict:
    if not values:
        return {"n_24h": 0, "latest": None, "mean_24h": None, "max_24h": None,
                "change_6h": None, "slope_24h_per_hour": None, "latest_local_time": None}
    recent = values[-24:]
    numbers = np.asarray([row[2] for row in recent], dtype=float)
    change = float(numbers[-1] - numbers[-7]) if len(numbers) >= 7 else None
    slope = float(np.polyfit(np.arange(len(numbers)), numbers, 1)[0]) if len(numbers) >= 3 else None
    last = recent[-1]
    return {"n_24h": len(numbers), "latest": round(float(numbers[-1]), 1),
            "mean_24h": round(float(numbers.mean()), 1), "max_24h": round(float(numbers.max()), 1),
            "change_6h": round(change, 1) if change is not None else None,
            "slope_24h_per_hour": round(slope, 2) if slope is not None else None,
            "latest_local_time": f"{last[0]}T{last[1]:02d}:00+08:00"}


def _derive(raw_dir: Path, issue_date: str, cities: list[str]) -> dict:
    issue = date.fromisoformat(issue_date)
    series: dict[tuple[str, str], list[tuple[str, int, float]]] = defaultdict(list)
    for offset in (-2, -1, 0):
        day = issue + timedelta(days=offset)
        path = raw_dir / str(day.year) / f"china_cities_{day:%Y%m%d}.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            indexes = {city: header.index(city) for city in cities if city in header}
            for row in reader:
                if len(row) < 3 or row[2] not in POLLUTANTS:
                    continue
                hour = int(row[1])
                if offset == 0 and hour > 7:
                    continue
                for city, index in indexes.items():
                    value = row[index].strip() if index < len(row) else ""
                    if not value:
                        continue
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    if math.isfinite(number):
                        series[(city, row[2])].append((day.isoformat(), hour, number))
    output = {}
    for city in cities:
        output[city] = {
            pollutant: {"unit": POLLUTANT_UNITS[pollutant],
                        **_stats(series[(city, pollutant)])}
            for pollutant in POLLUTANTS
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=DEFAULT_EVIDENCE_ROOT / "manifests" / "open_evidence_v1.json")
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--coords", type=Path, default=Path("data/interim/city_coords.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/aq_obs/cities"))
    parser.add_argument("--issue-date")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    coords = json.loads(args.coords.read_text(encoding="utf-8"))
    cities = sorted(city for city, value in coords.items() if isinstance(value, list))
    dates = manifest["issue_dates"]
    if args.issue_date:
        dates = [value for value in dates if value == args.issue_date]
    if args.limit is not None:
        dates = dates[:args.limit]
    for number, issue_date in enumerate(dates, 1):
        city_rows = _derive(args.raw_dir, issue_date, cities)
        document = {
                "contract_version": EVIDENCE_VERSION,
                "issue_date": issue_date,
                "pollution": {"spatial_observations": {
                    "available": True,
                    "cutoff": f"{issue_date}T07:00:00+08:00",
                    "cutoff_note": "One-hour safety margin before the 08:00 BJT issue; no 08:00 value is used.",
                    "cities": city_rows,
                }},
            }
        target = args.root / "derived" / "by_issue" / f"{issue_date}.spatial_obs.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
                          encoding="utf-8")
        complete = sum(city_rows[city]["PM2.5"]["n_24h"] >= 20 for city in cities)
        print(f"[{number}/{len(dates)}] {issue_date}: {complete}/{len(cities)} cities with >=20 PM2.5 hours", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
