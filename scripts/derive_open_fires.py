#!/usr/bin/env python3
"""Derive legal, deterministic VIIRS fire evidence for every issue date.

Official LANCE daily NRT files are the preferred as-of source.  Historical
Collection-2 monthly files are used only as a reproducible fallback for a
sensor-day that has no archived NRT file.  The replay cutoff is strict: an
acquisition is visible only after the fixed six-hour latency in the evidence
contract.  Standard fallback rows are explicitly labelled as retrospective
proxies because standard and historical NRT processing are not identical.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from sitian.open_evidence import DEFAULT_EVIDENCE_ROOT, EVIDENCE_VERSION, issue_asof

UTC = timezone.utc
EARTH_KM = 6371.0088


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _acquired(day: str, hhmm: str) -> datetime | None:
    day = str(day).replace("-", "")
    raw_time = str(hhmm).replace(":", "")
    try:
        text = f"{int(raw_time):04d}"
        base = datetime.strptime(day, "%Y%m%d").replace(tzinfo=UTC)
        hours, minutes = int(text[:2]), int(text[2:])
        if hours == 24 and minutes == 0:
            return base + timedelta(days=1)
        if hours > 23 or minutes > 59:
            return None
        return base + timedelta(hours=hours, minutes=minutes)
    except (TypeError, ValueError):
        return None


def _distance_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    distance = 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return distance, bearing


def _monthly_paths(root: Path, issue: datetime) -> list[tuple[str, Path]]:
    months = {(issue - timedelta(hours=value)).strftime("%Y%m") for value in (0, 72)}
    paths = []
    base = root / "raw" / "firms" / "standard_monthly"
    for sensor_dir in sorted(base.glob("*")):
        if not sensor_dir.is_dir():
            continue
        for month in sorted(months):
            for path in sorted(sensor_dir.glob(f"*.{month}.C2.*.csv.gz")):
                if path.with_suffix(path.suffix + ".json").is_file():
                    paths.append((sensor_dir.name, path))
    return paths


def _daily_paths(root: Path, oldest: datetime, newest: datetime) -> list[tuple[str, Path]]:
    paths = []
    current = oldest.date()
    days = []
    while current <= newest.date():
        days.append(current)
        current += timedelta(days=1)
    base = root / "raw" / "firms" / "nrt_daily"
    for sensor_dir in sorted(base.glob("*")):
        if not sensor_dir.is_dir():
            continue
        sensor = sensor_dir.name
        for acquired_day in days:
            stamp = acquired_day.strftime("%Y%j")
            for path in sorted(sensor_dir.glob(f"*_NRT_{stamp}.txt")):
                if path.with_suffix(path.suffix + ".json").is_file():
                    paths.append((sensor, path))
    return paths


def _row_values(row: dict, processing: str) -> tuple[datetime, float, float, float] | None:
    if processing == "standard_science_quality":
        acquired = _acquired(row.get("YYYYMMDD", ""), row.get("HHMM", ""))
        lat_name, lon_name, frp_name = "Lat", "Lon", "FRP"
        # VIIRS Type 0 is presumed vegetation fire. Static sources, volcanoes
        # and offshore detections are not biomass-fire evidence.
        try:
            if int(float(row.get("Type", 0))) != 0:
                return None
        except (TypeError, ValueError):
            return None
    else:
        acquired = _acquired(row.get("acq_date", ""), row.get("acq_time", ""))
        lat_name, lon_name, frp_name = "latitude", "longitude", "frp"
        # The NRT daily format lacks Type. Low-confidence points are excluded,
        # and the remaining rows are described as active-fire candidates.
        if str(row.get("confidence", "")).lower() == "low":
            return None
    if acquired is None:
        return None
    try:
        return acquired, float(row[lat_name]), float(row[lon_name]), max(
            0.0, float(row.get(frp_name) or 0.0)
        )
    except (KeyError, TypeError, ValueError):
        return None


@lru_cache(maxsize=12)
def _read_source_rows(path_text: str, processing: str, south: float, north: float,
                      west: float, east: float) -> tuple[tuple[datetime, float, float, float], ...]:
    """Parse each monthly source once while consecutive issue dates reuse it.

    The manifest dates are chronological and each replay window touches at most
    two adjacent months.  Caching domain-filtered rows avoids decompressing the
    same large FIRMS CSV for every city-day without changing visibility rules.
    """
    path = Path(path_text)
    opener = gzip.open if path.suffix == ".gz" else open
    output = []
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            values = _row_values(row, processing)
            if values is None:
                continue
            acquired, lat, lon, frp = values
            if south <= lat <= north and west <= lon <= east:
                output.append((acquired, lat, lon, frp))
    return tuple(output)


def _read_visible(root: Path, issue: datetime, domain: dict, latency_hours: int,
                  lookback_hours: int) -> tuple[list[dict], list[dict]]:
    newest = issue - timedelta(hours=latency_hours)
    oldest = issue - timedelta(hours=lookback_hours)
    rows, provenance = [], []
    monthly = _monthly_paths(root, issue)
    daily = _daily_paths(root, oldest, newest)
    nrt_days: set[tuple[str, date]] = set()
    for sensor, path in daily:
        try:
            meta = json.loads(
                path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8")
            )
            nrt_days.add((sensor, datetime.fromisoformat(meta["acquisition_day"]).date()))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            match = re.search(r"_NRT_(20\d{5})\.txt$", path.name)
            if match:
                nrt_days.add((sensor, datetime.strptime(match.group(1), "%Y%j").date()))
    # Read NRT first. Standard rows are retained only for sensor-days with no
    # archived NRT file, so hindsight processing cannot silently replace the
    # source that was actually available at issue time.
    sources = [(sensor, path, "near_real_time") for sensor, path in daily]
    sources += [(sensor, path, "standard_science_quality") for sensor, path in monthly]
    seen = set()
    for sensor, path, processing in sources:
        source_rows = _read_source_rows(
            str(path), processing, float(domain["south"]), float(domain["north"]),
            float(domain["west"]), float(domain["east"]),
        )
        for acquired, lat, lon, frp in source_rows:
            if not (oldest <= acquired <= newest):
                continue
            if (
                processing == "standard_science_quality"
                and (sensor, acquired.date()) in nrt_days
            ):
                continue
            key = (sensor, acquired, round(lat, 4), round(lon, 4))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"sensor": sensor, "acquired": acquired, "lat": lat,
                         "lon": lon, "frp_mw": frp, "processing": processing})
        meta = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        observed_processing = meta.get("processing", processing)
        provenance.append({"sensor": sensor, "relative_path": str(path.relative_to(root)),
                           "processing": observed_processing,
                           "asof_role": ("operational_nrt" if observed_processing == "near_real_time"
                                         else "retrospective_proxy_fallback"),
                           "sha256": meta.get("sha256") or _sha256(path),
                           "bytes": path.stat().st_size})
    return rows, provenance


def _top_clusters(rows: list[dict], limit: int = 12) -> list[dict]:
    cells: dict[tuple[int, int], dict] = defaultdict(lambda: {"count": 0, "frp": 0.0,
                                                              "lat_sum": 0.0, "lon_sum": 0.0})
    for row in rows:
        # Two-degree cells are intentionally deterministic and model-free.
        key = (math.floor(row["lat"] / 2), math.floor(row["lon"] / 2))
        cell = cells[key]
        cell["count"] += 1
        cell["frp"] += row["frp_mw"]
        cell["lat_sum"] += row["lat"]
        cell["lon_sum"] += row["lon"]
    ranked = sorted(cells.values(), key=lambda value: (value["frp"], value["count"]), reverse=True)
    return [{"lat": round(cell["lat_sum"] / cell["count"], 2),
             "lon": round(cell["lon_sum"] / cell["count"], 2),
             "detections": cell["count"], "frp_sum_mw": round(cell["frp"], 1)}
            for cell in ranked[:limit]]


def _cities(rows: list[dict], coords: dict) -> dict:
    row_lats = np.asarray([row["lat"] for row in rows], dtype=float)
    row_lons = np.asarray([row["lon"] for row in rows], dtype=float)
    row_lats_rad = np.radians(row_lats)
    output = {}
    for city, location in coords.items():
        if not isinstance(location, list) or len(location) != 2:
            continue
        lat, lon = map(float, location)
        p1 = math.radians(lat)
        dp = row_lats_rad - p1
        dl = np.radians(row_lons - lon)
        a = np.sin(dp / 2) ** 2 + math.cos(p1) * np.cos(row_lats_rad) * np.sin(dl / 2) ** 2
        distances = 2 * EARTH_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        within_800 = distances <= 800
        nearby_indexes = np.flatnonzero(within_800)
        within_300_indexes = np.flatnonzero(distances <= 300)
        nearest_index = None if nearby_indexes.size == 0 else int(
            nearby_indexes[np.argmin(distances[nearby_indexes])]
        )
        nearest_distance = None
        nearest_bearing = None
        if nearest_index is not None:
            nearest_distance, nearest_bearing = _distance_bearing(
                lat, lon, float(row_lats[nearest_index]), float(row_lons[nearest_index])
            )
        output[city] = {
            "within_100km": int(np.count_nonzero(distances <= 100)),
            "within_300km": int(np.count_nonzero(distances <= 300)),
            "within_800km": int(nearby_indexes.size),
            # Preserve the original row-order summation so existing evidence
            # remains byte-stable despite vectorized distance calculations.
            "frp_within_300km_mw": round(sum(rows[index]["frp_mw"]
                                               for index in within_300_indexes), 1),
            "frp_within_800km_mw": round(sum(rows[index]["frp_mw"]
                                               for index in nearby_indexes), 1),
            "nearest": None if nearest_index is None else {
                "distance_km": round(nearest_distance),
                "bearing_from_city_deg": round(nearest_bearing),
                "frp_mw": round(rows[nearest_index]["frp_mw"], 1),
                "acquired": rows[nearest_index]["acquired"].isoformat().replace("+00:00", "Z"),
            },
        }
    return output


def _complete_existing(path: Path, issue_date: str) -> bool:
    """True only when a previously derived file exists for exactly this issue date."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (document.get("issue_date") == issue_date
            and document.get("contract_version") == EVIDENCE_VERSION
            and isinstance(document.get("pollution", {}).get("fires"), dict))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=DEFAULT_EVIDENCE_ROOT / "manifests" / "open_evidence_v1.json")
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--coords", type=Path, default=Path("data/interim/city_coords.json"))
    parser.add_argument("--issue-date")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip issue dates whose derived file already exists for this issue date")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    coords = json.loads(args.coords.read_text(encoding="utf-8"))
    dates = manifest["issue_dates"]
    if args.issue_date:
        dates = [value for value in dates if value == args.issue_date]
    if args.limit is not None:
        dates = dates[:args.limit]
    latency = int(manifest["firms"]["latency_hours"])
    lookback = int(manifest["firms"]["lookback_hours"])
    for number, issue_date in enumerate(dates, 1):
        target = args.root / "derived" / "by_issue" / f"{issue_date}.fires.json"
        if args.skip_existing and _complete_existing(target, issue_date):
            print(f"[{number}/{len(dates)}] {issue_date}: complete existing file -> skip", flush=True)
            continue
        issue = issue_asof(issue_date)
        rows, provenance = _read_visible(args.root, issue, manifest["domain"], latency, lookback)
        document = {
            "contract_version": EVIDENCE_VERSION,
            "issue_date": issue_date,
            "pollution": {"fires": {
                "available": bool(provenance),
                "window": {"oldest_acquisition": (issue - timedelta(hours=lookback)).isoformat().replace("+00:00", "Z"),
                           "newest_acquisition": (issue - timedelta(hours=latency)).isoformat().replace("+00:00", "Z")},
                "availability_semantics": manifest["firms"]["standard_processing_note"],
                "detection_semantics": (
                    "Nominal/high-confidence NRT active-fire candidates are preferred; "
                    "standard Type=0 vegetation-fire rows are retrospective proxy fallbacks "
                    "only for sensor-days without an archived NRT file."
                ),
                "asof_nrt_source_available": any(
                    row.get("processing") == "near_real_time" for row in provenance
                ),
                "active_fire_candidate_detections": len(rows),
                "near_real_time_detections": sum(
                    row["processing"] == "near_real_time" for row in rows
                ),
                "standard_type0_detections": sum(
                    row["processing"] == "standard_science_quality" for row in rows
                ),
                "frp_sum_mw": round(sum(row["frp_mw"] for row in rows), 1),
                "regional_clusters": _top_clusters(rows),
                "cities": _cities(rows, coords),
                "raw": provenance,
            }},
        }
        target = args.root / "derived" / "by_issue" / f"{issue_date}.fires.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
                          encoding="utf-8")
        print(f"[{number}/{len(dates)}] {issue_date}: {len(rows)} visible fires -> {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
