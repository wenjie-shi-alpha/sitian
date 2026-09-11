#!/usr/bin/env python3
"""Fetch as-of-legal CAMS total-column forecast trajectories from Earth Engine.

This is a distinct evidence channel from ``fetch_open_cams.py``: Earth Engine
provides total-column trace gases, while the ADS path requests model level 137.
The two products must never be used as silent substitutes for one another.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from sitian.open_evidence import CAMS_TRACE_GAS_LEAD_HOURS, DEFAULT_EVIDENCE_ROOT

DATASET = "ECMWF/CAMS/NRT"
KIND = "cams_column_trajectory"
SCHEMA_VERSION = "cams-earthengine-columns-v1"
LEAD_HOURS = tuple(CAMS_TRACE_GAS_LEAD_HOURS)
VARIABLES = (
    ("carbon_monoxide", "total_column_carbon_monoxide_surface", "kg m-2"),
    ("nitrogen_dioxide", "total_column_nitrogen_dioxide_surface", "kg m-2"),
    ("sulphur_dioxide", "total_column_sulphur_dioxide_surface", "kg m-2"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jobs(manifest: dict, issue_date: str | None = None) -> list[dict]:
    """Return stable issue-date/cycle jobs without coverage-dependent rechunking."""
    rows = manifest["cams"]["cycles"]
    if issue_date is not None:
        rows = [row for row in rows if row["issue_date"] == issue_date]
    jobs: list[dict] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda value: value["issue_date"]):
        cycle = row["cycle"]
        if cycle in seen:
            continue
        if not cycle.endswith("12:00:00Z"):
            raise ValueError(f"Earth Engine CAMS job is not a 12Z cycle: {row}")
        seen.add(cycle)
        jobs.append({
            "issue_date": row["issue_date"],
            "cycle": cycle,
            "available_at": row["available_at"],
        })
    return jobs


def _target_paths(out_dir: Path, cycle: str) -> tuple[Path, Path]:
    stamp = cycle.replace("-", "").replace(":", "")
    stamp = stamp.replace("T", "T").removesuffix("Z")
    target = out_dir / f"{KIND}_{stamp}.tif"
    return target, Path(str(target) + ".json")


def _is_complete(target: Path, sidecar: Path, cycle: str) -> bool:
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        return (
            target.is_file()
            and target.stat().st_size > 0
            and target.stat().st_size == int(metadata["bytes"])
            and metadata["dataset"] == DATASET
            and metadata["kind"] == KIND
            and metadata["schema_version"] == SCHEMA_VERSION
            and metadata["forecast_reference_time"] == cycle
            and metadata["leadtime_hour"] == list(LEAD_HOURS)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _band_names() -> list[str]:
    return [
        f"fh{lead:03d}_{short_name}_column"
        for lead in LEAD_HOURS
        for short_name, _source_name, _unit in VARIABLES
    ]


def _build_stack(ee: Any, cycle: str) -> tuple[Any, list[str]]:
    initialization = cycle.removesuffix("Z")
    source_names = [source_name for _short, source_name, _unit in VARIABLES]
    collection = (
        ee.ImageCollection(DATASET)
        .filter(ee.Filter.eq("model_initialization_datetime", initialization))
        .filter(ee.Filter.inList("model_forecast_hour", list(LEAD_HOURS)))
    )
    actual_hours = sorted(collection.aggregate_array("model_forecast_hour").getInfo())
    if actual_hours != list(LEAD_HOURS):
        raise RuntimeError(
            f"incomplete Earth Engine CAMS trajectory for {cycle}: {actual_hours}"
        )
    images = []
    names = _band_names()
    for lead in LEAD_HOURS:
        selected = (
            ee.Image(collection.filter(ee.Filter.eq("model_forecast_hour", lead)).first())
            .select(source_names)
            .rename([
                f"fh{lead:03d}_{short_name}_column"
                for short_name, _source_name, _unit in VARIABLES
            ])
        )
        images.append(selected)
    stack = images[0]
    for image in images[1:]:
        stack = stack.addBands(image)
    return stack, names


def _download_url(stack: Any, ee: Any, domain: dict) -> str:
    region = ee.Geometry.Rectangle(
        [domain["west"], domain["south"], domain["east"], domain["north"]],
        proj=None,
        geodesic=False,
    )
    # Omitting scale/crs preserves the native CAMS 0.4-degree grid rather than
    # introducing a second interpolation in the mirror download.
    return stack.getDownloadURL({"region": region, "format": "GEO_TIFF"})


def _inspect_geotiff(path: Path, expected_names: list[str]) -> dict:
    import rasterio

    try:
        with rasterio.open(path) as dataset:
            info = {
                "size": [dataset.width, dataset.height],
                "geoTransform": list(dataset.transform.to_gdal()),
                "coordinateSystem": {"wkt": dataset.crs.to_wkt() if dataset.crs else None},
                "bands": [{"description": value} for value in dataset.descriptions],
            }
    except rasterio.errors.RasterioIOError as exc:
        raise RuntimeError(f"cannot read GeoTIFF {path}: {exc}") from exc
    bands = info.get("bands", [])
    if len(bands) != len(expected_names):
        raise RuntimeError(
            f"expected {len(expected_names)} bands in {path}, found {len(bands)}"
        )
    descriptions = [band.get("description") for band in bands]
    populated = [value for value in descriptions if value]
    if populated and descriptions != expected_names:
        raise RuntimeError(f"unexpected band order in {path}: {descriptions[:6]}")
    size = info.get("size", [])
    if len(size) != 2 or min(size) <= 0:
        raise RuntimeError(f"invalid raster dimensions in {path}: {size}")
    return {
        "size": size,
        "geo_transform": info.get("geoTransform"),
        "coordinate_system": info.get("coordinateSystem", {}).get("wkt"),
        "band_descriptions_preserved": bool(populated),
    }


def _stream_to_file(response: requests.Response, temporary: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with temporary.open("wb") as destination:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            destination.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    expected = response.headers.get("Content-Length")
    if expected is not None and byte_count != int(expected):
        raise RuntimeError(f"download size mismatch: {byte_count} != {expected}")
    return byte_count, digest.hexdigest()


def _write_sidecar(path: Path, metadata: dict) -> None:
    temporary = Path(str(path) + ".part")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _download_job(ee: Any, session: requests.Session, job: dict, domain: dict,
                  out_dir: Path, attempts: int, retry_seconds: float,
                  timeout_seconds: float) -> None:
    target, sidecar = _target_paths(out_dir, job["cycle"])
    if _is_complete(target, sidecar, job["cycle"]):
        print(f"skip {target.name}", flush=True)
        return
    stack, names = _build_stack(ee, job["cycle"])
    temporary = Path(str(target) + ".part")
    started_at = _utc_now()
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            url = _download_url(stack, ee, domain)
            with session.get(
                url,
                stream=True,
                timeout=(30, timeout_seconds),
            ) as response:
                response.raise_for_status()
                byte_count, digest = _stream_to_file(response, temporary)
                etag = response.headers.get("ETag")
            raster = _inspect_geotiff(temporary, names)
            temporary.replace(target)
            completed_at = _utc_now()
            _write_sidecar(sidecar, {
                "contract_version": "open-evidence-v1",
                "schema_version": SCHEMA_VERSION,
                "dataset": DATASET,
                "provider": "ECMWF",
                "mirror": "Google Earth Engine",
                "kind": KIND,
                "semantics": "total_column_mass",
                "issue_date": job["issue_date"],
                "forecast_reference_time": job["cycle"],
                "available_at": job["available_at"],
                "leadtime_hour": list(LEAD_HOURS),
                "variables": [
                    {"name": short, "source_band": source, "unit": unit}
                    for short, source, unit in VARIABLES
                ],
                "bands": names,
                "domain": domain,
                "raster": raster,
                "bytes": byte_count,
                "sha256": digest,
                "retrieval": {
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                    "download_host": urlparse(url).hostname,
                    "http_etag": etag,
                    "attempt": attempt,
                },
                "non_substitution_note": (
                    "Total-column gases are a separate evidence channel and must not be "
                    "used to fill missing CAMS model-level-137 values."
                ),
            })
            print(
                f"done {target.name} {byte_count / 1e6:.2f} MB "
                f"{(time.monotonic() - started):.1f}s",
                flush=True,
            )
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                break
            wait = retry_seconds * (2 ** (attempt - 1))
            print(
                f"retry {target.name} in {wait:g}s ({attempt}/{attempts}): {exc}",
                flush=True,
            )
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "manifests" / "open_evidence_v1.json",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--issue-date")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--retry-seconds", type=float, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--project", help="optional Earth Engine quota project")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require shard-count >= 1 and 0 <= shard-index < shard-count")
    if args.attempts < 1 or min(args.retry_seconds, args.timeout_seconds) <= 0:
        parser.error("attempts and timeout/backoff values must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_jobs = _jobs(manifest, args.issue_date)
    shard_jobs = all_jobs[args.shard_index::args.shard_count]
    out_dir = args.root / "raw" / "cams_earthengine"
    missing = []
    for job in shard_jobs:
        target, sidecar = _target_paths(out_dir, job["cycle"])
        if not _is_complete(target, sidecar, job["cycle"]):
            missing.append(job)
    print(json.dumps({
        "dataset": DATASET,
        "kind": KIND,
        "all_jobs": len(all_jobs),
        "shard_jobs": len(shard_jobs),
        "missing_shard_jobs": len(missing),
        "shard": [args.shard_index, args.shard_count],
        "lead_hours": list(LEAD_HOURS),
        "bands_per_cycle": len(_band_names()),
        "domain": manifest["domain"],
    }, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    import ee

    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "sitian-open-evidence/1.0"
    failed = []
    for job in missing:
        try:
            _download_job(
                ee, session, job, manifest["domain"], out_dir,
                args.attempts, args.retry_seconds, args.timeout_seconds,
            )
        except Exception as exc:
            failed.append({"issue_date": job["issue_date"], "cycle": job["cycle"],
                           "error": str(exc)[:300]})
            print(f"FAILED {job['cycle']}: {exc}", flush=True)
    if failed:
        print(json.dumps({"failed": failed}, ensure_ascii=False), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
