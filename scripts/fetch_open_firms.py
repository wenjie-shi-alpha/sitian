#!/usr/bin/env python3
"""Fetch public VIIRS active-fire locations used by open evidence.

NASA's science-quality monthly ASCII files are served from the public UMD
SFTP endpoint documented in the VIIRS Collection-2 active-fire guide.  They
are retained as reproducible fallbacks.  The official LANCE NRT HTTPS archive
supplies the preferred as-of daily files.  NRT and standard files stay in
separate directories and retain explicit processing labels in their sidecars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

from sitian.open_evidence import DEFAULT_EVIDENCE_ROOT

HOST = "fuoco.geog.umd.edu"
USER = "fire"
PUBLIC_PASSWORD = os.environ.get("FIRMS_SFTP_PASSWORD", "burnt")
SENSORS = {
    "noaa20": ("VJ114IMGML", "VJ114IMGML"),
    "noaa21": ("VJ214IMGML", "VJ214IMGML"),
}
NRT_PRODUCTS = {
    "noaa20": ("noaa-20-viirs-c2", "J1", "VJ114IMGTDL"),
    "noaa21": ("noaa-21-viirs-c2", "J2", "VJ214IMGTDL"),
}
NRT_ROOT = "https://nrt3.modaps.eosdis.nasa.gov/archive/FIRMS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_sftp(commands: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([
        "sshpass", "-p", PUBLIC_PASSWORD,
        "sftp", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
        "-o", "BatchMode=no", f"{USER}@{HOST}",
    ], input=commands, text=True, capture_output=True, timeout=timeout, check=False)


def _available(sensor: str) -> dict[str, str]:
    directory, prefix = SENSORS[sensor]
    remote_dir = f"data/VIIRS/C2/{directory}"
    result = _run_sftp(f"ls {remote_dir}\nbye\n", timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    pattern = re.compile(rf"({prefix}\.(20\d{{4}})\.C2\.\d+\.csv\.gz)")
    return {match.group(2): f"{remote_dir}/{match.group(1)}"
            for match in pattern.finditer(result.stdout)}


def _months(manifest: dict) -> list[str]:
    days = [date.fromisoformat(value) for value in manifest["issue_dates"]]
    start, end = min(days) - timedelta(days=3), max(days)
    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append(current.strftime("%Y%m"))
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def _wanted_days(manifest: dict) -> tuple[date, date]:
    days = [date.fromisoformat(value) for value in manifest["issue_dates"]]
    return min(days) - timedelta(days=3), max(days)


def _nrt_available(sensor: str, start: date, end: date) -> list[dict]:
    directory, platform, product = NRT_PRODUCTS[sensor]
    listing_url = f"{NRT_ROOT}/{directory}/Global/.json"
    with urllib.request.urlopen(listing_url, timeout=60) as response:
        listing = json.load(response)
    pattern = re.compile(
        rf"^{platform}_VIIRS_C2_Global_{product}_NRT_(20\d{{5}})\.txt$"
    )
    output = []
    for item in listing.get("content", []):
        match = pattern.match(item.get("name", ""))
        if not match or item.get("resourceType") != "File":
            continue
        acquired_day = datetime.strptime(match.group(1), "%Y%j").date()
        if start <= acquired_day <= end:
            output.append({
                "sensor": sensor,
                "day": acquired_day,
                "url": item["downloadsLink"],
                "name": item["name"],
                "bytes": int(item["size"]),
                "mtime": int(item.get("mtime") or 0),
            })
    return sorted(output, key=lambda value: value["day"])


def _fetch_nrt(job: dict, root: Path) -> tuple[str, str]:
    out_dir = root / "raw" / "firms" / "nrt_daily" / job["sensor"]
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / job["name"]
    part = target.with_suffix(target.suffix + ".part")
    sidecar = target.with_suffix(target.suffix + ".json")
    if (target.is_file() and target.stat().st_size == job["bytes"] and
            sidecar.is_file()):
        return "skip", target.name
    # NASA Earthdata accepts credentials from ~/.netrc when wget is told to
    # send them without waiting for an authentication challenge.  --continue
    # makes the multi-GB historical batch safely restartable.
    command = [
        "wget", "--no-verbose", "--auth-no-challenge=on", "--continue",
        "--timeout=60", "--tries=8", "-O", str(part), job["url"],
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=1800,
                            check=False)
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:] or result.stdout[-1000:])
    if not part.is_file() or part.stat().st_size != job["bytes"]:
        actual = part.stat().st_size if part.is_file() else 0
        raise RuntimeError(f"size mismatch: expected {job['bytes']}, got {actual}")
    part.replace(target)
    sidecar.write_text(json.dumps({
        "contract_version": "open-evidence-v1",
        "source": "NASA LANCE FIRMS VIIRS Collection-2 daily active-fire locations",
        "processing": "near_real_time",
        "sensor": job["sensor"],
        "acquisition_day": job["day"].isoformat(),
        "source_url": job["url"],
        "source_mtime_epoch": job["mtime"],
        "availability_semantics": (
            "Historical replay exposes each acquisition only after the fixed six-hour "
            "latency in the evidence contract. NRT remains the preferred as-of source; "
            "standard processing is only a labelled fallback for missing sensor-days."
        ),
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return "done", f"{target.stat().st_size/1e6:.1f} MB {target.name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=DEFAULT_EVIDENCE_ROOT / "manifests" / "open_evidence_v1.json")
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--sensor", choices=("noaa20", "noaa21", "both"), default="both")
    parser.add_argument("--processing", choices=("standard", "nrt", "both"), default="both")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sensors = ("noaa20", "noaa21") if args.sensor == "both" else (args.sensor,)
    wanted_months = set(_months(manifest))
    start_day, end_day = _wanted_days(manifest)
    standard_jobs = []
    if args.processing in {"standard", "both"}:
        for sensor in sensors:
            for month, remote in sorted(_available(sensor).items()):
                if month in wanted_months:
                    standard_jobs.append((sensor, month, remote))
    nrt_jobs = []
    if args.processing in {"nrt", "both"}:
        for sensor in sensors:
            nrt_jobs.extend(_nrt_available(sensor, start_day, end_day))
    if args.limit is not None:
        standard_jobs = standard_jobs[:args.limit]
        nrt_jobs = nrt_jobs[:args.limit]
    print(json.dumps({
        "standard_jobs": len(standard_jobs), "nrt_jobs": len(nrt_jobs),
        "months_requested": sorted(wanted_months),
        "nrt_days_requested": [start_day.isoformat(), end_day.isoformat()],
        "nrt_bytes": sum(job["bytes"] for job in nrt_jobs),
        "nrt_jobs_by_sensor": {
            sensor: sum(job["sensor"] == sensor for job in nrt_jobs) for sensor in sensors
        },
        "sensors": sensors,
    }, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    failures = []
    for number, (sensor, month, remote) in enumerate(standard_jobs, 1):
        out_dir = args.root / "raw" / "firms" / "standard_monthly" / sensor
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / Path(remote).name
        sidecar = target.with_suffix(target.suffix + ".json")
        if target.is_file() and target.stat().st_size and sidecar.is_file():
            print(f"[standard {number}/{len(standard_jobs)}] skip {target.name}", flush=True)
            continue
        # The public science archive is deliberately low-bandwidth.  ``reget``
        # resumes a partial monthly file after a disconnect instead of spending
        # the next run downloading the same tens of megabytes again.
        try:
            result = _run_sftp(f'lcd "{out_dir}"\nreget "{remote}"\nbye\n', timeout=7200)
        except subprocess.TimeoutExpired as exc:
            failures.append((sensor, month, f"timed out after {exc.timeout}s; partial file retained"))
            print(f"[standard {number}/{len(standard_jobs)}] TIMEOUT {sensor} {month}; rerun resumes", flush=True)
            continue
        if result.returncode or not target.is_file() or not target.stat().st_size:
            failures.append((sensor, month, result.stderr[-500:]))
            print(f"[standard {number}/{len(standard_jobs)}] FAILED {sensor} {month}", flush=True)
            continue
        sidecar.write_text(json.dumps({
            "contract_version": "open-evidence-v1",
            "source": "NASA VIIRS Collection-2 monthly active-fire locations",
            "processing": "standard_science_quality",
            "sensor": sensor,
            "month": month,
            "source_url": f"sftp://{HOST}/{remote}",
            "availability_semantics": (
                "Acquisition timestamps are retained. Standard processing is not claimed "
                "to be byte-identical to the historical NRT feed."
            ),
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[standard {number}/{len(standard_jobs)}] done {target.stat().st_size/1e6:.1f} MB {target.name}", flush=True)
    if nrt_jobs:
        workers = max(1, args.workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(_fetch_nrt, job, args.root): job for job in nrt_jobs}
            completed = 0
            for future in as_completed(pending):
                completed += 1
                job = pending[future]
                try:
                    status, detail = future.result()
                    print(f"[nrt {completed}/{len(nrt_jobs)}] {status} {detail}", flush=True)
                except Exception as exc:
                    failures.append((job["sensor"], job["day"].isoformat(), str(exc)))
                    print(f"[nrt {completed}/{len(nrt_jobs)}] FAILED {job['sensor']} "
                          f"{job['day']}: {exc}", flush=True)
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
