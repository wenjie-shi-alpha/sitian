#!/usr/bin/env python3
"""Download the missing open CAMS composition evidence to the external disk."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cdsapi
from ecmwf.datastores import Client

from sitian.open_evidence import (
    CAMS_AEROSOL_LEAD_HOURS,
    CAMS_TRACE_GAS_LEAD_HOURS,
    DEFAULT_EVIDENCE_ROOT,
)

DATASET = "cams-global-atmospheric-composition-forecasts"


def _client() -> Client:
    """Build the async client from the same ADS credentials used by cdsapi."""
    legacy = cdsapi.Client(quiet=True)
    return Client(url=legacy.url, key=legacy.key, verify=legacy.verify,
                  maximum_tries=10, sleep_max=30, cleanup=False)


def _publish_from_stage(staged: Path, target: Path, lock_path: Path) -> tuple[int, str]:
    """Serialize writes to the slower external disk and publish atomically."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with staged.open("rb") as source, temporary.open("wb") as destination:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(target)
    return target.stat().st_size, digest.hexdigest()


def _consecutive_chunks(days: list[date], size: int) -> list[tuple[date, date]]:
    runs: list[tuple[date, date]] = []
    if not days:
        return runs
    start = previous = days[0]
    for day in days[1:]:
        if day != previous + timedelta(days=1) or (day - start).days + 1 > size:
            runs.append((start, previous))
            start = day
        previous = day
    runs.append((start, previous))
    return runs


def _covered_reference_days(out_dir: Path, kind: str) -> set[date]:
    """Return days backed by a complete NetCDF + provenance sidecar."""
    covered: set[date] = set()
    for sidecar in out_dir.glob(f"{kind}_*.nc.json"):
        target = sidecar.with_suffix("")
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            first, last = map(date.fromisoformat, meta["forecast_reference_dates"])
            if (not target.is_file() or target.stat().st_size <= 0 or
                    target.stat().st_size != int(meta.get("bytes", -1))):
                continue
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        current = first
        while current <= last:
            covered.add(current)
            current += timedelta(days=1)
    return covered


def _retrieve_with_deadline(client: Client, request: dict, archive: Path,
                            timeout_minutes: float, poll_seconds: float) -> dict:
    """Submit/resume one ADS job and download it with parallel HTTP ranges."""
    request_digest = hashlib.sha256(json.dumps(
        request, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    state_path = Path(str(archive) + ".request.json")
    remote = None
    submitted_at = datetime.now(timezone.utc)
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("request_sha256") == request_digest:
                candidate = client.get_remote(state["ads_request_id"])
                if candidate.status in ("accepted", "running", "successful"):
                    remote = candidate
                    submitted_at = datetime.fromisoformat(
                        state["submitted_at"].replace("Z", "+00:00")
                    )
                    print(f"resuming request_id={candidate.request_id} status={candidate.status}",
                          flush=True)
        except Exception as exc:
            print(f"discard stale request state {state_path.name}: {exc}", flush=True)
    if remote is None:
        remote = client.submit(DATASET, request)
        submitted_at = datetime.now(timezone.utc)
        state_temporary = Path(str(state_path) + ".part")
        state_temporary.write_text(json.dumps({
            "dataset": DATASET,
            "ads_request_id": remote.request_id,
            "submitted_at": submitted_at.isoformat().replace("+00:00", "Z"),
            "request_sha256": request_digest,
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        state_temporary.replace(state_path)
    request_id = remote.request_id
    started = time.monotonic()
    print(f"submitted request_id={request_id}", flush=True)
    deadline = time.monotonic() + timeout_minutes * 60
    results_ready = False
    try:
        while True:
            if remote.results_ready:
                results_ready = True
                results = remote.get_results()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"ADS result {request_id} was ready but had no download time remaining"
                    )
                command = [
                    "aria2c", "--continue=true", "--auto-file-renaming=false",
                    "--allow-overwrite=true", "--file-allocation=none",
                    # ECMWF's generated object-store URLs are commonly throttled as
                    # one object.  Eight tiny ranges did not increase aggregate
                    # throughput and repeatedly tripped the 1 KiB/s per-connection
                    # abort, throwing away useful transfer time.  Four resumable
                    # ranges are enough to tolerate a stalled connection; a slow
                    # but live response must be allowed to continue.
                    "--max-connection-per-server=4", "--split=4", "--min-split-size=4M",
                    "--max-tries=0", "--retry-wait=5", "--connect-timeout=30",
                    "--timeout=120", "--lowest-speed-limit=0", "--summary-interval=30",
                    "--console-log-level=notice", "--dir", str(archive.parent),
                    "--out", archive.name, results.location,
                ]
                try:
                    completed = subprocess.run(command, check=False, timeout=remaining)
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError(
                        f"parallel download for {request_id} exceeded the job deadline"
                    ) from exc
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"aria2c failed for ADS request {request_id}: exit {completed.returncode}"
                    )
                if archive.stat().st_size != results.content_length:
                    raise RuntimeError(
                        f"download size mismatch for {request_id}: "
                        f"{archive.stat().st_size} != {results.content_length}"
                    )
                completed_at = datetime.now(timezone.utc)
                return {
                    "ads_request_id": request_id,
                    "submitted_at": submitted_at.isoformat().replace("+00:00", "Z"),
                    "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    remote.delete()
                except Exception as exc:
                    print(f"warning: could not delete timed-out {request_id}: {exc}", flush=True)
                raise TimeoutError(
                    f"ADS request {request_id} exceeded {timeout_minutes:g} minutes and was cancelled"
                )
            time.sleep(min(poll_seconds, remaining))
            remote.update()
    except BaseException:
        # A local interruption must not leave an invisible server-side job
        # consuming the user's ADS queue. Successful jobs are left intact.
        if not results_ready:
            try:
                remote.delete()
            except Exception:
                pass
        raise


def _request(client: Client, kind: str, d0: date, d1: date, variables: list[str],
             leads: list[int], area: list[float], out_dir: Path,
             stage_dir: Path, timeout_minutes: float, poll_seconds: float) -> None:
    target = out_dir / f"{kind}_{d0}_{d1}.nc"
    sidecar = target.with_suffix(".nc.json")
    if target.is_file() and target.stat().st_size > 0 and sidecar.is_file():
        print(f"skip {target.name}", flush=True)
        return
    request = {
        "date": [f"{d0}/{d1}"],
        "time": ["12:00"],
        "type": ["forecast"],
        "leadtime_hour": [str(value) for value in leads],
        "area": area,
        "data_format": "netcdf_zip",
        "variable": variables,
    }
    if kind == "trace_gases":
        request["model_level"] = ["137"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    archive = stage_dir / f"{kind}_{d0}_{d1}.zip"
    # ADS occasionally rejects an accepted job because the dataset-wide queue
    # is temporarily full. Rapid resubmission only extends the throttle, so
    # queue-limit responses get a longer, bounded backoff and more attempts.
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            retrieval = _retrieve_with_deadline(
                client, request, archive, timeout_minutes, poll_seconds
            )
            break
        except Exception as exc:
            message = str(exc)
            if ("too large" in message.lower() or "cost limits" in message.lower()) and d0 < d1:
                midpoint = d0 + (d1 - d0) // 2
                _request(client, kind, d0, midpoint, variables, leads, area, out_dir,
                         stage_dir, timeout_minutes, poll_seconds)
                _request(client, kind, midpoint + timedelta(days=1), d1,
                         variables, leads, area, out_dir, stage_dir,
                         timeout_minutes, poll_seconds)
                return
            if attempt == max_attempts - 1:
                raise
            queue_limited = "queued requests for this dataset" in message.lower()
            wait = (min(120 * (attempt + 1), 900) if queue_limited
                    else min(60 * (attempt + 1), 300))
            print(f"retry {target.name} after {wait}s: {message[:160]}", flush=True)
            time.sleep(wait)
    with zipfile.ZipFile(archive) as zipped:
        names = [name for name in zipped.namelist() if name.endswith(".nc")]
        if len(names) != 1:
            raise RuntimeError(f"expected one NetCDF in {archive}, got {names}")
        staged_netcdf = stage_dir / f"{kind}_{d0}_{d1}.nc"
        staged_temporary = staged_netcdf.with_suffix(".nc.extracting")
        with zipped.open(names[0]) as source, staged_temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1 << 20)
        staged_temporary.replace(staged_netcdf)
    archive.unlink(missing_ok=True)
    Path(str(archive) + ".aria2").unlink(missing_ok=True)
    Path(str(archive) + ".request.json").unlink(missing_ok=True)
    byte_count, digest = _publish_from_stage(
        staged_netcdf, target, stage_dir / ".external_publish.lock"
    )
    staged_netcdf.unlink(missing_ok=True)
    sidecar.write_text(json.dumps({
        "contract_version": "open-evidence-v1",
        "dataset": DATASET,
        "kind": kind,
        "forecast_reference_dates": [str(d0), str(d1)],
        "forecast_reference_time": "12:00Z",
        "fixed_available_at_lag_hours": 10,
        "variables": variables,
        "leadtime_hour": leads,
        "area": area,
        "bytes": byte_count,
        "sha256": digest,
        "retrieval": retrieval,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"done {target.name} {target.stat().st_size/1e6:.1f} MB", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=DEFAULT_EVIDENCE_ROOT / "manifests" / "open_evidence_v1.json")
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--stage-dir", type=Path,
                        default=Path.home() / ".cache" / "sitian" / "cams_stage",
                        help="fast local staging; completed NetCDF is atomically published to --root")
    parser.add_argument("--kind", choices=("aerosol", "trace_gases", "both"), default="both")
    parser.add_argument("--issue-date")
    parser.add_argument("--chunk-days", type=int,
                        help="override both kind-specific chunk sizes")
    parser.add_argument("--aerosol-chunk-days", type=int, default=32)
    parser.add_argument("--trace-gas-chunk-days", type=int, default=8)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    # ADS occasionally needs just over two hours to materialize an otherwise
    # healthy request.  A two-hour cutoff cancels that work and sends the same
    # block to the back of the queue, so keep a wider server-side deadline.
    parser.add_argument("--job-timeout-minutes", type=float, default=240)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require shard-count >= 1 and 0 <= shard-index < shard-count")
    if min(args.aerosol_chunk_days, args.trace_gas_chunk_days) < 1:
        parser.error("chunk sizes must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cycles = manifest["cams"]["cycles"]
    if args.issue_date:
        cycles = [row for row in cycles if row["issue_date"] == args.issue_date]
    reference_days = sorted({date.fromisoformat(row["cycle"][:10]) for row in cycles})
    area_def = manifest["domain"]
    area = [area_def["north"], area_def["west"], area_def["south"], area_def["east"]]
    definitions = {
        "aerosol": (manifest["cams"]["aerosol_variables"],
                    list(CAMS_AEROSOL_LEAD_HOURS)),
        "trace_gases": (manifest["cams"]["trace_gas_variables"],
                        list(CAMS_TRACE_GAS_LEAD_HOURS)),
    }
    kinds = ("aerosol", "trace_gases") if args.kind == "both" else (args.kind,)
    out_dir = args.root / "raw" / "cams"
    out_dir.mkdir(parents=True, exist_ok=True)
    missing_by_kind = {
        kind: [day for day in reference_days if day not in _covered_reference_days(out_dir, kind)]
        for kind in kinds
    }
    chunk_days = {
        "aerosol": args.chunk_days or args.aerosol_chunk_days,
        "trace_gases": args.chunk_days or args.trace_gas_chunk_days,
    }
    all_jobs = [(kind, d0, d1) for kind in kinds
                for d0, d1 in _consecutive_chunks(missing_by_kind[kind], chunk_days[kind])]
    jobs = all_jobs[args.shard_index::args.shard_count]
    print(json.dumps({"jobs": len(jobs), "all_jobs": len(all_jobs),
                      "reference_days": len(reference_days),
                      "missing_reference_days": {
                          kind: len(days) for kind, days in missing_by_kind.items()
                      }, "kinds": kinds, "area": area, "chunk_days": chunk_days,
                      "shard": [args.shard_index, args.shard_count],
                      "trace_gas_lead_hours": len(CAMS_TRACE_GAS_LEAD_HOURS)},
                     ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    if shutil.which("aria2c") is None:
        raise SystemExit("aria2c is required for resumable parallel CAMS result downloads")
    client = _client()
    failed = []
    for kind, d0, d1 in jobs:
        variables, leads = definitions[kind]
        try:
            _request(client, kind, d0, d1, list(variables), leads, area, out_dir,
                     args.stage_dir, args.job_timeout_minutes, args.poll_seconds)
        except Exception as exc:
            failed.append((kind, str(d0), str(d1), str(exc)[:200]))
            print(f"FAILED {kind} {d0}..{d1}: {exc}", flush=True)
    if failed:
        print(json.dumps({"failed": failed}, ensure_ascii=False), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
