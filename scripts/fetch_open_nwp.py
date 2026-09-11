#!/usr/bin/env python3
"""Range-download the fixed GFS/IFS synoptic subset in the evidence manifest.

Only complete GRIB messages for the registered fields are materialized.  Raw
files live on the external evidence disk and are keyed by source/cycle/step,
never duplicated by city.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from sitian.open_evidence import (
    DEFAULT_EVIDENCE_ROOT,
    EVIDENCE_VERSION,
    NWP_FIELDS,
    NWP_RADIATION_FIELDS,
    parse_utc,
)

IFS_ROOT = os.environ.get("SITIAN_IFS_ROOT", "https://storage.googleapis.com/ecmwf-open-data")
GFS_ROOT = os.environ.get("SITIAN_GFS_ROOT", "https://noaa-gfs-bdp-pds.s3.amazonaws.com")
RETRYABLE = {429, 500, 502, 503, 504}

GFS_NAME = {"gh": "HGT", "u": "UGRD", "v": "VGRD", "t": "TMP",
            "r": "RH", "w": "VVEL", "msl": "PRMSL", "dswrf": "DSWRF"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(session: requests.Session, url: str, *, headers: dict | None = None,
         timeout: tuple[int, int] = (15, 180)) -> requests.Response:
    for attempt in range(6):
        try:
            response = session.get(url, headers=headers, stream=True, timeout=timeout)
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(5 * (attempt + 1), 30))
            continue
        if response.status_code not in RETRYABLE:
            return response
        response.close()
        if attempt < 5:
            time.sleep(min(5 * (attempt + 1), 30))
    raise AssertionError("unreachable")


def _content(session: requests.Session, url: str, *, headers: dict | None = None,
             timeout: tuple[int, int] = (15, 180), expected: int | None = None) -> bytes:
    """Read a complete response, retrying disconnects that happen after headers."""
    for attempt in range(6):
        response = None
        try:
            response = _get(session, url, headers=headers, timeout=timeout)
            response.raise_for_status()
            body = response.content
            if expected is not None and len(body) != expected:
                raise RuntimeError(f"response length {len(body)} != {expected}")
            return body
        except (requests.RequestException, RuntimeError):
            if attempt == 5:
                raise
            time.sleep(min(5 * (attempt + 1), 30))
        finally:
            if response is not None:
                response.close()
    raise AssertionError("unreachable")


def _head_length(session: requests.Session, url: str) -> int | None:
    for attempt in range(6):
        try:
            response = session.head(url, timeout=(15, 60))
            if response.ok and response.headers.get("Content-Length"):
                return int(response.headers["Content-Length"])
            return None
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(5 * (attempt + 1), 30))
    return None


def _urls(source: str, cycle_text: str, step: int) -> tuple[str, str, str]:
    cycle = parse_utc(cycle_text)
    if source == "ifs":
        stamp = cycle.strftime("%Y%m%d%H%M%S")
        base = (f"{IFS_ROOT}/{cycle:%Y%m%d}/{cycle:%H}z/ifs/0p25/oper/"
                f"{stamp}-{step}h-oper-fc")
        return base + ".grib2", base + ".index", f"ifs_{cycle:%Y%m%d%H}_f{step:03d}.grib2"
    if source == "gfs":
        base = (f"{GFS_ROOT}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
                f"gfs.t{cycle:%H}z.pgrb2.0p25.f{step:03d}")
        return base, base + ".idx", f"gfs_{cycle:%Y%m%d%H}_f{step:03d}.grib2"
    raise ValueError(source)


def _product_fields(source: str, product: str) -> tuple[tuple[str, int | None], ...]:
    if product == "synoptic":
        return tuple(NWP_FIELDS)
    if product == "radiation":
        return (NWP_RADIATION_FIELDS[source],)
    raise ValueError(product)


def _ifs_ranges(text: str, step: int, wanted_fields=NWP_FIELDS) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    wanted = set(wanted_fields)
    selected: list[tuple[int, int]] = []
    identities: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        parameter = str(row.get("param"))
        level = (None if str(row.get("levtype")) == "sfc" or parameter == "msl"
                 else int(row.get("levelist", -1)))
        key = (parameter, level)
        if (str(row.get("type")) != "fc" or str(row.get("step")) != str(step)
                or key not in wanted):
            continue
        offset, length = int(row["_offset"]), int(row["_length"])
        selected.append((offset, length))
        identities.append({"parameter": parameter, "level_hpa": level,
                           "offset": offset, "length": length})
    if len(selected) != len(wanted):
        found = {(row["parameter"], row["level_hpa"]) for row in identities}
        raise RuntimeError(f"IFS index supplied {len(found)}/{len(wanted)} fields; missing={sorted(wanted-found, key=str)}")
    return sorted(selected), identities


def _gfs_ranges(text: str, content_length: int | None, wanted_fields=NWP_FIELDS) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    rows: list[tuple[int, str, str]] = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) >= 6:
            rows.append((int(parts[1]), parts[3], parts[4]))
    wanted = {
        (GFS_NAME[p],
         "mean sea level" if p == "msl" else "surface" if level is None else f"{level} mb")
        for p, level in wanted_fields
    }
    selected: list[tuple[int, int]] = []
    identities: list[dict[str, Any]] = []
    for index, (offset, parameter, level) in enumerate(rows):
        if (parameter, level) not in wanted:
            continue
        end = rows[index + 1][0] if index + 1 < len(rows) else content_length
        if end is None or end <= offset:
            raise RuntimeError(f"cannot determine GFS range length for {parameter}:{level}")
        selected.append((offset, end - offset))
        identities.append({"parameter": parameter, "level": level,
                           "offset": offset, "length": end - offset})
    found = {(row["parameter"], row["level"]) for row in identities}
    if len(found) != len(wanted):
        raise RuntimeError(f"GFS index supplied {len(found)}/{len(wanted)} fields; missing={sorted(wanted-found)}")
    return sorted(selected), identities


def _spans(ranges: list[tuple[int, int]], gap_limit: int = 256 * 1024,
           span_limit: int = 64 * 1024 * 1024) -> list[tuple[int, int]]:
    spans: list[list[int]] = []
    for offset, length in sorted(ranges):
        if spans:
            old_offset, old_length = spans[-1]
            gap = offset - (old_offset + old_length)
            end = max(old_offset + old_length, offset + length)
            if gap <= gap_limit and end - old_offset <= span_limit:
                spans[-1][1] = end - old_offset
                continue
        spans.append([offset, length])
    return [(offset, length) for offset, length in spans]


def _download_one(source: str, product: str, record: dict, root: Path,
                  verify: bool = False) -> dict:
    data_url, index_url, filename = _urls(source, record["cycle"], int(record["step_hour"]))
    cycle = parse_utc(record["cycle"])
    base = "nwp" if product == "synoptic" else "nwp_radiation"
    if product == "radiation":
        filename = filename.replace(".grib2", ".radiation.grib2")
    target = root / "raw" / base / source / f"{cycle:%Y%m%d%H}" / filename
    metadata = target.with_suffix(target.suffix + ".json")
    if target.is_file() and metadata.is_file():
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        if target.stat().st_size == meta.get("bytes") and (not verify or _sha256(target) == meta.get("sha256")):
            return {"status": "skip", "path": str(target), "bytes": target.stat().st_size}

    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers["User-Agent"] = "sitian-open-evidence/1"
        index_text = _content(session, index_url).decode("utf-8", "replace")
        wanted_fields = _product_fields(source, product)
        if source == "ifs":
            ranges, fields = _ifs_ranges(index_text, int(record["step_hour"]), wanted_fields)
        else:
            length = _head_length(session, data_url)
            ranges, fields = _gfs_ranges(index_text, length, wanted_fields)

        descriptor, temporary = tempfile.mkstemp(prefix=f".{filename}.", dir=target.parent)
        os.close(descriptor)
        candidate = Path(temporary)
        payloads: dict[tuple[int, int], bytes] = {}
        try:
            for span_offset, span_length in _spans(ranges):
                body = _content(
                    session, data_url,
                    headers={"Range": f"bytes={span_offset}-{span_offset + span_length - 1}"},
                    timeout=(15, 300), expected=span_length,
                )
                for identity in ranges:
                    offset, length = identity
                    if span_offset <= offset and offset + length <= span_offset + span_length:
                        start = offset - span_offset
                        payloads[identity] = body[start:start + length]
            if set(payloads) != set(ranges):
                raise RuntimeError(f"source omitted selected GRIB messages for {data_url}")
            with candidate.open("wb") as handle:
                for identity in ranges:
                    handle.write(payloads[identity])
            if candidate.read_bytes()[-4:] != b"7777":
                raise RuntimeError(f"downloaded GRIB has an incomplete tail: {data_url}")
            candidate.replace(target)
        finally:
            candidate.unlink(missing_ok=True)

    meta = {
        "contract_version": EVIDENCE_VERSION,
        "source": source,
        "product": product,
        "cycle": record["cycle"],
        "step_hour": int(record["step_hour"]),
        "valid_time": record["valid_time"],
        "available_at": record["available_at"],
        "source_url": data_url,
        "source_index_url": index_url,
        "fields": fields,
        "grib_messages": len(ranges),
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }
    metadata.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"status": "done", "path": str(target), "bytes": meta["bytes"]}


def _unique_records(manifest: dict, source: str, product: str,
                    issue_date: str | None) -> list[dict]:
    unique: dict[tuple[str, int], dict] = {}
    records = (manifest["nwp"][source] if product == "synoptic" else
               manifest["nwp_radiation"]["records"][source])
    for record in records:
        if issue_date and record["issue_date"] != issue_date:
            continue
        unique[(record["cycle"], int(record["step_hour"]))] = record
    return [unique[key] for key in sorted(unique)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=DEFAULT_EVIDENCE_ROOT / "manifests" / "open_evidence_v1.json")
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--source", choices=("gfs", "ifs", "both"), default="both")
    parser.add_argument("--product", choices=("synoptic", "radiation", "both"),
                        default="synoptic")
    parser.add_argument("--issue-date")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = ("gfs", "ifs") if args.source == "both" else (args.source,)
    products = ("synoptic", "radiation") if args.product == "both" else (args.product,)
    jobs = [
        (source, product, record)
        for source in sources
        for product in products
        for record in _unique_records(manifest, source, product, args.issue_date)
    ]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    print(json.dumps({"jobs": len(jobs), "sources": sources, "products": products,
                      "root": str(args.root)}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    counts = {"done": 0, "skip": 0, "failed": 0}
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_download_one, source, product, record, args.root, args.verify):
                (source, product, record)
            for source, product, record in jobs
        }
        for number, future in enumerate(as_completed(futures), 1):
            source, product, record = futures[future]
            try:
                result = future.result()
                counts[result["status"]] += 1
                total_bytes += result["bytes"]
                print(f"[{number}/{len(jobs)}] {result['status']} "
                      f"{result['bytes']/1e6:.2f} MB {result['path']}", flush=True)
            except Exception as exc:
                counts["failed"] += 1
                print(f"[{number}/{len(jobs)}] FAILED {source}/{product} "
                      f"{record['cycle']} f{record['step_hour']}: {exc}", flush=True)
    print(json.dumps({**counts, "materialized_gb": round(total_bytes / 1e9, 3)},
                     ensure_ascii=False), flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
