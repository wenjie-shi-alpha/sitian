#!/usr/bin/env python3
"""Fetch small, globally reproducible terrain and anthropogenic-emission priors."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import requests

from sitian.open_evidence import DEFAULT_EVIDENCE_ROOT, EVIDENCE_VERSION

ETOPO_URL = (
    "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/"
    "60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc"
)
GFS_OROGRAPHY_DATA = (
    "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20250331/00/atmos/"
    "gfs.t00z.pgrb2.0p25.f000"
)
EDGAR_ROOT = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/htap_v32/"
    "gridmaps_05x05/emissions/annual"
)
EDGAR_POLLUTANTS = ("PM2.5", "PM10", "NOx", "SO2", "CO", "NMVOC", "NH3", "BC", "OC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, metadata: dict) -> None:
    sidecar = target.with_suffix(target.suffix + ".json")
    if target.is_file() and target.stat().st_size and sidecar.is_file():
        print(f"skip {target}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    for attempt in range(5):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 300)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                with partial.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_content(1 << 20):
                        if chunk:
                            handle.write(chunk)
            break
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    partial.replace(target)
    sidecar.write_text(json.dumps({"contract_version": EVIDENCE_VERSION, "source_url": url,
                                   "bytes": target.stat().st_size, "sha256": _sha256(target),
                                   **metadata}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"done {target} {target.stat().st_size/1e6:.1f} MB", flush=True)


def _gfs_orography(root: Path) -> None:
    target = root / "raw" / "static" / "gfs_orography" / "gfs_0p25_orography_2025033100.grib2"
    sidecar = target.with_suffix(target.suffix + ".json")
    if target.is_file() and target.stat().st_size and sidecar.is_file():
        print(f"skip {target}", flush=True)
        return
    index_url = GFS_OROGRAPHY_DATA + ".idx"
    index = requests.get(index_url, timeout=(20, 60))
    index.raise_for_status()
    rows = []
    for line in index.text.splitlines():
        parts = line.split(":")
        if len(parts) >= 6:
            rows.append((int(parts[1]), parts[3], parts[4]))
    selected = next((number for number, row in enumerate(rows)
                     if row[1] == "HGT" and row[2] == "surface"), None)
    if selected is None:
        raise RuntimeError("GFS index has no HGT:surface field")
    offset = rows[selected][0]
    end = rows[selected + 1][0] if selected + 1 < len(rows) else None
    if end is None:
        head = requests.head(GFS_OROGRAPHY_DATA, timeout=(20, 60))
        head.raise_for_status()
        end = int(head.headers["Content-Length"])
    response = requests.get(GFS_OROGRAPHY_DATA, headers={"Range": f"bytes={offset}-{end - 1}"},
                            timeout=(20, 300))
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != end - offset:
        raise RuntimeError("GFS orography range response is incomplete")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(response.content)
    partial.replace(target)
    sidecar.write_text(json.dumps({
        "contract_version": EVIDENCE_VERSION,
        "dataset": "NOAA GFS 0.25 degree surface geopotential height",
        "source_url": GFS_OROGRAPHY_DATA,
        "source_index_url": index_url,
        "cycle": "2025-03-31T00:00:00Z",
        "parameter": "HGT:surface",
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
        "use_note": "Static model-orography prior; not a time-varying forecast field.",
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"done {target} {target.stat().st_size/1e6:.1f} MB", flush=True)


def _safe_extract(archive: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            target = (directory / info.filename).resolve()
            if directory.resolve() not in target.parents and target != directory.resolve():
                raise RuntimeError(f"unsafe ZIP member {info.filename!r}")
        zipped.extractall(directory)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--kind", choices=("terrain", "emissions", "both"), default="both")
    parser.add_argument("--pollutant", choices=EDGAR_POLLUTANTS)
    parser.add_argument("--etopo", action="store_true",
                        help="also fetch the optional 491 MB ETOPO source; GFS orography is the default")
    args = parser.parse_args()
    if args.kind in ("terrain", "both"):
        _gfs_orography(args.root)
        if args.etopo:
            _download(ETOPO_URL, args.root / "raw" / "static" / "etopo2022" /
                      "ETOPO_2022_v1_60s_N90W180_bed.nc",
                      {"dataset": "NOAA NCEI ETOPO 2022 bedrock elevation",
                       "doi": "10.25921/fd45-gt74", "resolution": "60 arc-second"})
    if args.kind in ("emissions", "both"):
        pollutants = (args.pollutant,) if args.pollutant else EDGAR_POLLUTANTS
        base = args.root / "raw" / "static" / "edgar_htap_v32" / "2020"
        for pollutant in pollutants:
            filename = f"edgar_HTAPv32_2020_{pollutant}.zip"
            archive = base / filename
            _download(f"{EDGAR_ROOT}/{pollutant}/{filename}", archive,
                      {"dataset": "EDGAR HTAP v3.2 annual emissions",
                       "year": 2020, "resolution": "0.5x0.5 degree",
                       "pollutant": pollutant,
                       "use_note": "Static source prior, not a 2025-2026 emissions observation."})
            _safe_extract(archive, base / pollutant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
