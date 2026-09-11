#!/usr/bin/env python3
"""Derive compact city terrain and anthropogenic-source priors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from sitian.open_evidence import DEFAULT_EVIDENCE_ROOT, EVIDENCE_VERSION

POLLUTANTS = ("PM2.5", "PM10", "NOx", "SO2", "CO", "NMVOC", "NH3", "BC", "OC")
DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry(lat: float, lon: float, latitudes: np.ndarray,
              longitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    p1, p2 = math.radians(lat), np.radians(lat_grid)
    dp, dl = p2 - p1, np.radians(lon_grid - lon)
    a = np.sin(dp / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    distance = 2 * 6371.0088 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    y = np.sin(dl) * np.cos(p2)
    x = math.cos(p1) * np.sin(p2) - math.sin(p1) * np.cos(p2) * np.cos(dl)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return distance, bearing


def _direction_masks(bearing: np.ndarray) -> dict[str, np.ndarray]:
    indexes = np.floor(((bearing + 22.5) % 360.0) / 45.0).astype(int)
    return {name: indexes == index for index, name in enumerate(DIRECTIONS)}


def _terrain(root: Path, coords: dict) -> tuple[dict, dict]:
    import cfgrib

    path = root / "raw" / "static" / "gfs_orography" / "gfs_0p25_orography_2025033100.grib2"
    datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    dataset = next(value for value in datasets if "orog" in value)
    latitudes = np.asarray(dataset.latitude)
    longitudes = np.asarray(dataset.longitude)
    lat_mask = (latitudes >= 10) & (latitudes <= 60)
    lon_mask = (longitudes >= 60) & (longitudes <= 150)
    lats, lons = latitudes[lat_mask], longitudes[lon_mask]
    values = np.asarray(dataset.orog.isel(latitude=np.where(lat_mask)[0],
                                          longitude=np.where(lon_mask)[0]).load(), dtype=float)
    output = {}
    for city, location in coords.items():
        if not isinstance(location, list):
            continue
        lat, lon = map(float, location)
        distance, bearing = _geometry(lat, lon, lats, lons)
        i, j = int(np.argmin(abs(lats - lat))), int(np.argmin(abs(lons - lon)))
        elevation = float(values[i, j])
        row = {"elevation_m": round(elevation), "radius": {}}
        for radius in (100, 300):
            sample = values[distance <= radius]
            row["radius"][str(radius)] = {
                "mean_elevation_m": round(float(np.nanmean(sample))),
                "max_elevation_m": round(float(np.nanmax(sample))),
                "relief_m": round(float(np.nanmax(sample) - np.nanmin(sample))),
            }
        sectors = {}
        annulus = (distance >= 30) & (distance <= 300)
        for name, mask in _direction_masks(bearing).items():
            sample = values[annulus & mask]
            sectors[name] = {"mean_elevation_m": round(float(np.nanmean(sample))),
                             "max_elevation_m": round(float(np.nanmax(sample))),
                             "barrier_above_city_m": round(max(0.0, float(np.nanmax(sample)) - elevation))}
        row["directional_terrain_30_300km"] = sectors
        row["basin_index_m"] = round(row["radius"]["300"]["mean_elevation_m"] - elevation)
        output[city] = row
    for value in datasets:
        value.close()
    meta_path = path.with_suffix(path.suffix + ".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return output, {"relative_path": str(path.relative_to(root)), "sha256": meta.get("sha256") or _sha256(path),
                    "bytes": path.stat().st_size}


def _sector_group(name: str) -> str:
    lower = name.lower()
    if "shipping" in lower:
        return "shipping"
    if "aviation" in lower:
        return "aviation"
    if "energy" in lower:
        return "energy"
    if any(value in lower for value in ("industry", "fugitive", "solvent")):
        return "industry"
    if any(value in lower for value in ("transport", "brake", "tyre")):
        return "ground_transport"
    if "residential" in lower:
        return "residential"
    if "waste" in lower:
        return "waste"
    if "agricultur" in lower:
        return "agriculture"
    return "other"


def _emissions(root: Path, coords: dict) -> tuple[dict, list[dict]]:
    by_city = {city: {} for city, value in coords.items() if isinstance(value, list)}
    provenance = []
    base = root / "raw" / "static" / "edgar_htap_v32" / "2020"
    for pollutant in POLLUTANTS:
        path = base / pollutant / f"edgar_HTAPv32_2020_{pollutant}.nc"
        if not path.is_file():
            continue
        with Dataset(path) as dataset:
            latitudes = np.asarray(dataset.variables["lat"][:], dtype=float)
            longitudes = np.asarray(dataset.variables["lon"][:], dtype=float)
            lat_mask = (latitudes >= 10) & (latitudes <= 60)
            lon_mask = (longitudes >= 60) & (longitudes <= 150)
            lats, lons = latitudes[lat_mask], longitudes[lon_mask]
            lat_indexes, lon_indexes = np.where(lat_mask)[0], np.where(lon_mask)[0]
            sectors: dict[str, np.ndarray] = {}
            for name, variable in dataset.variables.items():
                if name in ("lat", "lon"):
                    continue
                values = np.asarray(np.ma.filled(
                    variable[lat_indexes[0]:lat_indexes[-1] + 1,
                             lon_indexes[0]:lon_indexes[-1] + 1], 0.0), dtype=float)
                group = _sector_group(name)
                sectors[group] = sectors.get(group, np.zeros_like(values)) + values
            total = sum(sectors.values(), np.zeros((len(lats), len(lons)), dtype=float))
            for city, location in coords.items():
                if not isinstance(location, list):
                    continue
                distance, bearing = _geometry(float(location[0]), float(location[1]), lats, lons)
                radius = {str(km): round(float(np.nansum(total[distance <= km])), 2)
                          for km in (100, 300, 800)}
                within300 = {name: float(np.nansum(values[distance <= 300]))
                             for name, values in sectors.items()}
                sector_total = sum(within300.values())
                shares = {name: round(value / sector_total, 3) if sector_total > 0 else None
                          for name, value in sorted(within300.items())}
                annulus = (distance > 30) & (distance <= 800)
                directional = {name: round(float(np.nansum(total[annulus & mask])), 2)
                               for name, mask in _direction_masks(bearing).items()}
                by_city[city][pollutant] = {
                    "total_tonnes_per_year_by_radius_km": radius,
                    "sector_fraction_within_300km": shares,
                    "dominant_sectors": [name for name, value in
                                         sorted(within300.items(), key=lambda item: item[1], reverse=True)[:3]
                                         if value > 0],
                    "directional_total_30_800km_tonnes_per_year": directional,
                }
        archive = base / f"edgar_HTAPv32_2020_{pollutant}.zip"
        meta_path = archive.with_suffix(archive.suffix + ".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        provenance.append({"pollutant": pollutant, "relative_path": str(archive.relative_to(root)),
                           "sha256": meta.get("sha256") or _sha256(archive), "bytes": archive.stat().st_size})
    return by_city, provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--coords", type=Path, default=Path("data/interim/city_coords.json"))
    args = parser.parse_args()
    coords = json.loads(args.coords.read_text(encoding="utf-8"))
    terrain, terrain_raw = _terrain(args.root, coords)
    emissions, emissions_raw = _emissions(args.root, coords)
    cities = {city: {"terrain": terrain.get(city), "anthropogenic_emissions": emissions.get(city, {})}
              for city, value in coords.items() if isinstance(value, list)}
    document = {
        "contract_version": EVIDENCE_VERSION,
        "static_context": {
            "notes": {"terrain": "GFS 0.25-degree model orography; sufficient for synoptic barriers, not urban terrain.",
                      "emissions": "EDGAR HTAP v3.2 year-2020 emissions are a static source prior, not contemporaneous truth."},
            "cities": cities,
            "raw": {"terrain": terrain_raw, "emissions": emissions_raw},
        },
    }
    target = args.root / "derived" / "static_context.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"120-city static context -> {target} ({target.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
