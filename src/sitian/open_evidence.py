"""Open, forecast-time evidence contract for the national replay pool.

The raw fields are shared by issue date; city cases only receive deterministic
summaries derived from this store.  Every record carries both a valid time and
an availability time so a future-valid *forecast* is legal while a future
observation is not.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable

EVIDENCE_VERSION = "open-evidence-v1"
NWP_TEMPORAL_PROFILE_VERSION = "hybrid-6h72-12h132-v2"
DEFAULT_EVIDENCE_ROOT = Path(
    os.environ.get("SITIAN_EVIDENCE_ROOT", "/mnt/eaget/sitian_open_evidence")
)
UTC = timezone.utc

# Deliberately conservative fixed publication lags.  They are part of the
# contract rather than inferred from file mtimes during a replay.
SOURCE_LATENCY_HOURS = {
    "gfs": 5.0,
    "ifs": 8.0,
    "cams": 10.0,
    "firms_nrt": 6.0,
}

# Common synoptic wall.  CAMS already supplies the surface boundary-layer,
# precipitation, cloud and 10-m fields, so these are the pressure-level fields
# needed to reason about systems, stability, vertical motion and transport.
PRESSURE_LEVELS = (925, 850, 700, 500, 200)
# Approximate standard-atmosphere geopotential heights.  Pressure-level fields
# below a city's model orography are not physical near-surface evidence (a
# material issue for the Tibetan Plateau).  The compact process view and case
# attachments use the closest level that remains at least 150 m above terrain.
PRESSURE_LEVEL_APPROX_HEIGHT_M = {
    925: 762.0,
    850: 1457.0,
    700: 3012.0,
    500: 5574.0,
    200: 11784.0,
}
TERRAIN_CLEARANCE_M = 150.0
NWP_FIELDS = (
    ("gh", 500),
    *((component, level) for component in ("u", "v") for level in PRESSURE_LEVELS),
    *(("t", level) for level in (925, 850, 700)),
    *(("r", level) for level in (925, 850, 700)),
    *(("w", level) for level in (850, 700, 500)),
    ("msl", None),
)

# Forecast-process cadence: resolve the first three days at six-hourly
# intervals, then retain a twelve-hourly trajectory through day five.  The
# two past analyses stay separate and are not part of this offset tuple.
NWP_FORECAST_VALID_OFFSETS = (*range(0, 73, 6), 84, 96, 108, 120, 132)
NWP_ANALYSIS_VALID_OFFSETS = (-24, -12)
NWP_SNAPSHOTS_PER_ISSUE = (
    len(NWP_ANALYSIS_VALID_OFFSETS) + len(NWP_FORECAST_VALID_OFFSETS)
)

# Radiation is stored as a one-message sidecar rather than rewriting every
# existing 21-message synoptic GRIB.  GFS supplies interval-mean flux, while
# IFS supplies accumulated energy; the derivation step normalizes both to the
# preceding-period mean W m-2.
NWP_RADIATION_FIELDS = {
    "gfs": ("dswrf", None),
    "ifs": ("ssrd", None),
}
NWP_RADIATION_VALID_OFFSETS = {
    "gfs": tuple(range(0, 133, 6)),
    # IFS SSRD is cumulative from the cycle.  One legal pre-issue anchor lets
    # the current-time field also be differenced into an ending six-hour flux.
    "ifs": (-6, *range(0, 133, 6)),
}

CAMS_AEROSOL_VARIABLES = (
    "total_aerosol_optical_depth_550nm",
    "total_fine_mode_aerosol_optical_depth_550nm",
    "dust_aerosol_optical_depth_550nm",
    "organic_matter_aerosol_optical_depth_550nm",
    "black_carbon_aerosol_optical_depth_550nm",
    "sulphate_aerosol_optical_depth_550nm",
    "nitrate_aerosol_optical_depth_550nm",
)
CAMS_TRACE_GAS_VARIABLES = (
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
)
CAMS_AEROSOL_LEAD_HOURS = (12, 36, 60, 84, 108)
# Preserve the full six-hourly composition trajectory.  These are forecast
# fields already available at issue time, not future observations.
CAMS_TRACE_GAS_LEAD_HOURS = tuple(range(0, 121, 6))


def lowest_pressure_level_above_terrain(
    elevation_m: float | int | None,
    *,
    levels: tuple[int, ...] = (925, 850, 700, 500),
    clearance_m: float = TERRAIN_CLEARANCE_M,
) -> int | None:
    """Closest configured pressure surface safely above model terrain.

    ``levels`` must be ordered from the surface upward.  Returning ``None`` is
    preferable to silently treating an upper-tropospheric field as a boundary-
    layer proxy when no suitable level exists.
    """
    if elevation_m is None:
        return None
    try:
        threshold = float(elevation_m) + float(clearance_m)
    except (TypeError, ValueError):
        return None
    for level in levels:
        if PRESSURE_LEVEL_APPROX_HEIGHT_M[level] >= threshold:
            return level
    return None


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError(f"timestamp lacks UTC offset: {value!r}")
    return moment.astimezone(UTC)


def issue_asof(issue: str | date) -> datetime:
    """The national product is issued at 08:00 BJT, i.e. 00:00 UTC."""
    day = date.fromisoformat(issue) if isinstance(issue, str) else issue
    return datetime.combine(day, time(0), tzinfo=UTC)


def available_at(source: str, cycle_or_observation: datetime) -> datetime:
    try:
        lag = SOURCE_LATENCY_HOURS[source]
    except KeyError:
        raise ValueError(f"unknown evidence source {source!r}") from None
    return cycle_or_observation.astimezone(UTC) + timedelta(hours=lag)


def latest_legal_cycle(asof: datetime, source: str) -> datetime:
    """Latest 6-hourly cycle whose fixed publication lag is before ``asof``."""
    moment = asof.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    while moment.hour not in (0, 6, 12, 18):
        moment -= timedelta(hours=1)
    while available_at(source, moment) > asof:
        moment -= timedelta(hours=6)
    return moment


@dataclass(frozen=True)
class NwpRequest:
    source: str
    issue_date: str
    role: str
    cycle: datetime
    step_hour: int
    valid_time: datetime
    available_at: datetime

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "issue_date": self.issue_date,
            "role": self.role,
            "cycle": _iso(self.cycle),
            "step_hour": self.step_hour,
            "valid_time": _iso(self.valid_time),
            "available_at": _iso(self.available_at),
        }


def nwp_requests(issue: str | date, source: str) -> list[NwpRequest]:
    """Past analyses plus one legal forecast trajectory on common valid times."""
    if source not in ("gfs", "ifs"):
        raise ValueError("source must be 'gfs' or 'ifs'")
    asof = issue_asof(issue)
    issue_text = asof.date().isoformat()
    records: list[NwpRequest] = []

    # Two genuine past analyses.  Their publication is complete before issue.
    for offset in NWP_ANALYSIS_VALID_OFFSETS:
        valid = asof + timedelta(hours=offset)
        records.append(NwpRequest(
            source=source,
            issue_date=issue_text,
            role=f"analysis{offset:+d}h",
            cycle=valid,
            step_hour=0,
            valid_time=valid,
            available_at=available_at(source, valid),
        ))

    cycle = latest_legal_cycle(asof, source)
    for offset in NWP_FORECAST_VALID_OFFSETS:
        valid = asof + timedelta(hours=offset)
        step = int((valid - cycle).total_seconds() // 3600)
        records.append(NwpRequest(
            source=source,
            issue_date=issue_text,
            role="current" if offset == 0 else f"forecast+{offset}h",
            cycle=cycle,
            step_hour=step,
            valid_time=valid,
            available_at=available_at(source, cycle),
        ))
    return records


def nwp_radiation_requests(issue: str | date, source: str) -> list[NwpRequest]:
    """Six-hourly radiation trajectory used to normalize both NWP sources."""
    if source not in ("gfs", "ifs"):
        raise ValueError("source must be 'gfs' or 'ifs'")
    asof = issue_asof(issue)
    issue_text = asof.date().isoformat()
    cycle = latest_legal_cycle(asof, source)
    records = []
    for offset in NWP_RADIATION_VALID_OFFSETS[source]:
        valid = asof + timedelta(hours=offset)
        records.append(NwpRequest(
            source=source,
            issue_date=issue_text,
            role=("normalization_anchor" if offset < 0 else
                  "current" if offset == 0 else f"forecast+{offset}h"),
            cycle=cycle,
            step_hour=int((valid - cycle).total_seconds() // 3600),
            valid_time=valid,
            available_at=available_at(source, cycle),
        ))
    return records


def issue_dates_from_cases(cases_root: str | Path) -> list[str]:
    dates: set[str] = set()
    for path in Path(cases_root).glob("*/*/case.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))["issue_date"]
            dates.add(date.fromisoformat(value).isoformat())
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return sorted(dates)


def build_download_manifest(issue_dates: Iterable[str]) -> dict:
    dates = sorted({date.fromisoformat(value).isoformat() for value in issue_dates})
    nwp = {
        source: [record.to_dict() for day in dates for record in nwp_requests(day, source)]
        for source in ("gfs", "ifs")
    }
    nwp_radiation_records = {
        source: [record.to_dict() for day in dates
                 for record in nwp_radiation_requests(day, source)]
        for source in ("gfs", "ifs")
    }
    cams_cycles = []
    for value in dates:
        asof = issue_asof(value)
        cycle = asof - timedelta(hours=12)
        cams_cycles.append({
            "issue_date": value,
            "cycle": _iso(cycle),
            "available_at": _iso(available_at("cams", cycle)),
            "aerosol_lead_hours": list(CAMS_AEROSOL_LEAD_HOURS),
            "trace_gas_lead_hours": list(CAMS_TRACE_GAS_LEAD_HOURS),
        })
    manifest = {
        "contract_version": EVIDENCE_VERSION,
        "generated_at": _iso(datetime.now(UTC)),
        "issue_time": "08:00 Asia/Shanghai (00:00 UTC)",
        "issue_dates": dates,
        "domain": {"north": 60.0, "west": 60.0, "south": 10.0, "east": 150.0},
        "nwp_temporal_profile": {
            "version": NWP_TEMPORAL_PROFILE_VERSION,
            "analysis_valid_offsets_hours": list(NWP_ANALYSIS_VALID_OFFSETS),
            "forecast_valid_offsets_hours": list(NWP_FORECAST_VALID_OFFSETS),
            "rationale": "six-hourly through +72h, twelve-hourly through +132h",
        },
        "nwp_fields": [{"parameter": p, "level_hpa": level} for p, level in NWP_FIELDS],
        "nwp_radiation": {
            "forecast_only": True,
            "canonical_parameter": "surface_downward_shortwave_radiation",
            "canonical_units": "W m-2",
            "source_parameters": {
                source: {"parameter": field[0], "level_hpa": field[1]}
                for source, field in NWP_RADIATION_FIELDS.items()
            },
            "valid_offsets_hours": {
                source: list(offsets)
                for source, offsets in NWP_RADIATION_VALID_OFFSETS.items()
            },
            "records": nwp_radiation_records,
            "normalization": (
                "Six-hourly GFS interval mean used directly; six-hourly IFS accumulated "
                "energy differenced and divided by elapsed seconds"
            ),
        },
        "nwp": nwp,
        "cams": {
            "cycles": cams_cycles,
            "aerosol_variables": list(CAMS_AEROSOL_VARIABLES),
            "trace_gas_variables": list(CAMS_TRACE_GAS_VARIABLES),
        },
        "firms": {
            "lookback_hours": 72,
            "latency_hours": SOURCE_LATENCY_HOURS["firms_nrt"],
            "sensors": ["VIIRS_NOAA20", "VIIRS_NOAA21"],
            "standard_processing_note": (
                "Archived daily NRT is preferred for as-of replay. Historical monthly "
                "science-quality locations are used only as explicitly labelled "
                "retrospective proxy fallbacks for sensor-days lacking NRT; they preserve "
                "acquisition time but are not asserted to be byte-identical to the NRT feed."
            ),
        },
    }
    assert_manifest_legal(manifest)
    return manifest


def assert_manifest_legal(manifest: dict) -> None:
    for source, records in manifest.get("nwp", {}).items():
        for record in records:
            cutoff = issue_asof(record["issue_date"])
            if parse_utc(record["available_at"]) > cutoff:
                raise ValueError(f"{source} record is unavailable at issue: {record}")
            # Positive valid times are legal only because this is a forecast.
            if record["step_hour"] < 0:
                raise ValueError(f"negative NWP forecast step: {record}")
    for record in manifest.get("cams", {}).get("cycles", []):
        if parse_utc(record["available_at"]) > issue_asof(record["issue_date"]):
            raise ValueError(f"CAMS cycle is unavailable at issue: {record}")
    for source, records in manifest.get("nwp_radiation", {}).get("records", {}).items():
        for record in records:
            if parse_utc(record["available_at"]) > issue_asof(record["issue_date"]):
                raise ValueError(f"{source} radiation is unavailable at issue: {record}")
            if record["step_hour"] < 0:
                raise ValueError(f"negative NWP radiation forecast step: {record}")
