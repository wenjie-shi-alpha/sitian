import importlib.util
import gzip
import json
from datetime import timedelta
from pathlib import Path

from sitian.open_evidence import (
    CAMS_AEROSOL_LEAD_HOURS,
    CAMS_TRACE_GAS_LEAD_HOURS,
    EVIDENCE_VERSION,
    NWP_FORECAST_VALID_OFFSETS,
    NWP_SNAPSHOTS_PER_ISSUE,
    NWP_TEMPORAL_PROFILE_VERSION,
    assert_manifest_legal,
    build_download_manifest,
    issue_asof,
    lowest_pressure_level_above_terrain,
    nwp_radiation_requests,
    nwp_requests,
)
from sitian.case import CaseBundle
from scripts.audit_expert_evidence_contract import _composition_complete
from scripts.attach_open_evidence import _synoptic


def _fire_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "derive_open_fires.py"
    spec = importlib.util.spec_from_file_location("derive_open_fires_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _coverage_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_open_evidence_coverage.py"
    spec = importlib.util.spec_from_file_location("audit_open_evidence_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synoptic_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "derive_open_synoptic.py"
    spec = importlib.util.spec_from_file_location("derive_open_synoptic_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_cycles_are_independent_and_legal():
    day = "2025-12-19"
    gfs = nwp_requests(day, "gfs")
    ifs = nwp_requests(day, "ifs")
    assert gfs[2].cycle == issue_asof(day) - timedelta(hours=6)
    assert ifs[2].cycle == issue_asof(day) - timedelta(hours=12)
    assert all(row.available_at <= issue_asof(day) for row in gfs + ifs)
    assert [row.valid_time for row in gfs] == [row.valid_time for row in ifs]
    assert len(gfs) == len(ifs) == NWP_SNAPSHOTS_PER_ISSUE == 20
    forecast_offsets = [
        int((row.valid_time - issue_asof(day)).total_seconds() // 3600)
        for row in gfs[2:]
    ]
    assert forecast_offsets == list(NWP_FORECAST_VALID_OFFSETS)
    assert forecast_offsets == [0, 6, 12, 18, 24, 30, 36, 42, 48,
                                54, 60, 66, 72, 84, 96, 108, 120, 132]
    assert gfs[-1].step_hour == 138
    assert ifs[-1].step_hour == 144


def test_transport_pressure_level_is_terrain_adaptive():
    assert lowest_pressure_level_above_terrain(267) == 925
    assert lowest_pressure_level_above_terrain(1982) == 700
    assert lowest_pressure_level_above_terrain(2643) == 700
    assert lowest_pressure_level_above_terrain(4247) == 500


def test_case_attachment_masks_below_ground_pressure_surfaces():
    def row(source, u500):
        return {
            "source": source, "available": True,
            "valid_time": "2026-01-01T00:00:00Z",
            "cities": {"拉萨": {
                "wind": {str(level): {"u_ms": u500 if level == 500 else 1.0,
                                       "v_ms": 0.0, "speed_ms": 1.0,
                                       "from_deg": 270.0}
                         for level in (925, 850, 700, 500, 200)},
                "temperature_c": {"925": 1.0, "850": 0.0, "700": -5.0},
                "relative_humidity_pct": {"925": 50.0, "850": 40.0, "700": 30.0},
                "omega_pa_s": {"850": 0.1, "700": 0.0, "500": -0.1},
                "stability": {"t925_minus_t850_c": 1.0,
                              "t850_minus_t700_c": 5.0,
                              "low_level_inversion_signal": False},
            }},
        }

    document = {"synoptic": {
        "sources": {"gfs": [row("gfs", 3.0)], "ifs": [row("ifs", 5.0)]},
        "cross_model": {"2026-01-01T00:00:00Z": {"拉萨": {
            "wind925_vector_diff_ms": 99.0,
        }}},
    }}
    attached = _synoptic(document, "拉萨", 4247)
    city = attached["sources"]["gfs"][0]["cities"]["拉萨"]
    assert city["transport_pressure_level_hpa"] == 500
    assert city["wind"]["925"] is None
    assert city["wind"]["700"] is None
    assert city["wind"]["500"]["u_ms"] == 3.0
    assert city["omega_pa_s"]["500"] == -0.1
    disagreement = attached["cross_model"]["2026-01-01T00:00:00Z"]["拉萨"]
    assert disagreement["wind925_vector_diff_ms"] is None
    assert disagreement["transport_wind_vector_diff_ms"] == 2.0


def test_manifest_has_past_current_future_without_truth():
    manifest = build_download_manifest(["2025-12-19", "2026-08-21"])
    assert manifest["contract_version"] == EVIDENCE_VERSION
    assert len(manifest["nwp"]["gfs"]) == 40
    assert len(manifest["nwp"]["ifs"]) == 40
    assert manifest["nwp_temporal_profile"]["version"] == NWP_TEMPORAL_PROFILE_VERSION
    assert manifest["nwp_radiation"]["canonical_units"] == "W m-2"
    assert len(manifest["nwp_radiation"]["records"]["gfs"]) == 46
    assert len(manifest["nwp_radiation"]["records"]["ifs"]) == 48
    assert "truth" not in str(manifest).lower()
    assert manifest["cams"]["cycles"][0]["trace_gas_lead_hours"] == list(
        CAMS_TRACE_GAS_LEAD_HOURS
    )
    assert len(CAMS_TRACE_GAS_LEAD_HOURS) == 21
    assert_manifest_legal(manifest)


def test_radiation_contract_is_six_hourly_and_ifs_has_legal_anchor():
    day = "2025-12-19"
    gfs = nwp_radiation_requests(day, "gfs")
    ifs = nwp_radiation_requests(day, "ifs")
    assert len(gfs) == 23
    assert len(ifs) == 24
    assert ifs[0].role == "normalization_anchor"
    assert ifs[0].valid_time == issue_asof(day) - timedelta(hours=6)
    assert ifs[0].step_hour == 6
    assert all(row.available_at <= issue_asof(day) for row in gfs + ifs)


def test_radiation_normalization_is_comparable_period_mean_flux():
    synoptic = _synoptic_module()
    gfs = [
        {"step_hour": 6, "radiation_available": True,
         "_radiation_city_raw": {"target": 100.0},
         "_radiation_metadata": {"units": "W m**-2", "start_step": 0, "end_step": 6},
         "cities": {"target": {}}},
        {"step_hour": 12, "radiation_available": True,
         "_radiation_city_raw": {"target": 200.0},
         "_radiation_metadata": {"units": "W m**-2", "start_step": 6, "end_step": 12},
         "cities": {"target": {}}},
    ]
    ifs = [
        {"step_hour": 12, "radiation_available": True,
         "_radiation_city_raw": {"target": 12 * 3600 * 100.0},
         "_radiation_metadata": {"units": "J m**-2", "start_step": 0, "end_step": 12},
         "cities": {"target": {}}},
        {"step_hour": 18, "radiation_available": True,
         "_radiation_city_raw": {"target": 12 * 3600 * 100.0 + 6 * 3600 * 200.0},
         "_radiation_metadata": {"units": "J m**-2", "start_step": 0, "end_step": 18},
         "cities": {"target": {}}},
    ]
    synoptic._normalize_radiation(gfs, "gfs")
    synoptic._normalize_radiation(ifs, "ifs")
    assert [row["cities"]["target"]["surface_downward_shortwave_w_m2"] for row in gfs] == [100.0, 200.0]
    assert [row["cities"]["target"]["surface_downward_shortwave_w_m2"] for row in ifs] == [100.0, 200.0]


def test_nrt_fire_parser_rejects_low_confidence_and_understands_iso_time():
    fires = _fire_module()
    row = {"latitude": "39.9", "longitude": "116.4", "acq_date": "2026-07-06",
           "acq_time": "18:00", "confidence": "high", "frp": "12.5"}
    acquired, lat, lon, frp = fires._row_values(row, "near_real_time")
    assert acquired.isoformat() == "2026-07-06T18:00:00+00:00"
    assert (lat, lon, frp) == (39.9, 116.4, 12.5)
    row["confidence"] = "low"
    assert fires._row_values(row, "near_real_time") is None


def test_nrt_fire_visibility_obeys_six_hour_cutoff(tmp_path):
    fires = _fire_module()
    directory = tmp_path / "raw" / "firms" / "nrt_daily" / "noaa20"
    directory.mkdir(parents=True)
    path = directory / "J1_VIIRS_C2_Global_VJ114IMGTDL_NRT_2026187.txt"
    path.write_text(
        "latitude,longitude,acq_date,acq_time,confidence,frp\n"
        "39.9,116.4,2026-07-06,18:00,high,12.5\n"
        "39.9,116.4,2026-07-06,18:01,high,99.0\n"
        "39.9,116.4,2026-07-06,17:59,low,99.0\n",
        encoding="utf-8",
    )
    path.with_suffix(".txt.json").write_text(json.dumps({
        "processing": "near_real_time", "bytes": path.stat().st_size,
    }), encoding="utf-8")
    rows, provenance = fires._read_visible(
        tmp_path, issue_asof("2026-07-07"),
        {"south": 10, "north": 60, "west": 60, "east": 150},
        latency_hours=6, lookback_hours=72,
    )
    assert len(rows) == 1
    assert rows[0]["acquired"].isoformat() == "2026-07-06T18:00:00+00:00"
    assert provenance[0]["processing"] == "near_real_time"
    assert provenance[0]["asof_role"] == "operational_nrt"


def test_nrt_fire_precedes_retrospective_standard_on_same_sensor_day(tmp_path):
    fires = _fire_module()
    nrt_dir = tmp_path / "raw" / "firms" / "nrt_daily" / "noaa20"
    nrt_dir.mkdir(parents=True)
    nrt = nrt_dir / "J1_VIIRS_C2_Global_VJ114IMGTDL_NRT_2026187.txt"
    nrt.write_text(
        "latitude,longitude,acq_date,acq_time,confidence,frp\n"
        "39.9,116.4,2026-07-06,18:00,high,12.5\n",
        encoding="utf-8",
    )
    nrt.with_suffix(".txt.json").write_text(json.dumps({
        "processing": "near_real_time",
        "acquisition_day": "2026-07-06",
        "bytes": nrt.stat().st_size,
    }), encoding="utf-8")

    standard_dir = tmp_path / "raw" / "firms" / "standard_monthly" / "noaa20"
    standard_dir.mkdir(parents=True)
    standard = standard_dir / "VJ114IMGML.202607.C2.04.csv.gz"
    with gzip.open(standard, "wt", encoding="utf-8") as handle:
        handle.write(
            "YYYYMMDD,HHMM,Lat,Lon,FRP,Type\n"
            "20260706,1800,40.1,116.6,999.0,0\n"
        )
    standard.with_suffix(".gz.json").write_text(json.dumps({
        "processing": "standard_science_quality", "bytes": standard.stat().st_size,
    }), encoding="utf-8")

    rows, provenance = fires._read_visible(
        tmp_path, issue_asof("2026-07-07"),
        {"south": 10, "north": 60, "west": 60, "east": 150},
        latency_hours=6, lookback_hours=72,
    )
    assert len(rows) == 1
    assert rows[0]["processing"] == "near_real_time"
    assert rows[0]["frp_mw"] == 12.5
    assert {item["asof_role"] for item in provenance} == {
        "operational_nrt", "retrospective_proxy_fallback",
    }


def test_trace_gas_coverage_requires_complete_issue_legal_trajectory(tmp_path):
    coverage = _coverage_module()
    day = "2025-04-01"
    raw = tmp_path / "raw" / "cams" / "trace.nc"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"netcdf")
    raw.with_suffix(".nc.json").write_text(json.dumps({"bytes": raw.stat().st_size}))
    cycle = issue_asof(day) - timedelta(hours=12)
    records = []
    for lead in CAMS_TRACE_GAS_LEAD_HOURS:
        records.append({
            "lead_hour": lead,
            "issue_relative_hour": lead - 12,
            "valid_time": (cycle + timedelta(hours=lead)).isoformat().replace("+00:00", "Z"),
            "available_at": (cycle + timedelta(hours=10)).isoformat().replace("+00:00", "Z"),
            "cities": {"target": {
                "carbon_monoxide_ppbv_approx": 100.0,
                "nitrogen_dioxide_ppbv_approx": 10.0,
                "sulphur_dioxide_ppbv_approx": 1.0,
            }},
        })
    derived = tmp_path / "derived" / "by_issue" / f"{day}.composition.json"
    derived.parent.mkdir(parents=True)
    document = {"pollution": {"composition": {"trace_gases": {
        "available": True,
        "cycle": cycle.isoformat().replace("+00:00", "Z"),
        "records": records,
        "raw": {"relative_path": "raw/cams/trace.nc"},
    }}}}
    derived.write_text(json.dumps(document))

    assert coverage._trace_gas_contract(tmp_path, [day])["fraction"] == 1.0
    document["pollution"]["composition"]["trace_gases"]["records"].pop()
    derived.write_text(json.dumps(document))
    result = coverage._trace_gas_contract(tmp_path, [day])
    assert result["fraction"] == 0.0
    assert result["invalid_samples"][0]["reasons"] == ["lead_hours"]


def test_aerosol_coverage_requires_values_timing_and_raw_provenance(tmp_path):
    coverage = _coverage_module()
    day = "2025-04-01"
    raw = tmp_path / "raw" / "cams" / "aerosol.nc"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"netcdf")
    raw.with_suffix(".nc.json").write_text(json.dumps({"bytes": raw.stat().st_size}))
    cycle = issue_asof(day) - timedelta(hours=12)
    records = [{
        "lead_hour": lead,
        "issue_relative_hour": lead - 12,
        "valid_time": (cycle + timedelta(hours=lead)).isoformat().replace("+00:00", "Z"),
        "available_at": (cycle + timedelta(hours=10)).isoformat().replace("+00:00", "Z"),
        "cities": {"target": {"aod550": {
            "total": 0.2, "fine": 0.18, "dust": 0.01, "organic_matter": 0.04,
            "black_carbon": 0.01, "sulphate": 0.06, "nitrate": 0.04,
        }}},
    } for lead in CAMS_AEROSOL_LEAD_HOURS]
    derived = tmp_path / "derived" / "by_issue" / f"{day}.composition.json"
    derived.parent.mkdir(parents=True)
    document = {"pollution": {"composition": {"aerosol": {
        "available": True,
        "cycle": cycle.isoformat().replace("+00:00", "Z"),
        "records": records,
        "raw": {"relative_path": "raw/cams/aerosol.nc"},
    }}}}
    derived.write_text(json.dumps(document))
    paths = tuple(("aod550", name) for name in (
        "total", "fine", "dust", "organic_matter", "black_carbon", "sulphate", "nitrate",
    ))

    assert coverage._composition_contract(
        tmp_path, [day], name="aerosol",
        expected_leads=CAMS_AEROSOL_LEAD_HOURS, value_paths=paths,
    )["fraction"] == 1.0
    document["pollution"]["composition"]["aerosol"]["records"][0]["cities"]["target"][
        "aod550"
    ]["dust"] = -1.0
    derived.write_text(json.dumps(document))
    result = coverage._composition_contract(
        tmp_path, [day], name="aerosol",
        expected_leads=CAMS_AEROSOL_LEAD_HOURS, value_paths=paths,
    )
    assert result["fraction"] == 0.0
    assert result["invalid_samples"][0]["reasons"] == ["scientific_values"]


def test_expert_contract_requires_all_three_composition_trajectories():
    bundle = CaseBundle("case", "2025-12-19", "target", 5)
    composition = {
        "aerosol": {
            "available": True,
            "records": [{
                "lead_hour": lead,
                "issue_relative_hour": lead - 12,
                "cities": {"target": {"aod550": {"total": 0.2},
                                       "dominant_component": "dust"}},
            } for lead in CAMS_AEROSOL_LEAD_HOURS],
        },
        "trace_gases": {
            "available": True,
            "records": [{
                "lead_hour": lead,
                "issue_relative_hour": lead - 12,
                "cities": {"target": {
                    "carbon_monoxide_ppbv_approx": 100.0,
                    "nitrogen_dioxide_ppbv_approx": 10.0,
                    "sulphur_dioxide_ppbv_approx": 1.0,
                }},
            } for lead in CAMS_TRACE_GAS_LEAD_HOURS],
        },
        "column_gases": {
            "available": True,
            "records": [{
                "lead_hour": lead,
                "issue_relative_hour": lead - 12,
                "cities": {"target": {
                    "carbon_monoxide_kg_m2": 0.001,
                    "nitrogen_dioxide_kg_m2": 0.00001,
                    "sulphur_dioxide_kg_m2": 0.000001,
                }},
            } for lead in CAMS_TRACE_GAS_LEAD_HOURS],
        },
    }

    assert _composition_complete(bundle, composition)
    composition["trace_gases"]["records"].pop()
    assert not _composition_complete(bundle, composition)
