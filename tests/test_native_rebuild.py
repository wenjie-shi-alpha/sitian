import gzip
import json
from copy import deepcopy
from datetime import timedelta

import pytest

from sitian.case import CaseBundle
from sitian.data_contract import issue_time
from sitian.env import EnvConfig, ForecastEnv
from sitian.native_data import NativeCams, NativeObservations, VARIABLES, normalize_series, parse_concentration
from sitian.native_meteorology import diffusion_conditions, native_weather
from sitian.scoring import _semantic_evidence_match


def write_gz(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")


def native_row(var="blh", values=None, step=6):
    cycle = issue_time("2025-04-01")-timedelta(hours=12)
    leads = list(range(0, 121, step))
    values = values if values is not None else [100.0]*len(leads)
    if var == "go3":
        values = [[v] for v in values]
    dims = ["forecast_period", "model_level"] if var == "go3" else ["forecast_period"]
    return {"canonical_variable": var, "city": "test", "issue_date": "2025-04-01",
            "unit": VARIABLES[var][0], "cycle": cycle.isoformat(), "lead_hours": leads,
            "valid_times": [(cycle+timedelta(hours=h)).isoformat() for h in leads],
            "dimensions": dims, "lead_dimension": "forecast_period", "values": values,
            "coordinates": {"forecast_period": {"values": leads, "unit": "hours"},
                            **({"model_level": {"values": [137.0], "unit": "1"}} if var == "go3" else {})},
            "source_sha256": "source", "missing_values": sum(v is None for v in (values if var != "go3" else [v[0] for v in values])),
            "selected_grid": {"lat": 40, "lon": 116}, "grib_step_type": "instant"}


def native_case(tmp_path):
    rows = []
    constants = dict(pm2p5=50e-9, pm10=80e-9, go3=1e-7, u10=3, v10=4, t2m=293.15,
                     d2m=283.15, blh=100, tp=.006, tcc=.5)
    for var, value in constants.items():
        step = 3 if var == "go3" else 6
        rows.append(native_row(var, [value]*len(range(0, 121, step)), step))
    path = tmp_path / "cams.gz"
    write_gz(path, rows)
    cams = NativeCams(path, {"source": {}})
    b = CaseBundle("test_2025-04-01", "2025-04-01", "test", 5)
    source, b.diagnostics = cams.build("test", b.issue_date, b.forecast_dates())
    b.guidance = {"sources": {"cams": source}}
    return b


def test_native_numerics_and_daily_coverage(tmp_path):
    b = native_case(tmp_path)
    s = b.guidance["sources"]["cams"]
    assert list(s["daily_pm25"]) == ["2025-04-02", "2025-04-03", "2025-04-04"]
    assert set(s["partial_daily_pm25"]) == {"2025-04-05"}
    assert s["daily_pm25"]["2025-04-02"] == 50
    assert s["daily_o3max"]["2025-04-02"] == 120
    assert s["measurement_contracts"]["daily_o3max"]["statistic"] != "daily_max_8h_mean"
    d = b.diagnostics["daily"]["2025-04-02"]
    assert d["wind_speed_ms"] == 5
    assert d["tmax_c"] == 20
    assert d["rain_mm"] is None
    assert d["blh_min_m"] == d["blh_night_min_m"] == 100
    assert native_weather(b, start_hour=120, end_hour=144)["available"] is False


@pytest.mark.parametrize("change", ["unit", "level", "mask", "time", "shape"])
def test_native_contract_rejects_ambiguous_data(change):
    r = native_row("go3", [1e-7]*41, step=3)
    if change == "unit": r["unit"] = "ug m-3"
    if change == "level": r["coordinates"]["model_level"]["values"] = [1.0]
    if change == "mask": r["missing_values"] = 1
    if change == "time": r["valid_times"][1] = r["valid_times"][0]
    if change == "shape": r["values"].pop()
    with pytest.raises(ValueError): normalize_series(r)


def test_overlap_never_splices_or_silently_ignores_conflicts(tmp_path):
    a, b = native_row(step=6), native_row(step=3)
    b["source_sha256"] = "other"
    path = tmp_path/"overlap.gz"
    write_gz(path, [a,b])
    index = NativeCams(path, {"source": {}, "other": {}})
    assert index.rows[("test", "2025-04-01")]["blh_m"]["source_sha256"] == "other"
    b["values"][2] = 999
    write_gz(path, [a,b])
    with pytest.raises(ValueError, match="conflicting"): NativeCams(path, {"source": {}, "other": {}})


def test_diffusion_pairing_gap_and_grounding(tmp_path):
    b = native_case(tmp_path)
    fields = b.diagnostics["native"]["cams"]["fields"]
    fields["blh_m"]["values"][4] = None  # issue+12h, breaks a span
    result = diffusion_conditions(b, start_hour=0, end_hour=30, weak_wind_ms=6)
    assert len(result["samples"]) == 4
    assert [r["sample_span_hours"] for r in result["matched_sample_spans"]] == [6, 6]
    assert all(r["ventilation_proxy_m2_s"] == 500 for r in result["samples"])
    assert all(not r["thresholds_met"] for r in diffusion_conditions(b, weak_wind_ms=5)["samples"])
    env = ForecastEnv(b, EnvConfig(guidance_bias_path=None))
    env.reset()
    out, _, _, _ = env.step({"name": "compute_diffusion_conditions", "args": {}})
    content = out["content"]
    registry = {content["evidence_ref"]: {"tool": "compute_diffusion_conditions", "content": content}}
    base = {"type": "diagnostic", "ref": content["evidence_ref"]}
    assert _semantic_evidence_match({**base, "field": "/samples/0/ventilation_proxy_m2_s", "value": 500}, registry, {"compute_diffusion_conditions"})
    assert not _semantic_evidence_match({**base, "field": "/parameters/weak_wind_ms", "value": 2}, registry, {"compute_diffusion_conditions"})


def test_raw_observation_time_window_and_truth_validity(tmp_path):
    cutoff = issue_time("2025-04-01")
    rows, sources = [], {}
    for h in range(-72, 48):
        t = cutoff+timedelta(hours=h)
        sha = t.strftime("%Y%m%d")
        sources[sha] = {"path": f"china_cities_{sha}.csv"}
        for pol in ("PM2.5", "PM10", "O3", "SO2", "NO2", "CO", "O3_8h"):
            rows.append({"date": sha, "hour": str(t.hour), "type": pol, "source_sha256": sha,
                         "city_values": {"test": "0" if h < 0 else "1234567"}})
    path = tmp_path/"obs.gz"
    write_gz(path, rows)
    obs = NativeObservations(path, sources)
    inputs = obs.inputs("test", "2025-04-01")
    assert len(inputs["PM2.5"]["times"]) == 71
    assert inputs["PM2.5"]["times"][-1].endswith("06:00:00+08:00")
    assert set(inputs["PM2.5"]["series"]["test"]) == {0}
    assert obs.spatial(["test"], "2025-04-01")["pollution"]["spatial_observations"]["cities"]["test"]["PM2.5"]["n_24h"] == 23
    assert obs.truth("test", ["2025-04-02"])["daily"]["2025-04-02"]["pm25_avg"] == 1234567
    for hour in range(5):
        obs.rows[("2025-04-02", hour, "PM2.5")]["city_values"]["test"] = "-1"
    with pytest.raises(ValueError, match="19 hours"): obs.truth("test", ["2025-04-02"])
    b = CaseBundle("test", "2025-04-01", "test", 5, observations=deepcopy(inputs))
    assert not b.audit_time_gate()
    b.observations["PM2.5"]["sample_available_at"][-1] = cutoff.isoformat()
    assert b.audit_time_gate()


@pytest.mark.parametrize("bad", [None, "nan", "-1", "inf", True, ""])
def test_invalid_native_concentrations(bad):
    assert parse_concentration(bad) is None


def test_large_evidence_has_actionable_narrowing_and_complete_json(tmp_path):
    b = native_case(tmp_path)
    b.evidence = {"pollution": {"composition": {"aerosol": {"records": [
        {"aod": .3, "description": "x"*1500} for _ in range(40)]}}}}
    env = ForecastEnv(b, EnvConfig(guidance_bias_path=None))
    env.reset()
    result, *_ = env.step({"name": "get_pollution_evidence", "args": {"detail":"full"}})
    assert result["content"]["reason"] == "response_requires_narrower_query"
    assert "evidence_ref" not in result["content"]
    assert env.tools_called == set()
    result, *_ = env.step({"name":"get_pollution_evidence", "args":{
        "detail":"full", "kind":"composition", "path":"/aerosol/records/0"}})
    assert result["ok"]
    assert result["content"]["composition"]["aerosol"]["records"][0]["aod"] == .3
    assert len(json.dumps(result["content"])) < env.cfg.max_tool_response_chars
    assert b.evidence["pollution"]["composition"]["aerosol"]["records"][0]["aod"] == .3
    b.evidence["pollution"]["composition"]["aerosol"]["available"] = True
    result, *_ = env.step({"name":"get_pollution_evidence", "args":{
        "detail":"full", "kind":"composition", "path":"/aerosol/available"}})
    content = result["content"]
    registry = {content["evidence_ref"]: {"tool":"get_pollution_evidence", "content":content}}
    assert not _semantic_evidence_match({"type":"model_guidance", "ref":content["evidence_ref"],
        "field":"/composition/aerosol/available", "value":True}, registry, {"get_pollution_evidence"})


def test_climatology_is_time_gated_when_used(tmp_path):
    b = native_case(tmp_path)
    b.meta["climatology"] = {"04": {"pm25": {"p50": 30}}}
    b.meta["climatology_provenance"] = {"available_at":"2025-04-01T12:00:00+08:00"}
    assert b.audit_time_gate()
    b.meta.pop("climatology")
    assert not b.audit_time_gate()
