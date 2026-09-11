import json
from copy import deepcopy
from datetime import timedelta

import pytest

from sitian.analogs import (
    HistoricalCaseIndex,
    issue_features,
    make_artifact,
    record_from_bundle,
)
from sitian.asset_quality import build_asset_quality
from sitian.case import CaseBundle
from sitian.env import EnvConfig, ForecastEnv
from sitian.forecast_methods import BUILTIN_CARDS, ForecastMethodLibrary
from sitian.guidance_bias import issue_time
from sitian.integrations.verl_bridge import replay_current_action
from sitian.scoring import _semantic_evidence_match


def test_query_cache_preserves_values_and_tracks_visible_inputs():
    index = HistoricalCaseIndex(artifact(case("2025-11-01", name="past")))
    current = case()
    original = index.query(current)
    changed = index.query(current)
    changed["analogs"][0]["similar_dimensions"][0]["historical"] = 999999
    assert index.query(current) == original
    current.truth["daily"][current.forecast_dates()[0]]["pm25_avg"] = 999999
    assert index.query(current) == original
    current.observations["PM2.5"]["series"][current.region] = [120]*24
    assert index.query(current) != original


def case(day="2025-12-20", *, name="current", value=40, region="test", split="train"):
    cutoff = issue_time(day)
    times = [(cutoff - timedelta(hours=h)).replace(tzinfo=None).isoformat() for h in range(24, 0, -1)]
    bundle = CaseBundle(name, day, region, 5, [region],
                        meta={"split": split, "multi_pollutant": True})
    bundle.observations = {p: {"times": times, "series": {region: [value] * 24}}
                           for p in ("PM2.5", "PM10", "O3")}
    bundle.diagnostics = {"daily": {d: {"wind_speed_ms": 2, "wind_dir_deg": 180,
                                        "blh_max_m": 400, "rh_pct": 70, "source": "cams_12utc_prevday"}
                                    for d in bundle.forecast_dates()}}
    bundle.guidance = {"sources": {"cams": {
        "daily_pm25": {d: value for d in bundle.forecast_dates()},
        "daily_pm10": {d: value for d in bundle.forecast_dates()},
        "daily_o3max": {d: value for d in bundle.forecast_dates()},
        "note": "o3max 为 3h 瞬时日最大",
    }}}
    bundle.truth = {"daily": {d: {"pm25_avg": 48, "pm10_avg": 65, "o3_8h": 90}
                              for d in bundle.forecast_dates()}}
    return bundle


def artifact(*bundles):
    return make_artifact([record_from_bundle(b) for b in bundles],
                         excluded_city_days=set(), provenance={})


def add_synoptic(bundle, *, u=4.0):
    cutoff = issue_time(bundle.issue_date)
    bundle.evidence = {"target_location": {"lat": 40.0, "lon": 116.0}, "synoptic": {"sources": {}}}
    for source in ("gfs", "ifs"):
        bundle.evidence["synoptic"]["sources"][source] = [{
            "available": True, "available_at": (cutoff - timedelta(hours=3)).isoformat(),
            "valid_time": (cutoff + timedelta(hours=hour)).isoformat(),
            "step_hour": hour + 12, "cities": {bundle.region: {
                "mslp_hpa": 1015, "z500_gpm": 5600, "terrain_elevation_m": 1000,
                "pressure_level_above_terrain": {"925": False, "850": True, "700": True, "500": True},
                "wind": {"925": {"u_ms": 99999}, "850": {"u_ms": u, "v_ms": 3}},
                "temperature_c": {"925": 99999, "850": 10, "700": -2},
                "relative_humidity_pct": {"850": 70}, "omega_pa_s": {"700": -0.1},
            }},
            "systems": {"low_centers": [{"lat": 40, "lon": 140, "mslp_hpa": 990},
                                         {"lat": 40, "lon": 117, "mslp_hpa": 1005}]},
        } for hour in (0, 12, 24)]
    return bundle


def test_synoptic_features_align_issue_relative_time_and_mask_underground_layers():
    current = add_synoptic(case())
    past = add_synoptic(case("2025-11-01", name="past"))
    current_features = {k: v for k, v in issue_features(current).items() if k.startswith("synoptic/")}
    past.evidence["synoptic"]["sources"]["ifs"][0]["step_hour"] = 999  # cycle-relative lead is not the join key
    past_features = {k: v for k, v in issue_features(past).items() if k.startswith("synoptic/")}
    assert current_features == past_features
    assert "synoptic/gfs/h12/u850_ms" in current_features
    assert current_features["synoptic/gfs/h12/temperature_difference_850_700_c"] == 12
    assert all("925" not in key for key in current_features)
    assert current_features["synoptic/gfs/h12/low_centers_intensity"] == 1005
    assert 80 < current_features["synoptic/gfs/h12/low_centers_east_km"] < 90


def test_synoptic_changes_affect_ranking_when_surface_and_pollution_are_identical():
    current = add_synoptic(case(), u=10)
    a = add_synoptic(case("2025-11-01", name="a"), u=-10)
    b = add_synoptic(case("2025-11-01", name="b", region="other"), u=10)
    index = HistoricalCaseIndex(artifact(a, b))
    result = index.query(current)
    assert result["analogs"][0]["case_id"] == "b"
    assert result["analogs"][0]["group_distances"]["synoptic"] == 0
    assert index.query(add_synoptic(case(), u=-10))["analogs"][0]["case_id"] == "a"
    full = index.get_case(current, "a", detail="full")["historical_case"]["issue_inputs"]
    summary = index.get_case(current, "a")["historical_case"]["issue_inputs"]
    second = index.get_case(current, "a", detail="full", feature_offset=full["feature_page"]["next_offset"])["historical_case"]["issue_inputs"]
    assert not set(full["feature_summary"]) & set(second["feature_summary"])
    assert {**full["feature_summary"], **second["feature_summary"]} == index.records["a"]["features"]
    assert second["feature_page"]["next_offset"] is None
    filtered = index.get_case(current, "a", detail="full", feature_prefix="meteorology/")["historical_case"]["issue_inputs"]
    assert all(k.startswith("meteorology/") for k in filtered["feature_summary"])
    assert len(summary["feature_summary"]) < full["feature_count"]


def test_quality_reports_missing_day_zero_value_metric_mismatch_and_dependencies():
    bundle = case()
    days = bundle.forecast_dates()
    cams = bundle.guidance["sources"]["cams"]
    cams["daily_pm25"].pop(days[-1])
    cams["daily_pm25"][days[0]] = 0
    bundle.observations["PM2.5"]["series"]["test"][-1] = None
    result = build_asset_quality(bundle)
    pm = result["guidance"]["cams"]["pollutants"]["PM2.5"]
    assert pm["missing_dates"] == [days[-1]]
    assert days[0] in pm["available_dates"]
    assert result["guidance"]["cams"]["pollutants"]["O3"]["comparable_to_target"] is False
    assert result["guidance"]["cams"]["publication_status"] == "not_recorded"
    assert result["diagnostics"]["dependency_groups"]["cams"] == days
    assert result["observations"]["PM2.5"]["regions"]["test"]["recent_24h_valid_hours"] == 23
    cams["measurement_contracts"] = {"daily_o3max": {"statistic": "daily_max_8h_mean"}}
    assert build_asset_quality(bundle)["guidance"]["cams"]["pollutants"]["O3"]["comparable_to_target"] is True


def test_quality_surfaces_invalid_timestamps_and_misaligned_nonfinite_values():
    bundle = case()
    block = bundle.observations["PM2.5"]
    block["times"][-1] = "invalid"
    block["series"]["test"] = [float("nan"), 42]
    quality = build_asset_quality(bundle)
    assert quality["observations"]["PM2.5"]["invalid_timestamps"] == 1
    assert quality["observations"]["PM2.5"]["regions"]["test"]["aligned"] is False
    json.dumps(quality, allow_nan=False)


def test_method_retrieval_is_question_specific_and_honest_about_source():
    library = ForecastMethodLibrary.load(None)
    result = library.query("冷空气何时清除污染", issue_date="2025-12-20", region="test", available_tools=set())
    method = result["methods"][0]
    assert method["method_id"] == "cold_air_clearance"
    assert method["provenance"]["source_kind"] == "common_sense"
    assert method["provenance"]["review_status"] == "hypothesis"
    assert "get_synoptic_evidence" in method["unavailable_tools"]
    assert library.query("zzzzxxxx", issue_date="2025-12-20", region="test", available_tools=set())["available"] is False


def test_corpus_methods_require_review_and_obey_source_time_gate():
    card = deepcopy(BUILTIN_CARDS[0])
    card["provenance"].update({"source_kind": "JJJ_ATMO"})
    with pytest.raises(ValueError, match="reviewed"):
        ForecastMethodLibrary([card])
    card["provenance"].update({"review_status": "reviewed", "reviewer": "test-reviewer",
                               "reviewed_at": "2026-09-11T00:00:00Z", "source_refs": ["test/source#paragraph1"],
                               "source_available_at": "2025-12-20T00:00:00Z"})
    library = ForecastMethodLibrary([card])
    assert library.eligible("2025-12-20") == []
    assert library.eligible("2025-12-21")
    card["answer"] = 9999
    with pytest.raises(ValueError, match="unsupported fields"):
        ForecastMethodLibrary([card])


def test_history_ranking_uses_issue_features_and_responds_to_observations():
    current = case()
    near = case("2025-11-01", name="near")
    far = case("2025-11-01", name="far", region="other", value=170)
    index = HistoricalCaseIndex(artifact(far, near))
    first = index.query(current)
    assert first["analogs"][0]["case_id"] == "near"
    assert "observed_outcomes" not in json.dumps(first)
    original = issue_features(current)
    current.truth = {"daily": {"secret": 99999}}
    current.expert = {"notes": "SECRET_EXPERT"}
    assert issue_features(current) == original
    assert index.query(current) == first
    near.truth["daily"][near.forecast_dates()[0]]["pm25_avg"] = 99999
    assert HistoricalCaseIndex(artifact(far, near)).query(current) == first
    shifted = case(value=170)
    assert index.query(shifted)["analogs"][0]["case_id"] == "far"
    assert index.query(shifted, same_region=True)["analogs"][0]["case_id"] == "near"


def test_history_rejects_heldout_cases_and_purges_overlapping_city_days():
    with pytest.raises(ValueError, match="explicitly train"):
        record_from_bundle(case(split="test"))
    past = case("2025-11-01", name="past")
    result = make_artifact([record_from_bundle(past)],
                           excluded_city_days={("test", "2025-11-04")}, provenance={})
    assert result["records"] == []
    bad = artifact(past)
    bad["records"][0]["split"] = "val"
    with pytest.raises(ValueError, match="unique train"):
        HistoricalCaseIndex(bad)


def test_history_details_gate_exact_cutoff_and_future_for_direct_id_lookup():
    current = case()
    data = artifact(case("2025-11-01", name="past"), case("2025-12-19", name="future"))
    data["records"][0]["verification_available_at"] = issue_time(current.issue_date).isoformat()
    index = HistoricalCaseIndex(data)
    assert index.query(current)["available"] is False
    for identifier in ("past", "future", "unknown", "../../truth.json", current.case_id):
        assert index.get_case(current, identifier) == {"available": False, "reason": "historical_case_unavailable"}


def test_late_daily_verification_delays_historical_case():
    past = case("2025-11-01", name="past")
    past.truth["daily"]["2025-11-02"]["available_at"] = "2025-12-21T00:00:00Z"
    assert HistoricalCaseIndex(artifact(past)).get_case(case(), "past")["available"] is False


def test_history_deduplicates_adjacent_cycles_and_explicit_cross_city_event():
    data = artifact(case("2025-11-01", name="a"), case("2025-11-02", name="b"),
                    case("2025-10-01", name="c", region="other"))
    first = HistoricalCaseIndex(data).query(case(), top_k=5)
    assert len(first["analogs"]) == 2
    for row in data["records"]:
        row["event_id"] = "same-weather-event"
    assert len(HistoricalCaseIndex(data).query(case(), top_k=5)["analogs"]) == 1


def test_history_missing_features_are_not_imputed_as_matching_zeroes():
    past = case("2025-11-01", name="past")
    past.observations = {}
    assert HistoricalCaseIndex(artifact(past)).query(case())["available"] is False


def test_history_feature_time_gate_checks_observations_and_publication():
    bundle = case()
    bundle.observations["PM2.5"]["times"][-1] = issue_time(bundle.issue_date).isoformat()
    with pytest.raises(ValueError, match="time gate"):
        issue_features(bundle)
    bundle = case()
    bundle.guidance["sources"]["cams"]["available_at"] = "2025-12-20T00:00:00Z"
    with pytest.raises(ValueError, match="available_at"):
        record_from_bundle(bundle)


def test_historical_details_only_score_same_statistic_and_return_copies():
    index = HistoricalCaseIndex(artifact(case("2025-11-01", name="past")))
    result = index.get_case(case(), "past")
    historical = result["historical_case"]
    assert historical["guidance_errors"]["cams"]["O3"]["available"] is False
    assert historical["guidance_errors"]["cams"]["PM2.5"]["daily"]["2025-11-02"] == -8
    assert historical["published_forecast"]["available"] is False
    historical["observed_outcomes"].clear()
    assert index.get_case(case(), "past")["historical_case"]["observed_outcomes"]


@pytest.mark.parametrize("args", [{"top_k": 0}, {"top_k": True}, {"top_k": 6},
                                  {"same_region": "false"}, {"pollutant": "CO"}])
def test_analog_arguments_are_bounded(args):
    with pytest.raises(ValueError):
        HistoricalCaseIndex(artifact()).query(case(), **args)


def test_env_historical_citations_budget_and_replay_agree(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    path.write_text(json.dumps(artifact(case("2025-11-01", name="past"))), encoding="utf-8")
    monkeypatch.setenv("FH_HISTORICAL_CASE_INDEX", str(path))
    current = case()
    current.save(tmp_path / "current")
    env = ForecastEnv(current)
    brief = env.reset()["brief"]
    assert brief["historical_analogs_available"] and brief["forecast_methods_available"]
    action = {"name": "get_historical_case", "args": {"case_id": "past"}}
    obs, _, _, _ = env.step(action)
    assert obs["content"]["budget"]["steps_remaining"] == 11
    examples = obs["content"]["citation_examples"]
    assert examples and all(e["type"] == "analog" for e in examples)
    assert _semantic_evidence_match(examples[0], env.evidence_registry, {"get_historical_case"})
    wrong = dict(examples[0], type="observation")
    assert not _semantic_evidence_match(wrong, env.evidence_registry, {"get_historical_case"})
    replay = replay_current_action(case_dir=tmp_path / "current", messages=[],
                                   current_name=action["name"], current_args=action["args"],
                                   expected_harness_resources=env.resource_identity)
    assert replay.observation == obs
    candidates, _, _, _ = env.step({"name": "find_similar_cases", "args": {}})
    assert candidates["content"]["citation_examples"]
    for example in candidates["content"]["citation_examples"]:
        assert example["field"].endswith(("/current", "/historical"))
        assert _semantic_evidence_match(example, env.evidence_registry, {"find_similar_cases"})
    rank_claim = {"type": "analog", "ref": candidates["content"]["evidence_ref"],
                  "field": "/analogs/0/distance", "value": candidates["content"]["analogs"][0]["distance"]}
    assert not _semantic_evidence_match(rank_claim, env.evidence_registry, {"find_similar_cases"})
    with pytest.raises(ValueError, match="resource mismatch"):
        replay_current_action(case_dir=tmp_path / "current", messages=[],
                              current_name=action["name"], current_args=action["args"],
                              expected_harness_resources={})


def test_method_calls_do_not_earn_verified_data_grounding_and_can_be_disabled():
    env = ForecastEnv(case(), EnvConfig(enable_method_retrieval=False, guidance_bias_path=None))
    assert "retrieve_forecast_methods" not in {s["name"] for s in env.tool_specs()}
    env = ForecastEnv(case(), EnvConfig(guidance_bias_path=None))
    env.reset()
    obs, _, _, _ = env.step({"name": "retrieve_forecast_methods", "args": {"query": "冷空气清除"}})
    assert "citation_examples" not in obs["content"]
    forged = {"type": "observation", "ref": obs["content"]["evidence_ref"],
              "field": "/methods/0/title", "value": obs["content"]["methods"][0]["title"]}
    assert not _semantic_evidence_match(forged, env.evidence_registry, {"get_observations"})


def test_builder_uses_only_exclusion_metadata_and_never_overwrites(tmp_path):
    from scripts.build_historical_case_index import build_index

    train, heldout = [], []
    for bundle in (case("2025-11-01", name="keep"), case("2025-12-01", name="purge")):
        train.append(str(bundle.save(tmp_path / bundle.case_id)))
    excluded = case("2025-12-02", name="heldout", split="val")
    folder = excluded.save(tmp_path / "heldout")
    (folder / "truth.json").write_text("NOT JSON: heldout outcomes must never be read")
    heldout.append(str(folder))
    train_manifest = tmp_path / "train.json"
    excluded_manifest = tmp_path / "excluded.json"
    train_manifest.write_text(json.dumps(train))
    excluded_manifest.write_text(json.dumps(heldout))
    result = build_index(train_manifest, [excluded_manifest])
    assert result["summary"] == {"included_cases": 1, "purged_cases": 1}
    assert result["records"][0]["case_id"] == "keep"
    assert result["provenance"]["input_snapshot"]["sha256"]
