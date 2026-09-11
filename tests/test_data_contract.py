from datetime import timedelta

import pytest

from sitian.assess import daily_signals, obs_trend
from sitian.case import CaseBundle
from sitian.data_contract import issue_time
from sitian.env import EnvConfig, ForecastEnv
from sitian.guidance_bias import GuidanceBiasIndex, records_from_bundle


def bundle():
    return CaseBundle("sample", "2025-01-03", "北京", 1, ["北京"],
                      observations={"PM2.5": {"times": ["2025-01-03T05:00", "2025-01-03T07:00"],
                                               "series": {"北京": [20, 24]}}})


def test_observations_use_elapsed_hours_across_gaps_and_timezone_forms():
    b = bundle()
    b.observations["PM2.5"]["times"][-1] = "2025-01-02T23:00:00Z"
    env = ForecastEnv(b, EnvConfig(guidance_bias_path=None))
    env.reset()
    result = env.step({"name": "get_observations", "args": {"last_hours": 2, "stride": 1}})[0]["content"]
    assert result["series"]["北京"] == [24]
    assert result["window_start"] == "2025-01-03T06:00:00+08:00"
    assert result["window_end_exclusive"] == "2025-01-03T08:00:00+08:00"
    trend = obs_trend(b.observations, "北京", b.issue_date)["PM2.5"]
    assert trend["last24h_valid_hours"] == 2
    assert trend["prev24h_valid_hours"] == 0
    assert "change_24h" not in trend


@pytest.mark.parametrize("source", ["observations", "diagnostics", "guidance", "evidence", "previous_forecast"])
def test_all_visible_publication_times_are_strictly_before_issue(source):
    b = bundle()
    target = getattr(b, source)
    if target is None:
        target = {}
        setattr(b, source, target)
    target["nested"] = {"available_at": "2025-01-03T00:00:00Z"}
    # observations are pollutant blocks; nested publication is still inspected.
    with pytest.raises(ValueError, match="time gate"):
        ForecastEnv(b, EnvConfig(guidance_bias_path=None))


def test_utc_observation_that_looks_like_previous_day_cannot_leak():
    b = bundle()
    b.observations["PM2.5"]["times"][-1] = "2025-01-03T01:00:00Z"
    assert b.audit_time_gate()


def test_bias_uses_late_truth_publication_and_excludes_instantaneous_o3():
    b = bundle()
    day = b.forecast_dates()[0]
    b.guidance = {"sources": {"cams": {"daily_pm25": {day: 20}, "daily_o3max": {day: 80},
                                          "note": "3h 瞬时日最大"}}}
    b.truth = {"daily": {day: {"pm25_avg": 24, "o3_8h": 55, "available_at": "2025-01-08T12:00:00+08:00"}}}
    rows = records_from_bundle(b)
    assert len(rows) == 1
    assert rows[0]["pollutant"] == "PM2.5"
    assert rows[0]["verification_available_at"] == "2025-01-08T12:00:00+08:00"


def test_duplicate_bias_records_do_not_satisfy_sample_threshold():
    b = bundle()
    day = b.forecast_dates()[0]
    b.guidance = {"sources": {"cams": {"daily_pm25": {day: 20}}}}
    b.truth = {"daily": {day: {"pm25_avg": 24}}}
    row = records_from_bundle(b)[0]
    index = GuidanceBiasIndex({"artifact_type": "guidance_bias_history", "records": [row] * 20})
    result = index.query(region="北京", issue_date="2025-01-10", horizon=1)["series"][0]
    assert result["available"] is False
    assert result["fallback_chain"][0]["n"] == 1
    row["error_guidance_minus_truth"] = float("nan")
    with pytest.raises(ValueError, match="historical error"):
        GuidanceBiasIndex({"artifact_type": "guidance_bias_history", "records": [row]})


def test_zero_boundary_layer_and_missing_cloud_remain_distinct_from_missing_and_clear():
    result = daily_signals({"daily": {"2025-01-04": {"wind_speed_ms": 1, "rh_pct": 70,
                                                       "blh_max_m": 0, "blh_m": 1500, "tmax_c": 30}}})
    assert "stagnation" in result["2025-01-04"]["flags"]
    assert "o3_potential" not in result["2025-01-04"]["flags"]


def test_duplicate_timezone_equivalent_observations_return_error():
    b = bundle()
    b.observations["PM2.5"]["times"] = ["2025-01-03T07:00", "2025-01-02T23:00:00Z"]
    env = ForecastEnv(b, EnvConfig(guidance_bias_path=None))
    env.reset()
    result = env.step({"name": "get_observations", "args": {}})[0]["content"]
    assert "unique and ordered" in result["error"]


def test_missing_recent_data_is_not_replaced_by_older_records():
    b = bundle()
    old = issue_time(b.issue_date) - timedelta(days=2)
    b.observations["PM2.5"]["times"] = [old.isoformat(), (old + timedelta(hours=1)).isoformat()]
    env = ForecastEnv(b, EnvConfig(guidance_bias_path=None))
    env.reset()
    result = env.step({"name": "get_observations", "args": {"last_hours": 24}})[0]["content"]
    assert result["available"] is False
    assert result["series"]["北京"] == []
