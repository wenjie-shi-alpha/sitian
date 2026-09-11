import pytest

from sitian.schema import pm25_to_level
from sitian.scoring import RewardConfig, _scalar_equal, extract_event, score_forecast

ISSUE = "2025-12-16"
DATES = ["2025-12-17", "2025-12-18", "2025-12-19", "2025-12-20", "2025-12-21", "2025-12-22"]
TRUTH = dict(zip(DATES, [95.0, 135.0, 168.0, 150.0, 52.0, 38.0]))
# 等级: 3, 4, 5, 4, 2, 2 → event(>=4): 12-18..20；process(>=3): 12-17..20


def _forecast(levels, ranges, process, evidence=None):
    return {
        "issue_date": ISSUE,
        "region": "beijing",
        "daily": [
            {"date": d, "aqi_level": lv, "pm25_range": list(rg),
             "primary_pollutant": (
                 None if lv == 1 else
                 ("PM2.5" if pm25_to_level(rg[1]) >= lv else "NO2")
             )}
            for d, lv, rg in zip(DATES, levels, ranges)
        ],
        "process": process,
        "evidence": evidence or [],
        "confidence": "medium",
    }


PERFECT_PROCESS = {"has_event": True, "start": "2025-12-17", "peak": "2025-12-19", "end": "2025-12-20"}


def test_extract_event():
    assert extract_event(TRUTH, 4) == ("2025-12-18", "2025-12-19", "2025-12-20")
    # 两段过程取包含峰值的一段
    two_runs = dict(zip(DATES, [130.0, 60.0, 140.0, 160.0, 40.0, 30.0]))
    assert extract_event(two_runs, 4) == ("2025-12-19", "2025-12-20", "2025-12-20")
    assert extract_event(dict(zip(DATES, [30.0] * 6)), 4) is None


@pytest.mark.parametrize("values", [[120, 140, 40], [120, 40, 140], [120, 120, 40]])
def test_perfect_midpoints_select_same_numeric_aqi_peak_as_truth(values):
    truth = dict(zip(DATES, values))
    fc = {"issue_date": ISSUE, "region": "beijing", "daily": [
        {"date": day, "pm25_range": [value, value]} for day, value in truth.items()]}
    score = score_forecast(fc, truth, issue_date=ISSUE, horizon=3, region="beijing")
    assert score.valid
    assert score.outcome_composite == 1.0
    assert score.details["turning"]["pred"] == score.details["turning"]["truth"]


@pytest.mark.parametrize("pollutant,value,field", [
    ("SO2", 600, "so2_range"), ("NO2", 200, "no2_range"), ("CO", 18, "co_range")])
def test_optional_gas_can_represent_true_primary_and_level(pollutant, value, field):
    truth = {DATES[0]: {"PM2.5": 20, "PM10": 30, "O3": 40, pollutant: value}}
    daily = {"date": DATES[0], "pm25_range": [20, 20], "pm10_range": [30, 30],
             "o3_range": [40, 40], field: [value, value]}
    score = score_forecast({"issue_date": ISSUE, "region": "beijing", "daily": [daily]},
                           truth, issue_date=ISSUE, horizon=1, region="beijing")
    assert score.valid and score.outcome_composite == 1.0
    assert score.details["normalized_forecast"]["daily"][0]["primary_pollutant"] == pollutant


def test_extreme_native_dust_day_is_expressible():
    truth = {DATES[0]: {"PM2.5": 492.5, "PM10": 2113.8, "O3": 87}}
    fc = {"issue_date": ISSUE, "daily": [{"date": DATES[0], "pm25_range": [492.5, 492.5],
                                         "pm10_range": [2113.8, 2113.8], "o3_range": [87, 87]}]}
    result = score_forecast(fc, truth, issue_date=ISSUE, horizon=1)
    assert result.valid and result.outcome_composite == 1.0


def test_perfect_forecast_scores_one():
    fc = _forecast(
        [3, 4, 5, 4, 2, 2],
        [(v - 10, v + 10) for v in TRUTH.values()],
        PERFECT_PROCESS,
        [{"type": "observation", "claim": "48h实况"}, {"type": "synoptic", "claim": "形势"}],
    )
    res = score_forecast(fc, TRUTH, issue_date=ISSUE, horizon=6, region="beijing",
                         expert_evidence_types={"observation", "synoptic"})
    assert res.valid
    assert res.composite == pytest.approx(1.0)
    assert res.outcome_composite == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in res.components.values() if v is not None)


def test_flat_level2_forecast():
    fc = _forecast([2] * 6, [(30, 60)] * 6, {"has_event": False, "start": None, "peak": None, "end": None})
    res = score_forecast(fc, TRUTH, issue_date=ISSUE, horizon=6, region="beijing",
                         expert_evidence_types={"observation"})
    # level: (0.5*1 + 0 + 0 + 0 + 1 + 1) / (1+2+2+2+1+1) = 2.5/9
    assert res.components["level"] == pytest.approx(2.5 / 9, abs=1e-4)
    assert res.components["event"] == 0.0
    # interval: 覆盖日得 1；未覆盖按标准 interval score 连续指数衰减。
    expected_interval = sum(
        __import__("math").exp(-10 * max(value - 60, 0) / 120)
        for value in TRUTH.values()
    ) / 6
    assert res.components["interval"] == pytest.approx(expected_interval, abs=1e-4)
    assert res.components["turning"] == 0.0
    assert res.components["evidence"] == 0.0
    expected = 0.35 * 2.5 / 9 + 0.15 * expected_interval
    assert res.composite == pytest.approx(expected, abs=1e-3)
    # Evidence-type matching is an auxiliary training signal and is excluded
    # from the fair forecast-outcome comparison score.
    assert res.outcome_composite > res.composite


def _ranges_for_levels(levels):
    """schema-v0.6.0: levels are derived from interval midpoints (HJ 633-2012 here)."""
    midpoint = {1: 20, 2: 50, 3: 95, 4: 130, 5: 200, 6: 300}
    return [(midpoint[level] - 5, midpoint[level] + 5) for level in levels]


def test_off_by_one_levels_and_csi():
    process = {"has_event": True, "start": DATES[0], "peak": DATES[2], "end": DATES[-1]}
    fc = _forecast([4, 5, 6, 5, 3, 3], _ranges_for_levels([4, 5, 6, 5, 3, 3]), process)
    res = score_forecast(fc, TRUTH, issue_date=ISSUE, horizon=6, region="beijing")
    assert res.components["level"] == pytest.approx(0.5)
    # pred事件日 {17,18,19,20} vs 真值 {18,19,20}: hits3 fa1 → CSI 0.75
    assert res.components["event"] == pytest.approx(0.75)
    # evidence 无专家参照 → 弃权并去权归一
    assert res.components["evidence"] is None
    assert sum(res.weights_used.values()) == pytest.approx(1.0)


def test_wide_interval_penalty_has_no_floor():
    ranges = [(0, 400)] * 6
    fc = _forecast([3, 4, 5, 4, 2, 2], ranges, PERFECT_PROCESS)
    res = score_forecast(fc, TRUTH, issue_date=ISSUE, horizon=6, region="beijing")
    assert res.components["interval"] == pytest.approx(
        __import__("math").exp(-(400 - RewardConfig().width_free) /
                               RewardConfig().width_scale), abs=1e-4
    )
    assert res.components["interval"] < 0.06


def test_near_interval_miss_gets_small_continuous_credit():
    fc = _forecast([3, 4, 5, 4, 2, 2], [(v - 15, v - 5) for v in TRUTH.values()], PERFECT_PROCESS)
    res = score_forecast(fc, TRUTH, issue_date=ISSUE, horizon=6, region="beijing")
    assert res.components["interval"] == pytest.approx(
        __import__("math").exp(-(2 / RewardConfig().interval_alpha * 5) /
                               RewardConfig().width_scale), abs=1e-4
    )
    assert all(not d["covered"] and d["miss"] == 5 for d in res.details["interval_diagnostics"]["PM2.5"])


def test_turning_abstains_when_no_true_event():
    truth = dict(zip(DATES, [30.0] * 6))
    fc = _forecast([1] * 6, [(20, 40)] * 6, {"has_event": False})
    res = score_forecast(fc, truth, issue_date=ISSUE, horizon=6, region="beijing")
    assert res.components["event"] is None
    assert res.components["turning"] is None
    assert "event" not in res.weights_used
    assert "turning" not in res.weights_used


def test_clean_false_alarm_still_scores_event_zero():
    truth = dict(zip(DATES, [30.0] * 6))
    fc = _forecast(
        [4, 1, 1, 1, 1, 1], [(125, 135)] + [(20, 40)] * 5,
        {"has_event": True, "start": DATES[0], "peak": DATES[0], "end": DATES[0]},
    )
    res = score_forecast(fc, truth, issue_date=ISSUE, horizon=6, region="beijing")
    assert res.components["event"] == 0.0
    assert res.details["event"]["false_alarms"] == 1


def test_event_near_miss_has_bounded_shaping_credit_but_hard_csi_zero():
    fc = _forecast(
        [2, 3, 3, 3, 2, 2], _ranges_for_levels([2, 3, 3, 3, 2, 2]),
        {"has_event": True, "start": DATES[1], "peak": DATES[1], "end": DATES[3]},
    )
    res = score_forecast(fc, TRUTH, issue_date=ISSUE, horizon=6, region="beijing")
    assert res.components["event"] == pytest.approx(0.25)
    assert res.details["event"]["hard_csi"] == 0.0
    assert res.details["event"]["near_miss_count"] == 3


def test_turning_partial_credit():
    process = {"has_event": True, "start": "2025-12-17", "peak": "2025-12-19", "end": "2025-12-21"}
    ranges = [(v - 10, v + 10) for v in TRUTH.values()]
    ranges[4] = (90, 100)  # day 5 forecast level 3 -> derived process ends 12-21
    fc = _forecast([3, 4, 5, 4, 3, 2], ranges, process)
    res = score_forecast(fc, TRUTH, issue_date=ISSUE, horizon=6, region="beijing")
    # 派生过程 17..21（峰 19）对真值 17..20：偏差 0,0,1 天 → (1 + 1 + 0.6)/3
    assert res.components["turning"] == pytest.approx((1 + 1 + 0.6) / 3, abs=1e-4)


def test_process_head_is_derived_and_supplied_dates_are_ignored():
    process = {"has_event": True, "start": DATES[0], "peak": DATES[2], "end": DATES[4]}
    fc = _forecast([3, 4, 5, 4, 2, 2], [(v - 10, v + 10) for v in TRUTH.values()], process)
    result = score_forecast(fc, TRUTH, issue_date=ISSUE, horizon=6, region="beijing")
    assert result.valid
    derived = result.details["normalized_forecast"]["process"]
    assert (derived["start"], derived["peak"], derived["end"]) == (DATES[0], DATES[2], DATES[3])
    assert result.components["turning"] == pytest.approx(1.0)


def test_invalid_forecast_gets_zero():
    res = score_forecast({"garbage": True}, TRUTH, issue_date=ISSUE, horizon=6, region="beijing")
    assert not res.valid
    assert res.composite == 0.0
    assert res.outcome_composite == 0.0
    assert res.errors


def test_truth_missing_raises():
    fc = _forecast([2] * 6, [(30, 60)] * 6, {"has_event": False})
    partial = {d: TRUTH[d] for d in DATES[:4]}
    with pytest.raises(ValueError, match="missing"):
        score_forecast(fc, partial, issue_date=ISSUE, horizon=6, region="beijing")


@pytest.mark.parametrize("value", [None, "", "   ", float("nan"), float("inf")])
def test_missing_or_nonfinite_tool_values_are_not_grounding_facts(value):
    assert not _scalar_equal(value, value)
