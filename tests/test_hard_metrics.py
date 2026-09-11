from sitian.hard_metrics import (
    failed_forecast_hard_counts, forecast_hard_counts, metric_improvements, scale_counts,
    sum_counts, summarize_hard_counts,
)


def _forecast(levels):
    days = ["2026-06-02", "2026-06-03"]
    event_indices = [index for index, level in enumerate(levels) if level >= 3]
    process = ({"has_event": False, "start": None, "peak": None, "end": None}
               if not event_indices else {
                   "has_event": True,
                   "start": days[event_indices[0]],
                   "peak": days[max(event_indices, key=lambda index: levels[index])],
                   "end": days[event_indices[-1]],
               })
    return {
        "daily": [
            {"date": day, "aqi_level": level, "primary_pollutant": "PM10",
             "pm25_range": [10, 30], "pm10_range": [60, 100], "o3_range": [80, 120]}
            for day, level in zip(days, levels)
        ],
        "process": process,
    }


def test_hard_metrics_are_additive_and_keep_shaping_out():
    truth = {
        "2026-06-02": {"PM2.5": 20, "PM10": 80, "O3": 100},
        "2026-06-03": {"PM2.5": 20, "PM10": 300, "O3": 100},
    }
    perfect = forecast_hard_counts(_forecast([2, 4]), truth)
    summary = summarize_hard_counts(perfect)
    assert summary["level"]["ordinal_mae"] == 0
    assert summary["event_daily_level_ge_4"]["csi"] == 1
    assert summary["primary_pollutant"]["macro_f1_supported_classes"] == 1
    assert summary["interval_80pct"]["PM10"]["coverage"] == 0.5
    assert summary["interval_80pct"]["PM10"]["midpoint_mae"] == 110
    assert summary["turning_level_ge_3_intention_to_treat"]["peak_mae_days"] == 0

    averaged = sum_counts([scale_counts(perfect, 0.5), scale_counts(perfect, 0.5)])
    assert summarize_hard_counts(averaged) == summary


def test_improvements_use_positive_is_better_orientation():
    truth = {"2026-06-02": 20.0, "2026-06-03": 160.0}
    model = summarize_hard_counts(forecast_hard_counts(_forecast([1, 5]), truth))
    baseline = summarize_hard_counts(forecast_hard_counts(_forecast([2, 2]), truth))
    delta = metric_improvements(model, baseline)
    assert delta["level_ordinal_mae"] > 0
    assert delta["event_csi"] > 0


def test_failed_rollout_is_counted_conservatively_not_dropped():
    truth = {
        "2026-06-02": {"PM2.5": 20, "PM10": 80, "O3": 100},
        "2026-06-03": {"PM2.5": 20, "PM10": 300, "O3": 100},
    }
    counts = failed_forecast_hard_counts(truth)
    summary = summarize_hard_counts(counts)

    assert summary["level"]["n_days"] == 2
    assert summary["level"]["ordinal_mae"] == 3.5
    assert summary["event_daily_level_ge_4"]["misses"] == 1
    assert summary["event_daily_level_ge_4"]["false_alarms"] == 1
    assert summary["process_level_ge_3"]["fn"] == 1
    assert summary["turning_level_ge_3_intention_to_treat"]["start_mae_days"] == 2
    assert summary["interval_80pct"]["PM10"]["n"] == 2
    assert summary["interval_80pct"]["PM10"]["coverage"] == 0


def test_primary_ties_use_set_accuracy_and_do_not_create_impossible_false_negatives():
    day = "2026-06-02"
    truth = {day: {"PM2.5": 36, "PM10": 52, "O3": 1}}
    forecast = {
        "daily": [{"date": day, "aqi_level": 2, "primary_pollutant": "PM10",
                   "pm25_range": [30, 40], "pm10_range": [45, 60],
                   "o3_range": [0, 10]}],
        "process": {"has_event": False},
    }
    summary = summarize_hard_counts(forecast_hard_counts(forecast, truth))
    primary = summary["primary_pollutant"]
    assert primary["set_membership_accuracy"] == 1
    assert primary["tied_truth_days"] == 1
    assert primary["macro_f1_supported_classes"] is None
    assert all(row["fn"] == 0 for row in primary["one_vs_rest_by_pollutant"].values())
