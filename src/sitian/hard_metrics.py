"""Non-shaped forecast metrics used for checkpoint graduation and reporting.

These metrics are deliberately separate from the dense RL reward.  Counts can
be averaged within a case across stochastic rollouts, summed across cases, and
then recomputed inside a date/process-block bootstrap without treating sibling
rollouts as independent observations.
"""
from __future__ import annotations

import math
from copy import deepcopy
from datetime import date
from typing import Any

from .schema import aqi_standard_for_date, daily_aqi, pm25_to_level
from .scoring import extract_event

POLLUTANTS = ("PM2.5", "PM10", "O3", "SO2", "NO2", "CO")
INTERVAL_FIELDS = {"PM2.5": "pm25_range", "PM10": "pm10_range", "O3": "o3_range"}


def forecast_hard_counts(
    forecast: dict, truth_daily: dict, *, aqi_standard: str | None = None,
    event_level: int = 4, process_level: int = 3, interval_alpha: float = 0.2,
) -> dict:
    """Return additive sufficient statistics for one valid forecast."""
    daily = {item["date"]: item for item in forecast["daily"]}
    truth_concs = {
        day: (dict(value) if isinstance(value, dict) else {"PM2.5": float(value)})
        for day, value in truth_daily.items() if day in daily
    }
    truth_levels = {}
    truth_primary = {}
    truth_magnitude = {}
    for day, concentrations in truth_concs.items():
        standard = (aqi_standard.get(day) if isinstance(aqi_standard, dict)
                    else aqi_standard) or aqi_standard_for_date(day)
        if len(concentrations) > 1:
            aq = daily_aqi(concentrations, standard=standard)
            truth_levels[day] = aq["level"]
            truth_primary[day] = set(aq["primary"])
            truth_magnitude[day] = float(aq["aqi"])
        else:
            truth_levels[day] = pm25_to_level(concentrations["PM2.5"], standard=standard)
            truth_primary[day] = set()
            truth_magnitude[day] = float(concentrations["PM2.5"])

    counts: dict[str, Any] = {
        "level": {"n": 0.0, "absolute_error_sum": 0.0,
                  "exact": 0.0, "within_one": 0.0},
        "event": {"hits": 0.0, "misses": 0.0, "false_alarms": 0.0},
        "process": {"tp": 0.0, "fn": 0.0, "fp": 0.0},
        "turning": {
            "n_truth_processes": 0.0, "detected": 0.0, "missing": 0.0,
            "start_absolute_error_sum": 0.0,
            "peak_absolute_error_sum": 0.0,
            "end_absolute_error_sum": 0.0,
        },
        "primary": {
            "classification": {"n": 0.0, "correct": 0.0,
                               "tied_truth_days": 0.0},
            **{pollutant: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "support": 0.0}
               for pollutant in POLLUTANTS},
        },
        "interval": {pollutant: {"n": 0.0, "covered": 0.0,
                                  "width_sum": 0.0, "interval_score_sum": 0.0,
                                  "midpoint_absolute_error_sum": 0.0,
                                  "midpoint_squared_error_sum": 0.0,
                                  "midpoint_error_sum": 0.0}
                     for pollutant in INTERVAL_FIELDS},
    }
    for day, item in daily.items():
        if day not in truth_levels:
            continue
        predicted_level, true_level = int(item["aqi_level"]), truth_levels[day]
        difference = abs(predicted_level - true_level)
        counts["level"]["n"] += 1
        counts["level"]["absolute_error_sum"] += difference
        counts["level"]["exact"] += int(difference == 0)
        counts["level"]["within_one"] += int(difference <= 1)
        predicted_event, true_event = predicted_level >= event_level, true_level >= event_level
        counts["event"]["hits"] += int(predicted_event and true_event)
        counts["event"]["misses"] += int(not predicted_event and true_event)
        counts["event"]["false_alarms"] += int(predicted_event and not true_event)

        true_set = truth_primary[day]
        if true_set:
            predicted_primary = item.get("primary_pollutant")
            classification = counts["primary"]["classification"]
            classification["n"] += 1
            classification["correct"] += int(predicted_primary in true_set)
            classification["tied_truth_days"] += int(len(true_set) > 1)
            # A submission has exactly one primary label while IAQI can tie.
            # Set-membership accuracy handles ties fairly; per-class F1 is
            # restricted to unique-primary days so a perfect forecast is not
            # required to emit several mutually exclusive labels.
            if len(true_set) == 1:
                true_primary = next(iter(true_set))
                for pollutant in POLLUTANTS:
                    actual = pollutant == true_primary
                    predicted = predicted_primary == pollutant
                    block = counts["primary"][pollutant]
                    block["support"] += int(actual)
                    block["tp"] += int(actual and predicted)
                    block["fp"] += int(not actual and predicted)
                    block["fn"] += int(actual and not predicted)

        for pollutant, field in INTERVAL_FIELDS.items():
            if pollutant not in truth_concs[day]:
                continue
            interval = item.get(field)
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                continue
            lo, hi = map(float, interval)
            observation = float(truth_concs[day][pollutant])
            miss = max(lo - observation, observation - hi, 0.0)
            block = counts["interval"][pollutant]
            block["n"] += 1
            block["covered"] += int(miss == 0)
            block["width_sum"] += hi - lo
            block["interval_score_sum"] += (hi - lo) + 2.0 / interval_alpha * miss
            midpoint_error = (lo + hi) / 2.0 - observation
            block["midpoint_absolute_error_sum"] += abs(midpoint_error)
            block["midpoint_squared_error_sum"] += midpoint_error ** 2
            block["midpoint_error_sum"] += midpoint_error

    truth_event = extract_event(
        truth_magnitude, process_level, levels=truth_levels
    )
    true_process = truth_event is not None
    predicted_process = bool((forecast.get("process") or {}).get("has_event"))
    counts["process"]["tp"] = int(true_process and predicted_process)
    counts["process"]["fn"] = int(true_process and not predicted_process)
    counts["process"]["fp"] = int(not true_process and predicted_process)
    if truth_event is not None:
        turning = counts["turning"]
        turning["n_truth_processes"] = 1
        if predicted_process:
            predicted = forecast["process"]
            turning["detected"] = 1
            for name, actual in zip(("start", "peak", "end"), truth_event):
                error = abs((date.fromisoformat(predicted[name])
                             - date.fromisoformat(actual)).days)
                turning[f"{name}_absolute_error_sum"] = error
        else:
            # Intention-to-treat: a missed process is not absent timing data.
            # Charge the full forecast horizon to each phase.
            turning["missing"] = 1
            for name in ("start", "peak", "end"):
                turning[f"{name}_absolute_error_sum"] = len(truth_levels)
    return counts


def failed_forecast_hard_counts(
    truth_daily: dict, *, aqi_standard: str | dict | None = None,
    event_level: int = 4, process_level: int = 3, interval_alpha: float = 0.2,
) -> dict:
    """Conservative hard counts for a rollout with no valid forecast.

    This keeps hard-metric comparisons intention-to-treat.  A failed rollout
    is wrong—not missing at random—so it receives the worst ordinal error,
    event/process error on every applicable day, false negatives for true
    primary pollutants, and a degenerate [0,0] interval.
    """
    truth_concs = {
        day: (dict(value) if isinstance(value, dict) else {"PM2.5": float(value)})
        for day, value in truth_daily.items()
    }
    truth_levels = {}
    truth_primary = {}
    truth_magnitude = {}
    for day, concentrations in truth_concs.items():
        standard = (aqi_standard.get(day) if isinstance(aqi_standard, dict)
                    else aqi_standard) or aqi_standard_for_date(day)
        if len(concentrations) > 1:
            result = daily_aqi(concentrations, standard=standard)
            truth_levels[day] = result["level"]
            truth_primary[day] = set(result["primary"])
            truth_magnitude[day] = float(result["aqi"])
        else:
            truth_levels[day] = pm25_to_level(
                concentrations["PM2.5"], standard=standard
            )
            truth_primary[day] = set()
            truth_magnitude[day] = float(concentrations["PM2.5"])

    counts: dict[str, Any] = {
        "level": {"n": 0.0, "absolute_error_sum": 0.0,
                  "exact": 0.0, "within_one": 0.0},
        "event": {"hits": 0.0, "misses": 0.0, "false_alarms": 0.0},
        "process": {"tp": 0.0, "fn": 0.0, "fp": 0.0},
        "turning": {
            "n_truth_processes": 0.0, "detected": 0.0, "missing": 0.0,
            "start_absolute_error_sum": 0.0,
            "peak_absolute_error_sum": 0.0,
            "end_absolute_error_sum": 0.0,
        },
        "primary": {
            "classification": {"n": 0.0, "correct": 0.0,
                               "tied_truth_days": 0.0},
            **{pollutant: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "support": 0.0}
               for pollutant in POLLUTANTS},
        },
        "interval": {pollutant: {"n": 0.0, "covered": 0.0,
                                  "width_sum": 0.0, "interval_score_sum": 0.0,
                                  "midpoint_absolute_error_sum": 0.0,
                                  "midpoint_squared_error_sum": 0.0,
                                  "midpoint_error_sum": 0.0}
                     for pollutant in INTERVAL_FIELDS},
    }
    for day, level in truth_levels.items():
        counts["level"]["n"] += 1
        counts["level"]["absolute_error_sum"] += max(level - 1, 6 - level)
        if level >= event_level:
            counts["event"]["misses"] += 1
        else:
            counts["event"]["false_alarms"] += 1
        true_set = truth_primary[day]
        if true_set:
            classification = counts["primary"]["classification"]
            classification["n"] += 1
            classification["tied_truth_days"] += int(len(true_set) > 1)
            if len(true_set) == 1:
                pollutant = next(iter(true_set))
                counts["primary"][pollutant]["support"] += 1
                counts["primary"][pollutant]["fn"] += 1
        for pollutant in INTERVAL_FIELDS:
            if pollutant not in truth_concs[day]:
                continue
            observation = float(truth_concs[day][pollutant])
            counts["interval"][pollutant]["n"] += 1
            counts["interval"][pollutant]["covered"] += int(observation == 0)
            counts["interval"][pollutant]["interval_score_sum"] += (
                2.0 / interval_alpha * abs(observation)
            )
            counts["interval"][pollutant]["midpoint_absolute_error_sum"] += abs(
                observation
            )
            counts["interval"][pollutant]["midpoint_squared_error_sum"] += (
                observation ** 2
            )
            counts["interval"][pollutant]["midpoint_error_sum"] -= observation
    true_process = extract_event(
        truth_magnitude, process_level, levels=truth_levels
    ) is not None
    counts["process"]["fn"] = int(true_process)
    counts["process"]["fp"] = int(not true_process)
    if true_process:
        counts["turning"]["n_truth_processes"] = 1
        counts["turning"]["missing"] = 1
        for name in ("start", "peak", "end"):
            counts["turning"][f"{name}_absolute_error_sum"] = len(truth_levels)
    return counts


def scale_counts(counts: dict, factor: float) -> dict:
    output = deepcopy(counts)

    def scale(value):
        if isinstance(value, dict):
            return {key: scale(item) for key, item in value.items()}
        return float(value) * factor

    return scale(output)


def sum_counts(items: list[dict]) -> dict:
    if not items:
        raise ValueError("counts must not be empty")

    def combine(values):
        if isinstance(values[0], dict):
            return {key: combine([value[key] for value in values]) for key in values[0]}
        return sum(float(value) for value in values)

    return combine(items)


def summarize_hard_counts(counts: dict) -> dict:
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    level = counts["level"]
    event = counts["event"]
    process = counts["process"]
    turning = counts["turning"]
    event_denominator = event["hits"] + event["misses"] + event["false_alarms"]
    primary = {}
    supported = []
    for pollutant, values in counts["primary"].items():
        if pollutant == "classification":
            continue
        f1 = ratio(2 * values["tp"], 2 * values["tp"] + values["fp"] + values["fn"])
        primary[pollutant] = {**values, "f1": f1}
        if values["support"] and f1 is not None:
            supported.append(f1)
    intervals = {
        pollutant: {
            "n": values["n"],
            "coverage": ratio(values["covered"], values["n"]),
            "mean_width": ratio(values["width_sum"], values["n"]),
            "mean_interval_score": ratio(values["interval_score_sum"], values["n"]),
            "midpoint_mae": ratio(
                values["midpoint_absolute_error_sum"], values["n"]
            ),
            "midpoint_rmse": (
                math.sqrt(values["midpoint_squared_error_sum"] / values["n"])
                if values["n"] else None
            ),
            "midpoint_bias": ratio(values["midpoint_error_sum"], values["n"]),
        }
        for pollutant, values in counts["interval"].items()
    }
    return {
        "level": {
            "n_days": level["n"],
            "ordinal_mae": ratio(level["absolute_error_sum"], level["n"]),
            "exact_accuracy": ratio(level["exact"], level["n"]),
            "within_one_accuracy": ratio(level["within_one"], level["n"]),
        },
        "event_daily_level_ge_4": {
            **event,
            "csi": ratio(event["hits"], event_denominator),
            "pod": ratio(event["hits"], event["hits"] + event["misses"]),
            "far": ratio(event["false_alarms"], event["hits"] + event["false_alarms"]),
        },
        "process_level_ge_3": {
            **process,
            "pod": ratio(process["tp"], process["tp"] + process["fn"]),
            "far": ratio(process["fp"], process["tp"] + process["fp"]),
        },
        "turning_level_ge_3_intention_to_treat": {
            **turning,
            "detection_rate": ratio(
                turning["detected"], turning["n_truth_processes"]
            ),
            "start_mae_days": ratio(
                turning["start_absolute_error_sum"], turning["n_truth_processes"]
            ),
            "peak_mae_days": ratio(
                turning["peak_absolute_error_sum"], turning["n_truth_processes"]
            ),
            "end_mae_days": ratio(
                turning["end_absolute_error_sum"], turning["n_truth_processes"]
            ),
            "missing_process_penalty_days": "forecast horizon per phase",
        },
        "primary_pollutant": {
            "set_membership_accuracy": ratio(
                counts["primary"]["classification"]["correct"],
                counts["primary"]["classification"]["n"],
            ),
            "n_days": counts["primary"]["classification"]["n"],
            "tied_truth_days": counts["primary"]["classification"][
                "tied_truth_days"
            ],
            "per_class_f1_excludes_tied_truth_days": True,
            "macro_f1_supported_classes": sum(supported) / len(supported) if supported else None,
            "one_vs_rest_by_pollutant": primary,
        },
        "interval_80pct": intervals,
    }


def metric_improvements(model: dict, baseline: dict) -> dict[str, float | None]:
    """Return positive-is-better paired improvements for graduation metrics."""
    pairs = {
        "level_ordinal_mae": (baseline["level"]["ordinal_mae"], model["level"]["ordinal_mae"]),
        "level_exact_accuracy": (model["level"]["exact_accuracy"], baseline["level"]["exact_accuracy"]),
        "event_csi": (model["event_daily_level_ge_4"]["csi"], baseline["event_daily_level_ge_4"]["csi"]),
        "event_pod": (model["event_daily_level_ge_4"]["pod"], baseline["event_daily_level_ge_4"]["pod"]),
        "event_far": (baseline["event_daily_level_ge_4"]["far"], model["event_daily_level_ge_4"]["far"]),
        "process_pod": (model["process_level_ge_3"]["pod"], baseline["process_level_ge_3"]["pod"]),
        "process_far": (baseline["process_level_ge_3"]["far"], model["process_level_ge_3"]["far"]),
        "primary_macro_f1": (
            model["primary_pollutant"]["macro_f1_supported_classes"],
            baseline["primary_pollutant"]["macro_f1_supported_classes"],
        ),
        "primary_set_membership_accuracy": (
            model["primary_pollutant"]["set_membership_accuracy"],
            baseline["primary_pollutant"]["set_membership_accuracy"],
        ),
    }
    for phase in ("start", "peak", "end"):
        pairs[f"turning_{phase}_mae_days"] = (
            baseline["turning_level_ge_3_intention_to_treat"][f"{phase}_mae_days"],
            model["turning_level_ge_3_intention_to_treat"][f"{phase}_mae_days"],
        )
    output = {}
    for name, (better, worse) in pairs.items():
        output[name] = None if better is None or worse is None else better - worse
    for pollutant in INTERVAL_FIELDS:
        model_row = model["interval_80pct"][pollutant]
        baseline_row = baseline["interval_80pct"][pollutant]
        if model_row["coverage"] is None or baseline_row["coverage"] is None:
            output[f"{pollutant}_coverage_calibration"] = None
        else:
            output[f"{pollutant}_coverage_calibration"] = (
                abs(baseline_row["coverage"] - 0.8) - abs(model_row["coverage"] - 0.8)
            )
        model_score, baseline_score = model_row["mean_interval_score"], baseline_row["mean_interval_score"]
        output[f"{pollutant}_interval_score"] = (
            None if model_score is None or baseline_score is None else baseline_score - model_score
        )
        model_mae, baseline_mae = model_row["midpoint_mae"], baseline_row["midpoint_mae"]
        output[f"{pollutant}_midpoint_mae"] = (
            None if model_mae is None or baseline_mae is None else baseline_mae - model_mae
        )
    return output
