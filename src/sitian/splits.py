"""时间切分工具：以 forecast target grain 防止滚动窗口重叠。"""
from __future__ import annotations

from datetime import date, timedelta


def purged_calibration_split(
    rows: list[dict], *, calibration_fraction: float = 0.2, purge_days: int = 5,
) -> tuple[list[dict], list[dict], dict]:
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    case_dates = sorted({(r["issue_date"], r["case_id"]) for r in rows})
    if len(case_dates) < 2:
        raise ValueError("need at least two cases")
    cut = max(1, int(len(case_dates) * (1 - calibration_fraction)))
    calibration_start = case_dates[min(cut, len(case_dates) - 1)][0]
    fit_before = (date.fromisoformat(calibration_start) - timedelta(days=purge_days)).isoformat()
    calibration = [r for r in rows if r["issue_date"] >= calibration_start]
    fitting = [r for r in rows if r["issue_date"] < fit_before]
    if not fitting or not calibration:
        raise ValueError("purged calibration split produced an empty side")

    fit_cases = {r["case_id"] for r in fitting}
    calibration_cases = {r["case_id"] for r in calibration}
    fit_targets = {(r["region"], r["day"]) for r in fitting}
    calibration_targets = {(r["region"], r["day"]) for r in calibration}
    overlap = fit_targets & calibration_targets
    if overlap:
        raise AssertionError(f"fit/calibration target leakage: {len(overlap)} city-days")
    meta = {
        "calibration_fraction": calibration_fraction,
        "purge_days": purge_days,
        "calibration_start_issue_date": calibration_start,
        "fit_issue_range": [min(r["issue_date"] for r in fitting),
                            max(r["issue_date"] for r in fitting)],
        "calibration_issue_range": [min(r["issue_date"] for r in calibration),
                                    max(r["issue_date"] for r in calibration)],
        "fit_cases": len(fit_cases),
        "calibration_cases": len(calibration_cases),
        "purged_cases": len({r["case_id"] for r in rows}) - len(fit_cases) - len(calibration_cases),
        "fit_calibration_city_day_overlap": 0,
    }
    return fitting, calibration, meta


def purged_selection_calibration_split(
    rows: list[dict], *, calibration_fraction: float = 0.2,
    selection_fraction_of_development: float = 0.125,
    purge_days: int = 5,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    """Create chronological train-only model-selection and calibration folds.

    The final calibration fold is never used to choose an algorithm.  Candidate
    algorithms are fitted on ``selection_fit`` and compared on ``selection``;
    the chosen algorithm may then be refitted on all ``final_fit`` rows before
    its interval width is estimated on the still untouched ``calibration`` fold.
    Both boundaries carry a forecast-horizon embargo.
    """
    if not 0 < selection_fraction_of_development < 1:
        raise ValueError("selection_fraction_of_development must be between 0 and 1")
    final_fit, calibration, calibration_meta = purged_calibration_split(
        rows, calibration_fraction=calibration_fraction, purge_days=purge_days
    )
    selection_fit, selection, selection_meta = purged_calibration_split(
        final_fit,
        calibration_fraction=selection_fraction_of_development,
        purge_days=purge_days,
    )
    selection_targets = {(row["region"], row["day"]) for row in selection}
    calibration_targets = {(row["region"], row["day"]) for row in calibration}
    if selection_targets & calibration_targets:
        raise AssertionError("selection/calibration target leakage")
    meta = {
        **calibration_meta,
        "model_selection_fraction_of_development": selection_fraction_of_development,
        "model_selection": selection_meta,
        "model_selection_calibration_city_day_overlap": 0,
        "final_point_fit_cases": len({row["case_id"] for row in final_fit}),
        "final_point_fit_issue_range": [
            min(row["issue_date"] for row in final_fit),
            max(row["issue_date"] for row in final_fit),
        ],
    }
    return selection_fit, selection, final_fit, calibration, meta
