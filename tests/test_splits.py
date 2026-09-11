from datetime import date, timedelta

from sitian.splits import (
    purged_calibration_split, purged_selection_calibration_split,
)


def test_purged_calibration_has_no_forecast_target_overlap():
    rows = []
    base = date(2026, 1, 1)
    for i in range(20):
        issue = base + timedelta(days=i)
        case_id = f"city_{issue.isoformat()}"
        for lead in range(1, 6):
            rows.append({
                "case_id": case_id,
                "issue_date": issue.isoformat(),
                "region": "city",
                "day": (issue + timedelta(days=lead)).isoformat(),
            })
    fitting, calibration, meta = purged_calibration_split(
        rows, calibration_fraction=0.2, purge_days=5)
    fit_targets = {(r["region"], r["day"]) for r in fitting}
    cal_targets = {(r["region"], r["day"]) for r in calibration}
    assert not (fit_targets & cal_targets)
    assert meta["fit_calibration_city_day_overlap"] == 0
    assert meta["purged_cases"] == 5


def test_three_way_train_only_selection_keeps_calibration_untouched():
    rows = []
    base = date(2025, 1, 1)
    for i in range(80):
        issue = base + timedelta(days=i)
        for city in ("a", "b"):
            case_id = f"{city}_{issue.isoformat()}"
            for lead in range(1, 6):
                rows.append({
                    "case_id": case_id,
                    "issue_date": issue.isoformat(),
                    "region": city,
                    "day": (issue + timedelta(days=lead)).isoformat(),
                })
    fit, selection, final_fit, calibration, meta = (
        purged_selection_calibration_split(rows, purge_days=5)
    )

    def targets(values):
        return {(row["region"], row["day"]) for row in values}

    assert not (targets(fit) & targets(selection))
    assert not (targets(final_fit) & targets(calibration))
    assert not (targets(selection) & targets(calibration))
    assert max(row["issue_date"] for row in selection) < min(
        row["issue_date"] for row in calibration
    )
    assert meta["fit_calibration_city_day_overlap"] == 0
    assert meta["model_selection"]["fit_calibration_city_day_overlap"] == 0
