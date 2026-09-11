import json

from sitian.case import CaseBundle
from sitian.integrations.verl_reward import compute_score
from sitian.scoring import score_forecast


def test_single_turn_reward_uses_full_multi_pollutant_truth(tmp_path):
    issue = "2026-06-01"
    days = [f"2026-06-0{i}" for i in range(2, 7)]
    bundle = CaseBundle(
        case_id="multi",
        issue_date=issue,
        region="test",
        horizon=5,
        meta={"multi_pollutant": True},
        truth={"daily": {
            day: {"pm25_avg": 20, "pm10_avg": 35, "o3_8h": 220,
                  "so2_avg": 5, "no2_avg": 15, "co_avg": 0.5}
            for day in days
        }},
    )
    bundle.save(tmp_path)
    forecast = {
        "issue_date": issue,
        "region": "test",
        "daily": [
            {"date": day, "aqi_level": 1, "primary_pollutant": None,
             "pm25_range": [10, 30], "pm10_range": [20, 50],
             "o3_range": [80, 120]}
            for day in days
        ],
        "process": {"has_event": False},
        "evidence": [],
        "confidence": "medium",
    }
    actual = compute_score(json.dumps(forecast), str(tmp_path))
    expected = score_forecast(
        forecast, bundle.truth_daily_full(), issue_date=issue, horizon=5, region="test"
    ).composite
    pm25_only = score_forecast(
        forecast, bundle.truth_daily(), issue_date=issue, horizon=5, region="test"
    ).composite
    assert actual == expected
    assert actual < pm25_only
