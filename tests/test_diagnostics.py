"""行为归因与气候态基线。"""
from sitian.case import CaseBundle
from sitian.diagnostics import strategy_attribution


def _bundle():
    return CaseBundle(
        case_id="x_2026-01-05", issue_date="2026-01-05", region="x", horizon=2, regions=["x"],
        observations={"PM2.5": {"times": [], "series": {"x": [100.0] * 24}}},
        diagnostics={"daily": {}},
        guidance={"sources": {"cams": {"daily_pm25": {"2026-01-06": 40.0, "2026-01-07": 45.0}}}},
        meta={"climatology": {"01": {"pm25": {"p50": 60.0, "p75": 90.0, "p90": 120.0}}}},
    )


def _fc(vals):
    return {"daily": [{"date": d, "pm25_range": [v - 10, v + 10]}
                      for d, v in zip(("2026-01-06", "2026-01-07"), vals)]}


def test_attribution_nearest():
    b = _bundle()
    assert strategy_attribution(_fc([42, 44]), b)["nearest"] == "guidance:cams"
    assert strategy_attribution(_fc([99, 101]), b)["nearest"] == "persistence"
    assert strategy_attribution(_fc([61, 59]), b)["nearest"] == "climatology"


def test_climatology_baseline_runs():
    from sitian.agents.scripted import ClimatologyAgent
    b = _bundle()
    agent = ClimatologyAgent(b)
    agent.begin({"issue_date": b.issue_date, "region": "x",
                 "forecast_example": {"daily": [{"date": "2026-01-06"}, {"date": "2026-01-07"}]}})
    action = agent.act({})
    daily = action["args"]["forecast"]["daily"]
    assert action["name"] == "submit_forecast"
    assert all(d["pm25_range"][0] <= 60.0 <= d["pm25_range"][1] for d in daily)
