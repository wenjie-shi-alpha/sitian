"""自我会商合成与离散度诊断。"""
from sitian.ensemble import aggregate_forecasts, ensemble_spread


def _fc(mids, levels, has_event=False, peak=None):
    return {
        "issue_date": "2026-01-05", "region": "x",
        "daily": [{"date": f"2026-01-0{6+i}", "aqi_level": lv,
                   "pm25_range": [m - 10, m + 10], "primary_pollutant": "PM2.5"}
                  for i, (m, lv) in enumerate(zip(mids, levels))],
        "process": {"has_event": has_event, "start": peak, "peak": peak, "end": peak},
        "evidence": [{"type": "observation", "claim": "x"}], "confidence": "medium",
    }


def test_aggregate_majority_and_median():
    ens = [_fc([50, 80], [2, 3]), _fc([54, 90], [2, 4]), _fc([100, 85], [3, 3])]
    agg = aggregate_forecasts(ens)
    assert agg["daily"][0]["aqi_level"] == 2          # 多数票
    assert agg["daily"][0]["pm25_range"] == [44.0, 64.0]  # 中位端点 (40,44,90)/(60,64,110)
    assert agg["daily"][1]["aqi_level"] == 3


def test_aggregate_event_vote():
    ens = [_fc([50, 60], [2, 2]),
           _fc([50, 60], [2, 2], has_event=True, peak="2026-01-07"),
           _fc([50, 60], [2, 2], has_event=True, peak="2026-01-07")]
    agg = aggregate_forecasts(ens)
    assert agg["process"]["has_event"] is True and agg["process"]["peak"] == "2026-01-07"
    # 少数报事件 → 合成无事件
    agg2 = aggregate_forecasts([ens[0], ens[0], ens[1]])
    assert agg2["process"]["has_event"] is False


def test_spread_metrics():
    tight = ensemble_spread([_fc([50, 60], [2, 2]), _fc([52, 61], [2, 2])])
    wide = ensemble_spread([_fc([30, 60], [1, 2]), _fc([90, 62], [3, 2])])
    assert wide["pm25_mid_std"] > tight["pm25_mid_std"]
    assert wide["level_disagreement"] > tight["level_disagreement"]
    assert ensemble_spread([_fc([50], [2])])["pm25_mid_std"] == 0.0


def test_single_member_passthrough():
    f = _fc([50], [2])
    assert aggregate_forecasts([f]) == f
