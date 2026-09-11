"""确定性评估层：信号判据单元测试。"""
from sitian.assess import climatology_context, daily_signals, guidance_meta, obs_trend
from datetime import datetime, timedelta


def test_obs_trend():
    obs = {"PM2.5": {"times": [(datetime(2025, 1, 1) + timedelta(hours=h)).isoformat() for h in range(48)],
                     "series": {"c": [50.0] * 24 + [80.0] * 24}}}
    t = obs_trend(obs, "c")
    assert t["PM2.5"]["change_24h"] == 30.0


def test_cold_air_flag_needs_north_and_jump():
    diag = {"daily": {
        "d1": {"wind_speed_ms": 1.5, "wind_dir_deg": 180},
        "d2": {"wind_speed_ms": 4.0, "wind_dir_deg": 330},   # 北向 + 跳升 2.5 → cold_air
        "d3": {"wind_speed_ms": 4.5, "wind_dir_deg": 330},   # 跳升不足
    }}
    s = daily_signals(diag)
    assert "cold_air" in s["d2"]["flags"] and "cold_air" not in s["d3"]["flags"]


def test_stagnation_flag():
    diag = {"daily": {"d1": {"wind_speed_ms": 1.0, "wind_dir_deg": 90, "rh_pct": 75, "blh_max_m": 400}}}
    assert "stagnation" in daily_signals(diag)["d1"]["flags"]


def test_guidance_spread():
    g = {"sources": {"a": {"daily_pm25": {"d": 100}}, "b": {"daily_pm25": {"d": 40}}}}
    m = guidance_meta(g)
    assert m["d"]["spread"] == 60.0 and m["d"]["n_sources"] == 2


def test_climatology_position():
    clim = {"01": {"pm25": {"p50": 30, "p75": 60, "p90": 90}}}
    ctx = climatology_context(clim, 1, {"PM2.5": {"last24h_mean": 70.0}})
    assert ctx["pm25_position"] == "above_p75"
    assert climatology_context(None, 1, {}) is None
