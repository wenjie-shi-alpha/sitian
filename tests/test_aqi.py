"""六项 IAQI / 日 AQI / 首要污染物（HJ 633-2012）与多污染物评分路径。"""
import pytest

from sitian.schema import (
    AQI_STANDARD_2012,
    AQI_STANDARD_2026,
    aqi_standard_for_date,
    aqi_to_level,
    daily_aqi,
    iaqi,
    pm25_to_iaqi,
    pm25_to_level,
)
from sitian.scoring import score_forecast


class TestIaqi:
    def test_breakpoint_values_exact(self):
        # 分段端点必须精确命中（HJ 633 表1）
        assert iaqi("PM2.5", 35) == 50
        assert iaqi("PM2.5", 75) == 100
        assert iaqi("PM10", 150) == 100
        assert iaqi("O3", 160) == 100
        assert iaqi("SO2", 475) == 150
        assert iaqi("NO2", 280) == 200
        assert iaqi("CO", 4) == 100

    def test_interpolation(self):
        assert iaqi("PM2.5", 55) == pytest.approx(75.0)  # (55-35)/(75-35)*50+50
        assert iaqi("PM10", 100) == pytest.approx(75.0)

    def test_o3_caps_at_300(self):
        assert iaqi("O3", 800) == 300
        assert iaqi("O3", 900) == 300  # 8h 口径超 800 封顶（标准改用 1h 评价）

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            iaqi("PM2.5", -1)

    def test_pm25_wrapper_consistent(self):
        for v in (0, 20, 35, 60, 115, 200, 400, 600):
            assert pm25_to_iaqi(v) == iaqi("PM2.5", v)

    def test_2026_particle_breakpoints(self):
        assert iaqi("PM2.5", 60, standard=AQI_STANDARD_2026) == 100
        assert iaqi("PM10", 120, standard=AQI_STANDARD_2026) == 100
        assert iaqi("PM2.5", 60, standard=AQI_STANDARD_2012) < 100
        assert iaqi("PM10", 120, standard=AQI_STANDARD_2012) < 100

    def test_standard_switch_date(self):
        assert aqi_standard_for_date("2026-02-28") == AQI_STANDARD_2012
        assert aqi_standard_for_date("2026-03-01") == AQI_STANDARD_2026

    def test_2026_level_is_stricter_for_pm25(self):
        assert pm25_to_level(70, standard=AQI_STANDARD_2012) == 2
        assert pm25_to_level(70, standard=AQI_STANDARD_2026) == 3


class TestDailyAqi:
    def test_clean_day_no_primary(self):
        r = daily_aqi({"PM2.5": 20, "O3": 80})
        assert r["level"] == 1 and r["primary"] == []

    def test_primary_is_max_iaqi(self):
        r = daily_aqi({"PM2.5": 60, "O3": 180, "PM10": 90})
        assert r["primary"] == ["O3"]  # IAQI: PM2.5≈81, O3≈118, PM10≈70
        assert r["level"] == 3

    def test_pm10_primary(self):
        r = daily_aqi({"PM2.5": 90, "PM10": 500})
        assert r["primary"] == ["PM10"] and r["level"] == 6

    def test_level_consistent_with_pm25_only(self):
        # 单 PM2.5 时与旧 pm25_to_level 一致
        for v in (10, 35, 50, 75, 100, 115, 150, 200, 250, 300):
            assert daily_aqi({"PM2.5": v})["level"] == pm25_to_level(v)

    def test_aqi_to_level_bounds(self):
        assert [aqi_to_level(x) for x in (50, 51, 100, 150, 200, 300, 301)] == [1, 2, 2, 3, 4, 5, 6]


def _forecast(days, extra=None):
    daily = []
    for d, level, rng in days:
        item = {
            "date": d, "aqi_level": level, "pm25_range": rng,
            "primary_pollutant": (
                None if level == 1 else
                ("PM2.5" if pm25_to_level(rng[1], standard=AQI_STANDARD_2026) >= level
                 else "NO2")
            ),
        }
        if extra and d in extra:
            item.update(extra[d])
        daily.append(item)
    event_rows = [item for item in daily if item["aqi_level"] >= 3]
    if event_rows:
        peak = max(event_rows, key=lambda item: item["aqi_level"])["date"]
        process = {"has_event": True, "start": event_rows[0]["date"],
                   "peak": peak, "end": event_rows[-1]["date"]}
    else:
        process = {"has_event": False}
    return {"issue_date": "2026-06-01", "region": "beijing", "daily": daily,
            "process": process, "evidence": [], "confidence": "medium"}


class TestMultiPollutantScoring:
    DAYS = ["2026-06-02", "2026-06-03"]

    def test_legacy_float_truth_unchanged(self):
        fc = _forecast([(self.DAYS[0], 2, [40, 70]), (self.DAYS[1], 2, [40, 70])])
        truth = {self.DAYS[0]: 50.0, self.DAYS[1]: 60.0}
        r = score_forecast(fc, truth, issue_date="2026-06-01", horizon=2)
        assert r.valid and r.components["primary"] is None
        assert "primary" not in r.weights_used

    def test_o3_event_day_uses_full_aqi_level(self):
        # PM2.5 优等但 O3 中度污染：等级真值必须按全 AQI 判
        fc = _forecast([(self.DAYS[0], 4, [10, 40]), (self.DAYS[1], 1, [10, 40])],
                       extra={self.DAYS[0]: {"primary_pollutant": "O3", "o3_range": [200, 260]},
                              self.DAYS[1]: {"o3_range": [50, 100]}})
        truth = {self.DAYS[0]: {"PM2.5": 20, "O3": 230},   # IAQI(O3)≈173 → 4 级
                 self.DAYS[1]: {"PM2.5": 20, "O3": 80}}
        r = score_forecast(fc, truth, issue_date="2026-06-01", horizon=2)
        assert r.valid
        assert r.components["level"] == 1.0
        assert r.components["primary"] == 1.0
        assert "primary" in r.weights_used

    def test_missing_o3_range_is_invalid_when_truth_has_o3(self):
        fc = _forecast([(self.DAYS[0], 2, [40, 70]), (self.DAYS[1], 2, [40, 70])])
        truth = {d: {"PM2.5": 50, "O3": 100} for d in self.DAYS}
        r = score_forecast(fc, truth, issue_date="2026-06-01", horizon=2)
        assert not r.valid
        assert any("o3_range is required" in error for error in r.errors)

    def test_pm10_interval_is_scored_in_full_multi_pollutant_truth(self):
        fc = _forecast(
            [(self.DAYS[0], 2, [40, 70]), (self.DAYS[1], 2, [40, 70])],
            extra={d: {"pm10_range": [60, 100], "o3_range": [80, 120]}
                   for d in self.DAYS},
        )
        truth = {d: {"PM2.5": 50, "PM10": 80, "O3": 100} for d in self.DAYS}
        good = score_forecast(fc, truth, issue_date="2026-06-01", horizon=2)
        assert good.valid
        assert good.details["interval_diagnostics"]["PM10"][0]["covered"] is True

        del fc["daily"][0]["pm10_range"]
        invalid = score_forecast(fc, truth, issue_date="2026-06-01", horizon=2)
        assert not invalid.valid
        assert any("pm10_range is required" in error for error in invalid.errors)

    def test_primary_abstains_on_all_clean_days(self):
        fc = _forecast(
            [(self.DAYS[0], 1, [5, 30]), (self.DAYS[1], 1, [5, 30])],
            extra={d: {"o3_range": [40, 80]} for d in self.DAYS},
        )
        truth = {d: {"PM2.5": 15, "O3": 60} for d in self.DAYS}
        r = score_forecast(fc, truth, issue_date="2026-06-01", horizon=2)
        assert r.components["primary"] is None

    def test_scoring_uses_standard_by_valid_date(self):
        fc = _forecast([(self.DAYS[0], 3, [60, 80]), (self.DAYS[1], 3, [60, 80])])
        truth = {d: 70.0 for d in self.DAYS}
        r = score_forecast(fc, truth, issue_date="2026-06-01", horizon=2)
        assert r.components["level"] == 1.0
        assert set(r.details["aqi_standard_by_day"].values()) == {AQI_STANDARD_2026}
