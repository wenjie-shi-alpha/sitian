import pytest

from sitian.schema import (
    EVIDENCE_TYPE_ALIASES,
    example_forecast,
    example_forecast_compact,
    forecast_tool_schema,
    pm25_to_iaqi,
    pm25_to_level,
    validate_forecast,
)


def test_pm25_level_boundaries():
    assert pm25_to_level(0) == 1
    assert pm25_to_level(35) == 1
    assert pm25_to_level(35.1) == 2
    assert pm25_to_level(75) == 2
    assert pm25_to_level(115) == 3
    assert pm25_to_level(150) == 4
    assert pm25_to_level(250) == 5
    assert pm25_to_level(251) == 6
    with pytest.raises(ValueError):
        pm25_to_level(-1)


def test_pm25_iaqi():
    assert pm25_to_iaqi(35) == pytest.approx(50)
    assert pm25_to_iaqi(75) == pytest.approx(100)
    assert pm25_to_iaqi(55) == pytest.approx(75)
    assert pm25_to_iaqi(9999) == 500.0


def test_example_forecast_is_valid():
    obj = example_forecast("2025-12-18", 6, "beijing")
    fc, errors = validate_forecast(obj, issue_date="2025-12-18", horizon=6, region="beijing")
    assert errors == []
    assert fc is not None
    assert len(fc.daily) == 6
    assert fc.daily[0].date == "2025-12-19"


def test_policy_schema_uses_intervals_not_categorical_confidence():
    example = example_forecast("2026-06-01", 5, "beijing", multi_pollutant=True)
    policy_properties = forecast_tool_schema(
        "2026-06-01", 5, "beijing", multi_pollutant=True
    )["properties"]["forecast"]["properties"]
    assert "confidence" not in example
    assert "confidence" not in policy_properties
    assert {"pm25_lo", "pm25_hi", "pm10_lo", "pm10_hi", "o3_lo", "o3_hi"}.issubset(
        policy_properties
    )


def test_optional_gas_columns_are_visible_and_roundtrip_in_all_formats():
    from sitian.schema import forecast_format_description
    issue = "2025-12-18"
    brief = forecast_format_description(issue, 2, "beijing", multi_pollutant=True)
    schema = forecast_tool_schema(issue, 2, "beijing", multi_pollutant=True)["properties"]["forecast"]
    assert set(brief["optional_columns"]) <= set(schema["properties"])
    assert not set(brief["optional_columns"]) & set(schema["required"])
    assert "mg/m³" in brief["optional_columns"]["co_lo"]
    base = {"issue_date": issue, "region": "beijing"}
    table = {"pm25_range": [[20, 20], [20, 20]], "no2_range": [[200, 210], [210, 220]],
             "co_range": [[0.5, 1.5], [0.5, 1.5]]}
    compact, errors = validate_forecast({**base, "daily": table}, issue_date=issue, horizon=2)
    assert not errors
    flat = {**base, "pm25_lo": [20, 20], "pm25_hi": [20, 20],
            "no2_lo": [200, 210], "no2_hi": [210, 220], "co_lo": [0.5, 0.5], "co_hi": [1.5, 1.5]}
    for payload in (flat, compact.to_dict()):
        fc, errors = validate_forecast(payload, issue_date=issue, horizon=2)
        assert not errors and fc.to_dict() == compact.to_dict()
        assert fc.daily[0].primary_pollutant == "NO2"
    for bad in ({**flat, "no2_hi": [210]}, {**flat, "no2_lo": [-1, 210]},
                {**flat, "co_hi": [float("inf"), 1.5]}, {**flat, "co_hi": [True, 1.5]},
                {**base, "daily": {**table, "co_range": None}}):
        fc, errors = validate_forecast(bad, issue_date=issue, horizon=2)
        assert fc is None and errors


def test_process_head_is_derived_from_daily_intervals():
    """schema-v0.6.0: the policy submits intervals; level/process are derived."""
    obj = example_forecast("2025-12-18", 6, "beijing")
    fc, errors = validate_forecast(
        obj, issue_date="2025-12-18", horizon=6, region="beijing"
    )
    assert errors == [] and fc is not None
    assert [d.aqi_level for d in fc.daily] == [2] * 6
    assert fc.process is not None and fc.process.has_event is False

    obj["daily"][2]["pm25_range"] = [90, 110]   # midpoint 100 -> level 3
    obj["daily"][3]["pm25_range"] = [140, 160]  # midpoint 150 -> level 4 (peak)
    obj["daily"][4]["pm25_range"] = [80, 90]    # level 3
    # A supplied, contradictory process head is ignored rather than rejected.
    obj["process"] = {"has_event": False, "start": None, "peak": None, "end": None}
    fc, errors = validate_forecast(
        obj, issue_date="2025-12-18", horizon=6, region="beijing"
    )
    assert errors == [] and fc is not None
    assert [d.aqi_level for d in fc.daily] == [2, 2, 3, 4, 3, 2]
    assert fc.process.has_event is True
    assert (fc.process.start, fc.process.peak, fc.process.end) == (
        obj["daily"][2]["date"], obj["daily"][3]["date"], obj["daily"][4]["date"]
    )


def test_national_multi_pollutant_contract_requires_pm10_and_o3_ranges():
    obj = example_forecast("2025-12-18", 6, "beijing", multi_pollutant=True)
    fc, errors = validate_forecast(
        obj, issue_date="2025-12-18", horizon=6, region="beijing",
        require_pm10_range=True, require_o3_range=True,
    )
    assert errors == [] and fc is not None
    del obj["daily"][0]["pm10_range"]
    fc, errors = validate_forecast(
        obj, issue_date="2025-12-18", horizon=6, region="beijing",
        require_pm10_range=True, require_o3_range=True,
    )
    assert fc is None
    assert any("pm10_range is required" in error for error in errors)

    obj = example_forecast("2025-12-18", 6, "beijing", multi_pollutant=True)
    del obj["daily"][0]["o3_range"]
    fc, errors = validate_forecast(
        obj, issue_date="2025-12-18", horizon=6, region="beijing",
        require_pm10_range=True, require_o3_range=True,
    )
    assert fc is None
    assert any("o3_range is required" in error for error in errors)


def test_level_and_primary_are_derived_from_interval_midpoints():
    obj = example_forecast("2026-06-01", 2, "test", multi_pollutant=True)
    obj["daily"][0].update(
        aqi_level=1, primary_pollutant="PM2.5",   # contradictory, ignored
        pm25_range=[200, 250], pm10_range=[10, 20], o3_range=[20, 40],
    )
    obj["daily"][1].update(
        pm25_range=[10, 20], pm10_range=[10, 20], o3_range=[20, 40],
    )
    fc, errors = validate_forecast(
        obj, issue_date="2026-06-01", horizon=2, region="test",
        require_pm10_range=True, require_o3_range=True,
    )
    assert errors == [] and fc is not None
    assert fc.daily[0].aqi_level == 5 and fc.daily[0].primary_pollutant == "PM2.5"
    assert fc.daily[1].aqi_level == 1 and fc.daily[1].primary_pollutant is None


def test_derived_primary_follows_the_highest_iaqi_midpoint():
    obj = example_forecast("2026-06-01", 1, "test", multi_pollutant=True)
    obj["daily"][0].update(
        pm25_range=[36, 45], pm10_range=[115, 119], o3_range=[20, 40],
    )
    fc, errors = validate_forecast(
        obj, issue_date="2026-06-01", horizon=1, region="test",
        require_pm10_range=True, require_o3_range=True,
    )
    assert errors == [] and fc is not None
    assert fc.daily[0].primary_pollutant == "PM10"
    assert fc.daily[0].aqi_level == 2


def test_supplied_categorical_heads_are_type_checked_but_not_used():
    obj = example_forecast("2026-06-01", 1, "test")
    obj["daily"][0]["aqi_level"] = "bad"
    fc, errors = validate_forecast(
        obj, issue_date="2026-06-01", horizon=1, region="test"
    )
    assert fc is None and any("aqi_level, if given" in error for error in errors)
    obj = example_forecast("2026-06-01", 1, "test")
    obj["process"] = "not-an-object"
    fc, errors = validate_forecast(
        obj, issue_date="2026-06-01", horizon=1, region="test"
    )
    assert fc is None and any("process, if given" in error for error in errors)


def test_policy_tool_schema_has_no_categorical_or_process_heads():
    properties = forecast_tool_schema(
        "2026-06-01", 5, "beijing", multi_pollutant=True
    )["properties"]["forecast"]["properties"]
    assert "process" not in properties
    assert "daily" not in properties
    assert "aqi_level" not in properties and "primary_pollutant" not in properties
    assert {"pm25_lo", "pm25_hi", "pm10_lo", "pm10_hi", "o3_lo", "o3_hi"} <= set(properties)


def _base_obj():
    return example_forecast("2025-12-18", 6, "beijing")


@pytest.mark.parametrize("alias,canonical", EVIDENCE_TYPE_ALIASES.items())
def test_evidence_type_aliases_are_normalized(alias, canonical):
    obj = _base_obj()
    obj["evidence"] = [{"type": alias, "claim": "来自已调用工具的证据"}]
    fc, errors = validate_forecast(
        obj, issue_date="2025-12-18", horizon=6, region="beijing"
    )
    assert errors == []
    assert fc is not None
    assert fc.evidence == [{"type": canonical, "claim": "来自已调用工具的证据"}]
    # validator 不应原地改写模型的原始提交，轨迹仍可审计。
    assert obj["evidence"][0]["type"] == alias


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda o: o["daily"].pop(), "exactly 6"),
        (lambda o: o["daily"][0].update(date="2025-12-25"), "daily[0].date"),
        (lambda o: o["daily"][1].update(aqi_level=9), "aqi_level"),
        (lambda o: o["daily"][2].update(pm25_range=[-5, 40]), "lo <= hi"),
        (lambda o: o["daily"][3].update(primary_pollutant="PM99"), "primary_pollutant"),
        (lambda o: o.update(region="tianjin"), "region"),
        (lambda o: o.update(process="not-an-object"), "process, if given"),
        (lambda o: o.update(evidence=[{"type": "vibes", "claim": "x"}]), "evidence[0].type"),
        (lambda o: o.update(confidence="certain"), "confidence"),
    ],
)
def test_validation_errors(mutate, fragment):
    obj = _base_obj()
    mutate(obj)
    fc, errors = validate_forecast(obj, issue_date="2025-12-18", horizon=6, region="beijing")
    assert fc is None
    assert any(fragment in e for e in errors), errors


def test_compact_daily_table_is_equivalent_to_per_day_objects():
    compact = example_forecast_compact("2026-06-01", 5, "test", multi_pollutant=True)
    fc, errors = validate_forecast(
        compact, issue_date="2026-06-01", horizon=5, region="test",
        require_pm10_range=True, require_o3_range=True,
    )
    assert errors == [] and fc is not None
    assert [d.date for d in fc.daily] == [f"2026-06-0{k}" for k in range(2, 7)]
    assert fc.daily[0].pm25_range == (30.0, 60.0) and fc.daily[4].o3_range == (130.0, 190.0)
    short = dict(compact, daily={"pm25_range": compact["daily"]["pm25_range"][:4],
                                 "pm10_range": compact["daily"]["pm10_range"],
                                 "o3_range": compact["daily"]["o3_range"]})
    fc, errors = validate_forecast(
        short, issue_date="2026-06-01", horizon=5, region="test",
        require_pm10_range=True, require_o3_range=True,
    )
    assert fc is None and any("exactly 5" in error for error in errors)
    schema = forecast_tool_schema("2026-06-01", 5, "test", multi_pollutant=True)
    forecast = schema["properties"]["forecast"]
    assert "daily" not in forecast["properties"]
    assert forecast["properties"]["pm25_lo"]["minItems"] == 5
    assert "process" not in forecast["required"]


def test_flat_column_form_is_equivalent():
    flat = {"issue_date": "2026-06-01", "region": "test",
            "pm25_lo": [30, 35, 40, 45, 50], "pm25_hi": [60, 65, 70, 75, 80],
            "pm10_lo": [50, 60, 70, 80, 90], "pm10_hi": [100, 110, 120, 130, 140],
            "o3_lo": [90, 100, 110, 120, 130], "o3_hi": [150, 160, 170, 180, 190],
            "evidence": []}
    fc, errors = validate_forecast(
        flat, issue_date="2026-06-01", horizon=5, region="test",
        require_pm10_range=True, require_o3_range=True,
    )
    assert errors == [] and fc is not None
    assert fc.daily[0].pm25_range == (30.0, 60.0) and fc.daily[4].o3_range == (130.0, 190.0)
    assert fc.daily[2].date == "2026-06-04"
    bad = dict(flat, pm25_hi=[60, 65, 70, 75])
    fc, errors = validate_forecast(
        bad, issue_date="2026-06-01", horizon=5, region="test",
        require_pm10_range=True, require_o3_range=True,
    )
    assert fc is None and any("exactly 5" in error for error in errors)


def test_swapped_interval_bounds_are_accepted_as_the_same_interval():
    obj = example_forecast("2026-06-01", 1, "test")
    obj["daily"][0]["pm25_range"] = [70, 40]
    fc, errors = validate_forecast(obj, issue_date="2026-06-01", horizon=1, region="test")
    assert errors == [] and fc is not None
    assert fc.daily[0].pm25_range == (40.0, 70.0)
