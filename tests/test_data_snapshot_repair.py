import json

import pytest

from scripts.repair_data_snapshot import copy_corrected, profile, repair_diagnostics


def test_snapshot_repairs_labels_without_mutating_source(tmp_path):
    source = tmp_path / "original"
    case = source / "train" / "example"
    case.mkdir(parents=True)
    original = case / "diagnostics.json"
    original.write_text(json.dumps({"daily": {"2026-01-02": {
        "wind_dir_deg": 0, "wind_dir": "NNE", "rain_mm": 2,
    }}}))
    before = original.read_bytes()
    out = tmp_path / "snapshot"
    receipt = copy_corrected(source, out)
    assert receipt["changed_day_labels"] == 1
    result = json.loads((out / "train/example/diagnostics.json").read_text())
    assert result["daily"]["2026-01-02"] == {"wind_dir_deg": 0, "wind_dir": "N", "rain_mm": 2}
    assert original.read_bytes() == before
    assert repair_diagnostics(result) == []
    with pytest.raises(FileExistsError):
        copy_corrected(source, out)
    with pytest.raises(ValueError, match="separate"):
        copy_corrected(source, source / "nested")


def test_profile_deduplicates_event_targets_not_forecast_cases(tmp_path):
    cases = []
    for i, issue in enumerate(("2025-12-30", "2025-12-31")):
        case = tmp_path / str(i)
        case.mkdir()
        docs = {
            "case": {"region": "x", "issue_date": issue, "horizon": 2 - i,
                     "meta": {"stratum": "pm10_primary"}},
            "truth": {"daily": {d: {"pm25_avg": 150, "pm10_avg": 400, "o3_8h": 20,
                        "so2_avg": 5, "no2_avg": 5, "co_avg": 0.2}
                        for d in (["2025-12-31", "2026-01-01"] if i == 0 else ["2026-01-01"])}},
            "guidance": {"sources": {"cams": {}}}, "diagnostics": {"daily": {}},
        }
        for name, doc in docs.items():
            (case / f"{name}.json").write_text(json.dumps(doc))
        cases.append(case)
    result = profile(cases)
    assert result["event_cases"] == 2
    assert result["target_day_occurrences"] == 3
    assert result["event_unique_city_days"] == 2
    assert result["event_cities"] == 1
    assert result["strata"] == {"pm10_primary": 2}
    with pytest.raises(ValueError, match="Duplicate"):
        profile([cases[0], cases[0]])
