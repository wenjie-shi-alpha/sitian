from datetime import timedelta

from scripts.build_guidance_bias import DEFAULT_MANIFESTS
from sitian.case import CaseBundle
from sitian.guidance_bias import (
    GuidanceBiasIndex,
    issue_time,
    records_from_bundle,
)


def _artifact(records):
    return {
        "artifact_type": "guidance_bias_history",
        "verification_policy": {"rule": "test"},
        "provenance": {"test": True},
        "records": records,
    }


def _record(*, region="北京", season="DJF", available, error, truth_event=False,
            guidance_event=False):
    return {
        "case_id": f"case-{available}-{error}",
        "region": region,
        "source": "cams",
        "pollutant": "PM2.5",
        "lead_days": 1,
        "target_date": "2025-12-20",
        "target_season": season,
        "verification_available_at": available,
        "error_guidance_minus_truth": error,
        "absolute_error": abs(error),
        "truth_event": truth_event,
        "guidance_event": guidance_event,
    }


def test_main_experiment_bias_history_defaults_to_purged_train_only():
    assert DEFAULT_MANIFESTS == [
        "data/interim/train_without_winter_or_spatial_holdout.json"
    ]


def test_records_use_conservative_verification_time_and_no_raw_truth():
    bundle = CaseBundle(
        case_id="北京_2025-12-19", issue_date="2025-12-19", region="北京", horizon=1,
        guidance={"sources": {"cams": {"daily_pm25": {"2025-12-20": 120}}}},
        truth={"daily": {"2025-12-20": {"pm25_avg": 150}}},
    )
    records = records_from_bundle(bundle)
    assert len(records) == 1
    assert records[0]["verification_available_at"] == "2025-12-21T12:00:00+08:00"
    assert records[0]["error_guidance_minus_truth"] == -30
    assert "truth" not in records[0]
    assert "guidance_value" not in records[0]


def test_strict_time_gate_excludes_equal_cutoff_and_aggregates_prior_only():
    cutoff = issue_time("2026-01-15")
    prior = (cutoff - timedelta(days=2)).isoformat()
    equal = cutoff.isoformat()
    records = [dict(_record(available=prior, error=-20, truth_event=True), case_id=f"past-{i}") for i in range(12)]
    records += [dict(_record(available=equal, error=999), case_id=f"equal-{i}") for i in range(12)]
    result = GuidanceBiasIndex(_artifact(records)).query(
        region="北京", issue_date="2026-01-15", horizon=1
    )
    item = result["series"][0]
    assert item["scope"] == "city_season"
    assert item["n"] == 12
    assert item["mean_error"] == -20
    assert item["eligible_window"]["last_verification"] == prior


def test_fallback_avoids_small_city_season_sample():
    cutoff = issue_time("2026-01-15")
    prior = (cutoff - timedelta(days=2)).isoformat()
    records = [dict(_record(available=prior, error=-10), case_id=f"winter-{i}") for i in range(5)]
    records += [dict(_record(available=prior, error=5, season="JJA"), case_id=f"summer-{i}") for i in range(15)]
    result = GuidanceBiasIndex(_artifact(records)).query(
        region="北京", issue_date="2026-01-15", horizon=1
    )
    item = result["series"][0]
    assert item["scope"] == "city_all_seasons"
    assert item["n"] == 20
    assert item["fallback_chain"][0] == {
        "scope": "city_season", "n": 5, "minimum": 12
    }
    assert "direction" not in item
    assert "adjustment" not in item
