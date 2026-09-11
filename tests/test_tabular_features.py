from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.eval_tabular_baseline import (
    _assessment_features, _evidence_features, _guidance_features, _process_features,
)
from sitian.synth import make_case


def test_tabular_guidance_projection_keeps_all_sources_and_disagreement():
    bundle = make_case("accumulation", seed=4)
    days = bundle.forecast_dates()
    bundle.guidance = {"sources": {}}
    for source_index, source in enumerate(("cams", "cmaq", "naqp"), start=1):
        bundle.guidance["sources"][source] = {
            "daily_pm25": {day: 10 * source_index for day in days},
            "daily_pm10": {day: 20 * source_index for day in days},
            "daily_o3max": {day: 30 * source_index for day in days},
        }

    values = _guidance_features(bundle, days)[days[0]]
    assert values[:9] == [10, 20, 30, 20, 40, 60, 30, 60, 90]
    assert values[9:12] == pytest.approx([20, 10 * (2 / 3) ** 0.5, 20])
    assert values[12:15] == pytest.approx([40, 20 * (2 / 3) ** 0.5, 40])
    assert values[15:18] == pytest.approx([60, 30 * (2 / 3) ** 0.5, 60])
    assert len(values) == 198


def test_tabular_open_evidence_projection_uses_full_trajectory_and_transport_pool():
    records = [
        {
            "available": True,
            "valid_time": f"2026-06-{index + 1:02d}T00:00:00Z",
            "cities": {"city": {"mslp_hpa": 1000 + index}},
            "systems": {},
        }
        for index in range(20)
    ]
    evidence = {
        "synoptic": {
            "sources": {"gfs": records, "ifs": deepcopy(records)},
            "cross_model": {},
        },
        "pollution": {
            "composition": {},
            "source_context": {
                "transport_candidate_pool": [{
                    "city": "upstream", "distance_km": 200,
                    "bearing_from_target_deg": 10, "angle_to_inflow_deg": 5,
                    "PM2.5": {"latest": 80},
                }],
            },
        },
    }
    bundle = SimpleNamespace(region="city", evidence=evidence)
    first = np.asarray(_evidence_features(bundle, "2026-06-02"), dtype=float)

    changed = deepcopy(evidence)
    # Change a non-target-day trajectory value and a deep-query-only pool value.
    changed["synoptic"]["sources"]["gfs"][10]["cities"]["city"]["mslp_hpa"] = 900
    changed["pollution"]["source_context"]["transport_candidate_pool"][0][
        "distance_km"
    ] = 600
    second = np.asarray(
        _evidence_features(SimpleNamespace(region="city", evidence=changed), "2026-06-02"),
        dtype=float,
    )
    assert len(first) == len(second)
    assert np.count_nonzero(np.nan_to_num(first, nan=-99999) !=
                            np.nan_to_num(second, nan=-99999)) == 2


def test_tabular_process_projection_is_fixed_width_and_finite_or_nan():
    bundle = make_case("accumulation", seed=4)
    # Missing evidence still maps to a fixed-width vector; the formal evidence
    # audit independently rejects incomplete cases before fitting/evaluation.
    values = np.asarray(_process_features(bundle), dtype=np.float32)
    assert len(values) == 2140
    assert np.all(np.isfinite(values) | np.isnan(values))


def test_tabular_assessment_projection_matches_policy_visible_flags():
    bundle = make_case("accumulation", seed=4)
    values = np.asarray(_assessment_features(bundle), dtype=np.float32)
    assert len(values) == bundle.horizon * 6 + 4
    assert np.all(np.isfinite(values) | np.isnan(values))
