from sitian.process_evidence import _future_inflow_scenarios, _nearby_systems


def test_nearby_system_signals_are_target_relative_and_source_specific():
    rows = [
        ("gfs", {"systems": {
            "low_centers": [
                {"lat": 40.0, "lon": 117.0, "mslp_hpa": 998.0},
                {"lat": 20.0, "lon": 100.0, "mslp_hpa": 990.0},
            ],
            "trough_signals": [
                {"lat": 38.0, "lon": 116.0, "z500_zonal_anomaly_gpm": -45.0}
            ],
        }}),
        ("ifs", {"systems": {
            "low_centers": [
                {"lat": 39.0, "lon": 118.0, "mslp_hpa": 1001.0}
            ],
        }}),
    ]
    result = _nearby_systems(rows, {"lat": 39.9, "lon": 116.4})
    assert result["low"]["gfs"][:3] == [40.0, 117.0, 998.0]
    assert result["low"]["ifs"][:3] == [39.0, 118.0, 1001.0]
    assert result["low"]["gfs"][3] < result["low"]["ifs"][3]
    assert result["trough"]["gfs"][2] == -45.0


def test_future_wind_shift_routes_issue_time_candidates_by_direction():
    def city(name, bearing, pm25):
        return {"city": name, "distance_km": 200,
                "bearing_from_target_deg": bearing, "angle_to_925_inflow_deg": None,
                "PM2.5": {"latest": pm25, "change_6h": 3},
                "PM10": {"latest": pm25 + 20, "change_6h": 4},
                "O3": {"latest": 80, "change_6h": -2}}

    context = {"transport_candidate_pool": [city("north", 5, 90), city("south", 182, 120)],
               "static": {"target": {}}}
    trajectory = {
        "2026-01-01T08:00+08:00": {
            "issue_relative_hour": 0, "wind925_ms_from_spread": [3.0, 2.0, 10]
        },
        "2026-01-02T08:00+08:00": {
            "issue_relative_hour": 24, "wind925_ms_from_spread": [4.0, 185.0, 12]
        },
    }
    scenarios = _future_inflow_scenarios(context, trajectory)
    assert "north" in scenarios["N"]["issue_time_observation_candidates"]
    assert "south" not in scenarios["N"]["issue_time_observation_candidates"]
    assert "south" in scenarios["S"]["issue_time_observation_candidates"]
    assert scenarios["N"]["idealized_advective_reach_km"]["12h"] == 130
    north = scenarios["N"]["issue_time_observation_candidates"]["north"]
    assert north["estimated_travel_hours_at_mean_target_wind"] == 18.5
    assert "not a Lagrangian trajectory" in scenarios["N"]["screening_note"]
