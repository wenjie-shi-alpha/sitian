import pytest
import json
from datetime import timedelta

from sitian.agents.scripted import (
    ExpertReplayAgent,
    GuidanceFollowAgent,
    PersistenceAgent,
    run_baselines,
    run_episode,
)
from sitian.env import EnvConfig, ForecastEnv
from sitian.schema import example_forecast
from sitian.synth import make_case


@pytest.fixture()
def bundle():
    return make_case("accumulation", seed=1)


def test_guidance_agent_full_episode(bundle):
    env = ForecastEnv(bundle)
    out = run_episode(env, GuidanceFollowAgent())
    assert out["info"]["reason"] == "submitted"
    assert 0.0 < out["reward"] <= 1.0
    assert out["reward"] == out["info"]["score"]["composite"]
    assert out["steps"] == 2  # 查指导 + 提交


def test_guidance_tail_uses_carry_forward(bundle):
    dates = bundle.forecast_dates()
    bundle.guidance = {"sources": {"cams": {
        "daily_pm25": {dates[0]: 30, dates[1]: 40},
        "daily_o3max": {dates[0]: 100, dates[1]: 120},
    }}}
    bundle.meta["multi_pollutant"] = True
    env = ForecastEnv(bundle)
    out = run_episode(env, GuidanceFollowAgent())
    submitted = out["transcript"][-1]["action"]["args"]["forecast"]["daily"]
    assert submitted[-1]["pm25_range"] == submitted[1]["pm25_range"]
    assert submitted[-1]["o3_range"] == submitted[1]["o3_range"]


def test_persistence_agent_runs(bundle):
    env = ForecastEnv(bundle)
    out = run_episode(env, PersistenceAgent())
    assert out["info"]["reason"] == "submitted"
    assert 0.0 <= out["reward"] <= 1.0


def test_expert_beats_persistence_on_event_case(bundle):
    results = run_baselines(bundle)
    assert set(results) == {"persistence", "guidance", "expert"}
    # 积累过程个例：持续性外推必然漏报过程，专家应显著占优
    assert results["expert"]["composite"] > results["persistence"]["composite"]


def test_invalid_submit_returns_errors_not_done(bundle):
    env = ForecastEnv(bundle)
    env.reset()
    obs, reward, done, info = env.step({"name": "submit_forecast", "args": {"forecast": {"bad": 1}}})
    assert not done
    assert reward == 0.0
    assert obs["content"]["accepted"] is False
    assert obs["content"]["errors"]
    assert "evidence_ref" not in obs["content"]


def test_budget_exhaustion(bundle):
    env = ForecastEnv(bundle, EnvConfig(max_steps=2))
    env.reset()
    _, _, done, _ = env.step({"name": "list_data_assets", "args": {}})
    assert not done
    _, reward, done, info = env.step({"name": "list_data_assets", "args": {}})
    assert done
    assert reward == 0.0
    assert info["reason"] == "budget_exhausted"


def test_unknown_tool_consumes_step(bundle):
    env = ForecastEnv(bundle)
    env.reset()
    obs, _, done, info = env.step({"name": "hack_the_truth", "args": {}})
    assert not done
    assert obs["ok"] is False
    assert info["steps_used"] == 1


def test_unscoreable_case_accepts_valid_submit(bundle):
    bundle.truth = None
    env = ForecastEnv(bundle)
    env.reset()
    fc = example_forecast(bundle.issue_date, bundle.horizon, bundle.region)
    obs, reward, done, info = env.step({"name": "submit_forecast", "args": {"forecast": fc}})
    assert done
    assert reward == 0.0
    assert info["reason"] == "submitted_unscoreable"


def test_expert_replay_requires_expert(bundle):
    bundle.expert = None
    with pytest.raises(ValueError, match="no expert"):
        ExpertReplayAgent(bundle)


def test_tool_specs_openai_style(bundle):
    specs = ForecastEnv(bundle).tool_specs("openai")
    names = {s["function"]["name"] for s in specs}
    assert "submit_forecast" in names and "get_observations" in names
    assert all(s["type"] == "function" for s in specs)


def test_unavailable_optional_tools_are_not_advertised(bundle):
    bundle.previous_forecast = None
    bundle.meta.pop("analogs", None)
    names = {item["function"]["name"] for item in ForecastEnv(bundle).tool_specs("openai")}
    assert "get_previous_forecast" not in names
    assert "find_similar_cases" not in names

    bundle.previous_forecast = {"daily": []}
    bundle.meta["analogs"] = [{"case_id": "past"}]
    names = {item["function"]["name"] for item in ForecastEnv(bundle).tool_specs("openai")}
    assert {"get_previous_forecast", "find_similar_cases"}.issubset(names)


def test_tool_outputs_expose_legal_submission_evidence_types(bundle):
    env = ForecastEnv(bundle)
    env.reset()
    guidance, _, _, _ = env.step({"name": "get_model_guidance", "args": {}})
    assert next(iter(guidance["content"])) == "evidence_ref"
    assert guidance["content"]["submission_evidence_type"] == "model_guidance"
    assessment, _, _, _ = env.step({"name": "get_assessment", "args": {}})
    mapping = assessment["content"]["submission_evidence_type_mapping"]
    assert mapping["observation_trend"] == "observation"
    assert mapping["guidance_disagreement"] == "model_guidance"


def test_guidance_bias_tool_is_time_gated_aggregate_only(bundle, tmp_path):
    from sitian.guidance_bias import issue_time, season

    cutoff = issue_time(bundle.issue_date)
    prior = (cutoff - timedelta(days=2)).isoformat()
    future = (cutoff + timedelta(days=2)).isoformat()
    records = []
    for available, error in ((prior, -20.0), (future, 9999.0)):
        records.extend({
            "case_id": f"history-{available}-{i}", "region": bundle.region,
            "source": "cams", "pollutant": "PM2.5", "lead_days": 1,
            "target_date": bundle.forecast_dates()[0],
            "target_season": season(bundle.forecast_dates()[0]),
            "verification_available_at": available,
            "error_guidance_minus_truth": error, "absolute_error": abs(error),
            "truth_event": False, "guidance_event": False,
        } for i in range(12))
    artifact = tmp_path / "guidance_bias.json"
    artifact.write_text(json.dumps({
        "artifact_type": "guidance_bias_history",
        "verification_policy": {"rule": "strict test"},
        "provenance": {"synthetic": True}, "records": records,
    }))
    env = ForecastEnv(bundle, EnvConfig(guidance_bias_path=str(artifact)))
    env.reset()
    assert "get_guidance_bias" in {row["name"] for row in env.tool_specs()}
    obs, _, _, _ = env.step({
        "name": "get_guidance_bias",
        "args": {"source": "cams", "pollutant": "PM2.5"},
    })
    series = obs["content"]["series"]
    assert series and series[0]["mean_error"] == -20.0
    assert series[0]["n"] == 12
    blob = json.dumps(obs["content"])
    assert "9999" not in blob
    assert obs["content"]["submission_evidence_type"] == "model_guidance"


def test_open_evidence_tools_are_registered_only_when_present(bundle):
    assert "get_synoptic_evidence" not in {s["name"] for s in ForecastEnv(bundle).tool_specs()}
    bundle.evidence = {
        "contract_version": "open-evidence-v1",
        "synoptic": {"sources": {"gfs": [{"valid_time": "2025-12-16T00:00:00Z",
                                               "available_at": "2025-12-15T23:00:00Z"}]}},
        "pollution": {"composition": {"dust_fraction": 0.2}},
    }
    env = ForecastEnv(bundle)
    env.reset()
    names = {s["name"] for s in env.tool_specs()}
    assert {"get_synoptic_evidence", "get_pollution_evidence"} <= names
    obs, _, _, _ = env.step({"name": "get_pollution_evidence", "args": {"kind": "composition"}})
    assert obs["content"]["composition"]["dust_fraction"] == 0.2
    assert obs["content"]["submission_evidence_type_mapping"]["composition"] == "model_guidance"
    synoptic, _, _, _ = env.step({"name": "get_synoptic_evidence", "args": {}})
    assert synoptic["content"]["submission_evidence_type_mapping"][
        "circulation_or_weather_system"
    ] == "synoptic"


def test_pollution_summary_preserves_six_hour_trace_gas_trajectory(bundle):
    records = [
        {"lead_hour": lead, "valid_time": f"2025-12-15T{(12 + lead) % 24:02d}:00:00Z",
         "cities": {bundle.region: {"carbon_monoxide_ppbv_approx": 100 + lead,
                                     "nitrogen_dioxide_ppbv_approx": 10 + lead,
                                     "sulphur_dioxide_ppbv_approx": 1 + lead}}}
        for lead in range(0, 121, 6)
    ]
    bundle.evidence = {
        "contract_version": "open-evidence-v1",
        "pollution": {"composition": {
            "aerosol": {"available": False, "records": []},
            "trace_gases": {"available": True, "cycle": "2025-12-15T12:00:00Z",
                            "records": records},
            "column_gases": {
                "available": True,
                "cycle": "2025-12-15T12:00:00Z",
                "semantics": "total_column_mass",
                "unit": "kg m-2",
                "records": [
                    {"lead_hour": lead,
                     "valid_time": f"2025-12-15T{(12 + lead) % 24:02d}:00:00Z",
                     "cities": {bundle.region: {
                         "carbon_monoxide_kg_m2": 0.001 + lead / 1e6,
                         "nitrogen_dioxide_kg_m2": 0.00001 + lead / 1e8,
                         "sulphur_dioxide_kg_m2": 0.000001 + lead / 1e9,
                     }}}
                    for lead in range(0, 121, 6)
                ],
            },
        }},
    }
    env = ForecastEnv(bundle)
    env.reset()
    obs, _, _, _ = env.step({
        "name": "get_pollution_evidence",
        "args": {"kind": "composition", "detail": "summary"},
    })
    trajectory = obs["content"]["composition"]["trace_gases"]["trajectory_6h"]
    assert [row["lead_hour"] for row in trajectory] == list(range(0, 121, 6))
    assert len(trajectory) == 21
    columns = obs["content"]["composition"]["column_gases"]
    assert columns["semantics"] == "total_column_mass"
    assert "not a surface concentration" in columns["interpretation"]
    assert len(columns["trajectory_6h"]) == 21
