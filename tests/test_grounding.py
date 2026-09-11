"""grounding 分量：证据引用必须对应真实调用过的工具（规则化防伪造证据）。"""
import pytest

from sitian.scoring import EVIDENCE_TOOL_MAP, score_forecast


def _fc(evidence):
    return {"issue_date": "2026-06-01", "region": "beijing",
            "daily": [{"date": "2026-06-02", "aqi_level": 2,
                       "primary_pollutant": "PM2.5", "pm25_range": [40, 70]},
                      {"date": "2026-06-03", "aqi_level": 2,
                       "primary_pollutant": "PM2.5", "pm25_range": [40, 70]}],
            "process": {"has_event": False}, "evidence": evidence, "confidence": "medium"}


TRUTH = {"2026-06-02": 50.0, "2026-06-03": 60.0}
KW = dict(issue_date="2026-06-01", horizon=2)


def test_abstains_without_tool_history():
    r = score_forecast(_fc([{"type": "observation", "claim": "x"}]), TRUTH, **KW)
    assert r.components["grounding"] is None
    assert "grounding" not in r.weights_used


def test_supported_and_fabricated_evidence():
    ev = [{"type": "observation", "claim": "实况回落"},
          {"type": "model_guidance", "claim": "指导显示清洁"}]
    # 只查过实况但两条都没有语义引用：仅得类型过渡分。
    r = score_forecast(_fc(ev), TRUTH, tools_called={"get_observations"}, **KW)
    assert r.components["grounding"] == 0.1
    # 两类都查过但未结构化引用，仍不能拿满分。
    r2 = score_forecast(_fc(ev), TRUTH, tools_called={"get_observations", "get_model_guidance"}, **KW)
    assert r2.components["grounding"] == 0.2
    assert r2.composite > r.composite
    assert r2.outcome_composite == r.outcome_composite


def test_no_evidence_scores_zero():
    r = score_forecast(_fc([]), TRUTH, tools_called={"get_observations"}, **KW)
    assert r.components["grounding"] == 0.0


def test_grounding_declares_that_free_text_claim_is_not_machine_verified():
    evidence = [{
        "type": "observation", "claim": "这段自然语言可能解释错误",
        "ref": "e1", "field": "/latest", "value": 42.0,
    }]
    result = score_forecast(
        _fc(evidence), TRUTH, tools_called={"get_observations"},
        evidence_registry={
            "e1": {"tool": "get_observations", "content": {"latest": 42.0}}
        },
        **KW,
    )
    audit = result.details["grounding"]
    assert audit["verification_scope"] == "tool+semantic_section+json_pointer+scalar_value"
    assert audit["natural_language_claim_verified"] is False


def test_prior_knowledge_cannot_hack_grounding_reward():
    ev = [{"type": "expert_prior", "claim": "12月气候背景偏污染"}]
    r = score_forecast(_fc(ev), TRUTH, tools_called=set(), **KW)
    assert r.components["grounding"] == 0.0


def test_open_evidence_tools_ground_their_claim_types():
    ev = [{"type": "synoptic", "claim": "双模式显示低层偏北输送"},
          {"type": "observation", "claim": "上风向城市浓度正在下降"}]
    r = score_forecast(_fc(ev), TRUTH,
                       tools_called={"get_synoptic_evidence", "get_pollution_evidence"}, **KW)
    assert r.components["grounding"] == 0.2


@pytest.mark.parametrize(
    "alias,canonical,tool,pointer,content",
    [
        ("pollution_evidence", "diagnostic", "get_pollution_evidence",
         "/source_context/signal", {"source_context": {"signal": 1.5}}),
        ("assessment", "diagnostic", "get_assessment",
         "/daily_signals/signal", {"daily_signals": {"signal": 1.5}}),
        ("composition", "model_guidance", "get_pollution_evidence",
         "/composition/signal", {"composition": {"signal": 1.5}}),
    ],
)
def test_evidence_aliases_are_normalized_and_grounded(alias, canonical, tool, pointer, content):
    assert tool in EVIDENCE_TOOL_MAP[alias]
    r = score_forecast(
        _fc([{"type": alias, "claim": "工具证据", "ref": "e1",
              "field": pointer, "value": 1.5}]),
        TRUTH,
        tools_called={tool},
        evidence_registry={"e1": {"tool": tool, "content": content}},
        expert_evidence_types={canonical},
        **KW,
    )
    assert r.valid
    # One semantically valid scalar is useful but cannot earn full grounding.
    assert r.components["grounding"] == 0.5
    assert r.components["evidence"] == 1.0


@pytest.mark.parametrize(
    "evidence,registry",
    [
        ({"ref": "e1", "field": "/wind", "value": 9.9},
         {"e1": {"tool": "get_synoptic_evidence", "content": {"wind": 2.1}}}),
        ({"ref": "made-up", "field": "/wind", "value": 2.1},
         {"e1": {"tool": "get_synoptic_evidence", "content": {"wind": 2.1}}}),
        ({"ref": "e1", "field": "/missing", "value": 2.1},
         {"e1": {"tool": "get_synoptic_evidence", "content": {"wind": 2.1}}}),
        ({"ref": "e1", "field": "/wind", "value": 2.1},
         {"e1": {"tool": "get_observations", "content": {"wind": 2.1}}}),
    ],
)
def test_fabricated_semantic_assertions_get_no_type_fallback(evidence, registry):
    item = {"type": "synoptic", "claim": "伪造形势", **evidence}
    result = score_forecast(
        _fc([item]), TRUTH, tools_called={"get_synoptic_evidence"},
        evidence_registry=registry, **KW,
    )
    assert result.components["grounding"] == 0.0
    assert result.details["grounding"]["semantic_verified_count"] == 0
    assert result.details["grounding"]["structured_invalid_count"] == 1


@pytest.mark.parametrize("missing", [None, "", "   ", float("nan"), float("inf")])
def test_missing_tool_payload_cannot_become_semantic_grounding(missing):
    result = score_forecast(
        _fc([{"type": "synoptic", "claim": "缺失值不构成形势事实", "ref": "e1",
              "field": "/signal", "value": missing}]),
        TRUTH,
        tools_called={"get_synoptic_evidence"},
        evidence_registry={
            "e1": {"tool": "get_synoptic_evidence", "content": {"signal": missing}}
        },
        **KW,
    )
    assert result.components["grounding"] == 0.0
    assert result.details["grounding"]["semantic_verified_count"] == 0
    assert result.details["grounding"]["structured_invalid_count"] == 1


def test_process_terrain_diagnostic_is_groundable_but_time_coordinate_is_not():
    prefix = "/weather_trajectory_6h_to_72h_then_12h/2026-06-02T08:00+08:00"
    content = {
        "weather_trajectory_6h_to_72h_then_12h": {
            "2026-06-02T08:00+08:00": {
                "terrain_adaptive_low_level": {"pressure_level_hpa": 700},
                "issue_relative_hour": 24,
            }
        }
    }
    valid = score_forecast(
        _fc([{"type": "diagnostic", "claim": "高原城市使用700hPa低层风",
              "ref": "e1", "field": prefix + "/terrain_adaptive_low_level/pressure_level_hpa",
              "value": 700}]),
        TRUTH, tools_called={"get_process_evidence"},
        evidence_registry={"e1": {"tool": "get_process_evidence", "content": content}},
        **KW,
    )
    assert valid.components["grounding"] == 0.5

    metadata = score_forecast(
        _fc([{"type": "diagnostic", "claim": "时效坐标", "ref": "e1",
              "field": prefix + "/issue_relative_hour", "value": 24}]),
        TRUTH, tools_called={"get_process_evidence"},
        evidence_registry={"e1": {"tool": "get_process_evidence", "content": content}},
        **KW,
    )
    assert metadata.components["grounding"] == 0.0


def test_two_unique_semantic_assertions_and_two_types_are_required_for_full_credit():
    registry = {
        "e1": {"tool": "get_observations", "content": {"a": 12, "b": 18}},
        "e2": {"tool": "get_diagnostics", "content": {"wind": 2.5}},
    }
    first = {"type": "observation", "claim": "事实A", "ref": "e1",
             "field": "/a", "value": 12}
    duplicate = {**first, "claim": "重复包装同一事实"}
    one_unique = score_forecast(
        _fc([first, duplicate]), TRUTH, tools_called={"get_observations"},
        evidence_registry=registry, **KW,
    )
    assert one_unique.components["grounding"] == 0.5
    assert one_unique.details["grounding"]["unique_assertion_count"] == 1

    second = {"type": "observation", "claim": "事实B", "ref": "e1",
              "field": "/b", "value": 18}
    two_unique = score_forecast(
        _fc([first, second]), TRUTH, tools_called={"get_observations"},
        evidence_registry=registry, **KW,
    )
    assert two_unique.components["grounding"] == 0.5
    assert two_unique.details["grounding"]["semantic_verified_count"] == 2

    second_type = {"type": "diagnostic", "claim": "近地风事实", "ref": "e2",
                   "field": "/wind", "value": 2.5}
    diverse = score_forecast(
        _fc([first, second_type]), TRUTH,
        tools_called={"get_observations", "get_diagnostics"},
        evidence_registry=registry, **KW,
    )
    assert diverse.components["grounding"] == 1.0
    assert diverse.details["grounding"]["semantic_verified_type_count"] == 2


def test_same_scientific_fact_cannot_be_repackaged_by_numeric_type_or_replayed_ref():
    content = {"daily_signals": {"2026-01-02": {"wind_speed_ms": 2}}}
    registry = {
        "e1": {"tool": "get_assessment", "content": content},
        # A repeated identical query receives a different episode-local ref.
        "e2": {"tool": "get_assessment", "content": dict(content)},
    }
    field = "/daily_signals/2026-01-02/wind_speed_ms"
    repackaged = [
        {"type": "diagnostic", "claim": "弱风", "ref": "e1",
         "field": field, "value": 2},
        {"type": "synoptic", "claim": "同一弱风事实", "ref": "e2",
         "field": field, "value": 2.0},
    ]
    result = score_forecast(
        _fc(repackaged), TRUTH, tools_called={"get_assessment"},
        evidence_registry=registry, **KW,
    )
    assert result.components["grounding"] == 0.5
    assert result.details["grounding"]["unique_assertion_count"] == 1
    assert result.details["grounding"]["semantic_verified_count"] == 1


def test_invalid_structured_reference_cannot_fall_back_to_type_credit():
    registry = {"e1": {"tool": "get_observations", "content": {"latest": 42}}}
    evidence = [
        {"type": "observation", "claim": "wrong ref", "ref": "forged",
         "field": "/latest", "value": 42},
        {"type": "observation", "claim": "incomplete", "ref": "e1"},
    ]
    result = score_forecast(
        _fc(evidence), TRUTH, tools_called={"get_observations"},
        evidence_registry=registry, **KW,
    )
    assert result.components["grounding"] == 0.0
    assert result.details["grounding"]["structured_invalid_count"] == 2


def test_unavailable_tool_call_not_counted():
    from sitian.synth import make_case
    from sitian.env import ForecastEnv
    bundle = make_case("clean", seed=7)
    bundle.previous_forecast = None
    env = ForecastEnv(bundle)
    env.reset()
    env.step({"name": "get_previous_forecast", "args": {}})   # available=False
    env.step({"name": "get_observations", "args": {}})
    assert "get_previous_forecast" not in env.tools_called
    assert "get_observations" in env.tools_called


def test_env_passes_tool_history(tmp_path):
    from sitian.synth import make_case
    from sitian.env import ForecastEnv
    from sitian.agents.scripted import PersistenceAgent, run_episode
    bundle = make_case("clean", seed=7)
    env = ForecastEnv(bundle)
    out = run_episode(env, PersistenceAgent())
    comps = out["info"]["score"]["components"]
    assert comps["grounding"] == 0.5  # 两条实况事实有效，但同属一个证据类别


def test_tool_returns_copy_safe_citation_examples_that_reach_full_grounding():
    from sitian.synth import make_case
    from sitian.env import ForecastEnv

    bundle = make_case("clean", seed=8)
    env = ForecastEnv(bundle)
    task = env.reset()
    observation, _, _, _ = env.step({
        "name": "get_observations",
        "args": {"pollutant": "PM2.5", "region": bundle.region, "last_hours": 24},
    })
    examples = observation["content"]["citation_examples"]
    assert len(examples) >= 2
    assert all(item["ref"] == observation["content"]["evidence_ref"] for item in examples)
    diagnostic, _, _, _ = env.step({"name": "get_diagnostics", "args": {}})
    diagnostic_examples = diagnostic["content"]["citation_examples"]
    assert diagnostic_examples
    from sitian.schema import example_forecast
    forecast = example_forecast(bundle.issue_date, bundle.horizon, bundle.region)
    forecast["evidence"] = [
        {**examples[0], "claim": f"实况字段 {examples[0]['field']}"},
        {**diagnostic_examples[0],
         "claim": f"诊断字段 {diagnostic_examples[0]['field']}"},
    ]
    _, _, done, info = env.step({"name": "submit_forecast", "args": {"forecast": forecast}})
    assert done
    assert info["score"]["components"]["grounding"] == 1.0


def test_citation_examples_skip_missing_empty_and_nonfinite_values():
    from sitian.env import _scientific_scalars

    values = dict(_scientific_scalars(
        {"none": None, "empty": "  ", "nan": float("nan"),
         "inf": float("inf"), "valid": 2.5},
        "",
    ))
    assert values == {"/valid": 2.5}


def test_tuple_facts_are_verifiable_elementwise():
    from sitian.scoring import _scalar_equal
    assert _scalar_equal([9.7, 274, 5], [9.7, 274, 5])
    assert _scalar_equal([9.7, 274.0, 5], [9.7, 274, 5])
    assert not _scalar_equal([9.7, 274], [9.7, 274, 5])
    assert not _scalar_equal([9.7, None, 5], [9.7, None, 5])
    assert not _scalar_equal([[1, 2]], [[1, 2]])
    assert not _scalar_equal(9.7, [9.7])
