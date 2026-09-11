"""防泄漏看护：truth/expert 不得通过任何工具暴露；实况必须早于起报时刻。

哨兵法：把 truth/expert 换成全局唯一的哨兵值，穷举调用全部非提交工具，
断言哨兵不出现在任何工具输出里。工具代码但凡读了 truth/expert，测试即红。
"""
import json

from sitian.case import HIDDEN_FILES, VISIBLE_FILES
from sitian.env import ForecastEnv
from sitian.synth import make_case

SENTINEL = 7777.77


def _dump_all_tool_outputs(env: ForecastEnv) -> str:
    env.reset()
    chunks = []
    calls = [
        ("list_data_assets", {}),
        ("get_observations", {}),
        ("get_observations", {"pollutant": "PM2.5", "region": "beijing", "last_hours": 72, "stride": 1}),
        ("get_diagnostics", {}),
        ("get_model_guidance", {}),
        ("get_previous_forecast", {}),
        ("find_similar_cases", {}),
        ("get_assessment", {}),
    ]
    if env.bundle.evidence:
        calls.extend([
            ("get_synoptic_evidence", {}),
            ("get_pollution_evidence", {}),
        ])
    if "get_guidance_bias" in {row["name"] for row in env.tool_specs()}:
        calls.append(("get_guidance_bias", {}))
    for name, args in calls:
        obs, _, done, _ = env.step({"name": name, "args": args})
        chunks.append(json.dumps(obs, ensure_ascii=False))
        assert not done
    return "\n".join(chunks)


def test_truth_and_expert_never_leak_through_tools():
    bundle = make_case("accumulation", seed=3)
    for day in bundle.truth["daily"].values():
        day["pm25_avg"] = SENTINEL
    bundle.expert["forecast"]["daily"][0]["pm25_range"] = [SENTINEL, SENTINEL]
    bundle.expert["notes"] = "SENTINEL_NOTES_DO_NOT_LEAK"

    blob = _dump_all_tool_outputs(ForecastEnv(bundle))
    assert str(SENTINEL) not in blob
    assert "SENTINEL_NOTES_DO_NOT_LEAK" not in blob
    assert '"truth"' not in blob and '"expert"' not in blob


def test_visible_files_exclude_hidden():
    assert set(VISIBLE_FILES).isdisjoint(HIDDEN_FILES)
    assert "truth.json" in HIDDEN_FILES and "expert.json" in HIDDEN_FILES


def test_time_gate_audit():
    bundle = make_case("clean", seed=5)
    assert bundle.audit_time_gate() == []
    # 08:00 整点实况可能在整点后最多一小时才发布，不能用于 08:00 起报。
    bundle.observations["PM2.5"]["times"].append(f"{bundle.issue_date}T08:00")
    violations = bundle.audit_time_gate()
    assert violations and "08:00" in violations[0]


def test_open_evidence_availability_gate():
    bundle = make_case("clean", seed=5)
    bundle.evidence = {"synoptic": {"sources": {"gfs": [
        {"valid_time": "2026-01-10T00:00:00Z",
         "available_at": f"{bundle.issue_date}T01:00:00Z"}
    ]}}}
    violations = bundle.audit_time_gate()
    assert violations and "evidence" in violations[0]


def test_observations_end_before_issue_morning():
    bundle = make_case("guidance_misleading", seed=9)
    last = bundle.observations["PM2.5"]["times"][-1]
    assert last < f"{bundle.issue_date}T08:00"
