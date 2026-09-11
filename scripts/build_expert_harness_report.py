#!/usr/bin/env python3
"""Build the durable expert-requirements report from audited JSON artifacts.

The HTML itself is rendered by the Data Analytics portable-report builder.
This script only creates the source-backed artifact and its detailed appendix;
it never infers a PASS from a missing artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.scoring import reward_spec  # noqa: E402
from sitian.provenance import (  # noqa: E402
    identity_matches_file, score_implementation_matches,
)
from sitian.open_evidence import (  # noqa: E402
    EVIDENCE_VERSION,
    NWP_TEMPORAL_PROFILE_VERSION,
)

DEFAULT_REPORT_DIR = REPO_ROOT / "docs/reports/expert_forecast_requirements"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _identity(path: Path) -> dict:
    try:
        display_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(path)
    if not path.is_file():
        return {"path": display_path, "exists": False}
    raw = path.read_bytes()
    return {
        "path": display_path,
        "exists": True,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _pct(value) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _num(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(map(clean, headers)) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _same_reward_identity(left: dict, right: dict) -> bool:
    """Require the semantic reward identity, not merely a familiar filename."""
    return bool(
        left.get("version")
        and left.get("config_sha256")
        and left.get("version") == right.get("version")
        and left.get("config_sha256") == right.get("config_sha256")
    )


def _stored_identity_valid(identity: dict | None) -> bool:
    if not identity or not identity.get("path"):
        return False
    stored = Path(identity["path"])
    path = stored if stored.is_absolute() else REPO_ROOT / stored
    actual = _identity(path)
    return bool(
        actual.get("exists")
        and actual.get("sha256") == identity.get("sha256")
        and actual.get("bytes") == identity.get("bytes")
    )


def _effective_gates(
    preflight: dict,
    *,
    coverage_pass: bool,
    contract_pass: bool,
    reward_contract_pass: bool,
    formal_probe: bool,
    training_smoke_pass: bool,
) -> tuple[list[list[object]], dict]:
    """Invalidate stale gates when their upstream artifacts no longer match.

    The report may be rebuilt while a new data/reward contract is still being
    materialised.  In that interval an older ``rl_preflight.json`` must never
    lend a PASS to the current contract.
    """
    labels = {
        "data_and_reward_integrity": "数据、Reward 与专家证据合同",
        "format_and_trajectory": "格式与轨迹",
        "context_budget": "上下文预算",
        "sampling_efficiency": "采样效率",
        "structured_evidence_grounding": "结构化证据引用",
        "trainable_stack_and_hardware_smoke": "可训练栈与硬件冒烟",
    }
    current_reward = reward_spec()
    preflight_reward = preflight.get("reward") or {}
    reward_current = _same_reward_identity(preflight_reward, current_reward)
    overrides = {
        "data_and_reward_integrity": (
            coverage_pass and contract_pass and reward_contract_pass and reward_current
        ),
        "format_and_trajectory": formal_probe and reward_current,
        "context_budget": formal_probe and reward_current,
        "sampling_efficiency": formal_probe and reward_current,
        "structured_evidence_grounding": formal_probe and reward_current,
        "trainable_stack_and_hardware_smoke": training_smoke_pass and reward_current,
    }
    effective = {}
    rows = []
    recorded = preflight.get("preflight") or {}
    for name in labels:
        row = recorded.get(name) or {}
        passed = bool(row.get("pass")) and overrides.get(name, reward_current)
        effective[name] = passed
        rows.append([labels.get(name, name), "PASS" if passed else "BLOCKED"])
    return rows, {
        "reward_identity_current": reward_current,
        "preflight_reward": {
            "version": preflight_reward.get("version"),
            "config_sha256": preflight_reward.get("config_sha256"),
        },
        "current_reward": {
            "version": current_reward.get("version"),
            "config_sha256": current_reward.get("config_sha256"),
        },
        "effective": effective,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="data/interim/efr.json")
    parser.add_argument("--coverage", default="data/interim/open_evidence_coverage.json")
    parser.add_argument("--expert-contract",
                        default="data/interim/expert_evidence_contract_audit.json")
    parser.add_argument("--spatial-ood-audit",
                        default="data/interim/spatial_ood_audit.json")
    parser.add_argument("--probe", default="data/interim/probe_open_evidence_v079_n50_g8.json")
    parser.add_argument("--tabular",
                        default="data/interim/eval_tabular_open_evidence_v079.json")
    parser.add_argument("--preflight", default="data/interim/rl_preflight.json")
    parser.add_argument("--training-smoke", default="data/interim/training_smoke.json")
    parser.add_argument("--reward-contract",
                        default="data/interim/reward_contract_audit.json")
    parser.add_argument("--reward-sensitivity",
                        default="data/interim/reward_weight_sensitivity_v079.json")
    parser.add_argument("--verl-runtime-contract",
                        default="data/interim/verl_runtime_contract.json")
    parser.add_argument("--guidance-comparison",
                        default="data/interim/compare_v079_frozen_vs_guidance.json")
    parser.add_argument("--tabular-comparison",
                        default="data/interim/compare_v079_frozen_vs_tabular.json")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    paths = {
        "expert_analysis": REPO_ROOT / args.analysis,
        "coverage": REPO_ROOT / args.coverage,
        "expert_contract": REPO_ROOT / args.expert_contract,
        "spatial_ood_audit": REPO_ROOT / args.spatial_ood_audit,
        "probe": REPO_ROOT / args.probe,
        "tabular": REPO_ROOT / args.tabular,
        "preflight": REPO_ROOT / args.preflight,
        "training_smoke": REPO_ROOT / args.training_smoke,
        "reward_contract": REPO_ROOT / args.reward_contract,
        "reward_sensitivity": REPO_ROOT / args.reward_sensitivity,
        "verl_runtime_contract": REPO_ROOT / args.verl_runtime_contract,
        "guidance_comparison": REPO_ROOT / args.guidance_comparison,
        "tabular_comparison": REPO_ROOT / args.tabular_comparison,
    }
    data = {name: _load(path) for name, path in paths.items()}
    analysis = data["expert_analysis"]
    coverage = data["coverage"]
    contract = data["expert_contract"]
    spatial_ood = data["spatial_ood_audit"]
    probe = data["probe"]
    tabular = data["tabular"]
    preflight = data["preflight"]
    training_smoke = data["training_smoke"]
    reward_contract = data["reward_contract"]
    reward_sensitivity = data["reward_sensitivity"]
    runtime_contract = data["verl_runtime_contract"]
    guidance_cmp = data["guidance_comparison"]
    tabular_cmp = data["tabular_comparison"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    probe_summary = probe.get("summary") or {}
    attachment = coverage.get("case_attachments") or {}
    contract_pass = contract.get("preflight_contract_passed") is True
    evidence_pass = (
        coverage.get("ready_for_full_attachment") is True
        and coverage.get("contract_version") == EVIDENCE_VERSION
        and (coverage.get("nwp_temporal_profile") or {}).get("version")
            == NWP_TEMPORAL_PROFILE_VERSION
        and _stored_identity_valid(coverage.get("source_manifest"))
    )
    probe_reward = (probe.get("provenance") or {}).get("reward") or {}
    probe_reward_current = _same_reward_identity(probe_reward, reward_spec())
    formal_probe = (
        int(probe.get("n_cases") or 0) >= 50
        and int(probe.get("group_size") or 0) >= 8
        and int(probe.get("n_rollouts") or 0) >= 400
        and abs(float(probe.get("temperature", -999)) - 1.0) <= 1e-9
        and probe_reward_current
        and score_implementation_matches(probe.get("provenance") or {}, REPO_ROOT)
        and int(probe.get("group_size") or 0) == int(
            (runtime_contract.get("training_sampling_contract") or {}).get(
                "rollout_group_size") or 0
        )
    )
    raw_preflight_pass = preflight.get("preflight_passed") is True
    runtime_contract_pass = (
        runtime_contract.get("passed") is True
        and _same_reward_identity(runtime_contract.get("reward") or {}, reward_spec())
        and bool(runtime_contract.get("checks"))
        and all((runtime_contract.get("checks") or {}).values())
        and bool(runtime_contract.get("inputs"))
        and all(_stored_identity_valid(identity)
                for identity in (runtime_contract.get("inputs") or {}).values())
    )
    training_smoke_pass = (
        training_smoke.get("success") is True and runtime_contract_pass
    )
    reward_contract_pass = (
        reward_contract.get("passed") is True
        and score_implementation_matches(reward_contract, REPO_ROOT)
        and _same_reward_identity(reward_contract.get("reward") or {}, reward_spec())
    )
    reward_sensitivity_current = (
        bool(reward_sensitivity.get("scenarios"))
        and _same_reward_identity(
            reward_sensitivity.get("reward") or {}, reward_spec()
        )
        and score_implementation_matches(reward_sensitivity, REPO_ROOT)
        and _stored_identity_valid(reward_sensitivity.get("audit_implementation"))
        and _stored_identity_valid(reward_sensitivity.get("probe"))
        and _stored_identity_valid(reward_sensitivity.get("tabular"))
    )
    spatial_ood_ready = bool(
        spatial_ood.get("artifact_type") == "spatial_ood_split_audit"
        and len(spatial_ood.get("heldout_cities") or []) == 8
        and (spatial_ood.get("selection_policy") or {}).get(
            "validation_or_test_outcomes_used") is False
        and (spatial_ood.get("selection_policy") or {}).get(
            "power_rule_uses_outcomes") is False
        and all(value == 0 for value in (spatial_ood.get("integrity") or {}).values())
        and (spatial_ood.get("spatial_ood_val") or {}).get("issue_dates", 0) >= 30
        and (spatial_ood.get("spatial_ood_test") or {}).get("issue_dates", 0) >= 30
    )
    gate_rows, gate_freshness = _effective_gates(
        preflight,
        coverage_pass=evidence_pass,
        contract_pass=contract_pass,
        reward_contract_pass=reward_contract_pass,
        formal_probe=formal_probe,
        training_smoke_pass=training_smoke_pass,
    )
    effective_gates = gate_freshness["effective"]
    preflight_pass = bool(effective_gates) and all(effective_gates.values())
    nontraining_gates = {
        name: passed for name, passed in effective_gates.items()
        if name != "trainable_stack_and_hardware_smoke"
    }
    ready_for_training = bool(nontraining_gates) and all(nontraining_gates.values())
    status = (
        "已通过小模型可训练闭环门禁" if preflight_pass and training_smoke_pass else
        "冻结策略门禁已通过，等待可训练闭环" if ready_for_training else
        "数据/评测已完成，冻结策略门禁仍阻塞" if evidence_pass and contract_pass and formal_probe else
        "高频证据或正式探针仍在收尾"
    )

    evidence_sql = (
        "SELECT category, records, total_records, mention_rate "
        "FROM evidence_mentions ORDER BY mention_rate DESC LIMIT 5"
    )
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE evidence_mentions "
        "(category TEXT, records INTEGER, total_records INTEGER, mention_rate REAL)"
    )
    connection.executemany(
        "INSERT INTO evidence_mentions VALUES (?, ?, ?, ?)",
        [(row.get("category"), row.get("records"), row.get("total_records"),
          row.get("mention_rate")) for row in (analysis.get("evidence_mentions") or [])],
    )
    evidence_mentions = [
        {"category": row[0], "records": row[1], "total_records": row[2],
         "mention_rate": row[3]}
        for row in connection.execute(evidence_sql).fetchall()
    ]
    connection.close()
    evidence_dataset = [{
        "category": {
            "时序、转折与过程阶段": "过程时序",
            "实况态势与空间演变": "实况态势",
            "天气系统与环流": "天气系统",
            "风场与输送": "风场输送",
            "稳定度与边界层": "稳定边界层",
        }.get(row.get("category"), row.get("category")),
        "records": row.get("records"),
        "total_records": row.get("total_records"),
        "mention_rate": row.get("mention_rate"),
        "importance_class": (
            "词频下界；重要性另按通用/条件/校准分类"
        ),
    } for row in evidence_mentions]

    gate_table = (_markdown_table(["Preflight 门禁", "状态"], gate_rows)
                  if gate_rows else "尚无当前 reward 身份下的完整 preflight 产物。")
    gate_summary = ("\n".join(f"- {row[0]}：**{row[1]}**" for row in gate_rows)
                    if gate_rows else "尚无当前 reward 身份下的完整 preflight 产物。")
    executive = (
        f"## 执行摘要\n\n**当前状态：{status}。** "
        f"专家语料仅作机制设计参考；全国训练输入仍是起报时合法可得的开放数据。"
        f"新证据附件覆盖为 {_pct(attachment.get('fraction'))}，专家证据合同为 "
        f"{'PASS' if contract_pass else 'BLOCKED'}，Reward 可执行合同为 "
        f"{'PASS' if reward_contract_pass else 'BLOCKED'}；"
        f"veRL 原生运行时合同为 {'PASS' if runtime_contract_pass else 'BLOCKED'}；"
        f"整城空间 OOD 划分为 {'READY' if spatial_ood_ready else 'BLOCKED'}；"
        + (f"当前版本 frozen probe 为 {probe.get('n_cases', 0)} case × group "
           f"{probe.get('group_size', 0)}；" if probe_reward_current else
           f"当前版本 frozen probe 尚未生成（展示的 {probe.get('n_cases', 0)}×"
           f"{probe.get('group_size', 0)} 仅为旧 reward 冒烟）；")
        + f"可训练 LoRA 冒烟为 {'PASS' if training_smoke_pass else '尚未通过'}。\n\n"
        "Harness 已把初态、天气系统、多层扩散、未来风向对应的起报时上游实况、污染组成、"
        "模式订正和过程时序放入可审计工具。reward-v0.7.9 将含 grounding 的训练分与只含"
        "预报结果的 outcome composite 分离；论文毕业仍由硬 MAE/CSI/F1/区间指标和分块 CI 决定。"
        + ("\n\n旧 preflight 与当前 reward/证据合同身份不一致，已强制失效；"
           "下表只显示当前合同下的有效门禁。"
           if preflight and not gate_freshness["reward_identity_current"] else "")
    )
    evidence_table = (_markdown_table(
        ["专家证据类别", "命中记录", "总记录", "出现率"],
        [[row.get("category"), row.get("records"), row.get("total_records"),
          _pct(row.get("mention_rate"))] for row in evidence_dataset],
    ) if evidence_dataset else "当前专家分析产物未提供证据词频。")
    comparison_lines = [
        "## 当前可比较结果",
        "",
        "所有模型—基线比较使用 `outcome_composite`，不含交互式 grounding；训练仍优化含 grounding 的 composite，"
        "但 DAPO 按 outcome_composite 过滤零方差组，grounding-only 变化不能冒充数值梯度。",
        "",
        (f"- Frozen vs CAMS：n={guidance_cmp.get('n_cases', '—')}；模型 "
         f"{_num(guidance_cmp.get('model_mean'))}，基线 {_num(guidance_cmp.get('baseline_mean'))}，"
         f"差值 {_num(guidance_cmp.get('paired_difference'))}；"
         f"状态 {guidance_cmp.get('inference_status', '未生成')}。"),
        (f"- Frozen vs tabular：n={tabular_cmp.get('n_cases', '—')}；模型 "
         f"{_num(tabular_cmp.get('model_mean'))}，基线 {_num(tabular_cmp.get('baseline_mean'))}，"
         f"差值 {_num(tabular_cmp.get('paired_difference'))}；"
         f"状态 {tabular_cmp.get('inference_status', '未生成')}。"),
        "",
        "这些 frozen 结果只定义 RL 起点；是否扩到 8B 必须看训后独立验证和事件硬 CSI，不能由裸模型先验决定。",
        "空间 OOD test 来自八个完全不进训练的城市；它检验空间泛化，但暖季构成不能替代冬季事件前瞻评测。",
    ]
    decision_audit = analysis.get("coverage_audit") or []
    decision_matrix_lines = [
        "## 专家决策需求：四层缺口矩阵",
        "",
        "每项分别判断“资料存在、工具可见、策略实际使用、Reward/评测能检验”。"
        "前一层通过不自动推出后一层通过。",
        "",
    ]
    for row in decision_audit:
        decision_matrix_lines.extend([
            (f"{row.get('order')}. **{row.get('decision_task')}** "
             f"（{row.get('importance')} / {row.get('priority')}）"),
            (f"   数据：{row.get('raw_data')}；工具：{row.get('tool_exposure')}；"
             f"策略：{row.get('policy_use')}；Reward/测试：{row.get('reward_test')}。"),
            f"   剩余缺口：{row.get('principal_gap')}",
            "",
        ])
    capabilities = contract.get("capabilities") or {}
    capability_summary = (_markdown_table(
        ["逐 case 能力合同", "通过 case", "覆盖率", "Preflight 必需"],
        [[name, f"{row.get('passed_cases', 0)}/{row.get('cases', 0)}",
          _pct(row.get('fraction')), "是" if row.get('required_for_preflight') else "否"]
         for name, row in capabilities.items()],
    ) if capabilities else "正式专家证据合同尚未生成。")
    event_population = (
        ((preflight.get("preflight") or {}).get("data_and_reward_integrity") or {})
        .get("truth_event_population_audit") or {}
    )
    event_population_rows = [
        [
            split,
            (row or {}).get("cases"),
            (row or {}).get("by_sampling_stratum"),
            _pct((row or {}).get("sampling_event_stratum_recall")),
        ]
        for split, row in (event_population.get("by_split") or {}).items()
    ]
    event_population_table = (_markdown_table(
        ["split", "真值 AQI≥4 case", "按互斥采样层分布", "event 标签召回率"],
        event_population_rows,
    ) if event_population_rows else "正式真值事件总体审计尚未生成。")
    reward_audit_rows = [
        ["格式/字段投机", "schema 不合法或 daily/process 不一致", "整条提交 0 分；允许预算内自纠"],
        ["多输出头各自投机", "等级、首污、区间各自合理但联合不可能", "按 HJ 633 的 IAQI 尺度验证首污可达到等级并压过其他区间下界；process 头必填"],
        ["伪造工具返回", "模型文本冒充 tool result", "veRL bridge 只重放 assistant tool call，由隐藏 case 重算结果"],
        ["虚构引用", "不存在 ref/JSON Pointer/值", "严格解析工具+语义分区+Pointer+标量值，不匹配不给结构化引用分"],
        ["重复同一事实", "换 ref 或 42/42.0 凑满 grounding", "按工具返回内容哈希+JSON Pointer 去重，并要求 2 类证据"],
        ["真实但无关的引用", "机械复制两个正确标量赚 grounding", "仅作为低权重训练辅助且从 outcome 排除；实际证据利用必须通过同 seed 遮蔽剂量反应，当前不声称自动核验决策相关性"],
        ["干净日重复送分", "clean 同时拿 event/turning 满分", "无事件正确否定时相关分量弃权，level 承担评分"],
        ["交互模型比较占便宜", "agent 多出 grounding 分", "训练 composite 与公平 outcome_composite 分离"],
        ["超宽区间", "用无限宽区间追求覆盖", "按80% interval score连续指数衰减且无保底平台；毕业另看 coverage、width、原始 interval score"],
        ["成功条件化", "丢弃格式失败 rollout 后只报成功样本硬指标", "失败轨迹按最差保守计数进入 case 内平均，再按起报日分块 bootstrap"],
        ["事件层漏样", "把互斥采样标签 event 当成全部真实事件", "事件 probe/比较/门禁由隐藏完整真值 AQI≥4 定义；标签召回率仅作覆盖诊断"],
        ["因果套话", "自由文本机制解释看似合理", "不作为真值或奖励；保留为独立标注/遮蔽实验缺口"],
        ["Off-policy rollout", "AWQ 采样、BF16/LoRA 更新", "训练 rollout 由同步后的 trainable veRL/vLLM 策略生成"],
        ["工具 token 学习", "对环境返回反向传播", "veRL response_mask 仅保留 assistant token，并由冒烟审计"],
    ]
    reward_audit_table = _markdown_table(
        ["攻击面", "可能的假增益", "当前约束"], reward_audit_rows
    )
    executable_reward_rows = [
        [name, "PASS" if row.get("pass") else "BLOCKED", json.dumps(
            {key: value for key, value in row.items() if key != "pass"},
            ensure_ascii=False, allow_nan=False,
        )]
        for name, row in (reward_contract.get("checks") or {}).items()
    ]
    executable_reward_table = (_markdown_table(
        ["可执行 Reward 不变量", "状态", "观测"], executable_reward_rows
    ) if executable_reward_rows else "当前 reward 合同审计尚未生成。")
    sensitivity_rows = [[
        row.get("scenario"), _num(row.get("policy_mean")),
        _num(row.get("tabular_mean")), _num(row.get("paired_difference")),
        row.get("cluster_bootstrap_95pct"), row.get("direction"),
    ] for row in (reward_sensitivity.get("scenarios") or [])]
    sensitivity_table = (_markdown_table(
        ["权重情景", "策略", "Tabular", "配对差", "分块 95% CI", "方向"],
        sensitivity_rows,
    ) if sensitivity_rows else "正式 probe 与同信息集 tabular 完成后生成权重敏感性审计。")

    output_scope = contract.get("forecast_output_scope_audit") or {}
    output_scope_table = (_markdown_table(
        ["split", "六类首污真值 membership", "并列首污日"],
        [[
            split,
            json.dumps(
                (output_scope.get("truth_primary_memberships_by_split") or {}).get(split, {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            (output_scope.get("tied_primary_days_by_split") or {}).get(split, 0),
        ] for split in ("train", "val", "test")],
    ) if output_scope else "正式输出范围审计尚未生成。")

    source_rows = []
    for source_id, path in paths.items():
        identity = _identity(path)
        source_rows.append({
            "id": source_id,
            "label": source_id.replace("_", " ").title(),
            "path": identity["path"],
            "identity": identity,
            "query": {
                "engine": "sqlite" if source_id == "expert_analysis" else "artifact",
                "language": "sql" if source_id == "expert_analysis" else "json",
                "description": "Read-only audited project artifact; exact file identity is frozen below.",
                "tables_used": [identity["path"]],
                "executed_at": now,
                **({"sql": evidence_sql} if source_id == "expert_analysis" else {}),
            },
        })

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "专家预报 Harness：证据合同、Reward 与剩余门禁",
            "description": (
                "以 JJJ_ATMO 蒸馏语料作机制设计参考，对照全国开放证据、工具、"
                "reward-v0.7.9、硬指标与 RL 门禁；语料本身不进入训练。"
            ),
            "generatedAt": now,
            "cards": [],
            "charts": ([{
                "id": "evidence_mentions_chart",
                "title": "专家证据出现率（Top 5）",
                "subtitle": f"{(analysis.get('corpus') or {}).get('records', 0)} 条专家输出；词频仅是需求下界。",
                "type": "bar",
                "dataset": "evidence_mentions",
                "sourceId": "expert_analysis",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "category", "type": "nominal", "label": "证据类别"},
                    "y": {"field": "mention_rate", "type": "quantitative",
                          "label": "记录出现率", "format": "percent"},
                    "tooltip": [
                        {"field": "records", "type": "quantitative", "label": "命中记录"},
                        {"field": "total_records", "type": "quantitative", "label": "总记录"},
                    ],
                },
            }] if evidence_dataset else []),
            "sources": source_rows,
            "blocks": [
                {"id": "title", "type": "markdown",
                 "body": "# 专家预报 Harness：证据合同、Reward 与剩余门禁"},
                {"id": "executive_summary", "type": "markdown", "body": executive},
                {"id": "evidence_mentions", "type": "markdown", "body": (
                    "## 专家证据出现率（Top 5）\n\n"
                    "词频只是需求下界，不能代替通用必需/条件关键/校准增强的科学分类。"
                )},
                *([{"id": "evidence_mentions_chart_block", "type": "chart",
                    "chartId": "evidence_mentions_chart"}] if evidence_dataset else []),
                {"id": "decision_gap_matrix", "type": "markdown",
                 "body": "\n".join(decision_matrix_lines)},
                {"id": "case_contract", "type": "markdown", "body": (
                    "## 逐 case 证据合同\n\n" + capability_summary +
                    "\n\n合同只证明起报时合法可取且能送入工具，不证明策略已使用或因果解释正确。"
                )},
                {"id": "truth_event_population", "type": "markdown", "body": (
                    "## 真值事件总体与采样标签偏差\n\n" + event_population_table
                    + "\n\n互斥 stratum 只服务抽样平衡；事件毕业层始终由隐藏完整真值定义，"
                      "不会遗漏 PM10 主导或污染物切换事件。"
                )},
                {"id": "reward_attack_audit", "type": "markdown", "body": (
                    "## Reward 防投机审计\n\n" + reward_audit_table
                    + "\n\n### 可执行不变量\n\n" + executable_reward_table
                    + "\n\n### Outcome 权重敏感性\n\n" + sensitivity_table
                    + "\n\n这些不变量用于防已知投机；权重网格用于发现单一组合驱动的结论。"
                      "自由文本 claim 不在自动核验范围；二者都不能替代遮蔽实验、硬指标或证明模型具备预报技巧。"
                )},
                {"id": "forecast_output_scope", "type": "markdown", "body": (
                    "## 输出范围与尚未补齐的概率头\n\n" + output_scope_table
                    + "\n\n当前 AQI 与首污按六污染物评分，但 80% 浓度区间只覆盖 PM2.5/PM10/O3。"
                      "NO2/CO 首污仅出现在 train，SO2 首污在当前有效池未出现；因此暂不为极少数类别"
                      "增加三个高格式负担的概率头，但必须作为显式范围限制，不能声称六污染物均已校准。"
                )},
                {"id": "preflight", "type": "markdown",
                 "body": "## 开训门禁\n\n" + gate_summary},
                {"id": "comparisons", "type": "markdown",
                 "body": "\n".join(comparison_lines)},
                {"id": "boundary", "type": "markdown", "body": (
                    "## 科学边界\n\n"
                    "仍未完成的研究证据不会被数据覆盖率替代：时间安全 analog、独立污染机制标签、"
                    "SO2/NO2/CO 概率区间、自由文本因果核验、全传感器历史 NRT、"
                    "区间校准/spread-skill、证据剂量消融、"
                    + ("" if training_smoke_pass else "trainable policy 闭环、")
                    + "2026–27 冬季前瞻集"
                    "都必须单列。完整逐项审计见同目录 `appendix.md`。"
                )},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": now,
            "status": "ready",
            "datasets": {
                "headline": [{
                    "expert_records": (analysis.get("corpus") or {}).get("records"),
                    "meeting_dates": (analysis.get("corpus") or {}).get("meeting_dates"),
                    "attachment_rate": attachment.get("fraction"),
                    "expert_contract_pass": contract_pass,
                    "reward_contract_pass": reward_contract_pass,
                    "reward_sensitivity_current": reward_sensitivity_current,
                    "verl_runtime_contract_pass": runtime_contract_pass,
                    "formal_probe": formal_probe,
                    "first_submit_valid_rate": probe_summary.get("first_submit_valid_rate"),
                    "full_grounding_rate": (probe_summary.get("semantic_grounding") or {}).get(
                        "full_credit_rollout_rate"),
                    "usable_group_rate": probe_summary.get("usable_group_rate"),
                    "outcome_usable_group_rate_valid_only": probe_summary.get(
                        "outcome_usable_group_rate_valid_only"
                    ),
                    "event_decision_usable_group_rate_valid_only": probe_summary.get(
                        "event_decision_usable_group_rate_valid_only"
                    ),
                    "preflight_pass": preflight_pass,
                    "raw_preflight_pass": raw_preflight_pass,
                    "preflight_reward_identity_current": gate_freshness[
                        "reward_identity_current"
                    ],
                    "training_smoke_pass": training_smoke_pass,
                }],
                "evidence_mentions": evidence_dataset,
                "decision_chain": analysis.get("decision_chain") or [],
                "coverage_audit": analysis.get("coverage_audit") or [],
                "priority_backlog": analysis.get("prioritized_backlog") or [],
                "preflight_gates": [
                    {"gate": row[0], "status": row[1]} for row in gate_rows
                ],
                "preflight_freshness": [{
                    "reward_identity_current": gate_freshness[
                        "reward_identity_current"
                    ],
                    "preflight_reward_version": gate_freshness[
                        "preflight_reward"
                    ].get("version"),
                    "preflight_reward_config_sha256": gate_freshness[
                        "preflight_reward"
                    ].get("config_sha256"),
                    "current_reward_version": gate_freshness[
                        "current_reward"
                    ].get("version"),
                    "current_reward_config_sha256": gate_freshness[
                        "current_reward"
                    ].get("config_sha256"),
                }],
                "truth_event_population": [
                    {
                        "split": row[0], "truth_event_cases": row[1],
                        "by_sampling_stratum": row[2],
                        "sampling_event_stratum_recall": (
                            (event_population.get("by_split") or {}).get(row[0]) or {}
                        ).get("sampling_event_stratum_recall"),
                    }
                    for row in event_population_rows
                ],
            },
        },
        "sources": source_rows,
    }

    capability_rows = []
    for name, row in (contract.get("capabilities") or {}).items():
        capability_rows.append([
            name, f"{row.get('passed_cases', 0)}/{row.get('cases', 0)}",
            _pct(row.get("fraction")), "是" if row.get("required_for_preflight") else "否",
        ])
    coverage_rows = []
    for row in analysis.get("coverage_audit") or []:
        coverage_rows.append([
            row.get("order"), row.get("decision_task"), row.get("importance"),
            row.get("raw_data"), row.get("tool_exposure"), row.get("policy_use"),
            row.get("reward_test"), row.get("principal_gap"), row.get("priority"),
        ])
    backlog_rows = [[
        row.get("rank"), row.get("priority"), row.get("work"),
        row.get("deliverable"), row.get("success_gate"),
    ] for row in analysis.get("prioritized_backlog") or []]
    tabular_rows = []
    for split in ("val", "challenge_winter", "test",
                  "spatial_ood_val", "spatial_ood_test"):
        row = (tabular.get("results") or {}).get(split)
        if row:
            tabular_rows.append([
                split, row.get("n"), _num(row.get("composite")),
                _num(row.get("macro_stratum")),
                _num(((row.get("hard_metrics") or {}).get("level") or {}).get("ordinal_mae")),
                _num(((row.get("hard_metrics") or {}).get("event_daily_level_ge_4") or {}).get("csi")),
            ])
    gap_map = contract.get("known_nonblocking_research_gaps") or {}
    gap_labels = {
        "independent_pollution_mechanism_labels": "独立污染机制标签（不能用规则代理冒充真值）",
        "time_safe_historical_analog_index": "严格时间安全的全国历史相似过程索引",
        "previous_operational_forecast_archive": "全国上一版业务预报档案",
        "lagrangian_transport_trajectory": "拉格朗日输送轨迹（当前仅目标点风向/距离平流筛选）",
        "time_varying_emissions_and_control_measures": "逐时排放变化与临时管控信息",
        "surface_speciated_aerosol_observations": "全国地面颗粒物组分观测",
        "ensemble_spread_beyond_deterministic_gfs_ifs": "超出确定性 GFS/IFS 分歧的集合预报离散度",
        "so2_no2_co_probabilistic_interval_heads": "SO2/NO2/CO 概率区间输出头",
        "free_text_causal_claim_verification": "自由文本因果解释的独立核验真值",
        "decision_relevance_of_verified_citations": "已核验事实与具体预报结论之间的决策相关性真值",
        "all_fire_sensor_days_have_archived_nrt": "每个火点传感器日均有归档 NRT（当前允许显式 retrospective fallback）",
        "trained_policy_interval_calibration": "训后策略的独立区间校准",
        "trained_policy_spatial_ood_evaluation": "训后策略的整城空间 OOD 检验",
        "prospective_2026_27_winter_evidence": "2026–27 冬季完全前瞻证据",
    }
    training_checks = training_smoke.get("checks") or {}

    def backlog_status(rank: int) -> str:
        if rank == 1:
            return "已完成" if evidence_pass and contract_pass else "进行中"
        if rank == 2:
            required = (
                formal_probe
                and float(probe_summary.get("first_submit_valid_rate") or 0.0) >= 0.95
                and float((probe_summary.get("semantic_grounding") or {}).get(
                    "full_credit_rollout_rate") or 0.0) >= 0.50
                and float(probe_summary.get("usable_group_rate") or 0.0) >= 0.70
                and float(probe_summary.get(
                    "outcome_usable_group_rate_valid_only") or 0.0) >= 0.50
                and float(probe_summary.get(
                    "event_decision_usable_group_rate_valid_only") or 0.0) >= 0.50
            )
            return "已完成" if required else "进行中"
        if rank == 3:
            return "已完成" if (
                spatial_ood_ready and bool(tabular) and bool(guidance_cmp)
                and bool(tabular_cmp)
                and all(name in (tabular.get("results") or {}) for name in
                        ("spatial_ood_val", "spatial_ood_test"))
            ) else "进行中"
        if rank == 4:
            return "已完成" if training_smoke_pass else "待门禁后执行"
        if rank == 7:
            return "已完成" if (
                reward_sensitivity_current
                and int(reward_sensitivity.get("n_cases") or 0) >= 50
            ) else "待正式 probe 与 tabular"
        return "研究阶段待完成"

    appendix = [
        "# 专家空气质量预报：Harness、Reward 与缺口终审",
        "",
        f"生成时间：{now}",
        "",
        "用途：JJJ_ATMO 蒸馏语料只用于机制设计参考，不进入全国训练输入、SFT 文本、"
        "起报时证据或 reward 真值。词频只表示需求出现下界，不代表重要性排序。",
        "",
        "## 一、结论",
        "",
        executive.replace("## 执行摘要\n\n", ""),
        "",
        "## 二、专家决策链对 Harness 的逐项覆盖",
        "",
        (_markdown_table(
            ["序", "决策任务", "重要性", "原始数据", "工具暴露", "策略证据",
             "Reward/测试", "当前缺口", "优先级"], coverage_rows)
         if coverage_rows else "当前分析产物未提供覆盖矩阵。"),
        "",
        "## 三、逐 case 专家证据合同",
        "",
        (_markdown_table(["能力", "通过 case", "覆盖率", "Preflight 必需"], capability_rows)
         if capability_rows else "正式专家证据合同尚未生成。"),
        "",
        "该合同证明资料在起报时可取、语义分开且工具可暴露；它不证明策略已经使用，也不证明"
        "机制因果解释正确。后两项必须由 transcript 审计、遮蔽消融和结果指标验证。",
        "",
        "## 四、Reward 与评测尺",
        "",
        "- schema-v0.5.6：逐日 AQI 等级、首要污染物、PM2.5/PM10/O3 区间；首污需在 IAQI "
        "尺度可能压过其他区间下界，process 必填且与 AQI≥3 的 daily 头一致。",
        "- reward-v0.7.9 训练 `composite`：结果分量加低权重 grounding；伪造 ref/JSON Pointer/"
        "标量值不能得分，两条同类事实不能拿满。自由文本 claim 不作为因果真值。",
        "- 公平比较 `outcome_composite`：只含 level/event/interval/turning/primary，交互 agent、"
        "CAMS 和 tabular 使用相同权重分母。",
        "- 非塑形毕业指标：ordinal MAE、硬 CSI/POD/FAR、primary macro-F1、80% coverage/width/"
        "interval score；格式失败按保守最差计数进入 case 内平均，再按起报日整块 bootstrap。",
        "",
        "### Reward 防投机审计",
        "",
        reward_audit_table,
        "",
        "### 可执行 Reward 不变量",
        "",
        executable_reward_table,
        "",
        "### Outcome 权重敏感性",
        "",
        sensitivity_table,
        "",
        "不变量通过只能证明实现没有上述已知漏洞；权重网格也不能替代硬指标或业务技能评测。",
        "",
        "### 输出范围审计",
        "",
        output_scope_table,
        "",
        (output_scope.get("scientific_boundary") or
         "六类首污与三类概率区间的范围边界尚未生成。"),
        "",
        "## 五、当前同信息集 tabular",
        "",
        (_markdown_table(["split", "n", "outcome composite", "macro-stratum",
                          "ordinal MAE", "hard event CSI"], tabular_rows)
         if tabular_rows else "reward-v0.7.9 同信息集 tabular 尚未生成。"),
        "",
        "## 六、Preflight 与 Graduation",
        "",
        "### 真值事件总体审计",
        "",
        event_population_table,
        "",
        "互斥采样标签不等于事件真值；事件 probe、硬指标和毕业比较只用隐藏完整真值 AQI≥4。",
        "",
        "### 两层门禁",
        "",
        gate_table,
        "",
        "Preflight 不要求冻结模型先超过 CAMS。Graduation 才要求训后独立 val 超 CAMS、至少"
        "100 个 event case 的硬 CSI 与 outcome composite 均有正的分块 CI 下界、winter challenge "
        "不退化；论文阶段还要求同一隐藏事件层的 composite、硬 CSI 与峰值时序 MAE 超过强 "
        "tabular，不能把‘事件超 CAMS’和‘总体超 tabular’拼成机制结论；同时要求 outcome 权重网格稳定、"
        "遮蔽 synoptic+composition 后出现正的证据剂量反应。",
        "",
        "## 七、明确保留的研究缺口",
        "",
        *[f"- {'未完成' if value is False else '已完成'}：{gap_labels.get(name, name)}"
          for name, value in gap_map.items()],
        ("- 已完成：八聚类各留一整城的空间 OOD 划分；未完成：训后策略在该 test 上相对强 tabular 的分块检验。"
         if spatial_ood_ready else
         "- 未完成：证据类别遮蔽的 dose-response、留城市/留区制 OOD。"),
        ("- 已完成：trainable policy 50 step、checkpoint reload 后第 51 step、"
         "当前策略同步及工具 token loss mask 审计。"
         if training_smoke_pass else
         "- 未完成：trainable policy 50 step、当前策略同步与 checkpoint reload 的真实冒烟。"),
        "- 数据齐全不等于 RL 可学；正式 50×8 的格式、grounding、上下文与有效 group 门禁必须独立通过。",
        (f"- 已完成：锁定 veRL commit 的解析配置与 "
         f"{(runtime_contract.get('lifecycle') or {}).get('tools_loaded', 0)} 工具原生 "
         "create/execute/release 契约验证。"
         if runtime_contract_pass else
         "- 阻塞：veRL 解析配置或原生工具生命周期合同未通过。"),
        "",
        "可训练冒烟检查：",
        "",
        (_markdown_table(
            ["检查", "状态"],
            [[name, "PASS" if value else "BLOCKED"]
             for name, value in training_checks.items()],
        ) if training_checks else "可训练冒烟产物尚未生成。"),
        "",
        "## 八、剩余建设清单",
        "",
        (_markdown_table(["顺序", "优先级", "状态", "工作", "交付物", "成功门禁"],
                         [[row[0], row[1], backlog_status(int(row[0])), *row[2:]]
                          for row in backlog_rows])
         if backlog_rows else "当前分析产物未提供建设清单。"),
        "",
        "## 九、来源与可追溯性",
        "",
        _markdown_table(
            ["来源", "路径", "存在", "SHA256"],
            [[name, identity["path"], identity["exists"], identity.get("sha256", "—")]
             for name, identity in ((name, _identity(path)) for name, path in paths.items())],
        ),
        "",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "appendix.md").write_text("\n".join(appendix), encoding="utf-8")
    artifact_display = _identity(args.out_dir / "artifact.json")["path"]
    appendix_display = _identity(args.out_dir / "appendix.md")["path"]
    print(json.dumps({
        "status": status,
        "artifact": artifact_display,
        "appendix": appendix_display,
        "sources": {name: _identity(path) for name, path in paths.items()},
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
