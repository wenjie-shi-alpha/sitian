#!/usr/bin/env python3
"""Quantify expert-forecast evidence patterns and snapshot harness coverage.

The expert corpus is used only as a design reference.  This script does not
convert consultation text into training examples and does not attach it to
national cases.  Keyword results are deliberately labelled as lexical lower
bounds: absence of a keyword is not evidence that a concept was unimportant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.scoring import reward_spec  # noqa: E402
from sitian.provenance import (  # noqa: E402
    case_bundle_snapshot_matches, identity_matches_file, score_implementation_matches,
)
from sitian.open_evidence import (  # noqa: E402
    EVIDENCE_VERSION,
    NWP_TEMPORAL_PROFILE_VERSION,
)

DEFAULT_CORPUS_ROOT = Path(os.environ.get(
    "SITIAN_CORPUS_ROOT", str(REPO_ROOT.parent / "JJJ_ATMO" /
    "解压_20260807语料导出-2025年12月会商/20260807语料导出")
))


EVIDENCE_PATTERNS = {
    "天气系统与环流": r"高压|低压|槽|脊|环流|气压场|气压梯度|均压场|急流|冷空气|暖湿气流|天气形势",
    "风场与输送": r"风场|风向|风速|北风|南风|东风|西风|偏北|偏南|偏东|偏西|输送|传输|上游|下风向|辐合|辐散",
    "温度与冷暖平流": r"气温|温度|升温|降温|冷平流|暖平流|冷空气|增温|降温",
    "稳定度与边界层": r"边界层|混合层|逆温|稳定度|稳定层结|垂直扩散|抬升|下沉|垂直运动|扩散条件",
    "湿度与吸湿增长": r"湿度|相对湿度|水汽|湿增长|吸湿|高湿|低湿|雾",
    "实况态势与空间演变": r"实况|当前|目前|浓度|AQI|上升趋势|下降趋势|快速攀升|区域分布|空间分布|污染带|污染团",
    "模式指导、比较与订正": r"模式|预报结果|模式结果|模拟|CAMS|NAQP|CMAQ|EC|GFS|较昨日预报|前期预报|分歧|不确定|偏高|偏低|订正",
    "降水、云与辐射": r"降水|降雨|降雪|云量|多云|晴|辐射|日照|光照|光化学",
    "组分、前体物与二次转化": r"组分|硝酸盐|硫酸盐|铵盐|有机物|黑碳|一氧化碳|二氧化氮|二氧化硫|NO.?2|SO.?2|CO|VOCs?|前体物|二次转化|化学生成",
    "沙尘、火点与排放源": r"沙尘|扬尘|浮尘|火点|秸秆|燃烧|排放源|源区|源贡献|排放",
    "历史类比与上一版预报": r"历史同期|历史过程|相似过程|相似个例|类比|上一版|昨日预报|前期预报|此前预报",
    "时序、转折与过程阶段": r"起始|开始|峰值|高值|结束|缓解|清除|转折|转为|转好|转差|逐步|午后|夜间|早晨|上午|时起|持续至|提前|推迟",
}


TASK_PATTERNS = {
    "判定初始污染状态": r"当前|目前|实况|浓度|AQI|区域分布|污染形势|上升趋势|下降趋势",
    "识别天气系统演变": r"高压|低压|槽|脊|环流|气压场|冷空气|天气形势|系统控制|系统东移",
    "诊断水平输送与扩散": r"水平扩散|输送|传输|上游|下风向|风向|风速|风场|辐合|辐散",
    "诊断垂直混合与稳定度": r"垂直扩散|边界层|混合层|逆温|稳定度|稳定层结|抬升|下沉|垂直运动",
    "判断污染物生成与清除机制": r"吸湿|湿增长|二次转化|化学生成|光化学|降水清除|沉降|组分|前体物|沙尘|火点|燃烧",
    "融合模式并做偏差订正": r"模式|模拟|预报结果|较昨日预报|前期预报|分歧|不确定|偏高|偏低|订正",
    "识别事件阶段与关键时刻": r"起始|开始|峰值|高值|结束|缓解|清除|转折|午后|夜间|早晨|上午|时起|持续至|提前|推迟",
    "形成等级、首污与区间决策": r"AQI|等级|首要污染物|首污|浓度区间|范围|轻度污染|中度污染|重度污染|良|优",
}


# Expert-task to harness-layer audit.  These judgements are intentionally kept
# in executable data rather than only in prose so later tool/reward revisions
# can update the matrix and regenerate the report.
COVERAGE_AUDIT = [
    {
        "order": 1,
        "decision_task": "判定起报时的污染初态与区域态势",
        "importance": "通用必需",
        "key_evidence": "目标城市与区域 PM2.5/PM10/O3 小时实况、24h 趋势、空间热点",
        "raw_data": "强",
        "tool_exposure": "强",
        "policy_use": "仅小样本通过",
        "reward_test": "部分",
        "principal_gap": "process view 已给六污染物初态、趋势和热点；尚无正式 50×8 策略使用证据，结果 reward 不证明因果解释正确",
        "priority": "P1",
    },
    {
        "order": 2,
        "decision_task": "识别高低压、槽脊、冷空气及其演变",
        "importance": "通用必需",
        "key_evidence": "MSLP、500 hPa 高度场、多层风温湿、GFS/IFS 演变与分歧",
        "raw_data": "待合同解析",
        "tool_exposure": "强",
        "policy_use": "仅小样本通过",
        "reward_test": "弱",
        "principal_gap": "6h/12h 轨迹已进入 compact view，系统中心和槽脊仍需按需深查；全量覆盖完成前不能毕业",
        "priority": "P0",
    },
    {
        "order": 3,
        "decision_task": "判断水平扩散、上游输送与风向转折",
        "importance": "通用必需",
        "key_evidence": "地面至高空风场、上游城市污染、输送通道、地形与排放方位",
        "raw_data": "待合同解析",
        "tool_exposure": "较强",
        "policy_use": "未正式验证",
        "reward_test": "弱",
        "principal_gap": "已有地形自适应多层风转折、随时效来流情景、上游城市和静态源方位；当前仍是目标点平流筛选，不冒充拉格朗日轨迹",
        "priority": "P0",
    },
    {
        "order": 4,
        "decision_task": "判断边界层、逆温、垂直运动与静稳累积",
        "importance": "通用必需",
        "key_evidence": "BLH、925/850/700 hPa 温湿、850/700/500 hPa omega、风切变",
        "raw_data": "强",
        "tool_exposure": "强",
        "policy_use": "未正式验证",
        "reward_test": "弱",
        "principal_gap": "compact view 已给 RH925/700、omega700、逆温和逐日 BLH/RH/雨；仍需消融证明策略确实使用",
        "priority": "P1",
    },
    {
        "order": 5,
        "decision_task": "区分 PM2.5、O3、沙尘/PM10 与火点过程机制",
        "importance": "事件条件必需",
        "key_evidence": "AOD 组分、CO/NO2/SO2、温度/云/辐射、沙尘 AOD、火点、降水",
        "raw_data": "待合同解析",
        "tool_exposure": "较强",
        "policy_use": "未正式验证",
        "reward_test": "结果较强/机制弱",
        "principal_gap": "已有短波、AOD组分、气体、沙尘、火点和 PM10 区间；缺可独立标注的机制真值，不能把规则代理冒充专家判断",
        "priority": "P1",
    },
    {
        "order": 6,
        "decision_task": "融合多模式指导并做历史偏差订正",
        "importance": "通用必需",
        "key_evidence": "CAMS/CMAQ/NAQP、GFS/IFS 分歧、城市×季节×lead 历史误差",
        "raw_data": "强",
        "tool_exposure": "强",
        "policy_use": "未正式验证",
        "reward_test": "部分",
        "principal_gap": "guidance-bias 已按时间安全注册；需 frozen/RL 消融证明订正增值而非盲抄",
        "priority": "P1",
    },
    {
        "order": 7,
        "decision_task": "定位污染过程的起始、峰值、清除与转折时刻",
        "importance": "通用必需",
        "key_evidence": "高频天气转折、小时实况趋势、逐日污染指导及过程阶段",
        "raw_data": "待合同解析",
        "tool_exposure": "强",
        "policy_use": "待正式 50×8 探针",
        "reward_test": "稠密塑形+硬指标",
        "principal_gap": "已有高频过程视图、daily/process 一致性和硬 CSI；尚未证明真值 AQI>=4 事件层的组内方差及训后 CSI 增值",
        "priority": "P0",
    },
    {
        "order": 8,
        "decision_task": "表达情景分支、不确定性与可信区间",
        "importance": "校准必需",
        "key_evidence": "多模式/集合离散度、历史误差分位、情景触发条件",
        "raw_data": "较强",
        "tool_exposure": "较强",
        "policy_use": "未证",
        "reward_test": "较强",
        "principal_gap": "PM2.5/PM10/O3 80%区间与硬校准指标已覆盖；categorical confidence 已退出策略输出，仍需验证模式离散度与实际误差的 spread-skill",
        "priority": "P1",
    },
    {
        "order": 9,
        "decision_task": "对照上一版预报和历史相似过程",
        "importance": "校准增强",
        "key_evidence": "昨日预报、历史相似形势/污染过程、历史模式误差",
        "raw_data": "部分",
        "tool_exposure": "弱",
        "policy_use": "弱",
        "reward_test": "缺失",
        "principal_gap": "全国 case 当前无 previous_forecast/analog 输入；环境只在数据真实存在时注册工具，相似过程索引仍为 P2",
        "priority": "P2",
    },
    {
        "order": 10,
        "decision_task": "把结论锚定到可核验的证据断言",
        "importance": "科研可信必需",
        "key_evidence": "字段、有效时刻、空间对象、方向、来源版本与 hash",
        "raw_data": "强",
        "tool_exposure": "强",
        "policy_use": "待正式 50×8 探针",
        "reward_test": "较强",
        "principal_gap": "ref+JSON Pointer+标量值已抗伪造且需两条不重复断言；仍只核验事实，不把自由文本因果解释当真值",
        "priority": "P1",
    },
]


PRIORITIZED_BACKLOG = [
    {
        "rank": 1,
        "priority": "P0",
        "work": "完成 open-evidence-v1 高频合同与逐 case 重挂接",
        "deliverable": "GFS/IFS 6h→72h/12h→132h、辐射、CAMS 三通道、火点、静态源全部派生并严格附着",
        "success_gate": "原始/派生/attachment 均 100%，六污染物与时间门禁失败为 0",
    },
    {
        "rank": 2,
        "priority": "P0",
        "work": "冻结 v0.7.9 正式探针与格式/grounding 裁决",
        "deliverable": "至少 50 case×group8，记录首次合法、两条语义断言、过程工具使用和组内方差",
        "success_gate": (
            "首次合法>=95%，full-grounding rollout>=50%，总 reward 有效 group>=70%；"
            "outcome 与 event 决策有效 group 分别>=50%"
        ),
    },
    {
        "rank": 3,
        "priority": "P0",
        "work": "完成同信息集强基线与非塑形毕业指标",
        "deliverable": "重训 tabular；另对八个整城留出的空间 OOD val/test 报告同一套硬指标",
        "success_gate": "reward/manifest 一致；空间 OOD test 至少100 case、30个起报日块且所有主要结论有硬指标和分块CI",
    },
    {
        "rank": 4,
        "priority": "P0",
        "work": "完成 trainable 1.7B/4B LoRA 冒烟",
        "deliverable": "当前策略同步 rollout、assistant-only mask、void-turn 率门禁、Dr.GRPO、checkpoint reload",
        "success_gate": "50–200 step finite loss；同步与重载前后行为变化可复核",
    },
    {
        "rank": 5,
        "priority": "P1",
        "work": "验证污染类型条件路由和证据剂量反应",
        "deliverable": "PM2.5 高湿二次、O3 光化学、沙尘/PM10、火点烟霾分别遮蔽关键通道",
        "success_gate": "按过程类型的遮蔽实验呈剂量反应，而不是所有 case 无差别塞满工具输出",
    },
    {
        "rank": 6,
        "priority": "P1",
        "work": "先固定 compact 过程证据学习决策，再释放工具选择",
        "deliverable": "单轮/受控多轮课程，工具 token mask、void-turn 率门禁、Dr.GRPO、动态样本过滤",
        "success_gate": "小模型 smoke 闭环通过，event/turning 有有效样本率，工具调用不发生 lazy collapse",
    },
    {
        "rank": 7,
        "priority": "P1",
        "work": "预注册 reward 权重敏感性与 Pareto 稳定性分析",
        "deliverable": "在不改组件定义的前提下扫描合理权重网格，同时报告各硬指标、策略排序与事件层退化",
        "success_gate": "主要结论不依赖单一权重组合；任何权重选择均不能掩盖 MAE、事件命中或校准恶化",
    },
    {
        "rank": 8,
        "priority": "P2",
        "work": "建设时间安全的相似过程检索",
        "deliverable": "以起报时可见特征检索历史个例，返回相似度、可用截止和结果摘要",
        "success_gate": "严格时间门禁、城市/日期去重，并通过 analog 遮蔽消融证明净增值",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_valid(identity: dict | None) -> bool:
    if not identity or not identity.get("path"):
        return False
    stored = Path(identity["path"])
    path = stored if stored.is_absolute() else REPO_ROOT / stored
    return bool(
        path.is_file()
        and path.stat().st_size == identity.get("bytes")
        and _sha256(path) == identity.get("sha256")
    )


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _compile(patterns: dict[str, str]) -> dict[str, re.Pattern[str]]:
    return {name: re.compile(pattern, re.IGNORECASE) for name, pattern in patterns.items()}


def _shorten(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", "", text)
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _scan(records: list[dict], patterns: dict[str, str]) -> list[dict]:
    compiled = _compile(patterns)
    rows = []
    for name, regex in compiled.items():
        matched = [record for record in records if regex.search(record["output"])]
        rows.append({
            "category": name,
            "records": len(matched),
            "total_records": len(records),
            "mention_rate": round(len(matched) / len(records), 6) if records else None,
            "rank_basis": "output_text_lexical_lower_bound",
            "examples": [
                {
                    "meeting_date": item.get("metadata", {}).get("meeting_date"),
                    "topic_title": item.get("metadata", {}).get("topic_title"),
                    "excerpt": _shorten(item["output"]),
                }
                for item in matched[:2]
            ],
        })
    rows.sort(key=lambda item: (-item["records"], item["category"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def _tool_usage(probe: dict) -> list[dict]:
    rows = probe.get("rows", [])
    counts = Counter(tool for row in rows for tool in set(row.get("tools", [])))
    return [
        {
            "tool": tool,
            "rollouts": count,
            "total_rollouts": len(rows),
            "rollout_rate": round(count / len(rows), 6) if rows else None,
        }
        for tool, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _formal_probe(probe: dict) -> bool:
    """Return whether policy-use statements have the planned evidential scale."""
    observed = (probe.get("provenance") or {}).get("reward") or {}
    current = reward_spec()
    provenance = probe.get("provenance") or {}
    return (
        int(probe.get("n_cases") or 0) >= 50
        and int(probe.get("group_size") or 0) >= 8
        and int(probe.get("n_rollouts") or 0) >= 400
        and abs(float(probe.get("temperature", -999)) - 1.0) <= 1e-9
        and observed.get("version") == current.get("version")
        and observed.get("config_sha256") == current.get("config_sha256")
        and score_implementation_matches(provenance, REPO_ROOT)
        and all(_identity_valid(provenance.get(key)) for key in (
            "probe_script", "agent_driver", "environment_implementation",
            "process_evidence_implementation", "guidance_bias_implementation",
            "case_manifest", "standards_manifest",
        ))
        and case_bundle_snapshot_matches(
            provenance.get("case_input_snapshot"), REPO_ROOT
        )
    )


def _resolved_coverage_audit(
    coverage: dict, probe: dict, expert_contract: dict
) -> list[dict]:
    """Resolve volatile coverage wording from audited artifacts.

    Raw availability, tool exposure, observed policy use, and reward testability
    are deliberately separate claims.  In particular, complete downloads never
    get promoted to evidence that a policy used a field or learned its causal
    meaning.
    """
    coverage_provenance = coverage.get("provenance") or {}
    coverage_derivations = coverage_provenance.get("derivation_implementations") or {}
    ready = (
        coverage.get("ready_for_full_attachment") is True
        and coverage.get("contract_version") == EVIDENCE_VERSION
        and (coverage.get("nwp_temporal_profile") or {}).get("version")
            == NWP_TEMPORAL_PROFILE_VERSION
        and _identity_valid(coverage.get("source_manifest"))
        and _identity_valid(coverage_provenance.get("audit_implementation"))
        and _identity_valid(
            coverage_provenance.get("open_evidence_contract_implementation")
        )
        and len(coverage_derivations) == 6
        and all(_identity_valid(identity) for identity in coverage_derivations.values())
    )
    contract_provenance = expert_contract.get("provenance") or {}
    contract_pass = (
        expert_contract.get("preflight_contract_passed") is True
        and len(contract_provenance.get("case_manifests") or []) == 3
        and all(_identity_valid(identity)
                for identity in contract_provenance.get("case_manifests") or [])
        and all(_identity_valid(contract_provenance.get(key)) for key in (
            "audit_implementation", "process_evidence_implementation",
            "open_evidence_implementation", "guidance_bias_implementation",
        ))
    )
    formal = _formal_probe(probe)
    usage = {row["tool"]: row["rollout_rate"] for row in _tool_usage(probe)}
    grounding = (probe.get("summary") or {}).get("semantic_grounding") or {}

    def policy_status(*tools: str) -> str:
        if not formal:
            return "待正式 50×8 探针"
        rate = max((float(usage.get(tool) or 0.0) for tool in tools), default=0.0)
        if rate <= 0:
            return "正式探针未见使用"
        return f"已观测使用（最高 {rate:.1%}）"

    resolved = [dict(row) for row in COVERAGE_AUDIT]
    complete_raw_orders = {1, 2, 3, 4, 5, 6, 7, 8, 10}
    policy_tools = {
        1: ("get_process_evidence", "get_observations"),
        2: ("get_process_evidence", "get_synoptic_evidence"),
        3: ("get_process_evidence", "get_synoptic_evidence"),
        4: ("get_process_evidence", "get_synoptic_evidence"),
        5: ("get_process_evidence", "get_pollution_evidence"),
        6: ("get_model_guidance", "get_guidance_bias"),
        7: ("get_process_evidence", "get_synoptic_evidence"),
        8: ("get_model_guidance", "get_guidance_bias", "get_synoptic_evidence"),
        9: ("find_similar_cases", "get_previous_forecast"),
    }
    for row in resolved:
        order = int(row["order"])
        if order in complete_raw_orders:
            row["raw_data"] = (
                "强（逐 case 合同通过）" if ready and contract_pass
                else "阻塞（覆盖或合同未通过）"
            )
        if order in policy_tools:
            row["policy_use"] = policy_status(*policy_tools[order])
        elif order == 10:
            if not formal:
                row["policy_use"] = "待正式 50×8 探针"
            else:
                rate = grounding.get("full_credit_rollout_rate")
                row["policy_use"] = (
                    "正式探针无完整 grounding"
                    if rate is None else f"完整 grounding {float(rate):.1%}"
                )

    # Replace only stale operational clauses; scientific limitations remain.
    resolved[0]["principal_gap"] = (
        "process view 已给六污染物初态、趋势和热点；"
        + ("正式策略使用率见本行，" if formal else "仍待正式策略使用证据，")
        + "结果 reward 不证明因果解释正确"
    )
    resolved[1]["principal_gap"] = (
        "6h/12h 轨迹、系统中心和槽脊进入 compact/deep view；"
        + ("原始覆盖和合同已通过，仍需形势证据遮蔽消融" if ready and contract_pass
           else "全量覆盖完成前不能毕业")
    )
    resolved[6]["principal_gap"] = (
        "已有高频过程视图、daily/process 一致性和硬 CSI；"
        "仍需在不依赖互斥采样标签的真值 AQI>=4 事件层，同时证明相对 CAMS 与同信息强 tabular "
        "的训后 event/turning 增值及遮蔽消融"
    )
    resolved[9]["principal_gap"] = (
        "ref+JSON Pointer+标量值已抗伪造且需两条不重复断言；"
        "仍只核验事实，不把自由文本因果解释当真值"
    )
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument(
        "--open-evidence-coverage",
        type=Path,
        default=REPO_ROOT / "data/interim/open_evidence_coverage.json",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=REPO_ROOT / "data/interim/probe_open_evidence_v079_n50_g8.json",
    )
    parser.add_argument(
        "--reward-signal",
        type=Path,
        default=REPO_ROOT / "data/interim/group_signal_diagnostic_v079.json",
    )
    parser.add_argument(
        "--guidance-bias",
        type=Path,
        default=REPO_ROOT / "data/interim/guidance_bias_history_v1.json",
    )
    parser.add_argument(
        "--expert-contract",
        type=Path,
        default=REPO_ROOT / "data/interim/expert_evidence_contract_audit.json",
    )
    parser.add_argument(
        "--spatial-ood-audit",
        type=Path,
        default=REPO_ROOT / "data/interim/spatial_ood_audit.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data/interim/efr.json",
    )
    args = parser.parse_args()
    args.open_evidence_coverage = _repo_path(args.open_evidence_coverage)
    args.probe = _repo_path(args.probe)
    args.reward_signal = _repo_path(args.reward_signal)
    args.guidance_bias = _repo_path(args.guidance_bias)
    args.expert_contract = _repo_path(args.expert_contract)
    args.spatial_ood_audit = _repo_path(args.spatial_ood_audit)
    args.out = _repo_path(args.out)

    files = sorted(
        path for path in args.corpus_root.rglob("*_sft.jsonl")
        if "__MACOSX" not in path.parts
    )
    records = []
    manifest = []
    for path in files:
        line_count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("output"), str):
                    raise ValueError(f"missing output string: {path}:{line_number}")
                records.append(row)
                line_count += 1
        manifest.append({
            "relative_name": path.relative_to(args.corpus_root).as_posix(),
            "bytes": path.stat().st_size,
            "records": line_count,
            "sha256": _sha256(path),
        })

    aggregate_digest = hashlib.sha256()
    for item in manifest:
        aggregate_digest.update(item["relative_name"].encode("utf-8"))
        aggregate_digest.update(item["sha256"].encode("ascii"))

    coverage = json.loads(args.open_evidence_coverage.read_text(encoding="utf-8"))
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    reward_signal = (json.loads(args.reward_signal.read_text(encoding="utf-8"))
                     if args.reward_signal.is_file() else {})
    guidance_bias = json.loads(args.guidance_bias.read_text(encoding="utf-8"))
    expert_contract = (json.loads(args.expert_contract.read_text(encoding="utf-8"))
                       if args.expert_contract.is_file() else {})
    spatial_ood = (json.loads(args.spatial_ood_audit.read_text(encoding="utf-8"))
                   if args.spatial_ood_audit.is_file() else {})

    meeting_dates = Counter(
        row.get("metadata", {}).get("meeting_date") or "unknown" for row in records
    )
    artifact = {
        "artifact_type": "expert_forecast_requirements_analysis",
        "analysis_version": "2.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "design_boundary": {
            "corpus_role": "mechanism_design_reference_only",
            "not_used_as": ["national_training_input", "forecast_time_evidence", "reward_truth"],
            "importance_rule": (
                "importance is classified by universal necessity, conditional decisiveness, "
                "and calibration value; mention frequency is only a lexical lower bound"
            ),
        },
        "corpus": {
            "files": len(files),
            "records": len(records),
            "meeting_dates": len(meeting_dates),
            "output_characters": sum(len(row["output"]) for row in records),
            "aggregate_sha256": aggregate_digest.hexdigest(),
            "records_by_meeting_date": dict(sorted(meeting_dates.items())),
            "source_manifest": manifest,
        },
        "evidence_mentions": _scan(records, EVIDENCE_PATTERNS),
        "decision_task_mentions": _scan(records, TASK_PATTERNS),
        "coverage_audit": _resolved_coverage_audit(
            coverage, probe, expert_contract
        ),
        "prioritized_backlog": PRIORITIZED_BACKLOG,
        "harness_snapshot": {
            "open_evidence": {
                "contract_version": coverage.get("contract_version"),
                "nwp_temporal_profile": coverage.get("nwp_temporal_profile"),
                "source_manifest": coverage.get("source_manifest"),
                "issue_dates": coverage.get("issue_dates"),
                "case_attachments": coverage.get("case_attachments"),
                "ready_for_full_attachment": coverage.get("ready_for_full_attachment"),
                "synoptic": coverage.get("derived_by_issue", {}).get("synoptic"),
                "composition": coverage.get("derived_by_issue", {}).get("composition"),
                "spatial_observations": coverage.get("derived_by_issue", {}).get("spatial_observations"),
                "fires": coverage.get("derived_by_issue", {}).get("fires"),
                "trace_gases": coverage.get("required_by_issue", {}).get(
                    "cams_model_level_137_trace_gases"
                ),
            },
            "probe": {
                "model": probe.get("model"),
                "policy_stage": probe.get("policy_stage"),
                "n_rollouts": probe.get("n_rollouts"),
                "summary": probe.get("summary"),
                "tool_usage": _tool_usage(probe),
            },
            "reward_signal": {
                "reward": reward_signal.get("reward"),
                "overall": reward_signal.get("overall"),
                "event_core_signal": reward_signal.get("event_core_signal"),
                "by_component": reward_signal.get("by_component"),
            },
            "guidance_bias": guidance_bias.get("summary"),
            "spatial_ood": {
                "selection_policy": spatial_ood.get("selection_policy"),
                "heldout_by_cluster": spatial_ood.get("heldout_by_cluster"),
                "train": spatial_ood.get("train"),
                "validation": spatial_ood.get("spatial_ood_val"),
                "test": spatial_ood.get("spatial_ood_test"),
                "integrity": spatial_ood.get("integrity"),
                "scientific_boundary": (
                    "warm-season unseen-city evaluation; not evidence for winter-event OOD"
                ),
            },
        },
        "methodology": {
            "unit": "one distilled question-answer record",
            "text_scanned": "output only",
            "matching": "case-insensitive Chinese/Latin regular expressions; one hit maximum per category per record",
            "known_limitations": [
                "The corpus covers 24 consultation files in December 2025 and is not a national prevalence sample.",
                "Categories overlap because expert reasoning integrates evidence rather than using mutually exclusive steps.",
                "Low-frequency conditional evidence can be decisive for rare events; frequency is not an importance score.",
                "The corpus informs mechanism and tool design only and is not attached to replay cases.",
            ],
        },
        "provenance": {
            "script": {
                "path": "scripts/analyze_expert_forecast_requirements.py",
                "sha256": _sha256(Path(__file__)),
            },
            "inputs": [
                {"path": _display_path(path), "sha256": _sha256(path)}
                for path in (args.open_evidence_coverage, args.probe,
                             args.reward_signal, args.guidance_bias,
                             args.expert_contract, args.spatial_ood_audit)
                if path.is_file()
            ],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "out": str(args.out),
        "files": len(files),
        "records": len(records),
        "meeting_dates": len(meeting_dates),
        "top_evidence": artifact["evidence_mentions"][:5],
        "tool_usage": artifact["harness_snapshot"]["probe"]["tool_usage"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
