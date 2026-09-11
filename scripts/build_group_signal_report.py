#!/usr/bin/env python3
"""把 group-signal diagnostic 打包为 Data Analytics portable report artifact。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostic", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    diagnostic_path = REPO_ROOT / args.diagnostic
    data = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    overall = data["overall"]
    event = data["event_core_signal"]
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    stratum_rows = []
    stratum_chart_rows = []
    for row in data["by_stratum"]:
        enriched = {
            **row,
            "composite_signal_rate": row["variable_composite_rate"],
            "decision_signal_rate": row["variable_decision_component_rate"],
            "event_signal_rate": row["variable_event_component_rate"],
        }
        stratum_rows.append(enriched)
        for field, label in (
            ("composite_signal_rate", "Composite"),
            ("decision_signal_rate", "决策分量"),
            ("event_signal_rate", "Event 分量"),
        ):
            stratum_chart_rows.append({
                "stratum": row["stratum"],
                "signal": label,
                "signal_rate": enriched[field],
                "cases": row["cases"],
                "mean_reward_sigma": row["mean_reward_sigma"],
                "interval_only_groups": row["interval_only_groups"],
            })
    summary = [{
        "nonzero_group_rate": overall["nonzero_group_rate"],
        "meaningful_group_rate": overall["range_gt_0_01_group_rate"],
        "rank_rich_group_rate": overall["groups_with_3_or_4_unique_rewards"] / overall["cases"],
        "interval_only_group_rate": overall["interval_only_variable_groups"] / overall["cases"],
        "event_positive_rollouts": event["event_component_positive_rollouts"],
        "event_valid_rollouts": event["valid_event_rollouts"],
        "median_sigma": overall["median_sigma"],
    }]
    source_id = "round2_group_signal"
    source = {
        "id": source_id,
        "label": "Round-2 thinking-on group-signal diagnostic",
        "path": args.diagnostic,
        "query": {
            "engine": "python",
            "language": "python",
            "sql": (
                "SELECT *\n"
                f"FROM read_json_auto('{args.diagnostic}');"
            ),
            "description": (
                "DuckDB 查询读取已审阅的诊断快照；scripts/analyze_group_signal.py 对最终有效提交按 case 聚合，计算 population sigma、"
                "reward range、唯一值数、component 方差与阈值敏感性。"
            ),
            "executed_at": generated,
            "tables_used": [
                "data/interim/probe_thinking_on_v041_round2_n50_g4_paired.json"
            ],
            "filters": [
                "thinking_enabled=true",
                "reward.version=0.4.1",
                "submitted=true",
                "50 stratified cases; group_size=4; paired rollout seed base=7",
            ],
            "metric_definitions": [
                "非零组率 = case 内最终有效提交 reward 的 population sigma > 1e-12 的 case 数 / 50。",
                "有意义差值组率 = case 内 reward max-min > 0.01 的 case 数 / 50。",
                "决策分量 = level、event、turning、primary；任一分量组内 sigma > 1e-12 即记为变化。",
                "event 分量信号只统计 reward component.event 本身，不用互斥 event stratum 的 composite sigma 代替。",
            ],
        },
    }
    title = "Round-2 组内 Reward 信号诊断"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "评估 thinking-on 的 GRPO 组内区分度是否与 event 订正目标对齐。",
            "generatedAt": generated,
            "cards": [
                {
                    "id": "usable_signal",
                    "description": "非零差异与排除千分位微差后的组级覆盖率。",
                    "dataset": "summary",
                    "sourceId": source_id,
                    "metrics": [
                        {"label": "非零组率", "field": "nonzero_group_rate", "format": "percent"},
                        {"label": "range > 0.01", "field": "meaningful_group_rate", "format": "percent"},
                    ],
                },
                {
                    "id": "event_hits",
                    "description": "真实 event 层有效 rollout 中 event component 大于 0 的数量。",
                    "dataset": "summary",
                    "sourceId": source_id,
                    "metrics": [
                        {"label": "Event 命中 rollout", "field": "event_positive_rollouts", "format": "number"},
                        {"label": "有效 event rollout", "field": "event_valid_rollouts", "format": "number"},
                    ],
                },
                {
                    "id": "rank_richness",
                    "description": "组内至少出现三个不同 reward 的 case 比例。",
                    "dataset": "summary",
                    "sourceId": source_id,
                    "metrics": [
                        {"label": "3–4档 reward 组率", "field": "rank_rich_group_rate", "format": "percent"},
                        {"label": "仅 interval 变化", "field": "interval_only_group_rate", "format": "percent"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "signal_by_stratum",
                    "title": "各分层组内信号覆盖率",
                    "subtitle": "每层 10 case；比较 composite、决策分量与 event 分量是否在组内变化。",
                    "headerMarkdown": "同一层内三种口径使用相同 case 分母。",
                    "intent": "comparison",
                    "question": "thinking-on 的组内差异是否覆盖各层，并与 event 目标对齐？",
                    "rationale": "分组柱形图直接比较同一分层下三种同分母信号覆盖率。",
                    "comparisonContext": {
                        "denominator": "每层 10 case",
                        "grain": "case",
                        "normalization": "组内 population sigma > 1e-12",
                        "unit": "case rate",
                    },
                    "type": "bar",
                    "dataset": "stratum_signal_long",
                    "sourceId": source_id,
                    "encodings": {
                        "x": {"field": "stratum", "type": "nominal", "label": "分层"},
                        "y": {
                            "field": "signal_rate",
                            "type": "quantitative",
                            "label": "有组内变化的 case 比例",
                            "format": "percent",
                        },
                        "color": {"field": "signal", "type": "nominal", "label": "信号口径"},
                        "tooltip": [
                            {"field": "cases", "type": "quantitative", "label": "Case 数"},
                            {"field": "mean_reward_sigma", "type": "quantitative", "label": "平均 reward σ"},
                            {"field": "interval_only_groups", "type": "quantitative", "label": "仅 interval 变化组"},
                        ],
                    },
                    "palette": {"kind": "categorical", "name": "blue-orange-pink"},
                    "legend": {"position": "bottom", "sort": "spec", "interactive": True},
                    "valueFormat": "percent",
                    "yAxisTitle": "组率",
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "threshold_table",
                    "title": "Reward range 阈值敏感性",
                    "subtitle": "50 个 case；pair 分母为 291 个有效 rollout 对。",
                    "dataset": "threshold_sensitivity",
                    "sourceId": source_id,
                    "defaultSort": {"field": "threshold", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "threshold", "label": "阈值", "type": "number", "format": "0.000"},
                        {"field": "threshold_label", "label": "最小差值", "type": "text"},
                        {"field": "qualifying_groups", "label": "通过组数", "format": "number"},
                        {"field": "group_rate", "label": "通过组率", "format": "percent"},
                        {"field": "pair_rate", "label": "通过 pair 比例", "format": "percent"},
                    ],
                },
                {
                    "id": "component_table",
                    "title": "Reward 分量的组内变化覆盖",
                    "subtitle": "仅最终有效提交；每个 component 独立计算组内 population σ。",
                    "dataset": "component_signal",
                    "sourceId": source_id,
                    "defaultSort": {"field": "variable_groups", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "component", "label": "分量", "type": "text"},
                        {"field": "variable_groups", "label": "变化组数", "format": "number"},
                        {"field": "variable_rate", "label": "变化组率", "format": "percent"},
                        {"field": "mean_sigma_all_groups", "label": "全组平均 σ", "format": "number"},
                    ],
                },
            ],
            "sources": [{
                "id": source_id,
                "label": source["label"],
                "path": args.diagnostic,
            }],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": (
                        "## 技术结论：总体够做 smoke，event 核心信号仍不够\n\n"
                        "thinking-on 已把 **78%** 的 case 变成非零方差组；把微小差异过滤为 "
                        "`max-min > 0.01` 后仍有 **74%**，所以不能说 composite 完全没有区分度。"
                        "但只有 **46%** 的组产生 3–4 档 reward，且真实 event 层 39 条有效 rollout 中，"
                        "`event` component **0 条大于 0**。结论是：可启动 50-step 管线 smoke，"
                        "但不能据此启动长训或宣称模型已具备 event 订正学习信号。"
                    ),
                },
                {"id": "metrics", "type": "metric-strip", "cardIds": ["usable_signal", "event_hits", "rank_richness"]},
                {
                    "id": "alignment_finding",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": (
                        "## Composite 方差没有等价转化为 event 命中差异\n\n"
                        "event 层 10 个 case 中有 7 组 composite 可区分、6 组至少一个决策分量变化，"
                        "但 event 分量变化组为 **0**。这些差异主要来自 level、interval、turning 或 primary；"
                        "它们可以提供接近阈值的间接梯度，却尚未证明任何 rollout 跨过 AQI≥4 的事件门槛。"
                    ),
                },
                {"id": "stratum_chart", "type": "chart", "chartId": "signal_by_stratum", "layout": "full"},
                {
                    "id": "strength_finding",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": (
                        "## 大多数非零组不是纯浮点噪声，但排序仍偏粗\n\n"
                        "37/50 组的 reward range 超过 0.01，说明只按非零方差统计并未严重夸大覆盖率。"
                        "然而 11 组完全同分、16 组只有两档 reward；全部有效 rollout pair 中，"
                        "差值超过 0.01 的只有 **51.9%**。GRPO 会标准化组内 advantage，"
                        "因此还应过滤极小 range，避免把千分位差异放大成强更新。"
                    ),
                },
                {"id": "thresholds", "type": "table", "tableId": "threshold_table", "layout": "full"},
                {
                    "id": "component_finding",
                    "type": "markdown",
                    "sourceId": source_id,
                    "body": (
                        "## 连续 interval 是主要覆盖来源，真正决策差异只覆盖 29/50 组\n\n"
                        "interval 在 38 组中变化，而 level/event/turning/primary 任一变化的组只有 29 个；"
                        "另有 10 组只有 interval 变化。这个结构能优化区间数值，却可能让策略继续停留在"
                        "“不报事件、只微调区间”的局部最优。"
                    ),
                },
                {"id": "components", "type": "table", "tableId": "component_table", "layout": "full"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## 口径与实验范围\n\n"
                        "对象为 reward v0.4.1 的 Round-2 thinking-on：50 个分层 case、每组 4 条 rollout；"
                        "只对 197 条最终有效提交计算组内 population σ。非零阈值为 1e-12；"
                        "“决策分量”指 level、event、turning、primary，不含连续 interval 与 grounding。"
                    ),
                },
                {
                    "id": "method",
                    "type": "markdown",
                    "body": (
                        "## 方法与稳健性检查\n\n"
                        "每个 case 独立计算 reward σ、range、唯一 reward 数与 component σ；再按分层汇总。"
                        "同时用 0.001–0.10 的 range 阈值做敏感性分析，并检查 σ 集中度：前两组占 summed σ "
                        "的 26.9%，剔除后平均 σ 仍为 0.0228，因此总体差异并非完全由两个异常 case 驱动。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 限制：当前 artifact 只能证明 reward 区分，不能证明行为因果\n\n"
                        "probe 未保存最终 forecast 对象和完整证据选择，因此无法区分“不同预报得到同分”与"
                        "“完全相同预报”，也无法验证差异是否来自真正使用资料。event 只有 10 case/39 条有效 rollout；"
                        "本报告是训练可行性诊断，不是泛化效果估计。"
                    ),
                },
                {
                    "id": "recommendation",
                    "type": "markdown",
                    "body": (
                        "## 建议：保留 thinking-on，但把 smoke 目标改成 event 信号是否被激活\n\n"
                        "1. 50-step smoke 动态过滤 `reward range <= 0.01` 的组，并同时报告 exact-zero 与阈值过滤率。\n\n"
                        "2. 先接入 `get_guidance_bias` 做 frozen event panel；若 event component 仍全 0，"
                        "不要靠增加 group size 直接长训。\n\n"
                        "3. smoke 的继续条件至少包括：event case 中 event-component 非零组率上升、"
                        "event/turning 均分上升、格式率不退化；只看到 composite 上升不算通过。\n\n"
                        "4. 下一版 rollout artifact 保存规范化 forecast、adjustment diagnostics 与证据通道，"
                        "用消融验证差异确实来自资料订正。"
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 后续需要回答的问题\n\n"
                        "- guidance-bias 是否能让冻结策略在 event case 首次跨过 AQI≥4 门槛？\n\n"
                        "- 组内差异中，有多少是方向/转折变化，有多少只是区间边界微调？\n\n"
                        "- range 过滤阈值 0.01 在 50-step smoke 中是否改善梯度稳定性而不显著降低有效 batch？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "summary": summary,
                "stratum_signal": stratum_rows,
                "stratum_signal_long": stratum_chart_rows,
                "threshold_sensitivity": data["threshold_sensitivity"],
                "component_signal": data["by_component"],
            },
        },
        "sources": [source],
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
