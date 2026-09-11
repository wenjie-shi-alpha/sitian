import React from "react";

import {
  DataComponent,
  DataTable,
  EvidenceChart,
  ReportSection,
  RichNarrative,
  useDataApp,
} from "../../data-app-public.jsx";

const baselineSpec = {
  type: "horizontalBar",
  x: "system",
  y: "score",
  xLabel: "系统",
  yLabel: "outcome composite",
  valueDecimals: 3,
  startAtZero: true,
  colors: { score: "var(--chart-1)" },
};

const verdicts = [
  {
    label: "工程可行性",
    verdict: "基本成立",
    tone: "positive",
    detail: "数据、结构化环境、规则 reward、veRL 原生合同与 trainable 1.7B 基座已具备。",
  },
  {
    label: "小模型闭环",
    verdict: "待一次关键实测",
    tone: "pending",
    detail: "需要完成 50 步训练、从 checkpoint 恢复到第 51 步，并验收权重同步、loss mask 与 finite loss。",
  },
  {
    label: "效果与扩模",
    verdict: "尚未证明",
    tone: "negative",
    detail: "冻结 8B 仍落后于 CAMS 与同信息集 tabular；当前没有任何训后策略的独立验证结果。",
  },
];

const gateColumns = [
  { key: "gate", label: "门禁" },
  { key: "status", label: "状态", presentation: "status" },
  { key: "evidence", label: "完成版证据" },
];

const probeColumns = [
  { key: "metric", label: "指标" },
  {
    key: "value",
    label: "实测",
    renderCell: (value) => `${(Number(value) * 100).toFixed(value < 0.99 ? 2 : 1)}%`,
  },
  {
    key: "threshold",
    label: "门槛/参照",
    renderCell: (value, row) => `${row.metric.includes("void") ? "≤" : "≥"}${(Number(value) * 100).toFixed(0)}%`,
  },
  { key: "status", label: "判断", presentation: "status" },
];

function VerdictStrip() {
  return <div className="verdict-strip" aria-label="分层可行性判断">
    {verdicts.map((item) => <div className={`verdict-item verdict-${item.tone}`} key={item.label}>
      <div className="verdict-label">{item.label}</div>
      <div className="verdict-value">{item.verdict}</div>
      <div className="verdict-detail">{item.detail}</div>
    </div>)}
  </div>;
}

function EvolutionFigure({ rows }) {
  return <div className="route-flow" data-reviewed-rows aria-label="项目技术路线演进">
    {rows.map((row, index) => <React.Fragment key={row.stage}>
      <div className="route-step">
        <div className="route-index">{String(index + 1).padStart(2, "0")}</div>
        <div>
          <div className="route-name">{row.stage}</div>
          <div className="route-status">{row.status}</div>
          <div className="route-evidence">{row.evidence}</div>
        </div>
      </div>
      {index < rows.length - 1 && <div className="route-arrow" aria-hidden="true">→</div>}
    </React.Fragment>)}
  </div>;
}

function DataReadinessFigure({ rows }) {
  const byName = Object.fromEntries(rows.map((row) => [row.asset, row]));
  const raw = byName["全国原始 case"];
  const valid = byName["全国有效 case"];
  const train = byName["最终训练 manifest"];
  return <div className="data-flow" data-reviewed-rows aria-label="全国 case 数据与训练隔离">
    <div className="data-node">
      <span className="data-number">{raw?.count.toLocaleString("zh-CN")}</span>
      <span className="data-label">全国原始 case</span>
    </div>
    <div className="data-connector">
      <span>隔离 491 个观测无效 case</span>
      <span aria-hidden="true">→</span>
    </div>
    <div className="data-node">
      <span className="data-number">{valid?.count.toLocaleString("zh-CN")}</span>
      <span className="data-label">正式有效 case</span>
    </div>
    <div className="data-connector">
      <span>固定 train split；再隔离 challenge 与 OOD 城市</span>
      <span aria-hidden="true">→</span>
    </div>
    <div className="data-node data-node-final">
      <span className="data-number">{train?.count.toLocaleString("zh-CN")}</span>
      <span className="data-label">最终训练 manifest</span>
    </div>
  </div>;
}

export function ReportContent() {
  const {
    reviewedRows,
    visible,
    canEdit,
    mode,
    appTitle,
    setAppTitle,
  } = useDataApp();

  const evolution = reviewedRows("project_evolution");
  const dataAssets = reviewedRows("data_assets");
  const gates = reviewedRows("readiness_gates");
  const probe = reviewedRows("probe_metrics");
  const baselines = reviewedRows("baseline_comparison");
  const verification = reviewedRows("verification");

  const allSummarySources = {
    readiness_gates: gates,
    baseline_comparison: baselines,
    project_evolution: evolution,
  };

  return <article className="report-content" aria-label="LLM 微调技术可行性验证报告">
    <header className="report-hero">
      <div className="report-kicker">项目研究报告 · 证据截止 2026-09-03 02:22 PDT</div>
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        aria-label={canEdit && mode === "edit" ? "编辑报告标题" : undefined}
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}
        onKeyDown={canEdit && mode === "edit" ? (event) => {
          if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
        } : undefined}>{appTitle}</h1>
      <RichNarrative id="report:description" className="report-deck" label="编辑报告说明"
        value="本报告基于 `JJJ_ATMO` 的领域语料与早期训练工程，以及 `sitian` 当前已经完成的全国案例、Agent 环境、规则化评分、冻结策略探针和 veRL 接口产物，判断 LLM 微调是否值得进入下一阶段。这里的“可行”特指：是否具备开展受控小模型实验的条件，不等同于模型效果已经验证。" />
    </header>

    {visible("decision-summary") && <ReportSection id="decision-summary" title="结论摘要"
      queryId="readiness_gates" queryIds={["readiness_gates", "baseline_comparison", "project_evolution"]}
      sourceRowsByQuery={allSummarySources} showHeading={false}>
      <RichNarrative id="decision-summary:body" className="report-lead" label="编辑结论摘要"
        value={`## 结论摘要

综合判断为 **有条件可行**：当前项目已经跨过“只有语料和脚本”的阶段，具备启动 **Qwen3-1.7B/4B + LoRA 的短程闭环验证**的工程基础；但尚未跨过“训练确实发生且带来可泛化增益”的证据门槛。

- 建议批准的范围：完成当前合同下的冻结探针，随后开展 1.7B/4B、50+1 步的 LoRA/Agent RL 冒烟与小规模独立验证。
- 暂不建议批准的范围：直接投入 Qwen3-8B 长训，或对外宣称“微调后已优于数值模式/强监督模型”。
- 技术主线：让 SFT 只承担必要的格式与工具协议冷启动，数值预报能力由规则化结果 reward 驱动的 Agent RL 学习；若当前探针达标，可跳过通用 SFT。`} />
      <VerdictStrip />
    </ReportSection>}

    {visible("evidence-boundary") && <ReportSection id="evidence-boundary" title="验证范围与证据边界"
      queryId="project_evolution" sourceRows={evolution} showHeading={false}>
      <RichNarrative id="evidence-boundary:body" className="report-analysis" label="编辑验证范围"
        value={`## 验证范围与证据边界

本次验证覆盖四个层面：领域数据是否可用、训练对象与奖励是否定义清楚、训练栈是否能形成可审计闭环、冻结策略是否显示可学习空间。结论只使用已完成并能回溯到本地文件的证据。

报告截点时，schema-v0.6.4 / reward-v0.8.1 的新一轮 50×8 探针仍在运行，因此正文中的完成版探针数值来自 schema-v0.6.3 / reward-v0.8.0。新合同已经修复 evidence value 列表等接口问题，但在完整产物落盘前，不把进行中结果提前写成通过。`} />
    </ReportSection>}

    {visible("route-evolution") && <section className="report-section">
      <ReportSection id="route-evolution-context" title="从问答微调到可验证的预报决策训练"
        queryId="project_evolution" sourceRows={evolution} showHeading={false}>
        <RichNarrative id="route-evolution-context:body" className="report-analysis" label="编辑技术演进说明"
          value={`## 从问答微调到可验证的预报决策训练

\`JJJ_ATMO\` 已完成 SFT/GRPO/LoRA 的工程入口、会商材料结构化和预报智能体原型，证明领域资料能够被整理成模型输入；\`sitian\` 则把训练目标从“复述专家答案”改为“在时间门禁内调用工具并提交可评分预报”。这一步改变了可行性判断：语料规模不再是唯一瓶颈，真正关键的是结果 reward、同策略 rollout、数据隔离和独立评测。`} />
      </ReportSection>
      <DataComponent id="route-evolution" title="技术路线演进与当前停点" queryId="project_evolution"
        kind="custom" displayRows={evolution} sourceRows={evolution} description="状态描述来自项目文档和已完成训练产物盘点。">
        <EvolutionFigure rows={evolution} />
      </DataComponent>
    </section>}

    {visible("data-readiness") && <section className="report-section">
      <ReportSection id="data-readiness-context" title="数据基础足以支持受控实验，但两类语料不能混用"
        queryId="data_assets" sourceRows={dataAssets} showHeading={false}>
        <RichNarrative id="data-readiness-context:body" className="report-analysis" label="编辑数据结论"
          value={`## 数据基础足以支持受控实验，但两类语料不能混用

\`JJJ_ATMO\` 中可核对到 24 个领域 SFT 导出文件、272 条问答样本和 22 个有效校验对话文件（已排除 macOS 元数据副本）。这些材料适合提炼预报术语、任务分解、证据类别与专家审阅清单，但不应作为全国 episode 的结果真值，因为会商叙述可能包含回顾性判断，且与当前结构化输出、时间门禁和多污染物合同并不一一对应。

当前项目拥有 7,548 个全国 case，隔离 491 个观测无效 case 后保留 7,057 个；再完成 winter challenge 与空间 OOD 的隔离，最终训练 manifest 为 3,061 个。样本规模和隔离设计已足以支撑小模型实验。已有 6,172 条脚本生成的格式 SFT 轨迹属于 reward-v0.4、无 thinking 的旧合同，**不能直接用于当前 reward-v0.8.1/schema-v0.6.4**；如需 SFT，应按当前合同重新生成。`} />
      </ReportSection>
      <DataComponent id="data-readiness" title="全国 case 数据与训练隔离" queryId="data_assets"
        kind="custom" displayRows={dataAssets} sourceRows={dataAssets} description="数量单位为 case；最终训练集与验证、挑战及 OOD 目标按 manifest 隔离。">
        <DataReadinessFigure rows={dataAssets} />
      </DataComponent>
    </section>}

    {visible("readiness-status") && <section className="report-section">
      <ReportSection id="readiness-status-context" title="六项 preflight 中四项通过，两项仍是硬门槛"
        queryId="readiness_gates" sourceRows={gates} showHeading={false}>
        <RichNarrative id="readiness-status-context:body" className="report-analysis" label="编辑开训状态说明"
          value={`## 六项 preflight 中四项通过，两项仍是硬门槛

数据/reward 完整性、上下文预算、采样效率和结构化 grounding 已经通过完成版 preflight。未通过项集中在两处：一是首次提交合法率为 93.75%，低于 95% 门槛且 void-turn 审计未完成；二是没有 \`training_smoke.json\`，因而无法证明 LoRA 权重更新后已同步到 rollout 引擎、工具返回 token 已正确 mask、loss 有限且 checkpoint 可恢复。

“四项通过”不宜折算为 67% 就绪度，因为任一硬门槛失败都足以阻止自动开训。`} />
      </ReportSection>
      <DataComponent id="readiness-status" title="完成版 preflight 门禁" queryId="readiness_gates"
        kind="table" displayRows={gates} sourceRows={gates} description="完成版 artifact 为 preflight 2.3.0；进行中的新合同重跑未计入。">
        <DataTable rows={gates} columns={gateColumns} rowKey="gate" searchable={false}
          caption="LLM 微调开训门禁及完成版证据" />
      </DataComponent>
    </section>}

    {visible("probe-diagnostics") && <section className="report-section">
      <ReportSection id="probe-diagnostics-context" title="冻结策略已显示可学方差，但格式和轨迹纪律仍需收口"
        queryId="probe_metrics" sourceRows={probe} showHeading={false}>
        <RichNarrative id="probe-diagnostics-context:body" className="report-analysis" label="编辑探针说明"
          value={`## 冻结策略已显示可学方差，但格式和轨迹纪律仍需收口

50 case×8 rollout 的探针中，50/50 组保留有效差异，说明 GRPO 不会在起点因“组内全同”而失去梯度；完整 grounding rollout 率为 70.41%，process 工具使用率为 98.75%，表明模型基本会使用证据链。另一方面，首次提交合法率仍少 1.25 个百分点才达门槛，13.25% 的 rollout 含 void-turn；按当前研究要求，正式探针的 void-turn 率需不高于 5%。

因此，是否需要 SFT 应由新合同探针决定，而不是预设：如果 schema-v0.6.4 重跑后格式和 void-turn 均达标，直接进入小模型 Agent RL；若仍失败，只做小规模、thinking 设置一致的 tutor SFT。`} />
      </ReportSection>
      <DataComponent id="probe-diagnostics" title="冻结策略的格式、采样与证据指标" queryId="probe_metrics"
        kind="table" displayRows={probe} sourceRows={probe} description="比例来自完成版 400 条 rollout；void-turn 使用当前研究设计的 5% 要求。">
        <DataTable rows={probe} columns={probeColumns} rowKey="metric" searchable={false}
          caption="冻结 Qwen3-8B-AWQ 探针指标与门槛" />
      </DataComponent>
    </section>}

    {visible("baseline-gap") && <section className="report-section">
      <ReportSection id="baseline-gap-context" title="当前最重要的负证据：冻结模型尚未形成数值增值"
        queryId="baseline_comparison" sourceRows={baselines} showHeading={false}>
        <RichNarrative id="baseline-gap-context:body" className="report-analysis" label="编辑基线比较说明"
          value={`## 当前最重要的负证据：冻结模型尚未形成数值增值

在同一 50-case 探针样本上，冻结 Qwen3-8B-AWQ 的 outcome composite 为 0.535，CAMS 为 0.631，同信息集 tabular 为 0.683。模型相对 CAMS 的配对差为 -0.096，95% 分块 bootstrap 区间为 [-0.170, -0.023]；相对 tabular 的差为 -0.148，区间为 [-0.190, -0.109]。

这组结果不能否定“微调可行”，因为被测模型尚未训练；但它明确了证明责任：训练后至少需要先在独立 val 上超过 CAMS，最终还要与同信息集 tabular 比较。任何只报告 reward 上升、不报告这些对照的实验，都不足以验证业务可行性。`} />
      </ReportSection>
      <EvidenceChart id="baseline-gap" queryId="baseline_comparison" title="冻结策略与同样本基线的结果分"
        spec={baselineSpec} rows={baselines} sourceRows={baselines} height={300}
        description="outcome composite 排除 evidence 与 grounding；分数越高越好。" />
    </section>}

    {visible("training-route") && <ReportSection id="training-route" title="建议的最小可行验证路线"
      queryId="readiness_gates" queryIds={["readiness_gates", "probe_metrics", "baseline_comparison"]}
      sourceRowsByQuery={{readiness_gates: gates, probe_metrics: probe, baseline_comparison: baselines}}
      showHeading={false}>
      <RichNarrative id="training-route:body" className="report-analysis" label="编辑验证路线"
        value={`## 建议的最小可行验证路线

1. **先冻结合同。** 等待 schema-v0.6.4/reward-v0.8.1 的 50×8 探针完整落盘，重算首次合法率、void-turn、上下文预算、grounding 和有效组率。合同变化后不沿用旧结论。
2. **按条件决定 SFT。** 若首次合法率 ≥95%、最终提交率 ≥95%、void-turn ≤5%，则跳过通用 SFT；否则仅生成当前合同、thinking 设置一致的 rejection-sampled tutor 轨迹，做 ≤1 epoch、低学习率的格式/协议冷启动。JJJ_ATMO 的 272 条语料用于设计与抽检，不直接监督数值结果。
3. **跑通 1.7B/4B LoRA 闭环。** 在本地 RTX 4090 24 GB 上完成 50 步并从 checkpoint 恢复到第 51 步；必须同时确认 policy sync、工具 token loss mask、finite loss、adapter 变化和可重复恢复。
4. **先学决策，再释放工具选择。** 阶段 A 由 harness 预取固定证据，策略只提交预报；阶段 B 再从 A 的 checkpoint 继续训练多轮工具选择。这样可把“数值决策能否学会”和“agency 是否增值”分开验证。
5. **以对照决定是否扩模。** 小模型训练后在独立 val 上相对 CAMS 的起报日分块 bootstrap 95% 区间下界必须 >0，winter challenge 不退化，才申请云上 8B；论文级结论还需超过同信息集 tabular，并完成隐藏真值事件集与 2026–27 冬季前瞻评测。`} />
    </ReportSection>}

    {visible("risk-gates") && <ReportSection id="risk-gates" title="主要风险与停止条件"
      queryId="readiness_gates" queryIds={["readiness_gates", "data_assets", "probe_metrics"]}
      sourceRowsByQuery={{readiness_gates: gates, data_assets: dataAssets, probe_metrics: probe}}
      showHeading={false}>
      <RichNarrative id="risk-gates:body" className="report-analysis" label="编辑风险说明"
        value={`## 主要风险与停止条件

- **合同漂移风险：** schema/reward 每次升级都必须重做 probe、tabular 和 preflight；旧 6,172 条 SFT 轨迹不得直接复用。
- **门禁实现漂移：** 当前研究文档要求 void-turn ≤5%，而完成版 \`rl_preflight.py\` 默认阈值仍为 20%；开训前必须统一机器门禁，避免“脚本通过、研究标准未通过”。
- **格式学习掩盖数值学习：** 最终自纠成功不能替代首次提交合法率；SFT 只能解决格式/协议，不应被包装成预报能力提升。
- **off-policy 风险：** 训练 rollout 必须来自当前可训练策略。冻结 8B-AWQ 只能评测，不能为另一份 BF16/LoRA 权重采样。
- **事件与季节泛化风险：** 暖季 val/test 不能证明冬季重污染能力；winter challenge 只用于模型选择，2026–27 冬季才提供真正前瞻证据。
- **算力放大风险：** 本地 24 GB 仅承担 1.7B/4B 冒烟。若小模型不能稳定越过 CAMS，不进入 8B 长训，也不通过降低基线标准“制造成功”。`} />
    </ReportSection>}

    {visible("engineering-verification") && <ReportSection id="engineering-verification" title="工程复核与最终判断"
      queryId="verification" queryIds={["verification", "readiness_gates", "baseline_comparison"]}
      sourceRowsByQuery={{verification, readiness_gates: gates, baseline_comparison: baselines}}
      showHeading={false}>
      <RichNarrative id="engineering-verification:body" className="report-analysis" label="编辑最终判断"
        value={`## 工程复核与最终判断

本次报告生成期间实际复核了两套工程：\`JJJ_ATMO\` 单元测试 53 项通过，\`sitian\` 测试 202 项通过。测试结果说明代码与数据合同具有较好的回归基础，但不替代 GPU 训练与独立效果评测。

最终建议是：**将 LLM 微调立项定义为“小模型、短程、可审计的技术可行性验证”，当前可以继续；将“8B 长训与业务增值验证”列为条件触发项，当前不启动。** 最关键的下一份证据不是更多方案文字，而是新合同下完成的 50×8 preflight 与可复现的 50+1 步 \`training_smoke.json\`。`} />
      <div className="verification-line" data-reviewed-rows>
        {verification.map((row) => <span key={row.project}><strong>{row.project}</strong>：{row.passed} passed</span>)}
      </div>
    </ReportSection>}

    <RichNarrative id="report:disclosure" className="report-disclosure" label="编辑证据说明"
      value="证据说明：本报告不使用进行中探针的部分结果，不将代码通过、dry-run 成功或冻结模型探针解释为微调后效果。所有数量、门禁和基线比较均可从组件的来源面板回溯。" />
  </article>;
}
