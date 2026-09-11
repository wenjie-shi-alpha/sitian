# Agent RL 训练设计（成功率优先的 harness 配套）

核心判断：多轮工具调用 + 稀疏 reward 的 RL，主要死法是 **GRPO 组内方差归零**
（同 case 的 G 条 rollout 全格式失败 → 全 0；或 clean 日全拿相近高分）。
所有设计围绕"保住组内可学分差"展开。

## 0a. 开放证据数据状态（2026-09-01）

- 原逐日 `open-evidence-v1` 已完成 401/401 个起报日：synoptic、composition、spatial
  observations、fires 均为 100%；GFS/IFS 原始快照各 3,208/3,208。2026-09-01 冻结的
  `hybrid-6h72-12h132-v2` 补充契约把天气过程改为前 72h 每 6h、其后每 12h，并增加两源
  地表短波辐射；在 16,040 个形势快照和 18,847 个辐射 sidecar 达到 100% 前，旧 401/401
  不得表述成新时间契约已完成。
- 全国正式 train/val/test 的 7,548 个 case 已补齐六污染物小时实况；新高频形势、辐射、
  CAMS 组成、Earth Engine 总柱、火点和静态源正在按 manifest 补齐。旧 evidence attachment
  不得冒充新合同完成；最终器只在原始/派生覆盖 100%、逐 case 严格校验和时间门禁均通过后重挂接。
- 三类 CAMS 通道均为开训必需项：AOD/组分、ADS model-level-137 近地层 CO/NO2/SO2，
  以及 Earth Engine 总柱 CO/NO2/SO2。两类气体垂直语义严格分开，任一路不得填补另一条
  通道的缺值；当前完成度以最终 coverage artifact 为准，不沿用旧 401/401 表述。
- 同信息集 tabular 使用固定宽度投影；正式维数只由本轮产物记录，不沿用旧附件的 1,798/lead。
  v2.2 投影包含完整 20 时点 GFS/IFS、5 时点 AOD、21 时点两类气体轨迹、六污染物 48h 原始
  实况、五个目标槽的指导/偏差/地面诊断投影，以及最多 64 个方向均衡输送候选池；其中全国
  CAMS 污染指导和地面诊断只实有 issue+1…issue+4，第 5 天没有 CAMS 值，依靠已延伸至
  +132h 的 GFS/IFS 形势外推；baseline 的 carry 投影、lead 和缺测/偏差可用性均显式留档。
  它还复用与 agent
  完全相同的地形自适应通风、最近系统距离和未来来流确定性投影，避免 agent 靠独占
  `get_assessment` / `get_guidance_bias` 或 Harness 特征工程取得表观增益。训练矩阵固定为
  float32，正式维数、理论训练矩阵大小和实测进程峰值内存写入产物；
  purged calibration 与 val/winter challenge/test
  的 city-day target overlap 全为 0。全量近地层气体接入后，reward-v0.4 结果为 val 0.7533、
  winter challenge 0.6592、test 0.8213；winter event 为 0.4647，是 agent 增值必须正面超过的
  强基线切片。
- 全国低层输送不再固定取 925 hPa：按模式地形选择至少高于地面 150 m 的最低可用
  925/850/700/500 hPa 层，地下压力面在 agent 附件和 tabular 投影中同时置空。正式报告需增加
  高地形切片，确认增值不是压力层插值伪影。未来来流另给出目标点风速下的方向—距离平流筛选，
  且在字段中明确它不是拉格朗日轨迹；无有效相邻温度层对时通风状态只能报 `partial_profile`。
- 旧逐日合同的数据覆盖门禁曾通过；新高频合同仍在下载，不能继承旧门禁结果。历史
  v0.4.x thinking/alias 实验只用于解释为何不应预设 SFT。当前 reward-v0.7.9 已修复
  PM10/O3 输出合同、结构化引用与公平比较分离；正式 50×8 探针和同版本 tabular 将在新附件
  100% 后自动运行。总 preflight 仍需这两项实测以及 trainable stack/50-step smoke。

## 0. 探针实测结论（2026-08-28，Qwen3-8B-AWQ / 本地 4090）

| 指标 | 结果 |
|---|---|
| 格式通过率 | 24/24 = 100%（只作 provisional） |
| 平均步数 | 6.0（查全 5 个信息工具后提交，行为高度一致） |
| 冻结模型 composite | 0.633 |
| 同 case 对照 persistence / CAMS / climatology | 0.673 / 0.740 / 0.654 |
| 三基线事后择优参考 | 0.795 |
| 分量 | grounding 1.00、level 0.718、event 0.833、**turning 0.000、interval 0.357、primary 0.390** |
| 分层 | clean 0.727、o3 0.638、switch 0.627、pm10_primary 0.842、**event 0.236** |
| GRPO 零方差组 | 1/6 = 16.7%（仅小样本，50 case×group 8 正在复测） |
| token/episode | prompt 19.5K（累计）/ completion 654 |

开放证据接入后的定向 2-case×group4 探针补充了一条更严格的格式结论：8/8 rollout 最终提交，
平均 reward 0.583，但首次 submit 合法率是 **0/8**，总计 10 次无效提交；clean 组 4 条 reward
完全相同，历史互斥 `event` 采样层标准差 0.0577。`probe_model.py` v1.3 起会把每类 schema 校验错误写入产物，
避免“最终自纠成功”掩盖训练初期的全零格式梯度。该小样本当时只说明需要定位格式问题；
后续 Round-1 已证明主要根因是 harness alias validator，是否需要 SFT 改由 v0.4.1 Round-2 裁决。

读数：这 6 个 case 只显示“格式可能已会、event/process 可能是弱项”，
不能由 0.633 vs 0.740 得出稳健的模型差于 CAMS 结论。正式探针先在 case 内平均
8 条 rollout，再按 issue date/天气过程块做配对 bootstrap；格式失败轨迹保留 0 分。
当前只能说**可以做小成本 RL 管线冒烟**，不能由 100% 小样本格式通过率
推导“可直接 8B 长训”。
完整判断见 `RL_READINESS_AUDIT_2026-08-28.md`。

⚠️ 首轮探针当场发现并修复了一个 harness 缺陷：模型调用 `get_assessment` 后把证据类型写成
`"assessment"`（词表外），12 步反复失败耗尽预算。根因是系统提示只说"取自受控词表"却没列出
词表，且工具与词表无映射说明。开放证据工具加入后又出现首次提交非法，说明每次扩展工具/schema
都必须重跑首次提交门禁；**这类缺陷若留到训练期会造成全零方差组、直接毁掉梯度**。

历史版本已从本地执行记录核实：明确列出八类证据词表、工具到词表的映射，以及
"严禁填写 `pollution_evidence`、`assessment`" 的提示于 2026-08-29 00:47 PDT 已生效；
reward-v0.4 正式 50×8 thinking-off 探针于 2026-08-30 21:51 PDT 才启动。因此 82.25%
不是旧提示词的遗留结果：在提示和 submit JSON Schema 都明令禁止的情况下，仍产生
70 条 `pollution_evidence` 类型错误和 27 条 `assessment` 类型错误。结论是提示修复不足，
但不能直接推出需要 SFT：这些名称是模型对 harness 工具/字段的自然复述，不是预报能力错误。

Round-1 配对 50×4×2（reward v0.4.0）进一步完成了归因：两臂最终提交率均为 98.5%；
首次合法率 off/on 为 79.5%/51.5%，但仅有效提交的零方差组为 74%/22%，组内 σ 为
0.0063/0.0299，event σ 为 0.0033/0.0323。按 issue_date 整块 bootstrap，thinking-on
均分差 95% CI 为 `[-0.0174, 0.0072]`，event σ 差 CI 为 `[0.0072, 0.0452]`。
因此“thinking 导致格式退化”不成立：thinking 放大的是证据条目数量，进而放大同一个词表缺陷；
剔除格式噪声后，它提供了显著更强的可学分差。固定分析产物为
`data/interim/analysis_thinking_round1_v04_n50_g4_paired.json`。

## 0b. 基线矩阵（预注册）

| 组 | 配置 | 探针规模 | 回答的问题 |
|---|---|---:|---|
| 0 脚本阶梯 + tabular | persistence / guidance / climatology / oracle-switch / 训练期内选型的 HGB/LightGBM/XGBoost | 已有 | 强基线；tabular 是最终必须正面比较的对手，旧版本分数不跨 reward 版本沿用 |
| 1a 纯 LLM·仅简报 | 冻结 8B，无检索工具，单次结构化提交 | 50×1 | 裸知识/最小信息能力；不用于宣称 agency 增益 |
| 1b 纯 LLM·全证据预拼 | 冻结 8B，与主实验相同 thinking，无多轮工具 | 50×4 | agency 消融：信息集不变，只去掉按需检索和多轮交互 |
| 2 Harness + 8B frozen | thinking off/on 双臂 | 50×4×2 | thinking 价值、有效组内方差和 RL 起跑线 |
| 3 Harness + 商业模型 | 原生 reasoning，相同工具/schema/步数上限 | 50×1 | 能力上界参考；报 token/费用/延迟，不进主推断表 |
| 4 Harness + 8B RL | 训练与评测 thinking 分布一致 | 主实验 | 证明 RL 相对同构 frozen policy 的增益 |

1b 使用与 Harness 工具返回字节一致的冻结 snapshot，相同时间门禁、压缩方式和固定顺序；
记录 snapshot hash、输入 token 和任何截断。它可以看到所有当时可用证据，因而对 Harness 是一个保守对照。
输出仍使用同一 forecast schema，但禁用检索工具和环境重试。

**agency 的严格比较是 2 vs 1b**：模型、thinking、信息可得性、采样参数和输出 schema 均保持一致。
4 vs 1b 只能说明“RL + agency”端到端系统增益，不能单独归因给多轮交互。

## 0c. Thinking 训练臂裁决（Round-1 记录 + Round-2 预注册）

- 固定同一 50 个分层 case、group=4、温度、工具预算、prompt/schema/reward identity，只改
  thinking on/off。所有主指标先在 case 内平均，再按 issue date/过程块做 paired cluster bootstrap；
  不把同 case 的 4 条 rollout 当独立样本。
- Round-1 原规则和结果只作历史记录，不因结果反写：其格式门禁未过；均分非劣和有效提交
  的方差优势均已通过。修复单一 harness 缺陷后进入有明确理由的 corrective Round-2，
  不把 v0.4.0 与 v0.4.1 reward 曲线混画。
- event 层“可学分差”不计入格式失败造成的假方差：只在最终合法且 reward identity 一致的 rollout
  上计算 case 内 population σ；零方差统一按 `σ <= 1e-12`。必须同时报告全部 rollout 与仅有效提交
  两套口径，禁止把前者误标为后者。历史 event panel 只有 10 case/8 个 issue-date cluster，CI 作为支持证据，
  不夸大为大样本结论；主训后使用隐藏完整真值筛出至少 100 个 AQI≥4 event case，不能用互斥 `stratum=event` 代替。
- **v0.4.1 修复**：validator 接受并立即归一 `pollution_evidence→diagnostic`、
  `assessment→diagnostic`、`composition→model_guidance`；`EVIDENCE_TOOL_MAP` 保留别名来源映射。
  prompt、工具 schema、case、rollout seed、温度和预算均不变。validation 语义已变，reward 升为 v0.4.1。
- **Round-2 规则**：同一 50 case×4×2、同 seed 重跑。两臂首次合法率均 `>=95%` 才跳过格式 SFT；
  之后要求 thinking-on 的有效提交可用组率更高、总体与真值 AQI≥4 事件组内 σ 更高，且
  `delta = on-off` 的 case-first/issue-date-cluster bootstrap 均分 CI 下界 `>-0.03`，才选择
  thinking-on；否则选择 off。若格式门禁失败则状态是“先修格式”，不是自动选择 off。
  裁决由 `scripts/analyze_thinking_ablation.py` 复算，不能手工挑口径。
- 因 σ 选择 on 只表示它更适合 GRPO 产生相对优势，不可写成“冻结推理质量更高”。训练、
  checkpoint 选择和最终评测的 thinking 设置必须一致。

## 1. 冷启动（决定下限）

- 先测裸 Qwen3-8B 在合成个例上的**首次 submit 合法率**（`cli llm` + 合成套件）；
  最终靠环境反馈自纠成功不能冒充格式通过。首次合法率 ≥95% 才可跳过通用格式冷启动，
  但仍需组内方差、challenge 和训练栈门禁；否则先 SFT：
  `cli sft --manifest data/interim/train_without_winter_or_spatial_holdout.json --out data/interim/sft_open_evidence_v04.jsonl`
  （脚本基线轨迹 → OpenAI messages tool-calling 格式，与 rollout 驱动器同构，
  `--min-reward 0.3` 过滤差数值轨迹——SFT 只教格式与流程，数值质量交给 RL）。
  禁止改回 `valid_cases_train.json`：当前 manifest 同时清除了 winter challenge 的 city-day
  和 8 个整城空间 holdout；原始 train 仍含模型选择集与 OOD 城市个例。
  命令原子写入 JSONL，并生成同名 `.manifest.json`，固化源清单/output SHA256、reward 身份、过滤数量、
  分层构成以及与 winter challenge 的 case/city-day 零重叠审计。
  1 epoch、低 LR，过拟合格式即可。
- 旧 258MB SFT 数据是无 thinking 的脚本轨迹，不得直接用于 thinking-on 格式 SFT，以免压制
  思考分布。v0.4.1 Round-2 两臂均过 95% 即取消格式 SFT；若仍失败，thinking-on 必须使用
  rejection-sampled 自蒸馏轨迹。SFT 轨迹和后续 RL rollout 的 thinking 设置不得错配。

## 2. reward 结构（已实现，勿加密集 shaping）

> **2026-09-02 起 schema 0.6.0 / reward 0.8.0**：策略只提交 PM2.5/PM10/O3 区间；AQI 等级、
> 首要污染物和 AQI≥3 过程头由 harness 从区间中点按 HJ 633 派生。下文关于"联合一致性校验"
> 的描述为历史（v0.5.3–v0.5.6）：那些约束仍然成立，但改由构造保证而非格式门禁——正式 50×8
> 探针显示它们曾占首次提交失败的 45%。分量定义与权重不变。

当前 reward 版本是 `0.7.9`，forecast schema 是 `0.5.6`。训练、probe、baseline 和 eval 产物必须同时记录
version、config SHA256，以及 `scoring.py`/`schema.py` 内容 SHA256；任一项缺失或不同时拒绝 composite 比较，
不能把“版本号相同”冒充成“评分实现相同”。比较/消融/敏感性脚本也记录自身实现 hash。

v0.7.9 继承 v0.7.3–v0.7.8 的三污染物区间、clean 弃权、daily/process 一致性、公平比较分离与
结构化引用回退，并新增等级—首污—区间联合一致性：优日首污必须为空，非优日必须给首污；
三类区间不能必然推出更高等级，被声明为首污的区间不仅必须覆盖该等级，还必须能在 IAQI
尺度压过其他区间的最低值；`process` 头不得省略，has_event 必须和 AQI≥3 的逐日头一致，
start/end 必须是一个连续污染段的边界，peak 必须是该段最高预报等级日，且所选段包含全期最高等级。
全国输出缺 PM10 或 O3
区间均为格式失败；grounding 满分需至少两条不重复事实且
覆盖两个证据类别。更重要的是，`composite` 明确作为含 grounding 的训练目标，模型毕业与
CAMS/tabular 比较统一使用只含 level/event/interval/turning/primary 的
`outcome_composite`，避免交互 agent 与非交互数值基线使用不同权重分母。旧 v0.4.x–v0.6.x
artifact 均只作历史诊断，不能与新曲线直接比较。

- 无效提交 = 0；有效但差的预报因部分分天然拿 0.15+ ——格式梯度自动存在。
- 组内信号分三层审计：总 `composite`、有效提交的 `outcome_composite`、以及隐藏真值 AQI≥4 event case 的
  level/event/turning 决策分量。只有 grounding 变化的 group 不能证明数值预报可学；正式探针要求
  后两层均有足够的有效 group，且实质变化的 population sigma `>0.001`。
- 环境内提交失败返回错误可重试（耗步数）→ RL 可学格式自纠。
- **grounding 分量（0.05，仅训练）**：evidence 引用必须对应本 episode 真实工具返回的 evidence_ref，
  用 RFC 6901 指针定位非元数据标量且 value 精确一致；仅调用同类工具而无结构化断言最多得
  0.2 型别信用。两条同类事实最多得到 0.5；伪造 ref/数值/字段/工具类型均为 0；正确重复
  不重复得分，而同一 ref/pointer 后追加矛盾值会作为无效引用进入分母，不能被去重隐藏。
  自由文本 claim 不作因果真值评分，防止规则代理冒充专家判断。真实标量是否与具体预报
  结论相关也不由 grounding 自动判定；它只能由同 seed 证据遮蔽的 outcome 剂量反应验证，
  因此 grounding 始终是低权重训练辅助且不进入模型—基线公平比较。
- high-impact event 按 AQI>=4；过程转折按 AQI>=3；无真实过程时 turning 弃权，避免重复送分。
- 全国个例激活 primary（0.10），PM2.5/PM10/O3 区间共同进入 interval；等级/事件按全 AQI。
- 训练用 interval 分量按标准 80% interval score 做连续指数衰减；超宽区间不再有保底平台。
  它是为保住组内梯度设计的有界 surrogate，不是严格 proper scoring rule；
  它不能单独支撑校准结论。毕业必须另报 80% coverage、平均宽度和标准 interval score，且不得因
  surrogate 上升掩盖任一硬校准指标退化。
- **毕业不用塑形分偷换业务指标**：`outcome_composite` 只做统一结果汇总；论文同时报告
  ordinal MAE、PM2.5/PM10/O3 区间中点 MAE/RMSE/bias、硬 CSI/POD/FAR、漏报按完整 horizon
  计罚的 start/peak/end MAE、首污集合命中率与排除并列真值日的 macro-F1，以及 80%
  coverage/width/interval score，并按起报日或天气过程块 bootstrap。

## 3. 训练器配置要点（verl/TRL）——含外部已验证配方

可行性外证：Qwen2.5-3B/7B base 即可 RL 学出多轮搜索推理（Search-R1，veRL 栈开源）；
4–8B 在真实结果 reward 上学会预测（FutureWorld）；14B 预测校准打平 o1
（Outcome-based RL to Predict the Future）。这些工作支持技术可行性，但不能替代本任务的
正式 50×8 信号、训练冒烟和独立验证。

**红线四条（历史失败的头号原因清单，按常见度排序）：**
1. **工具返回 token 必须 loss mask**（Search-R1 的 retrieved-token masking：
   环境/工具输出的 token 不进梯度，只训模型自产 token。否则会把不可由策略生成的环境文本
   当监督目标，造成错误梯度与表面收敛）。
2. **void-turn 率门禁与训练期监控**：既无合法工具调用也无提交的 rollout 在正式探针中
   必须 `<=5%`。当前 veRL 路径保留其 0 分作为“必须调用工具”的负信号；若训练中该比例上升
   或出现梯度异常，再启用 assistant mask 清零/样本隔离，不能在没有审计的情况下声称已过滤。
3. **去 σ 归一化**（Outcome-based forecasting 的关键发现）：GRPO 按组标准差归一会
   过度压制大误差、催生过度自信——对"校准数值区间"型输出尤其致命。
   用 Dr.GRPO（去 σ）或 ReMax；这是预测域特有的坑。
4. **防 lazy collapse**（GRPO 懒惰崩溃，arXiv 2512.04220）：agent 学会少调用工具走
   likelihood 捷径。我们的 grounding 分量（引证须有真实工具调用支撑）是结构性反制。

常规项：零方差组过滤；token-mean 损失；KL≈0；clip-higher；group 8–16；温度 ~1.0；
max_steps=12。超参起点抄 Search-R1：rollout batch 512、response 上限与 turn 数
渐进解锁（先短后长，配合课程）。32,768-token rollout 中单次 prompt 的 preflight 上限为
24K，累计 prompt token 只做成本核算，不能误当上下文长度。
正式 50×8 方差探针固定 `temperature=1.0`，必须与训练采样温度一致；0.6 等低温冻结评测
只能描述确定性基线，不能用于否决 GRPO 的组内信号。默认训练与正式可学性探针均为 group=8；
若本地 1.7B 因显存临时降为 group=4，则必须另用 group=4 探针通过相同门禁，不能拿 group=8 的
可用率替代。小模型训练仍仅是闭环/显存冒烟，
正式训练使用 group≥8 并同步调整 mini-batch。
训练栈使用 **veRL 原生连续 token 多轮路径**；`response_mask` 在 assistant 生成 token 上为 1，
tool/user/system 与 veRL 插入的边界 token 为 0。`audit_trajectory_mask.py` 同时做两类互补审计：
直接执行锁定 commit 的 `QwenContinuousTokenBuilder`，以及用 Qwen3 chat template 对完整轨迹
独立重构 assistant mask。二者任一失败都禁止认定 loss mask 合格。RLFactory 仅作备选。

## 3.5 认知脚手架：确定性评估层（弱模型强 harness）

8B 模型弱在数值推理与跨源综合——预报员报文里的分析套路（趋势研判、静稳/冷空气识别、
指导可信度评估、气候背景定位）由 harness 固化为 `get_assessment` 工具（`sitian/assess.py`）：
实况 24h 趋势、逐日信号旗标（判据固定可调）、指导多源分歧、气候态分位
（2022..2025-03 窗前数据，无泄漏）。**边界：harness 只算信号（客观事实），
结论留给模型学**——否则 RL 学的是抄摘要。

`get_guidance_bias` 是当前最高杠杆的新工具，优先于继续扩张 CoT：

- 按城市 × 季节 × 污染物 × lead 统计指导历史偏差，输出样本数、均值/中位误差、分位数、
  event 漏报/空报率和窗口起止；只给可审计信号，不直接给“上调 30%”结论。
- 防泄漏截止用 `verification_available_at < issue_time`，不只是 target date 早于起报日；
  这样把观测发布延迟纳入门禁。窗口、样本及输出均记录 provenance。
- 城市×季节样本不足时按预注册阶梯 shrink/fallback 到城市全季节、污染区制或全国，
  不返回高方差小样本“偏差结论”。
- 接入后先用 frozen policy 复测 event/turning。若冻结模型即抬升，主要是信息缺口；
  若不抬升，则将“信号→订正”映射作为 RL 的决策缺口。
- 主离线实验的索引**只由 city-day 清洗后的 train manifest 构建**；val/test 真值不滚动回写。
  “用较早评测真值更新较晚评测 episode”虽符合实时业务，但会给 agent 一个冻结 tabular 没有的
  在线学习优势，因此只能作为单列的 operational-online 扩展实验。

v2 备选：evidence 升级为结构化断言（type+日期+字段+方向），claim 与数据的一致性
即可规则校验（要点进奖励函数的严谨路径）。

## 4. 课程（三段）

1. 合成剧本（格式 + 基本决策；clean → accumulation → guidance_misleading）。
2. 全国池 clean/o3/event 混合（指导基本可信的时段先学"会用指导"）。
3. 全分层（switch/pm10_primary/bust 比例拉满，学"何时不信指导"）。

## 4.5 防"假学"评测阶梯（fundamental 贡献的证明责任）

模型学"信号→决策"映射是真技能；假学 = 抄指导 / 抄气候态 / 抄持续性。三道防线：

1. **完整基线阶梯**：persistence / 各指导源直译 / climatology（当月 p50，`ClimatologyAgent`）
   / **oracle-switch**（逐 case 事后取上述三份完整预报中最好者，不可实现的参考）。
   超过它说明模型在该评分上好于三个基线的事后整份择优，但它不是任意数值融合策略的数学上限。
   冒烟已见必要性：北京 12-19 上
   climatology 0.794 > guidance 0.629，无此基线"抄 p50"会被误判为技能。
2. **行为归因**（`sitian/diagnostics.py::strategy_attribution`）：每份提交与各简单策略的
   MAE 距离 + nearest 标签，训练全程监控分布迁移（健康轨迹：早期贴 guidance，
   学成后在 bust 层脱离）。不读 truth，纯行为画像。
3. **OOD 三轴**（留城市/留区制/留时段）：当前已预注册“8 个污染型聚类各留 1 个完整城市”，
   联合清洗后的 train 为 3,061 case；OOD val/test 为 50/178 case、32/46 个起报日块。
   OOD test 不进 checkpoint 选择。留区制与真正前瞻时段仍由后续泛化章节完成。

## 4.6 会商机制与 GRPO（叙事框架 + 集合预报层）

形式对应：会商同场竞技考核 ≈ GRPO 同 case 组内相对优势（组=同一起报日，与业务考核
的公平性原则同构）。断裂处：GRPO 无先验多样性、无交流、无合成——只是会商的
"考核学习"半边。补齐"集体决策"半边的三层落法：

1. 叙事框架（论文动机）：相对优势学习 = 会商考核的计算形式化。
2. **自我会商合成**（`sitian/ensemble.py::aggregate_forecasts`）：推断期 k 份采样 →
   等级多数票 / 区间逐端点中位 / 过程投票 / medoid 打底。= 气象学集合预报范式；
   评测表加"单人 vs 自我会商"列。
3. **spread-skill 分析**（`ensemble_spread`）：集合离散度 × 事后真实误差的相关 +
   RL 前后离散度变化——"GRPO 熵坍缩是否摧毁政策的会商价值"，嫁接 RL 熵坍缩与
   气象 spread-skill 两个成熟概念的新问题，纯评测零训练风险。

多智能体辩论式会商（互看草稿再修正的 RL）为 future work，不做核心依赖。

## 4.7 订正轨迹与证据消融审计

**结构化订正记录尚未实现，属于 P2 诊断增强，不是当前开训门禁。** 若实现，只进 diagnostics，
reward 权重恒为 0；届时 `submit_forecast` 才增加可选 `adjustment` 块，按日记录
`target / anchor_source / anchor_value / direction / delta / final_value / drivers[]`。
当前 forecast 没有连续 AQI 中心值，因此拟议 v1 的 `target` 只取
`pm25_range_midpoint` 或 `o3_range_midpoint`，`final_value` 必须等于已提交区间中点；不为了做诊断
偷加一个不进评分的 AQI 数值。规则只校验：

1. `final_value = anchor_value + delta`（容差固定）；
2. `direction` 与 delta 符号一致；
3. `drivers` 必须是本 episode 真实调用过的工具/返回字段。

“调整方向与自由文本证据解释是否一致”只做盲化人工审计，不进 reward，防止密集 shaping 和自循环。
它是未来 schema 扩展；合并时必须升 schema 版本并重跑首次提交合法率门禁。

**证据消融审计已有可执行入口。** `probe_model.py --ablate-evidence` 可重复遮蔽
`synoptic / composition / fires / source_context`；copy-on-write 后由 process view 重新派生，
不修改 case 文件。full/masked 使用完全相同的 case、rep、rollout seed、模型、温度、工具预算和
reward identity；`compare_evidence_ablation.py` 先在 case 内平均，再按 issue_date 分块 bootstrap，
主估计只用不含 grounding 的 `outcome_composite`。示例：

```bash
python3 scripts/probe_model.py --split val --n 50 --group-size 4 --thinking \
  --out data/interim/probe_full.json
python3 scripts/probe_model.py --split val --n 50 --group-size 4 --thinking \
  --ablate-evidence synoptic --out data/interim/probe_without_synoptic.json
PYTHONPATH=src python3 scripts/compare_evidence_ablation.py \
  --full data/interim/probe_full.json --masked data/interim/probe_without_synoptic.json \
  --out data/interim/compare_evidence_synoptic.json
```

20 case 只用于发现异常；正式剂量反应至少 50 case 且需 ≥30 个独立起报日 cluster。冻结模型与
RL checkpoint 各跑一次，报告结果差、分层差和工具使用迁移。这是输入干预对输出的因果敏感度，
不能用 CoT 文本代替；guidance/observations 的遮蔽仍待专用、等工具面的实现以避免改变任务定义。

## 5. 训练中评测与诊断

- checkpoint 双轨选择：固定暖季 val 子集（分层抽 ~200）每 N 步跑
  temperature=0，同时在从 train 整起报日划出、city-day 零重叠的 winter challenge
  上监控 event/turning/switch。暖季 val 提升但冬季 challenge 明显退化的 checkpoint 不毕业。
- 暖季 val 内单列 8 个未见城市的 `spatial_ood_val` 作为选择诊断；最终空间结论只看从未参与
  训练/选择的 `spatial_ood_test`。它有 178 case 和 46 个起报日块，满足分块推断预算，但没有
  冬季 event/turning，因此不得外推成重污染空间泛化结论。
- 每次报 composite + macro-stratum + 分层分 + 格式通过率 + 平均步数；
  正式比较先在 case 内平均 rollout，再按起报日/过程块配对 bootstrap。
- 行为诊断（从 transcript 算，全规则）：guidance-copy 率（提交值与 CAMS 直译的相关）、
  persistence-copy 率、工具调用覆盖率、重复调用率。回答"policy 到底学会了什么"。
- 泄漏哨兵：官方未披露可审计的预训练 cutoff，不把“发布后日期”当作已证明的干净集；
  用记忆探针和真正前瞻冬季 holdout 做主证据，同时
  留意 val（2026-05/06）与 train 的 reward 差距变化。

### 执行顺序（当前 v0.7.9 合同）

1. 完成高频 open-evidence 合同、7,548 case 重挂接和专家证据契约 100% 审计；
2. 在 reward-v0.7.9 上重训同信息集 tabular，并跑 50 case×group8 frozen probe；
3. 同时门禁首次合法率、两类结构化引用、process 使用、有效 group 率和单次 prompt ≤24K；
4. 通过后做 trainable 1.7B/4B LoRA 50–200 step 闭环冒烟；失败若仅是格式/引用再做小型
   rejection-sampled tutor SFT，不把 SFT 预设为必需；
5. 小模型 RL 后先做同 policy frozen→RL、CAMS、tabular 与关键证据遮蔽比较；达到毕业条件才转 8B；
6. 最后一次性跑商业模型上界和完整 agency/证据剂量消融，避免合同变化造成返工。

## 5.5 开训前提与训后毕业分开

- **preflight**：数据/reward/专家证据合同完整性、格式通过率、两类可核验 grounding、
  单次 prompt 预算、过滤零方差后的有效采样率、trainable stack + 50–200 step 硬件冒烟。
  不要求冻结裸模型先超 CAMS。
- **graduation**：训后独立 val 对 CAMS 的分块 CI 下界 >0；≥100 个隐藏真值 AQI≥4 event case 上同样超 CAMS；
  winter challenge 不退化；论文阶段还要求同一隐藏事件层相对强 tabular 的 composite、硬 CSI 与
  intention-to-treat 峰值时序 MAE 三项分块 CI 下界均为正，不能用“事件超 CAMS + 总体超 tabular”拼接替代；
  另要求总体超强 tabular、默认/等权/单分量 ±25% 权重网格的
  分块 CI 下界均为正、在 ≥100-case/≥30 起报日的整城空间 OOD test 上超过同信息集 tabular
  并报告硬指标、遮蔽 synoptic+composition 后出现正的证据剂量反应，并有 2026–27 冬季前瞻证据。

## 5.6 AWQ 与硬件红线

- 本地 8B-AWQ 只做冻结评测和数据/方差探针。训练期 rollout 必须来自权重已同步的
  current trainable policy；不允许用 AWQ 采样、更新另一份 BF16 策略。
- 本机 RTX 4090 24 GB / WSL2 只承担 1.7B/4B LoRA 冒烟。8B 长训目标为云上
  ≥80 GB 级 A100/H100，优先 ≥2×80 GB。
- `training_smoke.json` 必须证明步数达标、policy sync、工具 token loss mask、
  finite loss 与 checkpoint reload；单纯“脚本能启动”不算通过。

**2026-09-03 起"栈冒烟"与"学习冒烟"分开**：栈冒烟只验证 policy sync、工具 token mask、
finite loss、checkpoint reload 四项机制，`SITIAN_SMOKE_STEPS=6`（finalizer 同步把
`--min-smoke-steps` 传给 preflight）；在 32K 上下文、12 轮工具循环下 1.7B 每个生成批次在 4090 上
要 1–3 小时，50 步已不是冒烟量级。50–200 步的学习冒烟与 8B 长训在云上 ≥80 GB 卡完成：
`SITIAN_LAYERED_SUMMON=True SITIAN_ROLLOUT_GPU_UTIL=0.6 SITIAN_ROLLOUT_MAX_NUM_SEQS=16
SITIAN_AGENT_WORKERS=8 FH_TRAINABLE_MODEL=<Qwen3-8B BF16 snapshot> SITIAN_SMOKE_STEPS=100
bash scripts/run_verl_smoke.sh`，改脚本或旋钮后先跑 config-only 与 `audit_verl_runtime_contract.py`。

## 5.7 可执行 veRL 闭环（实现状态）

当前不是“未来接口草图”，而是以下可执行合同：

- `scripts/prepare_verl_dataset.py` 从 winter city-day + spatial-city 联合 purged train、独立 val 和 winter challenge manifest
  生成 train/val/challenge、二者合并的 `selection.parquet` 与分层 smoke Parquet；训练池与
  winter challenge 的 case 和 `(city, forecast_date)` 重叠必须同时为 0。长训 checkpoint
  选择读取 `selection.parquet`，并按其中保留的 split 分别报告暖季 val 与冬季事件能力；
  manifest 还分别聚合 hash `case.json`、全部 agent 可见文件和 scorer-only `truth.json`，训练
  冒烟结束后重算核对，防止同一路径的 evidence attachment 被静默替换；全国目录出现
  `expert.json` 会直接失败；
  每行显式写 `agent_name=sitian_tool_agent`。JJJ_ATMO 文本、truth 和 expert 不进入 prompt 或
  `ground_truth`；case 路径只在模型不可见的 `tools_kwargs.create_kwargs` 中传给环境。
- `src/sitian/integrations/verl_bridge.py` 只读取 assistant 的 tool calls，忽略轨迹里可伪造的
  tool-result 文本，然后从隐藏 case bundle 确定性重放 `ForecastEnv`。因此跨轮 evidence registry、
  `ref/tool/语义分区/field/value` 结构化引用核验和 12 步预算与冻结探针一致；自由文本
  `claim` 不宣称被机器理解，其作用由同 seed 证据遮蔽和结果硬指标验证。
- `src/sitian/integrations/verl_runtime.py` 在 veRL 原生 token-in/token-out `ToolAgentLoop` 上增加
  首次合法 submit/预算耗尽终止语义；环境 reward 直接写入 `AgentLoopOutput.reward_score`，
  同时输出 `reward_extra_info.outcome_composite` 供 DAPO 在采样阶段丢弃并补采数值结果全同的组；
  被保留组仍优化含 grounding 的总 reward，避免 grounding-only 方差冒充数值预报可学性。
- `scripts/audit_verl_runtime_contract.py` 在固定 upstream commit 上读取 Hydra 完整解析配置，
  检查 Dr.GRPO、动态组过滤、LoRA-only checkpoint、同一 trainable policy 的异步 vLLM rollout，
  并真实执行全国主实验当前 10 个 native tool 的 create/execute/release 生命周期与两步轨迹重放；
  上一版预报和历史 analog 只在当前 case 确实可用时条件注册，避免死工具浪费 step/context；训练 Parquet
  manifest 固定该审计及其配置输入 hash，配置漂移会 fail closed。
- `scripts/run_verl_smoke.sh` 使用 Qwen3-1.7B BF16 base + PEFT LoRA rank 32、vLLM rollout、
  FSDP、Dr.GRPO (`norm_adv_by_std_in_grpo=False`) 和单卡共置权重同步。训练 50 步保存
  LoRA-only checkpoint，再从第 50 步真实恢复并跑到 51；`audit_training_smoke.py` 对日志、
  adapter hash、finite loss、policy sync 与 veRL 原生 response mask 逐项 fail closed。
- `scripts/finalize_open_evidence.py` 只在数据、reward 身份、正式 50×8 格式/grounding、
  prompt 预算与有效组率均通过后停止冻结 AWQ 服务并启动上述闭环；未通过时不会用一次
  无梯度或全零的“启动成功”冒充训练冒烟。

训练栈固定为 veRL commit `c2429f29a25d573f63d9bcc29e7ceb690817dce9`，独立 uv 环境避免
污染 harness 的 CPU 分析依赖。采用该实现的依据是 veRL 官方已明确支持异步 agent loop、
自定义 stateful tool、FSDP/PEFT LoRA 与 vLLM 权重同步：
[Agentic RL](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)、
[How to Extend veRL](https://verl.readthedocs.io/en/latest/extend_guide.html)、
[LoRA Support](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)。

## 6. 已知反 hacking 设计（scoring 既有）

宽区间衰减 + 覆盖保底、事件 CSI 防多数类、事件日 2× 加权、格式门禁 composite=0、
grounding 防证据伪造并要求证据类别广度；训练分与公平比较分隔离，避免工具行为分量抬高
对 tabular 的表观增益。新 hack 面出现时优先改 `RewardConfig` 配置并升 reward 版本。
