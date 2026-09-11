# Harness / Policy / Reward 责任切分（2026-09-02 评估）

> 目的：在 v2 高频证据挂接完成后，按预报第一性原理复核 harness 与 reward 的分工，
> 找出会降低 agent RL 成功率的结构性缺口，并给出不改动现有 reward 身份的推进顺序。
> 现状事实来自当日实测（工具返回、探针产物、finalizer 日志），不是推测。

## 1. 预报第一性原理 → 谁负责什么

城市逐日 AQI（D+1…D+5）的物理分解：

```
C(D+k) ≈ 初态(持续性衰减)
       + 扩散条件增量(边界层/风/湿度/降水/稳定度)
       + 输送(上游浓度 × 来流对齐 × 到达时间)
       + 化学(O3: 辐射/Tmax/NO2；二次PM: 高湿静稳)
       + 模式指导(CAMS/CMAQ/NAQP) − 已知偏差
不确定性随时效增大；D+5 无 CAMS 指导，只剩 GFS/IFS 形势。
```

| 角色 | 负责 | 不负责 | 现状 |
|---|---|---|---|
| **数据/Harness** | 起报时刻门禁下的证据；把上式每一项算成**确定性信号**（趋势、通风旗标、来流候选、指导偏差分位）；单位/覆盖/时间语义可审计；紧凑视图控制在 8B 上下文内 | 给结论、给"上调 30%"式订正 | 已实现 10 个工具；process view ≈18K 字符、单次 prompt ≤24K |
| **Policy（被训）** | 4 个真正的决策自由度：①锚定选择（信指导/信持续/订正）②事件日与过程转折定位 ③区间宽度（校准）④首要污染物 | 复述工具输出、编造证据 | turning 头已由 daily 联合约束，自由度收敛到 daily 三头 |
| **Reward** | 只对观测真值负责的结果分量；grounding 低权重、只核验事实不核验因果 | 密集塑形、LLM judge | v0.7.9，23 条防投机不变量 PASS |
| **基线阶梯** | 定义"学会"：persistence / CAMS / climatology / oracle-switch / 同信息集 tabular | — | 旧 reward-v0.4：val CAMS 0.740、oracle-switch 0.795、tabular 0.753、冻结 8B 0.617–0.633。**reward-v0.7.9 同信息集 tabular（2026-09-02，v2 证据，10,222 维，3,061 train）**：val 0.681 / winter 0.555 / test 0.754 / 空间 OOD test 0.676；event 分量 val 0.152、winter 0.080、OOD test 0.0 |
| **训练器** | Dr.GRPO、工具 token mask、零方差过滤、void-turn 门禁、checkpoint 重载 | — | veRL 合同 PASS，50 步冒烟未跑 |

结论：分工框架是对的，问题不在"缺工具"，而在**信号→决策的落差对 8B 太大**，以及
**D+5 证据不对称**。下面按对成功率的影响排序。

## 2. 结构性缺口（按影响排序）

### G1（P0）D+5 是"半盲日"：harness 只给 D+1…D+4 的逐日扩散诊断与指导锚点
- 实测 `get_process_evidence`：`daily_surface_dispersion` 与 `pollution_guidance_anchor`
  只有 4 天；D+5 仅剩形势轨迹上的 +120h 一个点（summary 视图 24h 抽稀，+132h 只在 full）。
  `get_assessment.daily_signals` 同样只有 4 天。
- 第一性原理上 D+5 仍可由 GFS/IFS 派生地形自适应低层风、RH、逆温、omega、辐射，
  只是没有 BLH/降水/CAMS。现在是让 8B 自己从 12h 轨迹外推，这是 turning/level 误差的集中点。
- **建议**：harness 为每个预报日生成同构的 `daily_signals` 行，D+5 标 `source=nwp_levels_only`，
  缺项显式 `null`；tabular 投影同步。不改 reward，不引入结论。

### G2（P0）事件分量在冻结策略上零梯度（强 tabular 同样塌陷）
- reward-v0.7.9 tabular 的 event 分量 val 0.152、winter 0.080、空间 OOD test 0.000：
  在"clean 正确否定弃权 + 近失误 0.25"的口径下，连梯度提升树也几乎不报 AQI≥4。
  这说明 event 层是该任务真正的难点，也是 agent 相对 tabular 最可能拿到正增值的层；
  但前提是策略先有"事件暴露"。
- v0.4.1 Round-2 诊断：10 个 event case × 39 条有效 rollout，`event_component_positive_rollouts = 0`，
  event σ = 0。策略从不报 AQI≥4，GRPO 在 event 分量上没有相对优势可学；
  只有 level 的 off-by-1 与 near-miss（0.25）在供梯度。
- **建议**（不改 reward）：正式 50×8 探针新增"事件暴露率"读数——event case 中至少一条 rollout
  报出 ≥4 的组占比。若 <30%，用 rejection-sampled 自蒸馏（只取自身采样中含 ≥4 且 outcome 高于组均值
  的轨迹）做 ≤1 epoch tutor，而不是用脚本基线轨迹 SFT。这是训练设计里已预留的路径，把触发条件量化。

### G3（P0）冻结策略输给 CAMS 的是"校准技能"而非"预报技能"
- 2026-08-28 探针分量：interval 0.357、primary 0.39、turning 0；level 0.718、event 0.833 并不差。
- `get_guidance_bias` 已含误差分位，信息足够；但策略要先学会"把分位变成区间"。
- **建议**：课程第一阶段用**单轮决策任务**（见 §3），让 RL 的信用分配只落在 daily 三头上；
  interval 权重 0.15 的稠密信号在单轮下最容易先学会。

### G4（P1）clean 组零方差的一部分来自区间免罚宽度是绝对值
- `width_free=30 µg/m³` 对 PM2.5≈15 的清洁日等于无约束（任意 [0,30] 都满分）；
  对 150–250 的事件日又过紧。v0.4.1 诊断 11/50 零方差组、10 组仅 interval 变化。
- **建议**：留到 reward 下一次升版时评估相对免罚宽度（如 `max(30, 0.4×指导均值)`），
  由 `audit_reward_weight_sensitivity.py` 预注册网格裁决；本轮不动 v0.7.9。

### G5（P1）turning 与 level 在过程 case 上是同一决策的两份分数
- schema v0.5.5 起 start/peak/end 由 daily 的 AQI≥3 连续段推出，turning 分量的自由度基本
  被 daily 吸收。作为塑形无害，但论文中不能把 turning 当独立技能证据；硬指标用 start/peak/end MAE。

### G7（P0，2026-09-02 探针实测新增）上下文预算没有驱动器层防线
- v2 证据下单个工具返回更大（process view ≈18K 字符、guidance bias ≈13K、synoptic/pollution 各 ≈9–10K），
  50×8 探针第 3 条 rollout 即触发 vLLM `HTTP 400: maximum context length is 32768`——
  `agents/llm_openai.py` 不截断工具返回，也不在临近上限时强制提交；该 rollout 记 0 分，
  会以"格式失败"的面目拉低首次合法率与有效组率。veRL 路径有右截断，探针路径没有，两者不同构。
- **根因实测（Qwen3 tokenizer，30 个 val case 中位数）**：v2 挂接后一次 `get_process_evidence`
  = 21.6K token、`get_pollution_evidence` = 25.8K、`get_synoptic_evidence` = 13.2K、
  `get_guidance_bias` = 5.4K；八个工具各调一次共 68K token，前 33 条 rollout 有 4 条溢出。
  数据契约没有错，错在**工具视图直接把 20 个快照 × 全字段、64 城候选池原样吐给策略**。
- **实测（压缩后 50×8）**：0/400 溢出；单次 prompt 中位 13.0K、p90 19.5K，仅 1 条 32.4K
  （策略过度取证），24K 门禁按最大值判会因这 1 条失败——应改为按 p95 或按"溢出即 0 分"口径。
- **已做（2026-09-02，只改视图层，builder/tabular/audit 不变）**：`compact_process_view`
  把每城常量（地形自适应层）提到头部、轨迹点改短键元组、6h 中间点只保留输送/通风核心、
  系统位置只在逐日点给 GFS 低/高压、来流情景取最多 2 个主扇区；synoptic 摘要 12h 骨架
  + 元组化系统信号 + cross_model 列表化；pollution 摘要候选池压成三污染物 [latest, change_6h]
  并限 8 城、气溶胶只留 3 个快照；guidance-bias 分位/事件率列表化并提升共享定义。
  结果：process 7.4K / synoptic 5.2K / pollution 4.0K / bias 3.5K token；四大工具各调一次
  + 固定提示 3.8K = **24.0K**，刚好贴着 24K 单次 prompt 门禁；veRL 侧 24,576 响应预算
  在"四大工具全调 + thinking"时仍偏紧。残余建议：①veRL 冒烟 `max_model_len` 28,672→32,768
  与探针对齐；②探针记录每条 rollout 的 `max_prompt_tokens` 分布，把"策略过度取证"与
  "harness 视图过大"分开报告。

### G8（P0，2026-09-02 正式 50×8 探针实测新增）联合一致性校验成了首次提交的主要死因
- 压缩视图后的正式探针（val 50 case × 8，T=1.0，thinking on）：提交率 90.3%、**首次合法率 81.5%**
  （门禁 95%）、无效提交 181 次、void-turn rollout 15%（门禁 5%）、0 条上下文溢出；
  零方差组 0/50，outcome 可变组 58%，单次 prompt 中位 13.0K / p90 19.5K / 最大 32.4K。
- 181 次无效提交的家族：首污 vs 三区间 151、process 起止 vs daily 连续段 110、
  区间必然推出更高等级 40、has_event vs daily 22、evidence 字段 16。
  这些都是 schema v0.5.3–v0.5.6 为堵"各输出头单独得分"而加的**联合约束**；它们在 reward
  层是正确的，但让策略在一次生成里同时满足四类隐式代数约束，对 8B 是格式税而不是预报能力。
- **建议（责任切分：一致性由 harness 派生，而非由策略"猜对"）**：
  ①`process` 头改为 harness 从 daily 派生（start/end/peak 由 AQI≥3 连续段与最高等级日确定），
  策略不再提交它——这与 G5 一致，turning 评分口径不变；
  ②首污与等级改为"从三区间派生的可行集合校验"：区间是唯一自由变量，等级/首污只需落在区间
  的 IAQI 可行域内，校验器把不可行的组合返回为**可执行的修正提示**（给出可行等级区间），
  而不是只报错；③evidence 字段的类型别名已归一，剩余 16 次是 `value` 非标量，提示中给出例子。
  这三项都是 schema 升版（须重跑首次合法率门禁），不动 reward 分量与权重。
- **G2 已被正式探针确认**：10 个真值 AQI≥4 的组、69 条有效 rollout，**0 个组有任何一条报出 ≥4**，
  event 分量均值 0.007。事件暴露率 = 0/10，触发 rejection-sampled tutor 的条件已满足，
  但自身采样中没有 ≥4 的轨迹可取——需要先用"事件日 near-miss 信用 + level 事件日 2× 权重"
  跑 RL 让暴露率脱离 0，或用真值侧筛选的事件 case 做极小规模格式示范（不进 reward）。

### G9（P0，reward-0.8.0 探针实测新增）void-turn 的根因是 tool_call 写进了 <think> 块
- schema 0.6.0 探针：无效提交 181→37、首次合法 86.3%、context/sampling 门禁 PASS、事件暴露 2/10 组；
  但 void-turn rollout 36.8%、提交率仍 90.3%。复现（南昌_2026-05-31）显示：冻结 Qwen3-8B 在
  thinking 模式下常把 `<tool_call>{...}</tool_call>` 直接写在 `<think>` 内，vLLM 的 hermes 解析器
  只在顶层提取，于是整轮成为"纯文本"void turn，驱动器反复催促直到 16 次调用耗尽。
- 这是**探针驱动器与训练运行时不同构**：veRL 的 HermesToolParser 用正则在整段响应文本里提取
  `<tool_call>`，不区分是否在 think 内。修法：`agents/llm_openai.py` 在响应无 tool_calls 时按同一
  规则从文本恢复工具调用（JSON 解析失败仍算 void），并记录恢复次数供审计。
- 剩余无效提交 37 次里 17 次是 `evidence.value` 传了数组，属策略行为，保留为 0 分负信号。
- **恢复器上线后复测（18:02Z）**：提交率 93.3%、首次合法 89.0%、无效提交 21、void rollout 31.3%，
  400 条里恢复次数为 0——复现 8 个种子显示 think 内的 `<tool_call>` **JSON 本身残缺**
  （括号不配对），veRL 的 hermes 解析同样会判为 void。结论：这是 thinking 模式下冻结 Qwen3-8B
  的行为缺陷，不是 harness 缺口；对 CAMS −0.164 [−0.250, −0.075]、对 tabular −0.216。
- **用户裁定（2026-09-02）：thinking-on 是硬约束**（可解释性），thinking-off 臂不测、已停。
- **根因再定位**：残缺 JSON 全部断在同一位置（char 203，逐日对象数组的第一项末尾，
  `]], [` 代替 `]}, {`）——冻结 8B 复制 forecast_example 的嵌套"对象数组套区间数组"结构时
  系统性错括号。这是**提交载荷的嵌套深度**问题，harness 可以负责。
- **已做（schema 0.6.1）**：`daily` 接受紧凑区间表 `{"pm25_range": [[lo,hi]×H], "pm10_range": …,
  "o3_range": …}`，日期由位置隐含；tool schema 与简报示例改为该形式（示例逐日数字略有不同，
  避免机械复制），逐日对象形式仍被接受（基线/tabular 不变）；顺带修掉 tool schema 里
  残留的 `process` 必填项；vLLM 加 `--reasoning-parser qwen3`。
  同 3 个高 void case × 8 seeds 的 A/B：void rollout 8/24→3/24，首次合法 22/24，
  提交 22/24；驱动器 tool_call 恢复保留（与 veRL hermes 同构）。

### G10（P0，schema 0.6.1 探针实测新增）简报里的数值示例被冻结策略整段照抄
- 0.6.1 探针（thinking-on，紧凑表）：提交 97.8%、**首次合法 95.5%（过门禁）**、void 13.5%、
  grounding full-credit 64.5%（过门禁）、0.6.1 tabular val 0.678 不变；但 outcome 可变组 44%
  （门禁 50%）、interval 0.446、primary 0.092，对 CAMS −0.157。
- 复核提交内容：**91% 的有效提交把 forecast_example 的占位区间原样照抄**（0.6.0 逐日对象
  示例是 74%）。冻结 8B 的"预报"大部分是示例模板；组内方差只来自不照抄的少数 rollout。
  这意味着此前所有冻结探针的 interval/level 读数都被示例锚定污染，而 RL 起步时的相对优势
  也会被同一模板吞掉。
- **修法（harness 不得喂"答案形状"的模板）**：简报不再给数值示例，只给结构说明
  （键名、长度 H、[下限, 上限]），数字必须由策略从证据得出；tool JSON schema 已含 minItems/
  maxItems/类型约束保证格式。用 3 个 case × 8 seeds 复测首次合法率与组内方差后再跑正式探针。
- **复测链（同 3 case × 8 seeds，thinking-on）**：
  ① 去示例 + 区间表：预报全部不同、组内 σ 0.21–0.34，但提交 17/24、void 11/24——冻结 8B 在
  嵌套 JSON 末尾错括号（`}]` 代替 `}}}`）；
  ② **schema 0.6.2 扁平列式**（forecast 顶层 pm25_lo/pm25_hi/…，各 H 个数字，深度 2）：
  提交 24/24、首次合法 22/24、void 2/24、预报 24/24 互不相同、组内 σ 0.08–0.20、均分 0.498
  且零照抄。结论：对 8B 而言"嵌套深度"与"数值示例"是两个独立的格式税，都由 harness 消除。
  正式 50×8 探针以 0.6.2 重跑。

### G11（决策，2026-09-03）0.6.2 正式探针后的两项门禁口径
- 0.6.2 正式 50×8（无数值示例、扁平列式、thinking-on）：提交 96.8%、首次合法 91.0%、void 15.5%、
  组内 σ 0.118、outcome/event 可变组 100%、事件暴露 6/10、grounding 70%、379/387 份预报互不相同；
  对 CAMS −0.099 [−0.174, −0.026]（此前 −0.157），对 tabular −0.151。冻结模型第一次给出"自己的"预报。
- 剩余无效提交 39 次里 22 次是 lo>hi（列式填写把上下限写反）：区间是无序的一对边界，
  schema 0.6.3 直接按序接受，不再拒绝。evidence.value 非标量 9 次保留为负信号。
- void 15.5%：全部是 thinking 内残缺 JSON（veRL 同判 void，训练期记 0 分负信号）。原 5% 门禁
  来自 SimpleTIR 对"无有效动作轨迹"的过滤建议；在 thinking-on 硬约束下把探针门禁定为 ≤20%，
  训练期继续监控且零方差组过滤不变——这是把"格式税"交给 RL 的负梯度，而不是继续改 harness。
- 上下文：p95 25.5K、32/400 超 24K。策略在无示例后调用更多工具，属合理探索；把探针显式预算
  与 veRL 冒烟统一到 32,768（响应 28,672 + 提示 4,096），门禁按 p95 ≤ 28K。

### G12（2026-09-03）0.6.3 探针：只剩首次合法率 93.75%，且失败全是"引用元组事实"
- 0.6.3 正式 50×8：提交 98.0%、首次合法 93.75%、void 13.25%（过 20%）、context p95 25.5K（过）、
  sampling/grounding/data 全过、事件暴露 8/10；对 CAMS −0.096 [−0.170, −0.023]。
- 23 次无效提交几乎全是 `evidence.value` 传了列表：压缩视图把大量事实编码成元组
  （`ll: [风速, 风向, 离散度, RH, ω, 逆温…]`），策略引用整个元组是自然且可核验的。
  schema 0.6.4 / reward 0.8.1：value 允许为 ≤8 项的标量列表，逐元素精确核验，部分匹配或含缺测不算事实。

### G13（2026-09-03）0.6.4 探针：策略侧五项门禁首次全部达标
- 0.6.4 / reward 0.8.1 正式 50×8（thinking-on）：提交 97.3%、**首次合法 95.0%**、void 15.3%、
  p95 prompt 25.4K、组内 σ 0.118、outcome/event 可变组 100%、grounding 70%、事件暴露 6/10；
  对 CAMS −0.102 [−0.176, −0.030]，对 tabular −0.154，均分（有效提交）0.566。
- 与 8 月 28 日的起点相比：首次合法率 100%（照抄示例）→ 真实 95%；事件暴露 0 → 6/10；
  预报互不相同 379/387；这才是 GRPO 可用的起跑线。剩余唯一门禁是可训练栈的 50 步冒烟。
- 技术性修正：上下文溢出（RuntimeError）行没有逐轮审计字段曾让 format 门禁判 False；
  探针 error 行现在带完整字段，preflight 把 error 行视为已审计的零分终止。

### G6（P2）已知未接：previous_forecast / analog 全国池均未注册；专家 evidence 分量全国弃权
- 不阻塞开训；接入前不得改基线投影。

## 3. 最高成功率的推进顺序（不改 reward 身份）

1. **数据（今日）**：v2 派生 → 覆盖门禁 → 7,548 case 重挂接 → 审计链 → tabular v0.7.9
   → 50×8 冻结探针 → preflight → 1.7B LoRA 50 步冒烟。全部由
   `scripts/finalize_open_evidence.py` 串行执行（已改为可断点续跑）。
2. **阶段 A：单轮决策 RL（主路径，不是消融）**：harness 预取 process view + assessment
   + guidance bias（与 1b 臂同一冻结 snapshot、同一字节序），策略只输出预报对象。
   去掉工具选择熵、void-turn 与首次提交格式失败三类主要死法；group 8、T=1.0、Dr.GRPO。
   毕业条件：val 对 CAMS 分块 CI 下界 >0。
3. **阶段 B：释放多轮工具选择**：在 A 的 checkpoint 上继续，同样 reward；
   agency 增益 = B vs A（同信息集），这正是论文预注册的 2 vs 1b 比较。
4. **阶段 C：证据遮蔽剂量反应 + OOD + 冬季前瞻**。

责任边界不变：harness 算信号、policy 做决策、reward 只认真值、基线定义"学会"。

## 4. 本日修复记录（数据侧）
- `derive_open_composition.py`：`_find_cycle_file` 引用未导入的 `Dataset`（NameError），
  改为 `_open_netcdf`；这是昨晚 finalizer 中断点。
- `derive_open_fires.py` / `derive_open_composition.py`：新增 `--skip-existing`；
  `finalize_open_evidence.py` 对 synoptic/fires/composition 传该参数，重跑不再重复 42 分钟形势派生。
- `finalize_open_evidence.py::_deliver_report`：外部 HTML 打包插件路径（0.2.9）已不存在，
  改为缺失时只产出 artifact.json/appendix.md 并告警，不再在训练冒烟前中止整条链。
- `finalize_open_evidence.py`：静态派生改用 Tethys 解释器（cfgrib）；新增 `--resume-from-probe`
  （数据/tabular 已在同一契约下通过时，直接从 50×8 探针起跑）。
- **G8+G1 已实施（schema 0.6.0 / reward 0.8.0）**：`validate_forecast` 从区间中点按 HJ 633 派生
  等级、首污与过程头（`derive_process`），提交的分类头只做类型检查；submit 工具返回
  `derived_heads` 与 `normalized_forecast`，探针/硬指标读取归一化对象；tool schema 与示例
  去掉 aqi_level/primary_pollutant/process；`daily_surface_dispersion` 为无 CAMS 诊断的预报日
  （D+5）生成 `nwp_levels_only` 行，`pollution_guidance_anchor` 标出 `days_without_guidance`；
  探针摘要新增 `event_exposure`（G2 读数）；preflight 上下文门禁改按 p95 并报告超预算条数（G7）。
  reward 不变量审计改为派生不变量（oracle 区间→真值等级、矛盾分类头不能改分、纯区间提交完整）。
- 视图层压缩（见 G7）：`process_evidence.compact_process_view`、env 的 synoptic/pollution/
  guidance-bias 摘要；`citation_examples` 8→6。199 个测试通过。
