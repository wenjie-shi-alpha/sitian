# 首轮 RL 预报技能试验

目标是检验 RL 是否改善同一个污染预报 Agent 的证据利用和最终预报。Agent 学习在给定
预报时点获取多源资料、综合判断并提交结构化预报；数值指导和统计订正既可以是资料，
也可以作为业务参照。不能仅凭“预报员定位”推断 Agent 优于这些方法。

## 已固定的实验

- 模型：原始 ModelScope Qwen/Qwen3-8B BF16，重新初始化 rank 32 LoRA，不载入工程 smoke 权重。
- 训练：完整 3,061 案例池，GRPO，50 个 optimizer steps，2 prompts/step、8 rollouts/prompt，
  DAPO 按 outcome_composite 过滤无差异组。50 步不是完整一个 epoch，约 100 个保留 prompt；
  过滤补采的 prompt 数可能更多。
- 评测：训前 step 0、第 25 步、第 50 步使用同一个原生 veRL 工具循环；同模型架构、
  32K 总上下文、12 轮上限、thinking 开启、temperature=1、每案例 1 次、相同案例种子。
  相同种子降低抽样差异，不保证不同执行调度和权重下逐 token 可重复。
- 案例：分层抽取 48 个暖季 val 和 16 个 winter challenge，64 个案例涉及 14 个 issue-week
  块。这个平衡面板不代表真实污染频率；test、留城 test 保持封存。
- 主要终点是第 50 步；第 25 步仅看趋势，不按同一面板挑最优 checkpoint。
- 最长运行 12 小时；异常退出记录状态，不自动增加步数或重新启动。

机器可读的案例清单、种子、输入与代码 SHA256、reward 版本和指标定义在
`data/experiments/rl_skill_pilot_20260909/protocol.json`。本轮新增遥测只进入 rollout 元数据，
不进入模型提示；训练 reward 不变。云端新 runtime contract 17 项检查通过。

## 如何判断增益

主比较是 **第 50 步 RL Agent − step 0 frozen Agent**。主指标是全部 64 案例的
outcome_composite 均值，未提交计零；训练中的 shaped reward 只作诊断。

同时报告提交率、等级 MAE、污染事件 CSI/POD/FAR、区间覆盖与 interval score，以及
PM2.5/PM10/O3 区间中点 MAE。完整分母的硬指标使用已明确约定的失败惩罚；例如失败时的
假警报计数是惩罚口径，不能解释成实际发布了假警报。两组都成功提交的案例另外配对比较，
帮助分辨格式收益和预报数值变化，但这是受训练影响的条件子集，不能当作无偏因果估计。

按 issue date 的 7 日块做配对 bootstrap，报告暖季/冬季与 stratum 分层。仅有 14 个块、
单次随机 rollout，置信区间只用于探索；本轮不能宣称全国、多污染类型泛化已获证实。
若只有提交率提高，结论是可用性改善；若硬预报指标也改善，才有扩大验证的技能信号。
若没有改善，应检查奖励分辨力、组内探索、KL、无效轨迹和案例难度，不修改终点后追认成功。

## Baseline 的职责

| 对照 | 回答的问题 |
|---|---|
| 同原生路径 frozen Agent | RL 本身是否带来增益，本轮主对照 |
| CAMS guidance、持续性、气候态 | Agent 是否提供业务增值，本轮同时报告 |
| 同信息集 MOS/机器学习 | 可用证据能被统计订正利用到什么程度，后续完整实验需要 |
| SFT-only（如果引入 SFT） | RL 是否比单纯示范学习多带来收益 |
| 固定证据/固定流程的 RL 消融 | 增益是否来自自主取证和决策，支撑 agency claim 所必需 |
| 可比时点、区域和提前量的人类业务预报 | 预报员层面的外部参照，需要匹配资料可用性 |

通过更大且固定的验证后，再一次性检验封存的时间 test、留城 test，并开展前瞻验证。
证据引用正确只能支持可追溯性；证据忠实性还需要遮蔽、替换等受控干预。

## 运行和交付

入口：`bash scripts/run_rl_skill_pilot.sh`。固定运行 ID 防止意外覆盖，不要重复运行。
日志：`data/experiments/rl_skill_pilot_20260909/train.log`；退出码：`training_exit_code`。
原生评测：`validation/{0,25,50}.jsonl`，包括每案例 normalized forecast、终止原因、分量和 token 数。
正常结束后自动生成 `comparison_step_25.json` 与 `comparison_step_50.json`。
模型保存到 `data/checkpoints/rl_skill_pilot_20260909/global_step_{25,50}`。

代码验证：7 项定向测试通过；另用完整 64 案例、相同合成两组且注入 10 个失败的分析检查，
确认失败保留、配对差为零、三个业务参照齐全。合成检查产物已删除，不计入真实评测。
正式结论须等真实模型评测完成，启动成功不等于技能增益成立。
