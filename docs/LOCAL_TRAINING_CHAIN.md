# 四卡增量试训与长训

2026-09-11，用户要求测试本身积累有效训练更新，并根据实际案例改进后选择配置长训。
主运行：`data/experiments/qwen3_8b_training_chain_20260911_r2/`。
systemd 用户服务：`sitian-training-chain-20260911-r2.service`。

## 已完成的 8 / 16 扩展对照

运行 `qwen3_8b_scaling_20260911` 正常完成。两组处理完全相同的 16 个案例、128 条轨迹，
每卡采样并发 16，推理显存预算 0.55，其余训练配置一致。

| 指标 | batch 8 × 2 步 | batch 16 × 1 步 |
|---|---:|---:|
| 累计步骤耗时 | 834.02 秒 | 871.14 秒 |
| 累计采样耗时 | 444.35 秒 | 481.53 秒 |
| 梯度更新耗时 | 240.89 秒 | 242.88 秒 |
| 序列 token/s，含提示与工具文本 | 2,659.86 | 2,607.81 |
| 案例/分钟 | 1.151 | 1.102 |
| 采样活跃期间至少两副本无请求的比例 | 36.36% | 10.13% |
| KV cache 抢占累计次数 | 0 | 3 |
| 整卡显存采样峰值（最大卡） | 62.96 GiB | 66.68 GiB |

全局 batch 增大确实减少了副本空闲，但尚未改善等量工作吞吐；batch16 案例吞吐低约 4.3%，
token 吞吐低约 2.0%。KV cache 峰值接近 100%，发生抢占，成为后续配置调整依据。
请求空闲为轮询代理指标，包括启动阶段及工具暂停，不是精确的尾部等待计时。

此前 batch8、每卡并发8的首步与本次 batch8、并发16首步对应相同8个案例：
案例吞吐提高16.4%，token吞吐提高10.4%。这是历史单步参考，不是稳定性能保证。

原始数据、计算结果和图：
`data/experiments/qwen3_8b_scaling_20260911/{comparison.json,comparison.md,resources.jsonl,occupancy.png,occupancy.svg}`。
复算脚本：`scripts/summarize_scaling_benchmark.py`、`scripts/plot_scaling_benchmark.py`。
绘图依赖使用独立 uv 临时环境，未修改训练环境。

## 连续更新的计数与恢复

主链从 `qwen3_8b_scaling_20260911_b16_s16/global_step_1` 开始，已有1个有效更新。
之前分别从原始模型启动的 batch4、batch8 及工程验收分支不能累加为这条链的更新次数。
后续先测试 batch32；显存低于85 GiB、主机可用内存高于32 GiB，且案例与token吞吐均不低于
batch16参考的95%，再测试batch64。每卡并发保持16，推理显存预算提高至0.65。
抢占计数作为效率诊断报告，不当作训练失败。

每一档都实际执行 GRPO，恢复四个 rank 的 LoRA、优化器、学习率调度器及随机状态。
完成后逐 rank 比较本地参数分片，要求参数有限且确有变化，优化器step与调度器step正确递增。
检查通过才进入下一档。选择较小 batch 时仍恢复最新 checkpoint，保留较大 batch 的更新。

从64条短测集进入3061条完整训练集时，仅复制actor checkpoint到新的`global_step_1`目录，
不携带旧数据迭代器；逐文件SHA256验证actor分片完全一致，原checkpoint不修改。
`full_data_entry/data_transition.json`明确记录这次迭代器重置，之后各段恢复`data.pt`。
全量训练数据与原审计数据字节一致；采样种子为20260909，奖励和数据隔离规则不变。

原`qwen3_8b_training_chain_20260911`在初始化期间停止，新增有效更新为0。
原因是在父试验结果完整返回后发现KV抢占，需要在下一档开始前调整显存预算；由`_r2`取代。

## 案例驱动修改

旧配置`configs/verl/sitian_tools.yaml`的提交描述仍要求`daily`、AQI等级和首要污染物，
与当前任务简报要求的顶层`pm25_lo/hi`、`pm10_lo/hi`、`o3_lo/hi`列冲突。
生成器`build_verl_tool_config.py`已经使用正确描述，运行中的旧生成产物未同步。

batch8/并发16的128条实际轨迹中，114条观察到成功提交，95条曾有工具/提交校验错误，
8条有闭合但无法解析的工具JSON。校验失败不等于最终失败；许多轨迹修正后成功提交。
例如`通辽_2025-09-06`的某条轨迹首次提交缺少`issue_date/region`，接到错误后补全成功。
同一案例最长与最短响应字符量相差3.14倍；字符包含工具返回，不能直接当作token或耗时。

续训使用重新生成的`configs/verl/sitian_tools_columns.yaml`，明确日期、地区和顶层区间列，
删除旧版daily/AQI要求。旧文件保留用于冻结对照协议。原生工具生命周期检查针对新文件执行。
不修补模型生成的无效JSON，不修改提交校验或奖励。新增回归测试验证恢复提交、训练集隔离
以及新版工具描述与生成器一致。

`scripts/review_training_cases.py`只允许训练集case_id，逐步写入来源路径、行号、失败片段、
工具错误、提交恢复情况与组内响应长度差异。只分析DAPO保留组，不把验证案例用于修改。
batch16旧描述/0.55预算与后续新描述/0.65预算的结果不能作为单因素batch因果对照；
batch32和batch64则使用相同的新描述、显存预算与完整训练数据，但权重、案例仍随训练推进。

## 自动选择及长训

在满足内存余量的配置中，先保留token吞吐达到最佳值95%的候选，再按案例/分钟选取，
案例吞吐也相差不超过5%时选择较小batch。规则在32档结果出来前写入`protocol.json`。
`selection.json`保存选择依据，随后自动从最新checkpoint继续到累计50个有效更新。
每5步保存checkpoint、保留最近4个；每档短测限时2小时，长训限时36小时。
失败时保留checkpoint和错误记录，不静默从头训练。该服务不依赖当前终端存活。

长训开始前，用已冻结的64案例面板验证当前checkpoint；这是续训基线step2或step3，
不是未经训练的step0。step25/50保存原生验证记录，结束后用相同case与种子做配对比较，
报告提交率、outcome及包含失败惩罚的硬指标。结果只能说明这段续训的变化。
被封存的测试集不进入训练或案例驱动修改。

运行进度：`phase`、`progress.json`、`profiles.json`、各段`train.log`。
恢复证据：`lineage.json`、`checkpoint_identity.json`、`checkpoint_update_audit.json`。
训练案例：各段`case_review/cases.{json,md}`。

验证：项目测试208通过、2跳过；新增3个回归测试通过；Python静态错误检查及Bash语法检查通过。
真实多卡恢复与参数变化审计将随每档完成写入运行目录，不能以配置检查替代。
