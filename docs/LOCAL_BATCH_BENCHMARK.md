# 四卡全局 batch=4 / 8 等量短测

日期：2026-09-11。运行 ID：`qwen3_8b_batch_benchmark_20260911`。
原 50 步任务已按用户要求停止；本次两组测试正常退出，GPU 已释放，未重启正式训练。

## 本次结果

只把全局 batch 和对应的 PPO mini batch 从 4 改为 8，采样并发等配置保持一致。
两组实际入选的 8 个案例完全相同，每案例 8 条轨迹，均为 64 条轨迹。
batch=4 做两次更新，batch=8 做一次更新；这是处理等量案例的运行效率比较，
不能据此判断两种更新方式的训练效果相同。

| 指标 | batch=4 × 2 步 | batch=8 × 1 步 |
|---|---:|---:|
| 步骤累计耗时 | 572.51 秒 | 556.94 秒 |
| 采样累计耗时 | 342.03 秒 | 337.29 秒 |
| 梯度更新累计耗时 | 139.42 秒 | 134.14 秒 |
| 总序列 token 数 | 1,217,031 | 1,209,371 |
| 四卡总序列 token/s | 2,125.76 | 2,171.47 |
| 整卡显存采样峰值（四卡范围） | 62.22–62.95 GiB | 62.22–62.95 GiB |
| actor 分配显存峰值 | 28.65 GiB | 28.41 GiB |

batch=8 的等案例吞吐提高 **2.8%**，token 吞吐提高 **2.2%**，等量数据耗时减少约 15.6 秒
（2.7%）。实际 token 数相差约 0.6%，均包括 prompt、模型响应和工具文本。
两组 loss、梯度范数有限，checkpoint 均成功保存，运行退出码均为 0。

结论：本次单独增加全局 batch 的收益很小，不足以从一次短测认定稳定提升。
原工程步骤累计 622 秒不能替代本次对照组；重新测试后可见多轮采样耗时有明显波动。
若继续优化，应单独测试采样并发上限或动态训练微批 token 上限，并保留相同样本量比较。
本次没有测试更高的采样并发，也没有执行新的 50 步训练。

## 控制条件与限制

- 4×RTX PRO 6000 Blackwell Max-Q，原始 Qwen3-8B、FSDP2、全新 LoRA rank/alpha=32/32。
- 相同候选数据 `qwen3_8b_4gpu_20260911_r2/dataset/smoke_train.parquet`、相同随机种子。
- 每案例 `rollout.n=8`；vLLM 每卡 `max_num_seqs=8`、TP=1、显存比例 0.55；agent workers=8。
- 训练与 log-prob 动态微批 token 上限均为 32,768，学习率 3e-6，DAPO 过滤保持启用。
- 不进行初始评估、不恢复 checkpoint；两组都从原始模型初始化。
- 先 batch=8，后 batch=4，每组仅一次短测。首次编译、缓存状态和随机采样可能影响结果。
- 使用日志中的 `timing_s/step` 累计值，不计模型加载；部分首次计算开销仍可能进入步骤计时。
- 整卡资源每 2 秒采样，极短峰值可能遗漏；actor 指标是框架分配统计，与整卡占用不同。
- 根据 rollout 输入文本 SHA256 核验两组案例完全重合；每个输入恰有 8 条轨迹。

## 产物与复核

运行目录：`data/experiments/qwen3_8b_batch_benchmark_20260911/`。

- `protocol.json`：试验设计、启动前代码身份和范围限制。
- `batch_{4,8}/resolved_config.yaml`、`runtime_audit.json`：解析配置和原生运行契约检查。
- `batch_{4,8}/train.log`、`rollouts/*.jsonl`：步骤指标和实际训练轨迹。
- `resources.jsonl`：整卡显存、GPU 利用率、功耗、主机可用内存。
- `comparison.json`、`comparison.md`：结果和计算口径。
- `exit_code` 及两个 arm 的 `exit_code`：均为 0；`phase` 为 `complete`。

checkpoint 位于 `data/checkpoints/qwen3_8b_batch_benchmark_20260911_batch_8/global_step_1/`
和 `data/checkpoints/qwen3_8b_batch_benchmark_20260911_batch_4/global_step_2/`。

复核本次统计：

```bash
python3 scripts/summarize_batch_benchmark.py \
  --run-dir data/experiments/qwen3_8b_batch_benchmark_20260911
```

`scripts/run_local_batch_benchmark.sh` 可用于同机后续等量复测，要求唯一
`SITIAN_BENCHMARK_ID`，并验证原本机冻结数据协议；不会覆盖已有运行锁。
每组限时 1 小时，完成或失败后退出，不会自动进入正式训练。
