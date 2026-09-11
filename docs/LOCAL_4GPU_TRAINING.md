# Qwen3-8B 四卡训练

首轮运行 ID：`qwen3_8b_4gpu_20260911_r2`。

## 配置

- 4×RTX PRO 6000 Blackwell；NCCL 四卡 all-reduce 检查通过。
- 原始 Qwen3-8B BF16，ModelScope 文件 SHA256 和 safetensors 内容结构核验通过。
- FSDP2 训练，vLLM rollout TP=1，四个单卡采样副本，8 个 agent workers。
- LoRA rank/alpha=32/32，learning rate=3e-6；每步 4 个保留 prompt，每 prompt 8 条轨迹。
- Dr.GRPO，不按组内标准差归一化；DAPO 按 outcome_composite 过滤零差异组。
- 总上下文 32K，最多 12 轮工具调用；完整 3,061 个训练案例池。
- 首轮 50 步，固定 step 0/25/50 的 64 案例面板；test 和空间 OOD test 不参与。

四卡提高的是采样和训练吞吐；未以同一工作量测量与单卡的加速倍数。
与原两 prompt 配置相比，本轮每步保留样本数增加，50 步约对应 200 个保留 prompt，
DAPO 补采可能消费更多，不能直接按旧实验的“50 步”比较训练量。

首次工程尝试 `qwen3_8b_4gpu_20260911` 在首次权重同步时触发 Ray 主机内存保护，未产生训练 checkpoint。
原因是 FSDP1 的分层 LoRA 导出回退到各 rank 完整 CPU 参数汇集；当前尝试改为 FSDP2，
原尝试的日志、退出状态和冻结协议保留。未关闭 Ray 内存保护。

## 启动流程

`scripts/run_local_4gpu.sh` 先运行 1 步工程训练，保存后恢复再运行 1 步。
工程审计要求有限 loss、权重同步、工具 token mask、恢复后权重变化和数据哈希同时通过。
通过后，正式 pilot 从原始模型重新初始化 LoRA，独立保存 checkpoint；异常不会自动重启。
工程阶段限时 1 小时，pilot 阶段限时 12 小时。

当前工程尝试已完成 step 1 并保存四份 rank checkpoint。该步包含 32 条轨迹，
平均 response 长度约 14,866 token；采样 269 秒、梯度更新 65 秒、权重同步 3.6 秒，
总计约 382 秒。此为首次工程步的实测值，包含启动后的首次计算开销，不能作为稳定吞吐
或四卡相对单卡的加速倍数。恢复检查和正式训练状态以本目录日志、审计文件为准。

本机服务：`sitian-qwen3-8b-4gpu-20260911-r2.service`，由用户级 systemd 管理，关闭终端不影响运行。
它不是开机自动启动服务；退出状态和文件记录应作为进度依据。

## 状态和产物

```bash
systemctl --user status sitian-qwen3-8b-4gpu-20260911-r2.service
cat data/experiments/qwen3_8b_4gpu_20260911_r2/phase
tail -f data/experiments/qwen3_8b_4gpu_20260911_r2/supervisor.log
nvidia-smi
```

运行目录 `data/experiments/qwen3_8b_4gpu_20260911_r2/` 中：

- `engineering.log`、`engineering_audit.json`：短程保存／恢复日志和检查结果。
- `train.log`、`tensorboard/`：正式训练日志和标量指标。
- `validation/{0,25,50}.jsonl`：原生工具循环逐案例评测。
- `comparison_step_{25,50}.json`：训练完成后的配对分析。
- `supervisor_exit_code`、`training_exit_code`：退出状态；未生成不表示已经成功。
- `protocol.json`：预先冻结的样本、种子、训练参数、代码身份和比较约定。
- `dataset/manifest.json`：迁移后的本机数据审计，保留原 manifest 身份和原始 gate 身份。
  Parquet 与 case 内容逐字节核验，不重写原数据；新 native-runtime 审计替换旧主机 runtime gate。

正式 checkpoint 位于 `data/checkpoints/qwen3_8b_4gpu_20260911_r2/global_step_{25,50}/`；
工程 checkpoint 位于带 `_engineering` 后缀的独立目录。

如需停止：

```bash
systemctl --user stop sitian-qwen3-8b-4gpu-20260911-r2.service
```

停止会终止该服务的子进程；只保留停止前已完整写出的 checkpoint。
恢复或更改配置应先核实退出原因和 checkpoint 完整性，不能删除运行锁后直接覆盖重跑。

训练成功和预报技能提升分别判断：先看工程与训练状态，再按冻结面板比较 outcome、提交率、
等级/事件/浓度区间指标。不能仅凭训练 reward 上升宣布预报能力提升。
