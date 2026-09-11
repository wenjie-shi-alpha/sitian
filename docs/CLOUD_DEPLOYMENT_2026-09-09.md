# 云端部署与工程验收

**结果：2026-09-09 云端 2 步训练 + 保存恢复 + 第 3 步更新完成，进程退出码 0，自动工程审计全部通过。** 张量级对比证实 504 个 LoRA 张量全部发生实际数值变化，优化器状态步数从 2 推进到 3，所有保存权重有限。尚未启动全训练集正式长训，也未据此判断 RL 预报技能增益。

最终 `global_step_3` 已完整备份到本机 `data/checkpoints/qwen3_8b_cloud_20260909_acceptance/global_step_3`，含优化器状态；11 个文件共 1,059,630,084 字节，全部与云端 SHA256 一致，见 `data/interim/cloud_checkpoint_backup_verification.json`。

本次部署使用 `.env` 中的 SSH 连接信息；凭据只在本机读取，未上传到云端项目目录。云端任务不依赖商业模型 API。

## 机器和目录

- GPU：单张 NVIDIA RTX PRO 6000 Blackwell Server Edition，97,887 MiB 显存，计算能力 12.0。
- 驱动：590.44.01；容器内存限额 120 GiB；数据盘容量 200 GB。
- 项目实际目录：`/root/autodl-tmp/sitian`，入口软链接 `/root/sitian`。
- veRL 实际目录：`/root/autodl-tmp/verl-upstream`，入口软链接 `/root/verl-upstream`。
- 模型：`/root/sitian/models/Qwen3-8B`，ModelScope 的 Qwen/Qwen3-8B 原始权重；必须逐文件通过已保存的官方 SHA256 清单。

## 运行环境

复制本机已安装且锁定的 veRL 环境，保持 veRL 提交 `c2429f29a25d573f63d9bcc29e7ceb690817dce9`。云端虚拟环境解释器指向云端 Python 3.12.3；入口路径保持不变。

| 包 | 版本 |
|---|---|
| PyTorch | 2.11.0+cu130 |
| vLLM | 0.24.0 |
| Transformers | 5.9.0 |
| FlashAttention | 2.8.3 |
| Ray | 2.55.1 |
| veRL | 0.10.0.dev0 |

`UV_NO_SYNC=1` 使用已经部署的环境，不在 Ray worker 启动时重新联网解析。测试另补装了 scikit-learn、SciPy、cdsapi 等辅助依赖，版本见 `data/interim/cloud_optional_deps.log`。本环境不含地理栅格预处理所需 GDAL。

## 已完成的检查

- 云端 BF16 FlashAttention 前向和反向通过，梯度有限；司天 veRL 运行模块导入通过：`data/interim/cloud_gpu_probe.log`。
- 训练集 3,061、验证集 781、冬季选择集 344 个案例的输入和隐藏真值快照，以及 parquet、切分清单、证据/奖励配置哈希全部匹配：`data/interim/cloud_dataset_verification.json`。
- 云端配置解析及工具生命周期合同通过：`data/interim/cloud_resolved_config.yaml`、`cloud_runtime_contract.json`。
- Qwen3-8B 原生工具 token 掩码通过：`data/interim/cloud_qwen3_8b_mask.json`。
- 云端测试 200 项通过、1 项跳过；命令为 `python -m pytest -q --ignore=tests/test_derive_open_composition.py`。GDAL 数据预处理测试文件未执行，见 `data/interim/cloud_pytest.log`。

实际训练证据见 `data/interim/qwen3_8b_cloud_20260909_acceptance_audit.json`，张量与优化器状态证据见同前缀的 `_tensor_audit.json`。运行期间出现个别工具 JSON 解析失败；工程验收通过不代表有效提交率达到业务要求。每步读取的是不同训练案例，不能用这 3 步的奖励变化声称学到了预报能力。

## 运行中发现并收口的问题

1. veRL V1 会将 `ppo_mini_batch_size` 再乘以 `rollout.n`。旧入口填写 16，导致每步 16 条真实轨迹外增加 112 条 loss mask 为零的最小填充序列。此次完整验收使用该旧配置，填充不贡献 PPO/熵/KL 损失。验收后入口已改为 2 个 prompt；直接调用锁定版本的批次函数验证得到 16 条轨迹、零额外填充，见 `cloud_batch_unit_check.json`。修正后的配置解析和工具合同也通过，见 `cloud_corrected_runtime_contract.json`；未另行重复整套参数训练。
2. V1 忽略 `max_num_gen_batches`，实际时间限制由外层 `timeout` 执行。此次任务从启动即设 60 分钟上限，正常完成后 GPU 已释放。
3. Ray 驱动日志的指标转发存在延迟，因此额外保留两阶段 TaskRunner 原始日志。最终驱动日志包含所需指标，自动审计正常通过。

## 短程验收入口

在模型完整性校验通过后，在云端执行：

```bash
cd /root/sitian
bash scripts/run_cloud_smoke.sh
```

默认进行 2 次优化器更新，保存检查点，重新启动并恢复，再进行 1 次更新。每次运行使用独立时间戳目录，不覆盖既有训练记录。使用 smoke_train/smoke_val，仅作为工程验收；不能视作全训练集正式训练，也不用于判断科学 claim。

配置保持现有奖励与数据合同：Qwen3-8B、LoRA rank 32、GRPO、每组 8 条 rollout、2 个 prompt、32K 上下文及原生多轮工具。单卡使用 `layered_summon=False`；vLLM 显存比例初值 0.55、并发序列 8。当前 veRL V1 ReplayBuffer 明确忽略 `max_num_gen_batches`，因此不能把该配置当作有效的重采样上限。入口通过外层 `timeout` 将验收限制为 3,600 秒，超时退出而不自动长训。

结果分别写入：

- `data/interim/<run_id>.log`
- `data/interim/<run_id>_mask.json`
- `data/interim/<run_id>_audit.json`
- `data/checkpoints/<run_id>/global_step_*`

正式长训需要另行使用完整 train.parquet 和独立验证/选点配置；本入口不自动转入长训。

## 本次代码调整

- 新增 `scripts/__init__.py`，防止 veRL 的同名 scripts 包遮蔽项目审计工具。
- 为现有 smoke 脚本增加实验名称、独立审计输出和重采样上限配置，保留原默认行为。
- 新增云端短程入口 `scripts/run_cloud_smoke.sh`。
- 修正 PPO mini-batch 的 prompt 单位，并使审计明确区分新日志单位和旧未标注记录。
- 新增官方模型哈希校验与检查点张量/优化器状态对比工具。
