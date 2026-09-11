# 司天 Agent RL 科学性与开训准备度审计

> **历史快照。** 本文冻结 2026-08-28 当时的 reward-v0.3.0 证据，用于追溯门禁如何形成；
> 当前可执行合同已升级到 schema-v0.5.6 / reward-v0.7.9，以
> [`TRAINING_DESIGN.md`](TRAINING_DESIGN.md) 和 `data/interim/rl_preflight.json` 为准。
> 旧表中 composite 与当前结果不可直接比较。

日期：2026-08-28

对象：`Qwen/Qwen3-8B-AWQ` 本地冻结策略 + 全国污染预报 harness

当前 reward：`0.3.0`（配置 SHA256 `f266b01d…b3593562`）

## 结论先行

**可以继续低成本的小模型 RL 管线冒烟；不启动 8B 多轮长训。** 这里的理由必须分成两层：

- `preflight` 只回答“能否安全开一次 50–200 step 的小模型 LoRA 冒烟”；不要求未训练策略先超过 CAMS。
- `graduation` 才回答“训后是否值得扩到 8B”与“是否达到论文证明强度”。这一层要求超过 CAMS、冬季 challenge 不退化，最终超过强监督 tabular 基线。

当前真正的硬阻塞是训练栈与硬件冒烟：本机是 WSL2 + RTX 4090 24 GB，环境有 `trl`，但无 `verl`/`peft`，也无可训 base checkpoint。本地 1.7B/4B LoRA 可用于闭环冒烟；8B BF16/LoRA 与 rollout 共卡不是 24 GB WSL2 上的可靠长训方案。

## 两层门禁

### A. Preflight：开小模型冒烟的前提

| 门禁 | 可机检标准 | 当前状态 |
|---|---|---|
| 数据与 reward 完整性 | valid manifests 存在；challenge/train 的 case-id 和 forecast city-day 重叠均为 0；reward version + config hash 一致；tabular 校准与评测 target 重叠为 0 | 数据侧已通过；待 50×8 新探针完成后由产物哈希最终锁定 |
| 格式与轨迹 | 成功提交率 ≥95% | 6-case 旧探针为 100%，但只作 provisional；50×8 正在复核 |
| 采样效率 | ≥50 case、group=8；过滤零方差后可用 group 率 ≥70% | 正在运行；零方差率是 rollout 预算指标，不是科学效果硬门槛 |
| 可训栈与硬件冒烟 | trainable base + `verl` + `peft`；≥50 step；验证策略权重同步、工具 token loss mask、finite loss、checkpoint 重载 | **未通过** |

### B. Graduation：训后的扩展与论文标准

| 阶段 | 最低标准 | 当前 |
|---|---|---|
| 扩到 8B | 独立 val 上对 CAMS 的配对差，按起报日/天气过程分块 bootstrap，95% CI 下界 > 0 | 未训练，不适用 |
| event 毕业 | ≥100 个隐藏真值 AQI≥4 case，同样相对 CAMS 的 CI 下界 > 0；不得用互斥采样标签代替 | 未训练，不适用 |
| checkpoint 选择 | 除暖季 val 外，冻结 checkpoint 必须在 purged winter challenge 上相对 CAMS 不退化 | challenge 已就绪，无训后 checkpoint |
| 论文效果 | 独立评测上超过强 tabular 基线，分块 CI 下界 > 0 | 未训练，不适用 |
| 前瞻证据 | 2026–27 冬季完全 prospective 评测 | 2026-12 起才能获取 |

`event composite ≥0.50` 已取消：这个绝对阈值没有经验或业务标定依据。新标准是在同一 event 层、同一 reward 版本下超过 CAMS。

## 冻结基线与数据读数

### 可实现对照

| split | n | persistence | CAMS | climatology | 事后三选一参考 | 强 tabular | tabular macro-stratum |
|---|---:|---:|---:|---:|---:|---:|---:|
| val（5–6 月） | 781 | 0.627 | 0.698 | 0.616 | 0.753 | **0.755** | 0.761 |
| winter challenge | 344 | 0.567 | 0.625 | 0.512 | 0.689 | **0.683** | 0.694 |
| test（7–8 月） | 1472 | 0.729 | 0.679 | 0.712 | 0.800 | **0.823** | 0.774 |

tabular 是训练期内时间折选出的 HGB/LightGBM/XGBoost 强监督套件，只用起报时可见的 CAMS、实况统计、诊断、气候态和时空特征；算法选择与区间校准使用彼此隔离、带 5 天 embargo 的训练切片，均不读取 val/test/challenge/OOD。因此，子刊级结论的证明责任是超过它，不是只超过 persistence。“事后三选一”只是不可实现的整份预报择优参考，不是任意数值融合的理论上限。

winter challenge 的 CAMS event 层为 0.542，tabular event 层为 0.529。这说明强 tabular 总分高并不等于 event 已解决；后续必须同时守住分层指标。

### 冬季 checkpoint-selection challenge

从原 valid train 池中按**完整起报日**选出 11 个冬季日期，challenge 仅保留 event/turning/switch 目标层：

- 344 case，85 城；event 130 / turning 146 / switch 68；
- 1,325 个唯一 forecast `(city, valid_date)` target；
- 所选日期前各 5 天的起报统一从 train 剔除，保留 train 3,240 case，剔除 1,564 case；
- challenge/train 的 case-id 重叠 = 0，forecast city-day 重叠 = 0。

它是模型选择的冬季守门集，**不是终评 test**。只有 11 个起报日/约 9 个连续过程块，适合防止暖季 val 选出冬季能力差的 checkpoint，不适合单独承担窄 CI 的论文推断。

### tabular 校准隔离

原“后 20%”切分已改成整起报日、带 5 天 embargo 的 purged split，而且校准样本不再参与产生预测的模型拟合：

- fit：2,532 case，2025-04-01–2026-01-07；
- purge：52 case；
- calibration：656 case，2026-02-11–2026-04-30；
- fit/calibration forecast city-day 重叠 = 0；
- calibration 与 val/winter challenge/test 的 forecast city-day 重叠均为 0；
- 实际训练 manifest 与三个评测集的 forecast city-day 重叠也均为 0。

## 冻结 AWQ 探针的正确解读

旧探针只有 6 case×4 rollout：冻结 Qwen3-8B-AWQ 平均 0.633，同 case CAMS 0.740，零方差 1/6。这只能写成**小样本描述性信号**，不能写成“数值技能显著低于 CAMS”。一个 case 的 4 条 rollout 不是 4 个独立样本。

新探针使用固定 seed、分层 50 case、group=8、temperature=1.0，共 400 rollout。正式比较规则是：

1. 格式失败 rollout 保留 reward=0，不做“只看成功轨迹”的条件化比较；
2. 先在 case 内平均 8 条 rollout；
3. 再对 case 做配对差，按 issue date 或预先规定的连续天气过程块整块 bootstrap；
4. 独立 cluster <30 时只报描述性区间，不下显著性结论。

这个探针只决定格式与采样预算，不是 graduation 结果。

## AWQ、on-policy 与硬件边界

- 本地 `Qwen3-8B-AWQ` 只用于**冻结评测和数据/方差探针**。
- 真正 RL 期间，rollout 必须由 veRL/SGLang/vLLM 托管的**当前可训策略**产生，LoRA 权重更新后同步到 rollout engine。
- 禁止用冻结 AWQ 采样、却更新另一份 BF16 策略；这会形成未经校正的 off-policy 错配。
- 本机 RTX 4090 24 GB + WSL2：1.7B/4B LoRA 单轮冒烟是合理目标；8B 长训目标是云上 ≥80 GB 级 A100/H100，优先 ≥2×80 GB，先做 100-step 显存/吞吐冒烟再预算长训。

`data/interim/training_smoke.json` 只有在同时记录 `success=true`、步数达标、`rollout_policy_sync_verified=true`、`tool_tokens_loss_masked=true`、`finite_losses=true` 和 `checkpoint_reload_verified=true` 时才能通过 preflight。

## reward、标准与可追溯性

- reward 已版本化为 `0.3.0`；每个 score details 以及 baseline/probe/eval 产物都写入 reward version、完整 config 和 config SHA256。
- 评测产物同时记录 case manifest 与标准 manifest 的 SHA256。reward 配置或标准变化后，旧曲线不再默认可比。
- HJ 633—2026 与 HJ 663—2026 已冻结到 [`references/standards`](../references/standards/README.md)；[`manifest.json`](../references/standards/manifest.json) 保存官方 URL、PDF hash、实现依据页码与审核截图 hash。

本轮已修复的 reward 语义问题包括：六污染物真值统一、HJ 633 新旧标准按日切换、event/turning 重复送分、AQI>=4 event 与 AQI>=3 process 阈值分离、区间 miss 稠密分、grounding 绕过与 CAMS 基线跨 split 变义。

## 成功率优先的执行顺序

1. **冻结数据与指标。** 只用 valid/purged manifests，所有曲线携带 reward/config/dataset/standard hash。
2. **完成 50×8 冻结策略探针。** 只用于格式、组内方差、成本与冷启动判断；不把小样本差值写成效果结论。
3. **1.7B/4B LoRA 单轮 50–200 step 冒烟。** 只验证 reward→gradient→policy sync→checkpoint reload→独立 val 的闭环。
4. **先固定证据包学数值，再释放工具选择。** 工具 token loss mask，void-turn 率门禁（正式探针要求 ≤5%，当前 veRL 路径不冒充已逐条过滤），group=8，零方差动态重采样，监控有效 token、KL、格式失败和每层可用 group 率。
5. **checkpoint 双轨选择。** 暖季 val 负责常规优化，winter challenge 是 event/turning/switch 守门；任一轨明显退化都不扩到 8B。
6. **只在 graduation 通过后转 8B 硬件。** 小模型先证明管线能学，云上 8B 先冒烟，再决定长训预算。

## 评测与论文的统计红线

- rollout 在 case 内先平均；不把 group=8 冒充成 8 个独立 case。
- paired bootstrap 以起报日或天气过程为 cluster，不按 city-case 独立重采样。
- 同时报 micro composite、macro-stratum、event CSI/POD/FAR、ordinal MAE、primary macro-F1、coverage/平均宽度/interval score。
- val 与 test 均是暖季，不能独立支撑冬季重污染结论。winter challenge 进模型选择；2026–27 冬季 prospective 集才是 event 时间泛化的最终证据。
- 当前 val/test 的城市在 train 中全部出现，真正 spatial OOD 仍未解决。论文前需要按城市或污染区制整组留出，并在训练 manifest 中剔除对应城市，而不是只从现有 val/test 里选城。

## 备选论文路线：tabular 作为工具

如果小模型 RL 最终打不过 tabular，不要通过降低对照标准挽救结论。可将 tabular 预测作为一个可调用工具，将任务明确改为“订正与融合”。对照应包括 tabular 单模型、CAMS/tabular 预注册加权融合、stacking/gating 等可实现融合；事后择优只作参考，不冒充可达上限。

## 复现入口

```bash
python3 scripts/audit_national_data.py
python3 scripts/build_winter_challenge.py
python3 scripts/eval_national.py --split val
python3 scripts/eval_national.py --split train \
  --manifest data/interim/challenge_winter.json --out-name challenge_winter
python3 scripts/eval_tabular_baseline.py
python3 scripts/probe_model.py --split val --n 50 --group-size 8 \
  --temperature 1.0 --policy-stage frozen_baseline \
  --out data/interim/probe_open_evidence_v079_n50_g8.json
python3 scripts/compare_probe_clustered.py \
  --probe data/interim/probe_open_evidence_v079_n50_g8.json --baseline guidance
# 训后事件毕业集必须由隐藏真值定义，不得用互斥采样标签 --stratum event 代替：
python3 scripts/probe_model.py --split val --truth-event-only --n 100 \
  --group-size 8 --temperature 1.0 --policy-stage trained --policy-id CHECKPOINT_ID \
  --out data/interim/probe_trained_v079_truth_event_n100_g8.json
python3 scripts/compare_probe_clustered.py \
  --probe data/interim/probe_trained_v079_truth_event_n100_g8.json \
  --baseline guidance --truth-event-only \
  --out data/interim/compare_trained_v079_truth_event_vs_guidance.json
python3 scripts/compare_probe_hard_metrics.py \
  --probe data/interim/probe_trained_v079_truth_event_n100_g8.json \
  --baseline guidance --truth-event-only \
  --out data/interim/compare_trained_v079_truth_event_vs_guidance_hard.json
python3 scripts/compare_probe_clustered.py \
  --probe data/interim/probe_trained_v079_truth_event_n100_g8.json \
  --baseline tabular --truth-event-only \
  --tabular data/interim/eval_tabular_open_evidence_v079.json --tabular-split val \
  --out data/interim/compare_trained_v079_truth_event_vs_tabular.json
python3 scripts/compare_probe_hard_metrics.py \
  --probe data/interim/probe_trained_v079_truth_event_n100_g8.json \
  --baseline tabular --truth-event-only \
  --tabular data/interim/eval_tabular_open_evidence_v079.json --tabular-split val \
  --out data/interim/compare_trained_v079_truth_event_vs_tabular_hard.json
python3 scripts/rl_preflight.py \
  --event-comparison data/interim/compare_trained_v079_truth_event_vs_guidance.json \
  --event-hard-comparison data/interim/compare_trained_v079_truth_event_vs_guidance_hard.json \
  --event-tabular-comparison data/interim/compare_trained_v079_truth_event_vs_tabular.json \
  --event-tabular-hard-comparison data/interim/compare_trained_v079_truth_event_vs_tabular_hard.json
```

机检总结见 [`rl_preflight.json`](../data/interim/rl_preflight.json)。它即使因为训练栈缺失而返回非零退出码，也不代表“RL 科学上注定失败”；它表示当前环境还没有满足安全开冒烟的全部前提。
