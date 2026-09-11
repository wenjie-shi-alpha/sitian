# 司天 sitian

全国城市空气质量预报智能体的 **Agent RL 项目**：可验证任务环境（harness）+ 真实个例池 + 规则化评分 + RL 训练入口。
名取"司天监"——古代观天象、司预报的官署。前身为 JJJ_ATMO 仓库的 forecast_harness 子目录，现为独立项目。

安装：`pip install -e '.[dev,imaging]'`（核心零依赖；图像层需 Pillow/numpy）。

本机完整开发/证据处理环境请运行 `bash scripts/setup_local.sh`，然后
`source .venv/bin/activate`。该入口使用 Python 3.12 和 `uv.lock`，同时安装测试需要的
analysis/evidence 依赖。GPU 安装、Blackwell 实测结果与旧数据路径迁移见
[`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md)。训练与推理共用独立的
`.local/verl-upstream/.venv`；复制来的旧 `.venv-vllm` 不作为启动入口。

当前专家决策需求、开放证据合同、策略使用证据、Reward 防投机与剩余门禁的可视化终审见
[`docs/reports/expert_forecast_requirements/report.html`](docs/reports/expert_forecast_requirements/report.html)；
逐项可追溯表见同目录 `appendix.md`。数据补齐后最终器会用正式 50×8 与真实训练冒烟自动覆盖更新。
当前全国池含 7,548 个原始 case；按逐小时观测有效性规则隔离 491 个后，正式训练/评测清单为
7,057 个 case。开放证据对全部原始 case 挂接，模型拟合与统计比较只消费有效清单。

## 设计原则

1. **预报是结构化决策，不是自由文本。** 提交物是 schema 校验的预报对象
   （逐日等级/首要污染物/PM2.5、PM10、O3 区间/过程转折点/证据引用），评分完全规则化
   可复算，不用 LLM judge。
2. **专家语料只设计机制，观测真值监督结果。** JJJ_ATMO 蒸馏语料仅用于梳理专家决策链、
   工具和审计合同，不进入全国 episode、SFT 输入或 reward 真值；等级/事件/区间/转折点
   只对观测真值负责，grounding 只核验工具引用而不冒充因果解释真值。
3. **一个 episode = 一个起报时次。** agent 多步调用工具（实况/诊断/模式指导/开放形势与污染机理/昨日预报/相似个例）
   后提交，稀疏 reward。episode 规模靠全国历史回放扩展，不靠专家会商语料条数。
4. **防泄漏是测试看护的硬约束。** truth/expert 永不通过工具暴露（`tests/test_leakage.py` 哨兵测试）；
   实况有起报时刻门禁（`CaseBundle.audit_time_gate`）。

## Quickstart

```bash
cd sitian
python3 -m pytest tests -q
PYTHONPATH=src python3 scripts/audit_reward_contract.py       # 23 条 Reward 防投机不变量
PYTHONPATH=src python3 -m sitian.cli gen --out cases/synthetic
PYTHONPATH=src python3 -m sitian.cli baselines --cases 'cases/synthetic/*'
PYTHONPATH=src python3 -m sitian.cli demo --case cases/synthetic/synth_accumulation_2025-12-16_s7

# LLM agent（对接任意 OpenAI 兼容服务，如 vllm serve Qwen/Qwen3-8B）
FH_BASE_URL=http://127.0.0.1:8000/v1 FH_MODEL=Qwen/Qwen3-8B \
  PYTHONPATH=src python3 -m sitian.cli llm --case cases/synthetic/synth_guidance_misleading_2025-12-19_s7
```

合成套件基线表（seed=7，验证评分器区分度）：

| case | persistence | guidance(盲从EC) | expert |
|---|---|---|---|
| accumulation（积累-清除过程） | 0.177 | 0.900 | 0.940 |
| clean（清洁时段） | 0.787 | 0.884 | 1.000 |
| guidance_misleading（EC 报早清除一天） | 0.215 | **0.704** | 0.942 |

预期行为全部出现：持续性基线漏报过程即崩；盲从模式指导在"低压推迟"型 bust 上被罚；专家稳居上限。

## 模块

```
src/sitian/
  schema.py      预报对象契约 v0.6.4：策略只提交扁平列式浓度区间；AQI 等级/首污/过程由 harness 从区间中点派生
  case.py        CaseBundle：个例数据包 IO、真值视图、时间门禁审计
  scoring.py     reward v0.8.1：结果分量 + 可核验 grounding（标量或元组事实）；训练 composite 与公平比较用
                 outcome_composite 分离，硬 CSI/MAE/F1/区间分另行报告
  env.py         ForecastEnv：gym 风格多步环境，条件注册开放证据工具，步数预算，稀疏 reward
  synth.py       合成个例生成器（accumulation / clean / guidance_misleading 三剧本，seed 可复现）
  agents/
    scripted.py    persistence / guidance / expert 三基线 + run_episode/run_baselines
    llm_openai.py  OpenAI 兼容多轮 tool-calling 驱动器（纯标准库），即 RL rollout 参考实现
  integrations/
    verl_reward.py  单轮兼容 reward 入口：compute_score(solution_str, case_dir) → [0,1]
    verl_bridge.py  从 token 轨迹确定性重放 ForecastEnv；忽略模型可伪造的 tool-result 文本
    verl_runtime.py veRL ToolAgentLoop/工具适配：终端环境 reward、工具 token mask、同策略 rollout
  cli.py         gen / demo / baselines / llm
cases/           个例数据包（契约见 cases/README.md）
```

## 评分与 reward

训练分 = Σ wᵢ·分量ᵢ，默认权重 level .35 / event .25 / interval .15 / turning .15 /
primary .10 / evidence .10 / grounding .05；缺失分量自动弃权去权归一。模型与 CAMS/tabular
的公平比较统一使用排除 evidence/grounding 的 `outcome_composite`。要点：

- **事件日加权**（默认 2×）+ **事件 CSI**：防"永远报良"的多数类投机；
- **区间稠密分**：按标准 80% interval score（宽度 + 未覆盖距离罚）做连续指数衰减，
  超宽区间无保底平台；同时输出未变换的 interval-score 诊断，降低 GRPO 组内零方差；
- **格式门禁**：校验失败 composite=0；环境内提交失败返回错误可重试（耗步数），支持 RL 学格式自纠；
- high-impact event 为 AQI>=4，污染 process/turning 为 AQI>=3；阈值、权重均在
  `RewardConfig` 可调。无真实 process 时 turning 弃权，不与 event 重复送分。
- 当前 reward 版本为 `0.8.1`、forecast schema 为 `0.6.4`（2026-09-03）：策略**只提交**
  PM2.5/PM10/O3 三类 80% 区间（扁平列式 `pm25_lo/pm25_hi/...`，各 H 个数字；逐日对象与
  区间表形式仍被接受；lo/hi 写反按序接受；任务简报只给结构说明、不给数值示例——冻结策略曾
  91% 原样照抄示例），AQI 等级、首要污染物与 AQI≥3 过程头由 harness 按 HJ 633
  从区间中点派生（联合一致性由构造保证，不再作为格式门禁；正式 50×8 探针曾有 45% 的
  首次提交因联合约束被拒）。提交里的 aqi_level/primary_pollutant/process 只做类型检查、不采纳。
  全国多污染物预报要求 PM2.5、PM10 与 O3 三类关键污染物区间；clean correct-negative 不再从
  event/turning 重复取分；event 的近失误信用只用于 RL 塑形，硬 CSI 单独报告；grounding
  必须用真实工具返回的 ref + JSON Pointer + 标量值核验，按“工具返回内容 hash + Pointer”
  去重，至少两条互不重复且覆盖两个证据类别的断言才满分；换 ref、42/42.0、null、空字符串、
  NaN/Inf 都不能凑科学事实。grounding 只进入训练分，
  不进入模型毕业的结果分。
  过程起止/峰值由派生的逐日等级确定：取包含最高等级的第一段连续 AQI≥3 污染段。所有新 probe/eval 产物都携带 reward 版本、
  config SHA256、case manifest 与 HJ 标准 manifest 哈希；不同 reward 身份的曲线不混比。

## RL 集成路径

1. **已实现的目标形态**：`SitianToolAgentLoop` 直接复用 veRL 的异步 token-in/token-out
   `ToolAgentLoop`；`agent_name=sitian_tool_agent` 显式写进 Parquet。工具调用由
   `ForecastEnv` 重放，首次合法 `submit_forecast` 立即终止并把环境训练 composite 作为
   `AgentLoopOutput.reward_score`，无需从生成文本二次解析 reward。
2. **同策略与 loss mask**：FSDP/PEFT LoRA 权重由 veRL 同步到同一次训练的 vLLM rollout；
   assistant 推理/工具动作 mask=1，system/user/tool-result mask=0。该结论由锁定 commit 的
   `QwenContinuousTokenBuilder` 直接审计，不只依赖框架外 tokenizer 重构。冻结 8B-AWQ 只做正式
   50×8 基线探针，结束后释放显存，绝不为另一份 BF16/LoRA 权重采样。
3. **优化与门禁**：Dr.GRPO（组内均值中心化、不除标准差）+ DAPO 零方差组补采样；
   50 步 LoRA 后必须从 checkpoint 真恢复并再跑 1 步，`training_smoke.json` 同时验收
   policy sync、finite loss、工具 mask 和权重变化。void-turn 不冒充有效轨迹：正式探针要求
   其比例不高于 5%，训练期保留零分负信号并持续监控。只有冻结探针的格式/grounding/有效组率
   门禁通过才自动启动；否则先定位 harness，必要时才做小型 tutor SFT。

数据和训练入口（最终数据附件完成后执行）：

```bash
bash scripts/setup_verl_stack.sh       # 固定官方 veRL commit + Qwen3-1.7B
PYTHONPATH=src python3 scripts/build_verl_tool_config.py
SITIAN_CONFIG_ONLY=1 bash scripts/run_verl_smoke.sh > data/interim/verl_resolved_smoke_config.yaml
.local/verl-upstream/.venv/bin/python scripts/audit_verl_runtime_contract.py
PYTHONPATH=src python3 scripts/prepare_verl_dataset.py
bash scripts/run_verl_smoke.sh         # 50 step + checkpoint reload 1 step
```

`prepare_verl_dataset.py` 强制用 `train_without_winter_or_spatial_holdout.json`，并生成保留 split 标签的
`selection.parquet`（warm-season val + winter challenge）；训练 manifest 会记录并复核 challenge
与训练池的 case 及 `(city, forecast_date)` 重叠均为 0。另以预注册 hash 从 8 个污染型聚类各留
1 个完整城市：其 val 子集可参与模型选择，178-case test 子集完全不进入训练或 checkpoint 选择。
它还对各 split 的 case 元数据、全部可见工具输入及 scorer-only truth 做内容聚合 hash；训练后
重算不一致即失败，且全国 RL 路径发现任何 `expert.json` 都拒绝构建。

机制验证可用 `probe_model.py --ablate-evidence synoptic|composition|fires|source_context`
在相同 case/rollout seed 下生成通道遮蔽臂，再由 `compare_evidence_ablation.py` 做 case-first、
issue-date cluster bootstrap；主估计使用 `outcome_composite`，不会把少拿 grounding 分误判为
预报技能下降。正式 probe 与同信息集 tabular 完成后，
`audit_reward_weight_sensitivity.py` 还会对默认、等权和每个结果分量 ±25% 的预注册网格做
case-first/起报日分块比较；它用于发现单一权重驱动的结论，不能替代硬指标。

## 模型选型建议

- **策略模型（被训）：Qwen3 系列。** 先用 1.7B/4B trainable base + LoRA 验证整条管线，
  再转 Qwen3-8B BF16 base + LoRA。当前本地 8B-AWQ 只是冻结评测/数据探针模型，
  不应当作可训权重或训练期的代理 rollout policy。
  快速迭代/管线调试用 Qwen3-4B（甚至 1.7B 冒烟）；算力允许时 Qwen3-14B 做上限探索
  （仓库已有其 SFT 配置）。**不建议 MoE（如 Qwen3-30B-A3B）做 RL**——推理快但训练侧支持不成熟；
  InternLM 生态弱于 Qwen，建议收敛到 Qwen 系。
- **rollout/推理服务**：vLLM 或 SGLang 的 OpenAI 兼容端点，`agents/llm_openai.py` 直接对接。
- **硬件边界（原实验环境）**：RTX 4090 24 GB / WSL2 用于 1.7B/4B LoRA 冒烟与冻结探针；
  8B BF16 + LoRA + rollout 长训目标为云上 ≥80 GB 级 GPU，优先 ≥2×80 GB，先通过
  100-step 策略同步/显存/吞吐冒烟。
  当前本机已迁移至原生 Linux、4×RTX PRO 6000 Blackwell 96 GB，适配结果见上述本机指南；
  单卡验证通过不等同于完成四卡分布式训练验收。
- **离线数据生产 VLM（不在线部署）**：Qwen2.5-VL-72B / GLM-4.5V 级别，批处理会商 slide →
  结构化转写，专家抽检校准；这属于 case 构建管线，不进环境。
- **相似个例检索**：bge-m3 embedding（+faiss），对接 `find_similar_cases` 占位接口。
  当前全国 7,548 个原始 case 均未注册 previous_forecast/analog，因此正式 agent 与同信息集
  tabular 都不消费该通道；未来接入后必须同步扩展基线投影并重做时间门禁。
- **评审判分不用 LLM**：全部规则化——这是本 harness 的立身之本，勿破。

## 全国池（v1，2026-08 设计定稿）

京津冀之上的主数据集：**120 城 × 多污染物 × Qwen3 发布日附近起始的评测窗（2025-04 → 2026-08）**。

- **城市选择**：31 省会强制 + 污染区制 k-means(k=8) 配额，完备性门槛 ≥95%/90%，
  方法学 `docs/CITY_SELECTION.md`（对审稿人可陈述、seed=7 可复现）。
- **多污染物**：horizon=5；六项 IAQI 全表（HJ 633）进 schema，评分等级/事件按全 AQI 口径，
  新增首要污染物分量（权重 0.10）与 `o3_range` 区间；2026-03-01 起自动切换
  HJ 633—2026，更早日期使用 HJ 633—2012。
- **分层抽样**：`scripts/build_frame.py`，真值侧六层
  pm10_primary/switch/event/o3/turning/clean；当前已构建 train 5248 / val 800 / test 1500；
  `pm10_primary` 只表示 PM10 首要污染物，不能直接解释成沙尘事件，
  切分带 6 天缓冲，测试段滚动持出（2026-27 冬季数据到位后做前瞻评测）。
- **数据管线**：观测=总站公开档案（`fetch_aq_obs.py`+`build_daily_all.py`）；
  指导+诊断=CAMS 12UTC 起报中国框服务端裁剪（`fetch_cams.py`，ADS key）；
  北方 68 城另附 cmaq/naqp d02 双源。CAMS 使用前日 12UTC 循环，cycle lead `0--120h`
  实际对应业务起报 `-12--+108h`，第 5 个目标日无 CAMS 指导，靠 agent 结合 GFS/IFS 外推。
- **开放多源证据层**：GFS+IFS 形势、CAMS 组成、全国空间实况、VIIRS 火点、GFS 地形和
  EDGAR 排放先验；时间门禁、数据量、派生和复跑命令见
  [`docs/OPEN_EVIDENCE_DESIGN.md`](docs/OPEN_EVIDENCE_DESIGN.md)。专家语料不进入全国 episode。
  VIIRS 采用官方逐日 NRT 优先、standard 月产品只补无 NRT 档案的传感器—日期；后者显式标为
  retrospective proxy，处理级别随 SHA sidecar 留档，不能冒充当时的原始 NRT feed。
- **历史指导偏差**：主离线实验只从 city-day 清洗后的 train manifest 建索引；val/test 真值不回流。
  按时间滚动吸收业务新真值只能作为独立 online 扩展，不能与冻结 tabular 的主对照混用。
- **构建**：`scripts/build_national_case.py --split train` → `cases/national/{split}/`。
  **当前状态（2026-09-01）**：已建成 7,548 个例（train 5,248 / val 800 / test 1,500），
  六污染物小时实况已补齐 7,548/7,548；高频 GFS/IFS、辐射、CAMS 三通道、火点和静态源
  正按冻结 manifest 补充。最终器只会在原始、派生、附件和时间门禁全部通过后重挂接并
  重建 reward-v0.7.9 基线/探针，旧 evidence attachment 不代表新合同完成。所有正式评分产物
  同时锁定 reward config、`scoring.py` 与 `schema.py` 的内容哈希。
- **评测**：`scripts/eval_national.py --split val` 出分层基线表；
  `scripts/eval_tabular_baseline.py` 是只用可见数值特征的强监督基线：在训练期内部以带
  embargo 的时间折选择 HGB/LightGBM/XGBoost，随后在未参与选型的更晚训练切片校准区间；
  val/test/challenge/OOD 均不参与选型或校准。
  冬季 checkpoint-selection challenge 有 344 case，从 train 按完整起报日划出并做
  `(city, forecast_valid_date)` 零重叠审计；它与暖季 val 一起选 checkpoint，不充当终评。
  空间 OOD 划分另从 8 个污染型聚类各固定留出 1 个非省会优先城市；联合清洗后 train 为
  3,061 case，OOD val/test 为 50/178 case（32/46 个起报日），整城不进入训练。暖季 OOD
  只检验空间泛化，不能替代 2026–27 冬季前瞻事件证据。
  当前审计和 go/no-go 见
  [`docs/RL_READINESS_AUDIT_2026-08-28.md`](docs/RL_READINESS_AUDIT_2026-08-28.md)。
- **模型纪律**：策略模型钉死初版 Qwen3-8B（2025-04-29 发布）。发布后日期只是
  评测污染风险的代理控制，因官方没有披露可审计的预训练 cutoff，不应称为已证明的
  “干净窗口”；换 checkpoint 前必须重做记忆探针。

## Roadmap

- [x] 真实个例构建器（JJJ：eaget d03 + 公开观测，`cases/real/` 1341 个可打分；全国：见上节）。
- [ ] **12 月专家基准**：19 个会商日全部入库后跑 `baselines`，产出"专家基线 vs 模型"评测表
      （目前 31 个 benchmark_dec2025 个例仅 1 个 expert.json；另有 expert 5 天 vs horizon 6 待修）。
- [ ] 相似个例索引（eaget 历史特征 + 向量检索）接通 `find_similar_cases`。
- [x] veRL multi-turn token-preserving rollout 适配器、Dr.GRPO/DAPO 与单卡 LoRA 冒烟配置。
- [ ] 预报订正任务变体（`previous_forecast` 已在环境中，补订正专用评分）。
- [x] 全国单城市 episode 的六污染物 AQI、首要污染物及 PM2.5/PM10/O3 区间合同。
- [ ] 同一 episode 联合预报多个城市（当前一个 episode 对应一个目标城市）。

## 图像层（三层方案，已实现）

原则：**感知离线/按需，决策只见文本**——策略模型永不吃像素；32,768-token rollout
上下文下，preflight 强制单次 prompt ≤24K，为工具调用与最终提交预留约 8K。

1. **确定性色标反演**（`analyze_chart` 工具 / `sitian.imaging.invert`）：产品图按版式裁出主图与色标条，
   从图例自校准调色板 → 逐像素分类 → 类别覆盖/象限分布/中心区统计；`PRODUCT_SCALES` 登记刻度后
   按色带相对位置插值出物理量（已校准并实图验证：boundary_layer、temp_inversion；
   其余产品照 invert.py 顶部注释的方法抄图例登记即可）。不经过任何模型，永远可用。
2. **趋势拼图 + 按需 VLM**（`describe_image` 工具）：同产品多时效图拼成一张证据板（对应业务
   把多图拼上一页 PPT 看趋势的做法），交给临时唤起的 VLM 转写为受控描述。VLM 不训练、不常驻：

   ```bash
   vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001   # 用时唤起，用完 ctrl-C
   export FH_VLM_BASE_URL=http://127.0.0.1:8001/v1 FH_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
   ```

   未配置 VLM 时工具返回 available=false，episode 不受阻塞。不做全量图像预处理。
3. **像素兜底**：个例 meta 带 image_cycle_dir 指向 eaget 原图，schema 有缺口时提取层可加字段重跑，信息永不销毁。

图像工具仅在个例带 `meta.image_cycle_dir` 时注册；"该看哪张图"由 RL 学习（专家 slide_pages 引用为行为种子）。
