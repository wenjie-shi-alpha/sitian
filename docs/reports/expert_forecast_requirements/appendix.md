# 专家空气质量预报：Harness、Reward 与缺口终审

生成时间：2026-09-03T17:17:39Z

用途：JJJ_ATMO 蒸馏语料只用于机制设计参考，不进入全国训练输入、SFT 文本、起报时证据或 reward 真值。词频只表示需求出现下界，不代表重要性排序。

## 一、结论

**当前状态：冻结策略门禁已通过，等待可训练闭环。** 专家语料仅作机制设计参考；全国训练输入仍是起报时合法可得的开放数据。新证据附件覆盖为 100.0%，专家证据合同为 PASS，Reward 可执行合同为 PASS；veRL 原生运行时合同为 PASS；整城空间 OOD 划分为 READY；当前版本 frozen probe 为 50 case × group 8；可训练 LoRA 冒烟为 尚未通过。

Harness 已把初态、天气系统、多层扩散、未来风向对应的起报时上游实况、污染组成、模式订正和过程时序放入可审计工具。reward-v0.7.9 将含 grounding 的训练分与只含预报结果的 outcome composite 分离；论文毕业仍由硬 MAE/CSI/F1/区间指标和分块 CI 决定。

## 二、专家决策链对 Harness 的逐项覆盖

| 序 | 决策任务 | 重要性 | 原始数据 | 工具暴露 | 策略证据 | Reward/测试 | 当前缺口 | 优先级 |
|---|---|---|---|---|---|---|---|---|
| 1 | 判定起报时的污染初态与区域态势 | 通用必需 | 强（逐 case 合同通过） | 强 | 已观测使用（最高 99.2%） | 部分 | process view 已给六污染物初态、趋势和热点；正式策略使用率见本行，结果 reward 不证明因果解释正确 | P1 |
| 2 | 识别高低压、槽脊、冷空气及其演变 | 通用必需 | 强（逐 case 合同通过） | 强 | 已观测使用（最高 99.2%） | 弱 | 6h/12h 轨迹、系统中心和槽脊进入 compact/deep view；原始覆盖和合同已通过，仍需形势证据遮蔽消融 | P0 |
| 3 | 判断水平扩散、上游输送与风向转折 | 通用必需 | 强（逐 case 合同通过） | 较强 | 已观测使用（最高 99.2%） | 弱 | 已有地形自适应多层风转折、随时效来流情景、上游城市和静态源方位；当前仍是目标点平流筛选，不冒充拉格朗日轨迹 | P0 |
| 4 | 判断边界层、逆温、垂直运动与静稳累积 | 通用必需 | 强（逐 case 合同通过） | 强 | 已观测使用（最高 99.2%） | 弱 | compact view 已给 RH925/700、omega700、逆温和逐日 BLH/RH/雨；仍需消融证明策略确实使用 | P1 |
| 5 | 区分 PM2.5、O3、沙尘/PM10 与火点过程机制 | 事件条件必需 | 强（逐 case 合同通过） | 较强 | 已观测使用（最高 99.2%） | 结果较强/机制弱 | 已有短波、AOD组分、气体、沙尘、火点和 PM10 区间；缺可独立标注的机制真值，不能把规则代理冒充专家判断 | P1 |
| 6 | 融合多模式指导并做历史偏差订正 | 通用必需 | 强（逐 case 合同通过） | 强 | 已观测使用（最高 48.8%） | 部分 | guidance-bias 已按时间安全注册；需 frozen/RL 消融证明订正增值而非盲抄 | P1 |
| 7 | 定位污染过程的起始、峰值、清除与转折时刻 | 通用必需 | 强（逐 case 合同通过） | 强 | 已观测使用（最高 99.2%） | 稠密塑形+硬指标 | 已有高频过程视图、daily/process 一致性和硬 CSI；仍需在不依赖互斥采样标签的真值 AQI>=4 事件层，同时证明相对 CAMS 与同信息强 tabular 的训后 event/turning 增值及遮蔽消融 | P0 |
| 8 | 表达情景分支、不确定性与可信区间 | 校准必需 | 强（逐 case 合同通过） | 较强 | 已观测使用（最高 88.0%） | 较强 | PM2.5/PM10/O3 80%区间与硬校准指标已覆盖；categorical confidence 已退出策略输出，仍需验证模式离散度与实际误差的 spread-skill | P1 |
| 9 | 对照上一版预报和历史相似过程 | 校准增强 | 部分 | 弱 | 正式探针未见使用 | 缺失 | 全国 case 当前无 previous_forecast/analog 输入；环境只在数据真实存在时注册工具，相似过程索引仍为 P2 | P2 |
| 10 | 把结论锚定到可核验的证据断言 | 科研可信必需 | 强（逐 case 合同通过） | 强 | 完整 grounding 70.2% | 较强 | ref+JSON Pointer+标量值已抗伪造且需两条不重复断言；仍只核验事实，不把自由文本因果解释当真值 | P1 |

## 三、逐 case 专家证据合同

| 能力 | 通过 case | 覆盖率 | Preflight 必需 |
|---|---|---|---|
| pollution_initial_state | 7057/7057 | 100.0% | 是 |
| synoptic_system_evolution | 7057/7057 | 100.0% | 是 |
| horizontal_transport | 7057/7057 | 100.0% | 是 |
| vertical_dispersion | 7057/7057 | 100.0% | 是 |
| pollutant_mechanism_evidence | 7057/7057 | 100.0% | 是 |
| terrain_adaptive_pressure_levels | 7057/7057 | 100.0% | 是 |
| guidance_and_calibration_inputs | 7057/7057 | 100.0% | 是 |
| process_timing_view | 7057/7057 | 100.0% | 是 |
| forecast_output_contract | 7057/7057 | 100.0% | 是 |
| time_gate | 7057/7057 | 100.0% | 是 |

该合同证明资料在起报时可取、语义分开且工具可暴露；它不证明策略已经使用，也不证明机制因果解释正确。后两项必须由 transcript 审计、遮蔽消融和结果指标验证。

## 四、Reward 与评测尺

- schema-v0.5.6：逐日 AQI 等级、首要污染物、PM2.5/PM10/O3 区间；首污需在 IAQI 尺度可能压过其他区间下界，process 必填且与 AQI≥3 的 daily 头一致。
- reward-v0.7.9 训练 `composite`：结果分量加低权重 grounding；伪造 ref/JSON Pointer/标量值不能得分，两条同类事实不能拿满。自由文本 claim 不作为因果真值。
- 公平比较 `outcome_composite`：只含 level/event/interval/turning/primary，交互 agent、CAMS 和 tabular 使用相同权重分母。
- 非塑形毕业指标：ordinal MAE、硬 CSI/POD/FAR、primary macro-F1、80% coverage/width/interval score；格式失败按保守最差计数进入 case 内平均，再按起报日整块 bootstrap。

### Reward 防投机审计

| 攻击面 | 可能的假增益 | 当前约束 |
|---|---|---|
| 格式/字段投机 | schema 不合法或 daily/process 不一致 | 整条提交 0 分；允许预算内自纠 |
| 多输出头各自投机 | 等级、首污、区间各自合理但联合不可能 | 按 HJ 633 的 IAQI 尺度验证首污可达到等级并压过其他区间下界；process 头必填 |
| 伪造工具返回 | 模型文本冒充 tool result | veRL bridge 只重放 assistant tool call，由隐藏 case 重算结果 |
| 虚构引用 | 不存在 ref/JSON Pointer/值 | 严格解析工具+语义分区+Pointer+标量值，不匹配不给结构化引用分 |
| 重复同一事实 | 换 ref 或 42/42.0 凑满 grounding | 按工具返回内容哈希+JSON Pointer 去重，并要求 2 类证据 |
| 真实但无关的引用 | 机械复制两个正确标量赚 grounding | 仅作为低权重训练辅助且从 outcome 排除；实际证据利用必须通过同 seed 遮蔽剂量反应，当前不声称自动核验决策相关性 |
| 干净日重复送分 | clean 同时拿 event/turning 满分 | 无事件正确否定时相关分量弃权，level 承担评分 |
| 交互模型比较占便宜 | agent 多出 grounding 分 | 训练 composite 与公平 outcome_composite 分离 |
| 超宽区间 | 用无限宽区间追求覆盖 | 按80% interval score连续指数衰减且无保底平台；毕业另看 coverage、width、原始 interval score |
| 成功条件化 | 丢弃格式失败 rollout 后只报成功样本硬指标 | 失败轨迹按最差保守计数进入 case 内平均，再按起报日分块 bootstrap |
| 事件层漏样 | 把互斥采样标签 event 当成全部真实事件 | 事件 probe/比较/门禁由隐藏完整真值 AQI≥4 定义；标签召回率仅作覆盖诊断 |
| 因果套话 | 自由文本机制解释看似合理 | 不作为真值或奖励；保留为独立标注/遮蔽实验缺口 |
| Off-policy rollout | AWQ 采样、BF16/LoRA 更新 | 训练 rollout 由同步后的 trainable veRL/vLLM 策略生成 |
| 工具 token 学习 | 对环境返回反向传播 | veRL response_mask 仅保留 assistant token，并由冒烟审计 |

### 可执行 Reward 不变量

| 可执行 Reward 不变量 | 状态 | 观测 |
|---|---|---|
| perfect_oracle_reaches_one | PASS | {"observed": 1.0} |
| level_error_strictly_reduces_outcome | PASS | {"perfect": 1.0, "degraded": 0.8124} |
| wider_intervals_are_penalized | PASS | {"narrow": 1.0, "wide": 0.0174} |
| maximum_width_intervals_have_no_reward_floor | PASS | {"maximum_width_interval_component": 0.0004, "maximum_width_outcome": 0.3201, "ordinary_wide_outcome": 0.4393} |
| clean_correct_negative_abstains_event_and_turning | PASS | {"event": null, "turning": null} |
| clean_false_alarm_is_penalized | PASS | {"event": 0.0, "clean_outcome": 1.0, "false_alarm_outcome": 0.5} |
| invalid_schema_is_zero | PASS | {"composite": 0.0, "outcome_composite": 0.0} |
| derived_heads_match_truth_for_oracle_intervals | PASS | {"daily": [["2026-06-02", 1, null], ["2026-06-03", 4, "PM2.5"], ["2026-06-04", 5, "PM10"], ["2026-06-05", 2, "PM2.5"]], "process": {"has_event": true, "start": "2026-06-03", "peak": "2026-06-04", "end": "2026-06-04"}} |
| supplied_categorical_heads_cannot_override_intervals | PASS | {"outcome": 1.0, "perfect": 1.0} |
| interval_only_submission_is_complete | PASS | {"outcome": 1.0} |
| derived_primary_is_the_highest_iaqi_midpoint | PASS | {"daily0": {"date": "2026-06-02", "aqi_level": 2, "pm25_range": [36.0, 45.0], "primary_pollutant": "PM10", "o3_range": [20.0, 40.0], "pm10_range": [115.0, 119.0]}} |
| two_facts_two_types_reach_full_grounding | PASS | {"observed": 1.0} |
| grounding_cannot_change_fair_outcome_score | PASS | {"without_tools": 1.0, "with_tools": 1.0} |
| tool_call_spam_without_citations_gets_zero_grounding | PASS | {"grounding": 0.0, "tools_called": 3} |
| extra_invalid_citation_cannot_preserve_full_grounding | PASS | {"clean_grounding": 1.0, "poisoned_grounding": 0.6667, "structured_invalid": 1, "unique_assertions": 2, "scored_items": 3} |
| duplicate_fact_cannot_reach_full_grounding | PASS | {"observed": 0.5} |
| replayed_ref_or_numeric_spelling_cannot_duplicate_fact | PASS | {"grounding": 0.5, "unique_assertions": 1} |
| missing_values_cannot_be_semantic_facts | PASS | {"grounding": 0.0, "semantic_verified": 0} |
| invalid_structured_citations_get_no_type_fallback | PASS | {"grounding": 0.0, "structured_invalid": 2} |
| unstructured_citation_retains_bounded_type_credit | PASS | {"grounding": 0.1, "type_only_items": 1} |
| standard_change_is_explicit | PASS | {"pm25_70_level_2012": 2, "pm25_70_level_2026": 3} |

### Outcome 权重敏感性

| 权重情景 | 策略 | Tabular | 配对差 | 分块 95% CI | 方向 |
|---|---|---|---|---|---|
| default | 0.529 | 0.683 | -0.154 | [-0.195031, -0.116996] | policy_below |
| equal_components | 0.495 | 0.670 | -0.175 | [-0.217886, -0.136207] | policy_below |
| level_minus25pct | 0.515 | 0.671 | -0.157 | [-0.197816, -0.120296] | policy_below |
| level_plus25pct | 0.540 | 0.692 | -0.152 | [-0.192765, -0.114357] | policy_below |
| event_minus25pct | 0.534 | 0.688 | -0.153 | [-0.193951, -0.116468] | policy_below |
| event_plus25pct | 0.525 | 0.679 | -0.154 | [-0.196822, -0.117982] | policy_below |
| interval_minus25pct | 0.529 | 0.678 | -0.148 | [-0.191421, -0.111054] | policy_below |
| interval_plus25pct | 0.529 | 0.688 | -0.159 | [-0.199865, -0.123376] | policy_below |
| turning_minus25pct | 0.534 | 0.698 | -0.163 | [-0.204308, -0.127087] | policy_below |
| turning_plus25pct | 0.524 | 0.669 | -0.145 | [-0.187268, -0.108482] | policy_below |
| primary_minus25pct | 0.532 | 0.680 | -0.148 | [-0.188443, -0.111211] | policy_below |
| primary_plus25pct | 0.527 | 0.686 | -0.159 | [-0.20152, -0.12288] | policy_below |

不变量通过只能证明实现没有上述已知漏洞；权重网格也不能替代硬指标或业务技能评测。

### 输出范围审计

| split | 六类首污真值 membership | 并列首污日 |
|---|---|---|
| train | {"CO": 1, "NO2": 166, "O3": 10063, "PM10": 1213, "PM2.5": 8116} | 96 |
| val | {"O3": 3247, "PM10": 140, "PM2.5": 71} | 4 |
| test | {"O3": 5451, "PM10": 273, "PM2.5": 16} | 4 |

AQI level and six-class primary pollutant are scored for all pollutants; calibrated concentration intervals are currently requested only for the three dominant pollutants. NO2/CO primary memberships occur only in train and SO2 is absent in the current valid pool, so this is an explicit output-scope limitation rather than an unreported validation advantage.

## 五、当前同信息集 tabular

| split | n | outcome composite | macro-stratum | ordinal MAE | hard event CSI |
|---|---|---|---|---|---|
| val | 781 | 0.678 | 0.696 | 0.480 | 0.000 |
| challenge_winter | 344 | 0.553 | 0.566 | 0.691 | 0.000 |
| test | 1472 | 0.754 | 0.735 | 0.366 | 0.000 |
| spatial_ood_val | 50 | 0.620 | 0.640 | 0.540 | 0.000 |
| spatial_ood_test | 178 | 0.676 | 0.726 | 0.440 | 0.000 |

## 六、Preflight 与 Graduation

### 真值事件总体审计

| split | 真值 AQI≥4 case | 按互斥采样层分布 | event 标签召回率 |
|---|---|---|---|
| train | 894 | {'event': 779, 'switch': 41, 'pm10_primary': 74} | 87.1% |
| val | 228 | {'pm10_primary': 14, 'event': 213, 'switch': 1} | 93.4% |
| test | 27 | {'pm10_primary': 15, 'event': 12} | 44.4% |

互斥采样标签不等于事件真值；事件 probe、硬指标和毕业比较只用隐藏完整真值 AQI≥4。

### 两层门禁

| Preflight 门禁 | 状态 |
|---|---|
| 数据、Reward 与专家证据合同 | PASS |
| 格式与轨迹 | PASS |
| 上下文预算 | PASS |
| 采样效率 | PASS |
| 结构化证据引用 | PASS |
| 可训练栈与硬件冒烟 | BLOCKED |

Preflight 不要求冻结模型先超过 CAMS。Graduation 才要求训后独立 val 超 CAMS、至少100 个 event case 的硬 CSI 与 outcome composite 均有正的分块 CI 下界、winter challenge 不退化；论文阶段还要求同一隐藏事件层的 composite、硬 CSI 与峰值时序 MAE 超过强 tabular，不能把‘事件超 CAMS’和‘总体超 tabular’拼成机制结论；同时要求 outcome 权重网格稳定、遮蔽 synoptic+composition 后出现正的证据剂量反应。

## 七、明确保留的研究缺口

- 未完成：独立污染机制标签（不能用规则代理冒充真值）
- 未完成：严格时间安全的全国历史相似过程索引
- 未完成：全国上一版业务预报档案
- 未完成：拉格朗日输送轨迹（当前仅目标点风向/距离平流筛选）
- 未完成：逐时排放变化与临时管控信息
- 未完成：全国地面颗粒物组分观测
- 未完成：超出确定性 GFS/IFS 分歧的集合预报离散度
- 未完成：SO2/NO2/CO 概率区间输出头
- 未完成：自由文本因果解释的独立核验真值
- 未完成：已核验事实与具体预报结论之间的决策相关性真值
- 未完成：每个火点传感器日均有归档 NRT（当前允许显式 retrospective fallback）
- 未完成：训后策略的独立区间校准
- 未完成：训后策略的整城空间 OOD 检验
- 未完成：2026–27 冬季完全前瞻证据
- 已完成：八聚类各留一整城的空间 OOD 划分；未完成：训后策略在该 test 上相对强 tabular 的分块检验。
- 未完成：trainable policy 50 step、当前策略同步与 checkpoint reload 的真实冒烟。
- 数据齐全不等于 RL 可学；正式 50×8 的格式、grounding、上下文与有效 group 门禁必须独立通过。
- 已完成：锁定 veRL commit 的解析配置与 10 工具原生 create/execute/release 契约验证。

可训练冒烟检查：

可训练冒烟产物尚未生成。

## 八、剩余建设清单

| 顺序 | 优先级 | 状态 | 工作 | 交付物 | 成功门禁 |
|---|---|---|---|---|---|
| 1 | P0 | 已完成 | 完成 open-evidence-v1 高频合同与逐 case 重挂接 | GFS/IFS 6h→72h/12h→132h、辐射、CAMS 三通道、火点、静态源全部派生并严格附着 | 原始/派生/attachment 均 100%，六污染物与时间门禁失败为 0 |
| 2 | P0 | 已完成 | 冻结 v0.7.9 正式探针与格式/grounding 裁决 | 至少 50 case×group8，记录首次合法、两条语义断言、过程工具使用和组内方差 | 首次合法>=95%，full-grounding rollout>=50%，总 reward 有效 group>=70%；outcome 与 event 决策有效 group 分别>=50% |
| 3 | P0 | 已完成 | 完成同信息集强基线与非塑形毕业指标 | 重训 tabular；另对八个整城留出的空间 OOD val/test 报告同一套硬指标 | reward/manifest 一致；空间 OOD test 至少100 case、30个起报日块且所有主要结论有硬指标和分块CI |
| 4 | P0 | 待门禁后执行 | 完成 trainable 1.7B/4B LoRA 冒烟 | 当前策略同步 rollout、assistant-only mask、void-turn 率门禁、Dr.GRPO、checkpoint reload | 50–200 step finite loss；同步与重载前后行为变化可复核 |
| 5 | P1 | 研究阶段待完成 | 验证污染类型条件路由和证据剂量反应 | PM2.5 高湿二次、O3 光化学、沙尘/PM10、火点烟霾分别遮蔽关键通道 | 按过程类型的遮蔽实验呈剂量反应，而不是所有 case 无差别塞满工具输出 |
| 6 | P1 | 研究阶段待完成 | 先固定 compact 过程证据学习决策，再释放工具选择 | 单轮/受控多轮课程，工具 token mask、void-turn 率门禁、Dr.GRPO、动态样本过滤 | 小模型 smoke 闭环通过，event/turning 有有效样本率，工具调用不发生 lazy collapse |
| 7 | P1 | 已完成 | 预注册 reward 权重敏感性与 Pareto 稳定性分析 | 在不改组件定义的前提下扫描合理权重网格，同时报告各硬指标、策略排序与事件层退化 | 主要结论不依赖单一权重组合；任何权重选择均不能掩盖 MAE、事件命中或校准恶化 |
| 8 | P2 | 研究阶段待完成 | 建设时间安全的相似过程检索 | 以起报时可见特征检索历史个例，返回相似度、可用截止和结果摘要 | 严格时间门禁、城市/日期去重，并通过 analog 遮蔽消融证明净增值 |

## 九、来源与可追溯性

| 来源 | 路径 | 存在 | SHA256 |
|---|---|---|---|
| expert_analysis | data/interim/efr.json | True | 41454580601b3ace9b19ed59ffc4f0fa1c463a641f511cb8b52a6a1ba9d70132 |
| coverage | data/interim/open_evidence_coverage.json | True | d03e7e4ceabec504186903b0560fa5fce43f9c9197fed9f586d9eea07395277f |
| expert_contract | data/interim/expert_evidence_contract_audit.json | True | 8d466cf8f0622652cd8034f0493437cc701896356dd17ede6ea8149f6c2a9ef4 |
| spatial_ood_audit | data/interim/spatial_ood_audit.json | True | 9a3ab91b3bf3da174e6e62ebef2f5f44921bcc61df35cab29a5358c9fa8d888e |
| probe | data/interim/probe_open_evidence_v079_n50_g8.json | True | 0e97ada4eeed31c8bb7f0d796c96020cba3b6c19c051b6fd30dc2f9c4da3e51b |
| tabular | data/interim/eval_tabular_open_evidence_v079.json | True | 414226e2ca0161a897a7249eb97955ca76e889ce7db5b35a2f2ba3c4557203a0 |
| preflight | data/interim/rl_preflight.json | True | edf3286006c1554a86d2e6f5a3197ea694b562848f0c27d008fe861d918d1e8f |
| training_smoke | data/interim/training_smoke.json | False | — |
| reward_contract | data/interim/reward_contract_audit.json | True | 2738bbd105d1b26657196ef973561224753f29239c9b1f3fd018c4daa6b0dae5 |
| reward_sensitivity | data/interim/reward_weight_sensitivity_v079.json | True | 1e67c55eeb85dd59778deff7626edfbe423f57212a16271c719529189de69f7b |
| verl_runtime_contract | data/interim/verl_runtime_contract.json | True | 6d9e319f2eeab9f5ed0398a088474cc6d6c5dcc4302a56a4887ce23ad1f2dba9 |
| guidance_comparison | data/interim/compare_v079_frozen_vs_guidance.json | True | e43078a56a60ebc18e79fa7cfe31c3f6a6b6a3c5cc635987ecacbe6447a7eb46 |
| tabular_comparison | data/interim/compare_v079_frozen_vs_tabular.json | True | 596627e316bdb10aed7b5d28756a59f114bbb7432d526b18714bd9ea0799b23c |
