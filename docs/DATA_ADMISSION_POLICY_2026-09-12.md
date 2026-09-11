# 数据排除与发布时间分层处理（2026-09-12）

本轮已完成候选池筛选和固定切分复核，结果写入独立目录 `data/admission_20260912/`。这是接收端重建的候选输入，**不是启动训练的命令或训练就绪证明**。

## 已经处理

- 从8,300个请求排除33个缺开放证据起报日的752个案例，以及原491个真值不合格案例，去重后共排除1,243个。
- 得到7,057个候选：原train 4,804、val 781、test 1,472。新发现的76个真值不合格、27个输入不足都包含在752个缺证据组中，不能再重复扣除。
- 原冻结RL train仍为3,061；val 781、winter selection 344、test 1,472；spatial OOD val/test仍为原成员。没有重抽样、让冬季选择案例回流训练或改变留出城市。
- 训练与val/winter selection/test的目标城市日重叠均为0，训练中没有留出城市。源清单SHA复核未变。
- 区域CMAQ/NAQP设为可选补充：缺失明确标记为缺失，不填零、不删掉整个有效case。同一case在冻结模型、RL和其他对照中必须使用同一来源集合。现有参考池804例包含CMAQ/NAQP，其余候选不因缺可选源被排除。
- 真值有效性和原构建输入门槛保持原要求。候选资格并不表示每种污染物最近24小时均无缺测，近期缺测仍由质量工具暴露。


工具实测完成：7,057例共98,798次调用，0失败。仍有1,793例被标记最近24小时观测不完整；这些质量警告保留，并非被悄悄清零或等同所有输入均不合格。原有D5和O3口径警告也未隐藏。

## 发布时间如何处理

分清三种情况：未记录、按规则估计、声明了时刻但没有独立核验。一个`available_at`字段或`available_at_verified=true`自声明不能升级为独立的发布凭据。未来/等于起报时刻的时间、无时区时间仍明确报错。

[CAMS官方实时供数说明](https://confluence.ecmwf.int/spaces/CKB/pages/116952716/SFTP-FTP-HTTPS%2Bdata%2Baccess%2Bto%2BCAMS%2Bglobal%2Bdata)给出了正常供数时间，12UTC循环通常在22UTC可用；并说明整个循环文件上传完成后会出现manifest。这个文档能支持正常计划，却不能证明某个历史文件实际何时到达。

[ADS产品文档](https://confluence.ecmwf.int/plugins/viewsource/viewpagesrc.action?pageId=514323232)明确提示服务可能延迟。[IFS开放数据说明](https://www.ecmwf.int/en/forecasts/datasets/open-data)也区分分发计划和滚动存档。当前文档并非覆盖全部2025—2026历史运行的逐文件日志。

因此，本轮保留既有CAMS/GFS/IFS的10/5/8小时延迟作为**显式旧假设**，没有写成实际发布事实；没有改写case中的未知时间，也没有根据文件修改时间填补它们。候选池的常规时间门禁通过仍仅说明在记录的时刻/假设下未越过起报边界。

`publication_audit.json` 记录51,007个未记录发布时间块、613,959个旧估计时间块。这些是嵌套记录数，不是文件数或独立案例数。当前脚本没有独立发布事件证明，因此`strict_publication_certified_cases.json`为空，含义是未签发严格发布时间证明，**不是证明这些案例当时全部不可用**。

探索性历史实验可以在预先声明的可用性假设下准备；如果要宣称每个输入确实在业务起报前已经发布，则需要补对应历史发布/到达凭据，或另立可验证的实验协议。不能通过修改标签让严格验收虚假通过。

## 产物与使用

- `candidate_pool.json`：7,057个参考case路径，供接收端限制新重建的案例范围。
- `candidate_splits/`：原冻结切分的候选清单，成员保持一致。
- `excluded_cases.json`：1,243个排除案例及原因，多项原因保留但人数去重。
- `candidate_audit.jsonl.gz`：逐case门禁、可选区域源和发布时间状态。
- `policy.json`、`publication_source_notes.json`：数据规则、时间假设和官方依据。
- `summary.json`：数据规模、切分隔离及源身份检查。
- `visible_tool_audit.json`：参考候选池的可见输入工具实测结果；不代替原生训练运行验收。

这些清单不把旧case自动当作最终新数据版本。接收端仍应使用已回传的原生数据和修复代码统一重建case，再建历史/偏差索引、Parquet与数据身份，并按新实验协议验收。D5缺指导和O3瞬时/八小时指标差异仍需保留可见标记。原训练暂停要求继续有效，未续跑旧checkpoint。

## 复跑

```bash
cd /root/sitian
python3 scripts/build_candidate_admission.py   --handoff data/local_handoff_20260912   --out data/admission_new
python3 scripts/audit_harness_inputs.py   --manifests data/admission_new/candidate_pool.json   --out data/admission_new/visible_tool_audit.json
```

两个入口都拒绝覆盖已有输出。策略测试涵盖缺少质量证据时拒绝、可选模式缺失不误删、多个排除原因、估计/自声明不伪装成核验，以及无时区或越过起报时间的拒绝。
