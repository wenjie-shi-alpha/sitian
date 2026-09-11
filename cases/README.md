# Case bundle 契约

一个个例（case）= 一个起报时次的目录：

```
cases/<split>/<case_id>/
  case.json               必需。{case_id, issue_date, region, horizon, regions, meta}
  observations.json       必需。起报前实况（agent 可见）
  diagnostics.json        必需。逐日诊断特征（agent 可见；未来日为 NWP 派生，属合法指导）
  guidance.json           必需。模式指导（agent 可见）
  evidence.json           全国合同必需。高频形势、组成、空间实况、火点、地形/排放及 provenance
  previous_forecast.json  可选。昨日发布的预报对象（agent 可见，订正任务用）
  truth.json              评分需要。未来逐日观测真值（工具永不暴露）
  expert.json             可选。专家预报+证据类型（评分参照与基线行，工具永不暴露）
```

## 字段格式

- `observations.json`：`{污染物: {"unit": "µg/m³|mg/m³", "times": ["YYYY-MM-DDTHH:00", ...], "series": {区域: [数值...]}}}`；全国多污染物 case 保留 PM2.5/PM10/O3/SO2/NO2/CO 六项起报前小时实况，CO 单位为 mg/m³，其余为 µg/m³。
  08:00 起报时仅允许严格早于 08:00 的记录（整点小时数据截止 07:00；`CaseBundle.audit_time_gate()` 机检），为实时监测数据发布时滞保留一小时安全窗。
- `diagnostics.json`：`{"daily": {日期: {synoptic, wind_dir, wind_speed_ms, inversion, blh_m, rh_pct, precip_mm, transport}}}`
  字段集随"论据覆盖率审计"演进（专家论据必须可被字段表达）。
- `guidance.json`：`{"sources": {源名: {"daily_pm25": {日期: 数值}, "note": ...}}}`
- `truth.json`：旧池可为 `{"daily": {日期: {"pm25_avg": float}}}`；全国池为六污染物日值，
  必须覆盖 issue+1..issue+horizon，且永不进入 agent 工具返回。
- `evidence.json`：由共享的 issue-date 原始场确定性裁切；含每条来源的 cycle、valid/available
  time、hash/sidecar。`get_process_evidence` 在运行时从中生成紧凑过程视图，不保存未来真值。
- `expert.json`：`{"forecast": <预报对象>, "evidence_types": [受控词表], "notes": str}`

## 目录规划

- `synthetic/`：合成剧本（`cli gen` 生成），训练管线冒烟与课程前段。
- `real/`：京津冀真实个例（已建成 1341 可打分，`scripts/build_real_case.py`）。
  命名 `{region}_{issue_date}`，如 `beijing_2025-12-19`。
- `national/{train,val,test}/`：全国 120 城多污染物个例（`scripts/build_national_case.py`）。
  horizon=5；truth 含六项日值（pm25_avg/pm10_avg/o3_8h/so2_avg/no2_avg/co_avg），评分走全 AQI；
  meta 携带 stratum/cluster/split/multi_pollutant。传统 CAMS 数值指导末日可能需要持平外推；
  高频 GFS/IFS 与开放证据覆盖到业务起报 `+132h`；CAMS 组成轨迹合法覆盖到 `+108h`。
  schema-v0.5.6 要求每天同时提交 PM2.5/PM10/O3 80% 区间，并保证等级、首污、区间与过程头联合自洽；未校准的 categorical confidence 不再进入策略输出。
- 切分规则：**按污染过程切 train/val/test，同一过程的多个起报日不得跨集**；
  评测采用滚动盲评（最新时段永远持出）。

## 旧京津冀真实池的数据来源（历史构建器）

| 文件 | 来源 |
|---|---|
| observations | 会商 slide 实况表格抽取（已在 runs/1219_structured_agent 验证）→ 长期切公开观测数据源 |
| diagnostics | eaget 图像特征管线（temp_inversion/boundary_layer/rain/10m_wind/高度层 → daily_matrix） |
| guidance | eaget 模式产品 + 会商 slide 中的模式预报页 |
| truth | 公开空气质量观测按日聚合（次日即可回填） |
| expert | 12 月会商语料蒸馏（预报结论 → 结构化预报对象；证据引用 → evidence_types） |

全国池不读取上述专家文件；JJJ_ATMO 蒸馏语料只用于设计工具、过程合同和审计清单。
