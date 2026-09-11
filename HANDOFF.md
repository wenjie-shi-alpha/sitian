# 数据与 Harness 交接（2026-09-12）

城市数据和准入交付已回到训练机，原始大文件继续留在9800X3D。训练保持暂停。

当前为Harness v2.2、reward v0.8.3。按用户接受的显式供数延迟假设重建7057个候选案例、固定切分、历史索引和Parquet。历史实际发布时间仍未核验。

详细范围、限制、检查结果与复现步骤见[离线训练准入说明](docs/TRAINING_READY_2026-09-12.md)。本机最终资产：

- 案例：`data/snapshots/offline_native_v3_20260912_r2/`
- 准入与训练资产：`data/training_ready/offline_native_v3_20260912_r3/`
- 机器可读结果：上述准入目录的`certificate.json`

只检查，不启动：

```bash
.venv/bin/python scripts/launch_offline_training.py \
  --bundle data/training_ready/offline_native_v3_20260912_r3 --check-only
```

以后开始训练使用此新入口，必须从Qwen3-8B基座开始，不能恢复旧数据协议的checkpoint；旧训练链入口已阻止误续跑。

新工具提供原生气象时序及通风代理计算。历史类比使用真实训练池并执行完整时段真值门禁；完整证据可按时次、子树或分页查询。缺测、供数估计、代理指标均显式标记。

仍缺降水累计区间语义、更密垂直温度廓线、严格按6小时延迟筛选的原生NRT火点、区域模式单位说明及专家方法来源/日期。当前问题火点汇总已屏蔽，降水日量不伪造，专家文本未冒充已核验方法卡。这些限制写入离线协议。

同步代码即可继续修改。数据、权重、证书不上传Git；原始NetCDF/GRIB无需传到训练机。换机器或修改输入/实现后，运行文档中的复现与准入命令，避免复用过期证书。
