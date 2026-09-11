# 全国开放证据层设计（open-evidence-v1）

## 1. 科学命题与边界

项目要检验的不是“LLM 是否比一个回归器更会拟合同一张表”，而是：在起报时刻合法可得的
环流、稳定度、输送、污染组成、空间实况、火点、地形和排放先验共同给定时，agent 能否学会
按天气过程组织证据并订正数值指导。JJJ_ATMO 会商语料只用于设计这套机制，不作为全国 case
的输入、SFT 文本或 reward 真值。

因此证据层遵守四条硬边界：

1. 未来信息只能来自起报前已发布的预报，不能来自未来分析或观测；
2. 原始场只保存一份，case 仅挂目标城市和相关区域信号的确定性切片；
3. 派生器只计算可复算事实，不输出“应报几级”等预报结论；
4. 同一数值摘要同时进入强 tabular 基线，agent 不能靠独占信息赢。

## 2. 证据清单与时间语义

| 层 | 开放源 | 起报日 08:00 BJT 可见内容 | 固定时间门禁/限制 |
|---|---|---|---|
| 天气形势 | NOAA GFS 0.25° | `t-24/t-12` 分析；`0--72h` 每 6h、`84--132h` 每 12h | 最新合法循环 `t-6h`，发布滞后按 5h |
| 天气形势 | ECMWF IFS Open Data 0.25° | 与 GFS 相同的共同有效时刻 | 最新合法循环 `t-12h`，发布滞后按 8h |
| 污染组成 | CAMS global composition / Earth Engine mirror | AOD/组分 5 个逐日时点；近地层和总柱 CO/NO2/SO2 21 个 6h 轨迹点 | 前日 12UTC 循环，cycle lead `0--120h` 对应业务时点 `-12--+108h`；两种垂直语义分开 |
| 空间实况 | 全国城市公开小时档案 | 120 城 PM2.5/PM10/O3/SO2/NO2/CO 24h 状态与趋势 | 最晚只用起报日 07:00 BJT，留 1h 安全边界 |
| 生物质燃烧 | NASA/UMD VIIRS C2 | 起报前 72h、考虑 6h 可用滞后的火点/FRP | 官方逐日 NRT 优先；仅在传感器—日期无 NRT 档案时使用 standard 回溯代理，并显式标注 |
| 地形 | NOAA GFS surface HGT | 0.25° 城市高程、盆地指标、八方位屏障 | 静态模式地形，不冒充城市尺度 DEM |
| 人为排放 | JRC EDGAR HTAP v3.2 | 9 类污染物、部门占比、半径/方位源强 | 2020 年静态先验，不冒充 2025–26 排放实况 |

GFS/IFS 每个形势快照只抽取 21 个 GRIB message：500 hPa 位势高度，925/850/700/500/200 hPa
风场，925/850/700 hPa 温湿，850/700/500 hPa 垂直速度和海平面气压。未来有效时刻合法的
依据是 `available_at <= issue_time`，而不是 `valid_time <= issue_time`。

地表向下短波辐射另存为单 message sidecar，避免为增加一个字段而重写既有形势 GRIB。GFS
`DSWRF` 是区间平均通量；IFS `SSRD` 是从循环起累计的能量。派生器先对 IFS 相邻六小时时刻做差，
再除以时段秒数，最终两源统一为上一时段平均 `W m-2`。这个量用于表示 O3 光化学生成条件，
不能从总云量线性插值替代。

开放来源：

- [NOAA GFS public archive](https://registry.opendata.aws/noaa-gfs-bdp-pds/)
- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [Copernicus CAMS global forecasts](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts)
- [Google Earth Engine ECMWF/CAMS/NRT mirror](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_CAMS_NRT)
- [NASA FIRMS active-fire data](https://firms.modaps.eosdis.nasa.gov/active_fire/) 与
  [LANCE FIRMS daily NRT archive](https://nrt3.modaps.eosdis.nasa.gov/archive/FIRMS/)
- [JRC EDGAR v8.1 / HTAP datasets](https://edgar.jrc.ec.europa.eu/emissions_data_and_maps)

## 3. 确定性派生

- `derive_open_synoptic.py`：城市各层风温湿/垂直速度、逆温信号，高低压中心、500 hPa
  槽脊信号和 GFS/IFS 分歧；不生成天气结论。
- `derive_open_composition.py`：总/细模态/沙尘/有机物/黑碳/硫酸盐/硝酸盐 AOD 与比例，
  区域源区信号，CO/NO2/SO2 近地层近似 ppbv，以及独立标识的总柱质量轨迹。总柱字段不得
  填补或冒充最低模型层缺值。
- `derive_spatial_observations.py`：起报前一小时截止的 120 城空间污染场。
- `derive_open_fires.py`：合法可见火点、FRP、目标城市不同半径统计和区域聚类。
- `derive_static_context.py`：地形屏障、盆地指标，以及 EDGAR 半径/部门/八方位排放先验。
- `attach_open_evidence.py`：把共享证据裁成目标城市 case；根据 GFS 模式地形，从
  925/850/700/500 hPa 中选择至少高于地形 150 m 的最低压力面来流筛选上风向城市，
  并把地形以下压力面字段显式置空（例如拉萨使用 500 hPa、西宁使用 700 hPa，而不是错误地
  使用 925 hPa），
  同时从起报时刻 120 城实况建立方向均衡的候选池，再按未来 6/12h 地形自适应压力层风向生成
  `forecast_inflow_scenarios`，并给出目标点风速下的理想化平流可达距离和候选城市旅行时间；
  这只是方向—速度—距离筛选，明确不冒充拉格朗日轨迹或源贡献。上游污染值始终取起报时已见
  实况，不读取未来观测。每次写入前运行
  `CaseBundle.audit_time_gate()`。

agent 先通过 `get_process_evidence` 读取约 6–12h 分辨率的初态—形势—扩散—输送—指导总览；
其中 `terrain_adaptive_low_level` 固化记录地形高程、所选压力层、层高净空、风/RH/omega 与
逆温可用性，防止全国尺度统一使用 925 hPa 造成高原城市地下层伪信号；若高原上空没有有效的
相邻温度层对，通风诊断标为 `partial_profile`，不能把“逆温未知”当作“无逆温”。
再用 `get_synoptic_evidence` 和 `get_pollution_evidence` 按需深挖。32k 本地模型默认使用
`detail=summary`：无显式 source 时返回 GFS 高频轨迹和 GFS/IFS 分歧（包括辐射分歧），IFS 轨迹可单独查询；污染工具
保留目标城市 CO/NO2/SO2 的完整 21 点六小时轨迹，同时压缩 AOD 区域源、上风向/区域候选和
目标城市静态先验。轨迹字段同时给出模式 `lead_hour=0--120` 和
`issue_relative_hour=-12--+108`，禁止把两者混称为业务起报后的 120 小时。
`detail=full` 保留全部格点派生摘要，用于审计与消融。六小时气体轨迹是起报时已经发布的
预报（其中首三个点描述业务起报前至起报时的模式状态），不是未来实况；同一 21x3 轨迹同步进入
tabular 基线。

## 4. 数据量与存储

`hybrid-6h72-12h132-v2` 时间契约下，401 个唯一起报日的清单包含 16,040 个形势快照，
另有 18,847 个六小时单字段辐射 sidecar（IFS 多一个起报前锚点用于累计量差分）。相对原来的
逐日 6,416 个快照，增量是 9,624 个形势文件和 18,847 个很小的辐射文件。按实测文件大小估算：

- 既有 NWP 子集约 108 GB；新增形势约 163 GB、辐射约 21 GB，最终 NWP 约 291 GB；
- CAMS ADS 成分约 3–6 GB，Earth Engine 总柱轨迹实测约 2.6 GB；
- VIIRS standard 月文件加 NRT 缺口约 3–4 GB（当前逐日 NRT 清单 564 文件、2.39 GB）；
- 静态原始/解压文件小于 0.3 GB；
- 派生文件和 case 切片约 1–3 GB。

最终全证据保守总预算约 **306–331 GB**，相对原逐日合同的增量约 **184 GB**；v2 相对已下载的
v1 高频合同仅再增加约 **15 GB**。原始与派生共享数据位于环境变量
`SITIAN_EVIDENCE_ROOT` 指向的目录（默认 `/mnt/eaget/sitian_open_evidence/`），不复制到仓库或
逐城市 case。原始盘只读恢复时，可在本机建立由符号链接组成的可写组合工作区并把该变量指向它；
派生和 case attachment 写本机工作区，原始盘保持只读。

## 5. 可复跑流程

```bash
pip install -e '.[evidence]'
export SITIAN_EVIDENCE_ROOT=/path/to/sitian_open_evidence
PYTHONPATH=src python3 scripts/build_open_evidence_manifest.py
PYTHONPATH=src python3 scripts/fetch_open_nwp.py --source gfs --product both --workers 8 --verify
PYTHONPATH=src python3 scripts/fetch_open_nwp.py --source ifs --product both --workers 8 --verify
for i in 0 1 2 3; do
  PYTHONPATH=src python3 scripts/fetch_open_cams.py --kind aerosol \
    --aerosol-chunk-days 32 --shard-count 4 --shard-index "$i" &
done
for i in 0 1 2 3; do
  PYTHONPATH=src python3 scripts/fetch_open_cams.py --kind trace_gases \
    --trace-gas-chunk-days 8 --shard-count 4 --shard-index "$i" &
done
wait
PYTHONPATH=src python3 scripts/fetch_cams_earthengine.py --shard-count 3 --shard-index 0
PYTHONPATH=src python3 scripts/fetch_cams_earthengine.py --shard-count 3 --shard-index 1
PYTHONPATH=src python3 scripts/fetch_cams_earthengine.py --shard-count 3 --shard-index 2
PYTHONPATH=src python3 scripts/fetch_open_firms.py --sensor both --processing both --workers 3
PYTHONPATH=src python3 scripts/fetch_open_static.py --kind both

PYTHONPATH=src python3 scripts/derive_open_synoptic.py
PYTHONPATH=src python3 scripts/derive_open_composition.py
PYTHONPATH=src python3 scripts/derive_spatial_observations.py
PYTHONPATH=src python3 scripts/derive_open_fires.py
PYTHONPATH=src /root/Tethys/.venv/bin/python scripts/derive_static_context.py
PYTHONPATH=src python3 scripts/attach_open_evidence.py
PYTHONPATH=src python3 scripts/audit_open_evidence_coverage.py
```

形势派生使用 `pygrib` 直接读取下载器固化的 21-message 紧凑 GRIB；输出与旧
`cfgrib/xarray` 路径逐字段一致，但避免重复索引和选择开销。机械盘批处理可用
`--num-shards N --shard-index I` 划分连续日期区间；并发数必须由实测吞吐决定，过多随机读会变慢。

长下载在 tmux 中运行时，可用 `scripts/finalize_open_evidence.py` 等待 NWP、CAMS、Earth Engine、
FIRMS 和静态源会话结束；它会对缺口最多重试三轮，再依次完成 401 日派生、覆盖门禁、7,548 个
case 的时间审计挂载、专家证据合同审计、reward-v0.7.9 同信息集 tabular，以及 50×8 frozen
probe 和聚类比较。任何下载、覆盖或专家证据门禁失败都会停止，不会把半成品静默挂到 case。

GFS 与 IFS 位于不同对象存储，生产下载应拆成两条各 8 worker 的会话，避免清单顺序让 IFS 等到
GFS 全部结束；finalizer 的补漏轮使用 12 worker。当前 CAMS 实测采用气溶胶四分片与气体四分片，
使单个慢对象不阻塞整类数据；若 ADS 返回队列限制，下载器会自动长退避，不应继续提高分片数。下载器通过异步任务接口记录每个
request id，单任务超过 240 分钟会主动取消并退避重投，避免把仍在健康生成的两小时任务过早
送回队尾。默认仍保留
气体 cycle lead `0--120h` 的全部 21 个六小时时效，提速不能靠减少 agent 可用证据实现；派生时
显式换算为业务相对 `-12--+108h`。

Earth Engine 通道按 issue date 的固定清单分片，不随已完成覆盖重新排片；每个文件保存同一起报的
21 x 3 总柱轨迹为原生 0.4° GeoTIFF，并记录 issue date、前日 12UTC cycle、available_at、波段语义、
字节数和 SHA256。2026-08-30 单日起报实测为 6.44 MB/7.48 s（0.86 MB/s），约为当时 ADS
对象下载速度的 20 倍。它用于解除 harness 构建阻塞并增加柱总量/输送证据，ADS 最低层下载仍在
后台继续；二者是互补通道，不是跨日期的替代填充值。

结果文件先由 `aria2c` 以 4 条 HTTP Range 连接下载到本机持久缓存
`~/.cache/sitian/cams_stage`，完成大小校验和
NetCDF 解包后，再通过跨进程文件锁顺序写入外置盘并原子改名；SHA256 在这次顺序复制中计算。
这样既能断点续传同一个 ADS request id，也避免多个慢速网络流直接并发写外置盘。`.request.json`
保存未完成任务的 request id 与请求摘要，进程重启时会续用同一服务端结果，不会重复生成任务。

2026-08-30 对同一参考日、同一空间框、同一 3 变量 x 21 时效请求做了并行格式试验：ADS 返回的
GRIB 为 4.0 MB，NetCDF zip 为 4.63 MB，两者服务端完成和下载墙钟时间近似。瓶颈主要在 ADS 调度与
链路而非 NetCDF 转换，因此生产链继续只用 NetCDF，避免为了约 18% 的传输体积引入第二套派生口径。

下载器均可重复运行：完整文件按 sidecar 字节数/SHA 跳过，NWP 用原子临时文件，FIRMS standard
用 SFTP `reget`、NRT 用 HTTPS range 续传。manifest、每个原始文件的来源 URL、处理级别、SHA256、
字节数和 contract version 均留档。NRT 日文件没有 standard 的 `Type` 字段，因此仅保留
nominal/high-confidence 检测并标为 active-fire candidate；同一传感器—日期同时存在两类文件时，
派生器使用当时可获得的 NRT，忽略该日 standard 行。standard 只补没有历史 NRT 档案的传感器—日期，
并始终标为 retrospective proxy，不能冒充当时的原始 NRT feed。
覆盖审计持续写入 `manifests/open_evidence_v1_coverage.json`；只有 NWP、四类逐日起报派生和
ADS model-level-137 气体轨迹语义审计均为 100%、静态上下文存在时，
`ready_for_full_attachment` 才会变为 true。气体语义审计逐日起报核对 cycle lead `0--120h` 的
21 个六小时时效、对应的业务相对 `-12--+108h`、cycle/valid_time/available_at、三气体非负
有限值以及原始文件 provenance。

## 6. 公平比较与实验责任

`eval_tabular_baseline.py` 的固定宽度投影读取相同 case evidence，包括天气层结/系统位置、模式分歧、
AOD/气体组成、空间实况、火点、地形和排放；CAMS/CMAQ/NAQP 的逐 lead 指导、多源离散度以及
`get_guidance_bias` 可见的同一套时间安全聚合也全部进入投影。投影显式铺平 20 时点 GFS/IFS、
5 时点 AOD、两类 21 时点气体轨迹、六污染物 48h 原始实况、五个目标槽的指导/偏差/地面诊断
投影（CAMS 实有前 4 天，第 5 天缺测状态显式保留）与
最多 64 个方向均衡候选池，而不是每个 lead 只取一张天气图或一天指导。
特征维数由每次正式构建产物记录并校验全行一致；缺证据时用 NaN，不允许从真值或 stratum
补特征。

最终机制证据必须来自预注册消融，而不是只看 composite：

1. 完整证据 agent vs 同信息集 tabular；
2. 去掉形势、组成、空间上游或静态源先验的 dose-response；
3. event/turning、留城市/留区制、冬季前瞻集分层；
4. case 内先平均 rollout，再按起报日/天气过程分块 bootstrap。

历史小探针只用于发现 harness 缺陷，不能做效果结论。reward-v0.6.0 的 2-case 引用探针达到
2/2 首次合法、process 工具使用 100%，结构化标量引用可核验 6/7，但 full-grounding 仅 1/2；这推动了
v0.7.9 对“两条事实 + 两个证据类别”的硬合同，并拒绝 null/空值/非有限数值、伪造、不完整或与已引事实矛盾的
结构化引用冒充事实；只有完全不提供结构化字段时才允许有上限的 type-only 信用。正式结论必须等待新附件上的 50×8 探针。
当前单次最大 prompt 的小样本读数约 23K tokens；preflight 使用 24K 显式上限，并同时读取
veRL 已审计的 28,672-token `max_model_len`，至少预留 4,096 token 给后续动作/提交；两者取更严者。
正式 probe 的 group 和 temperature 还必须与训练运行时一致。累计 prompt token 只报告成本，
不当作单次上下文。

## 7. 已知局限

- CAMS、GFS、IFS 本身都是模型证据，不是真值；多模式分歧必须保留。
- EDGAR 2020 只用于相对源型/空间先验，论文不得称为同期排放清单。
- 当前输送视图是目标点风向、距离和恒速到达时间筛选，不是随空间风场积分的拉格朗日轨迹，
  也不能证明源归因；实时排放变化、临时管控和突发源不在现有开放合同内。
- GFS/IFS 分歧是两套确定性模式的敏感性信息，不是集合预报概率分布；区间校准必须由训练期内
  历史误差与独立验证完成，不能把两模式 spread 直接解释为 80% 不确定性。
- 全国 case 没有上一版业务预报档案或时间安全的相似过程索引；二者是校准增强项，不作为当前
  preflight 的假定输入。
- CAMS AOD 组分属于模式证据，当前没有全国同尺度地面颗粒物组分观测或独立机制标签；污染类型
  的科学主张必须靠条件遮蔽实验和结果分层，不能由规则标签自证。
- VIIRS NRT 无 `Type` 字段，只能称为火点候选，不能直接等同生物质燃烧真值；standard 回溯代理
  虽保留观测时间，但不证明与当年的 NRT feed 字节一致，因此只在该传感器—日期缺 NRT 时补位，
  且 agent/tabular 都能看到其处理级别。
- 0.25° 模式地形能表示盆地和山脉，不能表示城市街谷；高分辨率 ETOPO/Copernicus DEM 是可选增强，
  不是当前决策链的硬依赖。
- 高原城市在现有 925/850/700 hPa 温湿层中可能没有可用的近地层热力廓线；harness 会返回
  `wind_only`/部分廓线状态并保留逐日地面 RH、风和 BLH，不用地下压力面补值。增加模式层/地面层
  热力廓线是后续增强项，不阻塞当前日尺度预报，但论文必须按地形高度分层报告。
- 专家语料不进入全国 episode，因此全国机制有效性最终必须由结果、消融和 OOD 证据证明。
