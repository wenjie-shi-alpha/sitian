# Local data

原始数据、派生数据、Parquet、实验结果和 checkpoint 不进入 Git。
`cases/real/`、`cases/national/`、`models/` 同样保留在本机，不随源码提交。
标准文档、报告和小型合成 case 则随源码保存。

新机器需复制经过审计的数据快照，或按 README 的数据管线重新生成。
新生成的 Parquet 对项目内 case 使用相对路径，veRL worker 按项目根目录解析。
旧 Parquet 可在 `configs/local.env` 中显式设置 `SITIAN_LEGACY_PROJECT_ROOT`，映射旧项目根目录；
启动前会验证每一条记录的目录和 case ID，Parquet 内容与哈希保持原样。
需要重新生成数据时运行 `scripts/prepare_verl_dataset.py`。
重建之前确认 `data/interim/train_without_winter_or_spatial_holdout.json` 等审计清单及全部 case 已就绪。
源数据身份校验失败时应核实快照，不能通过修改哈希跳过审计。
