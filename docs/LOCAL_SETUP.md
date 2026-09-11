# 本机适配与验证（2026-09-11）

## 结论与环境

本项目核心支持 Python ≥3.11；本机统一用 **Python 3.12.3**。锁定的 veRL
依赖组合要求 Linux / Python 3.12。系统没有 `python` 命令，激活虚拟环境后提供该命令。

| 项目 | 本机环境 / 验证组合 |
|---|---|
| 系统 | Ubuntu 24.04，原生 Linux x86_64，内核 6.17 |
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell Max-Q，单卡约 96 GB，SM 12.0 |
| NVIDIA 驱动 | 610.43.02，已验证，无需更换 |
| 系统 CUDA Toolkit | nvcc 13.1；nvidia-smi 显示驱动支持 CUDA 13.3 |
| 训练 / 推理 CUDA runtime | PyTorch wheel 自带 CUDA 13.0，与系统 Toolkit 版本不必完全一致 |
| veRL | 官方 commit `c2429f29a25d573f63d9bcc29e7ceb690817dce9` |
| GPU Python 包 | torch 2.11.0+cu130、vLLM 0.24.0、Transformers 5.9.0、PEFT 0.19.1、Ray 2.55.1、FlashAttention 2.8.3 |
| veRL 本地兼容约束 | NumPy 2.3.5，见 `configs/verl/local-compatibility.txt` |

Blackwell 需要包含相应架构的 CUDA/PyTorch 构建，不能只看系统是否安装了 nvcc。
官方说明：[PyTorch Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/)、
[vLLM GPU 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)。
本机采用 CUDA 13.0 wheels，并实际验证 `sm_120` 内核。

## 安装与运行

在项目根目录运行；需要 `uv`、Git 和网络，不需要 sudo：

```bash
bash scripts/setup_local.sh
source .venv/bin/activate
python -m pytest -q
fh baselines --cases 'cases/synthetic/*'

# 已有 models/Qwen3-8B 时无需再次下载 1.7B；新机器可不设置该变量。
SITIAN_SKIP_MODEL_DOWNLOAD=1 bash scripts/setup_verl_stack.sh
bash scripts/serve_local.sh
```

`serve_local.sh` 默认加载 `models/Qwen3-8B`，监听 `127.0.0.1:8000`，模型服务名为
`Qwen/Qwen3-8B`，支持 Hermes tool calling。可用 `CUDA_VISIBLE_DEVICES=0` 选择显卡，
用 `FH_MODEL_PATH`、`FH_PORT`、`FH_MAX_MODEL_LEN`、`FH_GPU_MEMORY_UTILIZATION` 调整服务。
用 `FH_BASE_URL=http://127.0.0.1:8000/v1 FH_MODEL=Qwen/Qwen3-8B fh llm --case ...` 连接。

安装设计：

- `.venv`：核心、测试、图像、分析和证据处理，版本锁在根目录 `uv.lock`。
  GRIB 使用 ecCodes 原生 wheel；GeoTIFF 使用带 GDAL 的 Rasterio wheel，避免系统
  `gdalinfo`、`osgeo` 和 sudo 安装依赖。保留 63 波段顺序、坐标和气体柱总量语义校验。
- `.local/verl-upstream/.venv`：由官方 frozen lock 安装 GPU 栈，再应用显式 NumPy 约束。
  上游 lock 的 NumPy 2.4.6 不满足 `mistral-common` 在 Python 3.12 上的 `<2.4` 要求。
  安装脚本执行 `uv pip check`；训练入口和 Ray worker 用 `--no-sync`，避免还原成冲突版本。
  更新依赖时重新执行安装脚本，不要单独升级 torch/vLLM。
- `serve_local.sh` 和模型冒烟脚本会设置 GPU 环境的 `bin` 到 PATH，确保 CUDA JIT 能找到 ninja。
  直接用旧虚拟环境中的 console script 会指向不存在的 `/root/sitian`，应使用上述新入口。
- WSL pinned-memory 和关闭 Ray 内存监控的旧设置仅在 WSL 内启用，原生 Linux 保持默认行为。

## 路径迁移

启动脚本使用当前项目根目录；`SITIAN_VERL_ROOT` 可覆盖独立的 veRL 位置。
Hugging Face 缓存遵循 `HF_HOME` / `HF_HUB_CACHE` / 当前用户目录。

`configs/local.env.example` 可复制为被 Git 忽略的 `configs/local.env`。
本机已配置以下两项，用于保留原 Parquet 快照并使用已有 8B 权重：

```bash
export SITIAN_LEGACY_PROJECT_ROOT="${SITIAN_LEGACY_PROJECT_ROOT:-/root/sitian}"
export FH_TRAINABLE_MODEL="${FH_TRAINABLE_MODEL:-${project_root}/models/Qwen3-8B}"
```

只有显式配置的旧根目录会被映射，新生成的项目内 case 路径使用相对路径。
训练前检查每条 Parquet 的目标 case 是否存在、case ID 是否吻合；不会修改数据快照或哈希。
该检查验证文件位置与身份，原有时间门禁、数据内容哈希及训练准入审计仍需通过。

外部气象场 / eaget 原始资料仍需单独提供：`SITIAN_EVIDENCE_ROOT`、`SITIAN_EAGET_ROOT`
可指定位置；`SITIAN_CORPUS_ROOT` 指定专家语料。现有历史 case 不因缺少原始下载而被删除。
`migrate_raw_ifs_to_mnt_c.sh` 是历史 WSL 磁盘迁移工具，不属于原生 Linux 安装流程。

## 可复跑验证

```bash
source .venv/bin/activate
python scripts/check_local_environment.py
python -m pytest -q
python scripts/audit_reward_contract.py

PYTHONPATH=src .local/verl-upstream/.venv/bin/python scripts/check_local_environment.py --gpu
.local/verl-upstream/.venv/bin/python scripts/smoke_local_model.py --model models/Qwen3-8B --mode inference
.local/verl-upstream/.venv/bin/python scripts/smoke_local_model.py --model models/Qwen3-8B --mode lora
PYTHONPATH=src .local/verl-upstream/.venv/bin/python -m pytest -q tests/test_trajectory.py tests/test_rl_skill_pilot.py

source scripts/local_env.sh
PYTHONPATH=src python scripts/check_verl_dataset_paths.py data/verl/*.parquet
SITIAN_CONFIG_ONLY=1 bash scripts/run_verl_smoke.sh > data/interim/local_resolved_config.yaml
PYTHONPATH=src .local/verl-upstream/.venv/bin/python scripts/audit_verl_runtime_contract.py \
  --resolved-config data/interim/local_resolved_config.yaml --out data/interim/local_verl_runtime_contract.json
```

已验证：全部四卡 BF16 矩阵运算；FlashAttention 前向 / 反向；Qwen3-8B vLLM
实际生成 `ready`；Qwen3-8B LoRA 单步 loss 有限且 504 个参数张量更新；veRL
配置解析、原生工具 create/execute/release、终端 reward 和 token mask；全部 5,391 条
迁移 Parquet 记录、4,186 个不同 case 路径。模型冒烟不会保存或覆盖基础模型权重。

项目测试 205 项通过，2 项训练环境测试在轻量环境跳过，并在 veRL 环境中另行通过
（该环境的两个测试文件共 8 项通过）。从暂存源码导出的无模型、无本机数据目录也执行测试，
确保首次 clone 不依赖忽略的本机文件。启动默认使用单卡；可设 `SITIAN_N_GPUS`，
显存检查遵循 `CUDA_VISIBLE_DEVICES`，多卡默认启用 layered summon。

初次适配验证覆盖本机软件与模型运行兼容性。后续已启动四卡 FSDP2 LoRA GRPO，
完成首个分布式更新并保存 checkpoint；恢复检查、50 步 pilot 及结果位置见
[四卡训练记录](LOCAL_4GPU_TRAINING.md)。预报技能仍需独立评估，现有训练门禁继续保留。

## Git 范围

跟踪源码、配置模板、依赖锁、小型合成 case、标准原件及报告。
忽略 `.env`、本机配置、虚拟环境、第三方源码、模型、真实 / 全国 case、派生数据、
checkpoint、日志、前端依赖、构建缓存及失败截图。Git 忽略规则不会删除本机现有文件。
