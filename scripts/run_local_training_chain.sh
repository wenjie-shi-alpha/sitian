#!/usr/bin/env bash
# Sequential scaling updates and long training, owned by one systemd cgroup.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"
cd "${project_root}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled UV_NO_SYNC=1
export PATH="${SITIAN_VERL_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export FH_TRAINABLE_MODEL="${project_root}/models/Qwen3-8B"
exec "${SITIAN_VERL_ROOT}/.venv/bin/python" scripts/run_local_training_chain.py "$@"
