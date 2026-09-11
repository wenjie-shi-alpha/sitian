#!/usr/bin/env bash
# One bounded engineering check, then a fresh 50-step scientific pilot.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"
cd "${project_root}"
: "${SITIAN_RUN_ID:?Set a unique SITIAN_RUN_ID}"
run_dir="${project_root}/data/experiments/${SITIAN_RUN_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SITIAN_N_GPUS=4 SITIAN_LAYERED_SUMMON=True
# FSDP2 retains per-parameter LoRA names for incremental adapter export.
# FSDP1 flattened names trigger a full CPU gather on every rank during sync.
export SITIAN_FSDP_STRATEGY=fsdp2
export SITIAN_ROLLOUT_GPU_UTIL=0.55 SITIAN_ROLLOUT_MAX_NUM_SEQS=8 SITIAN_AGENT_WORKERS=8
export SITIAN_ROLLOUT_N=8 SITIAN_TRAIN_BATCH_SIZE=4 SITIAN_PPO_MINI_BATCH_SIZE=4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export SITIAN_DATASET_MANIFEST="${run_dir}/dataset/manifest.json"
export SITIAN_LOGGERS='["console","tensorboard"]'
export TENSORBOARD_DIR="${run_dir}/tensorboard"
python_bin="${SITIAN_VERL_ROOT}/.venv/bin/python"
export PATH="${SITIAN_VERL_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${run_dir}"
mkdir "${run_dir}/supervisor.lock"
trap 'code=$?; printf "%s\n" "$code" > "${run_dir}/supervisor_exit_code"' EXIT
printf 'preflight\n' > "${run_dir}/phase"
"${python_bin}" scripts/evaluate_rl_skill_pilot.py --run-dir "${run_dir}" --check-only

printf 'four_gpu_smoke_and_restore\n' > "${run_dir}/phase"
SITIAN_EXPERIMENT_NAME="${SITIAN_RUN_ID}_engineering" \
SITIAN_SMOKE_STEPS=1 \
SITIAN_TRAIN_FILE="${run_dir}/dataset/smoke_train.parquet" \
SITIAN_VAL_FILE="${run_dir}/dataset/smoke_val.parquet" \
SITIAN_SMOKE_CKPT_DIR="${project_root}/data/checkpoints/${SITIAN_RUN_ID}_engineering" \
SITIAN_SMOKE_LOG="${run_dir}/engineering.log" \
SITIAN_MASK_AUDIT="${run_dir}/mask_audit.json" \
SITIAN_SMOKE_AUDIT="${run_dir}/engineering_audit.json" \
timeout --signal=TERM --kill-after=60s 3600 bash scripts/run_verl_smoke.sh

printf 'pilot_50_steps\n' > "${run_dir}/phase"
export SITIAN_TRAIN_FILE="${run_dir}/dataset/train.parquet"
bash scripts/run_rl_skill_pilot.sh
printf 'complete\n' > "${run_dir}/phase"
