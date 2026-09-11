#!/usr/bin/env bash
# Engineering acceptance only: two optimizer steps, save, restore, one more step.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"
run_id="${SITIAN_CLOUD_RUN_ID:-qwen3_8b_cloud_$(date -u +%Y%m%dT%H%M%SZ)}"
# The deployed environment is a copy of the pinned, installed local runtime.
export UV_NO_SYNC=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export SITIAN_VERL_ROOT="${SITIAN_VERL_ROOT}"
export FH_TRAINABLE_MODEL="${FH_TRAINABLE_MODEL:-${project_root}/models/Qwen3-8B}"
export SITIAN_EXPERIMENT_NAME="$run_id"
export SITIAN_SMOKE_STEPS="${SITIAN_SMOKE_STEPS:-2}"
export SITIAN_LAYERED_SUMMON=False
export SITIAN_ROLLOUT_GPU_UTIL="${SITIAN_ROLLOUT_GPU_UTIL:-0.55}"
export SITIAN_ROLLOUT_MAX_NUM_SEQS="${SITIAN_ROLLOUT_MAX_NUM_SEQS:-8}"
export SITIAN_AGENT_WORKERS="${SITIAN_AGENT_WORKERS:-2}"
export SITIAN_ROLLOUT_N="${SITIAN_ROLLOUT_N:-8}"
export SITIAN_TRAIN_BATCH_SIZE="${SITIAN_TRAIN_BATCH_SIZE:-2}"
# V1 ReplayBuffer ignores max_num_gen_batches; bound the whole acceptance run.
export SITIAN_SMOKE_CKPT_DIR="${project_root}/data/checkpoints/${run_id}"
export SITIAN_SMOKE_LOG="${project_root}/data/interim/${run_id}.log"
export SITIAN_MASK_AUDIT="${project_root}/data/interim/${run_id}_mask.json"
export SITIAN_SMOKE_AUDIT="${project_root}/data/interim/${run_id}_audit.json"
if [[ "${SITIAN_CONFIG_ONLY:-0}" != 1 ]]; then
  "${SITIAN_VERL_ROOT}/.venv/bin/python" "${project_root}/scripts/verify_modelscope_snapshot.py" \
    --model "${FH_TRAINABLE_MODEL}" \
    --out "${project_root}/data/interim/${run_id}_model_verification.json"
fi
exec timeout --signal=TERM --kill-after=60s "${SITIAN_CLOUD_MAX_SECONDS:-3600}" \
  bash "${project_root}/scripts/run_verl_smoke.sh"
