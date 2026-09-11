#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"
run_id="${SITIAN_RUN_ID:-rl_skill_pilot_20260909}"
run_dir="${project_root}/data/experiments/${run_id}"
export UV_NO_SYNC=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export FH_TRAINABLE_MODEL="${project_root}/models/Qwen3-8B"
export SITIAN_EXPERIMENT_NAME="$run_id" SITIAN_SKILL_PILOT=1 SITIAN_SMOKE_STEPS=50
export SITIAN_ROLLOUT_GPU_UTIL="${SITIAN_ROLLOUT_GPU_UTIL:-0.55}"
export SITIAN_ROLLOUT_MAX_NUM_SEQS="${SITIAN_ROLLOUT_MAX_NUM_SEQS:-8}"
export SITIAN_AGENT_WORKERS="${SITIAN_AGENT_WORKERS:-2}"
export SITIAN_ROLLOUT_N="${SITIAN_ROLLOUT_N:-8}"
export SITIAN_TRAIN_BATCH_SIZE="${SITIAN_TRAIN_BATCH_SIZE:-2}"
export SITIAN_PPO_MINI_BATCH_SIZE="${SITIAN_PPO_MINI_BATCH_SIZE:-${SITIAN_TRAIN_BATCH_SIZE}}"
export SITIAN_TRAIN_FILE="${SITIAN_TRAIN_FILE:-${project_root}/data/verl/train.parquet}"
export SITIAN_VAL_FILE="${run_dir}/panel.parquet" SITIAN_VALIDATION_DIR="${run_dir}/validation"
export SITIAN_SMOKE_CKPT_DIR="${project_root}/data/checkpoints/${run_id}"
export SITIAN_SMOKE_LOG="${run_dir}/train.log" SITIAN_SAVE_FREQ=25
if [[ "${SITIAN_CONFIG_ONLY:-0}" == 1 ]]; then
  exec bash "${project_root}/scripts/run_verl_smoke.sh"
fi
# A fixed run ID must never silently restart or replace a partially trained run.
mkdir -p "${run_dir}"
mkdir "${run_dir}/launched.lock"
python="${SITIAN_VERL_ROOT}/.venv/bin/python"
"$python" "${project_root}/scripts/evaluate_rl_skill_pilot.py" --run-dir "$run_dir" --check-only
"$python" "${project_root}/scripts/verify_modelscope_snapshot.py" --model "$FH_TRAINABLE_MODEL" --out "${run_dir}/model_verification.json"
set +e
timeout --signal=TERM --kill-after=60s 43200 bash "${project_root}/scripts/run_verl_smoke.sh"
status=$?
set -e
printf '%s\n' "$status" > "${run_dir}/training_exit_code"
if [[ "$status" == 0 ]]; then
  "$python" "${project_root}/scripts/evaluate_rl_skill_pilot.py" --run-dir "$run_dir" --step 25
  "$python" "${project_root}/scripts/evaluate_rl_skill_pilot.py" --run-dir "$run_dir" --step 50
fi
exit "$status"
