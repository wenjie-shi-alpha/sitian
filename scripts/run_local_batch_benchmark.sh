#!/usr/bin/env bash
# Equal-work comparison: global batch 8 x 1 step versus 4 x 2 steps.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"
cd "${project_root}"
: "${SITIAN_BENCHMARK_ID:?Set a unique benchmark ID}"
run_dir="${project_root}/data/experiments/${SITIAN_BENCHMARK_ID}"
source_run="${project_root}/data/experiments/qwen3_8b_4gpu_20260911_r2"
mkdir -p "${run_dir}"
mkdir "${run_dir}/launched.lock"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SITIAN_N_GPUS=4 SITIAN_FSDP_STRATEGY=fsdp2 SITIAN_LAYERED_SUMMON=True
export SITIAN_ROLLOUT_GPU_UTIL=0.55 SITIAN_ROLLOUT_MAX_NUM_SEQS=8 SITIAN_AGENT_WORKERS=8
export SITIAN_ROLLOUT_N=8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled UV_NO_SYNC=1
export PATH="${SITIAN_VERL_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export FH_TRAINABLE_MODEL="${project_root}/models/Qwen3-8B"
export SITIAN_TRAIN_FILE="${source_run}/dataset/smoke_train.parquet"
export SITIAN_VAL_FILE="${source_run}/dataset/smoke_val.parquet"
export SITIAN_DATASET_MANIFEST="${source_run}/dataset/manifest.json"
export SITIAN_LOGGERS='["console","tensorboard"]'
unset SITIAN_SKILL_PILOT SITIAN_CONFIG_ONLY
python_bin="${SITIAN_VERL_ROOT}/.venv/bin/python"
monitor_pid=""
cleanup() {
  code=$?
  trap - EXIT
  if [[ -n "${monitor_pid}" ]]; then kill "${monitor_pid}" 2>/dev/null || true; wait "${monitor_pid}" 2>/dev/null || true; fi
  printf '%s\n' "${code}" > "${run_dir}/exit_code"
  exit "${code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${python_bin}" scripts/evaluate_rl_skill_pilot.py --run-dir "${source_run}" --check-only
python3 scripts/monitor_local_training.py --run-dir "${run_dir}" &
monitor_pid=$!

# Run the requested batch-8 arm first. Both arms start from fresh LoRA with
# identical sampler/model seeds and input files, without validation or restore.
for batch in 8 4; do
  arm="${run_dir}/batch_${batch}"
  mkdir -p "${arm}"
  printf 'batch_%s\n' "${batch}" > "${run_dir}/phase"
  export SITIAN_TRAIN_BATCH_SIZE="${batch}" SITIAN_PPO_MINI_BATCH_SIZE="${batch}"
  export SITIAN_SMOKE_STEPS="$((8 / batch))"
  export SITIAN_EXPERIMENT_NAME="${SITIAN_BENCHMARK_ID}_batch_${batch}"
  export SITIAN_SMOKE_CKPT_DIR="${project_root}/data/checkpoints/${SITIAN_EXPERIMENT_NAME}"
  export SITIAN_SMOKE_LOG="${arm}/train.log" TENSORBOARD_DIR="${arm}/tensorboard"
  SITIAN_CONFIG_ONLY=1 bash scripts/run_verl_smoke.sh > "${arm}/resolved_config.yaml" 2> "${arm}/config.log"
  "${python_bin}" scripts/audit_verl_runtime_contract.py --resolved-config "${arm}/resolved_config.yaml" --out "${arm}/runtime_audit.json" > "${arm}/runtime_audit.log" 2>&1
  (
    cd "${SITIAN_VERL_ROOT}"
    timeout --signal=TERM --kill-after=60s 3600 \
      uv run --no-sync --frozen --all-packages --extra vllm --extra fsdp \
      python -m verl.trainer.main_ppo --config-path "${arm}" --config-name resolved_config \
      trainer.rollout_data_dir="${arm}/rollouts"
  ) > "${arm}/train.log" 2>&1
  test -d "${SITIAN_SMOKE_CKPT_DIR}/global_step_${SITIAN_SMOKE_STEPS}"
  printf '0\n' > "${arm}/exit_code"
done
printf 'complete\n' > "${run_dir}/phase"
