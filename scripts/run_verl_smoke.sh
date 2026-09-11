#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"
export UV_NO_SYNC=1
verl_root="${SITIAN_VERL_ROOT}"
model_path="${FH_TRAINABLE_MODEL:-}"
train_file="${SITIAN_TRAIN_FILE:-${project_root}/data/verl/smoke_train.parquet}"
val_file="${SITIAN_VAL_FILE:-${project_root}/data/verl/smoke_val.parquet}"
tool_config="${project_root}/configs/verl/sitian_tools.yaml"
agent_config="${project_root}/configs/verl/sitian_agent_loop.yaml"
checkpoint_dir="${SITIAN_SMOKE_CKPT_DIR:-${project_root}/data/checkpoints/verl_smoke_qwen3_1p7b_v079}"
log_file="${SITIAN_SMOKE_LOG:-${project_root}/data/interim/verl_training_smoke.log}"
steps="${SITIAN_SMOKE_STEPS:-50}"
experiment_name="${SITIAN_EXPERIMENT_NAME:-qwen3_1p7b_lora_reward_v079_smoke}"
mask_audit="${SITIAN_MASK_AUDIT:-${project_root}/data/interim/qwen3_1p7b_assistant_mask_audit.json}"
smoke_audit="${SITIAN_SMOKE_AUDIT:-${project_root}/data/interim/training_smoke.json}"
max_gen_batches="${SITIAN_MAX_GEN_BATCHES:-0}"
dataset_manifest="${SITIAN_DATASET_MANIFEST:-${project_root}/data/verl/manifest.json}"
# Hardware profile knobs.  Local defaults are the certified single-4090 values;
# an 80 GB-class cloud GPU sets SITIAN_LAYERED_SUMMON=True (multi-GPU FSDP
# weight sync), a larger rollout memory share and more concurrent sequences.
n_gpus="${SITIAN_N_GPUS:-1}"
if [[ ! "${n_gpus}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'SITIAN_N_GPUS must be a positive integer\n' >&2
  exit 2
fi
default_layered_summon=False
if (( n_gpus > 1 )); then default_layered_summon=True; fi
layered_summon="${SITIAN_LAYERED_SUMMON:-${default_layered_summon}}"
rollout_gpu_util="${SITIAN_ROLLOUT_GPU_UTIL:-0.42}"
rollout_max_num_seqs="${SITIAN_ROLLOUT_MAX_NUM_SEQS:-4}"
agent_workers="${SITIAN_AGENT_WORKERS:-2}"
rollout_n="${SITIAN_ROLLOUT_N:-8}"
train_batch_size="${SITIAN_TRAIN_BATCH_SIZE:-2}"
# veRL V1 interprets ppo_mini_batch_size as prompts and multiplies by rollout.n.
ppo_mini_batch_size="${SITIAN_PPO_MINI_BATCH_SIZE:-${train_batch_size}}"
optimizer_trajectory_batch_size="$((ppo_mini_batch_size * rollout_n))"
resume_step="$((steps + 1))"

if [[ -z "${model_path}" ]]; then
  model_path="$(find "${HF_HUB_CACHE}/models--Qwen--Qwen3-1.7B/snapshots" \
    -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true)"
fi
required_paths=(
  "${verl_root}/pyproject.toml" "${model_path}/config.json"
  "${tool_config}" "${agent_config}"
)
if [[ "${SITIAN_CONFIG_ONLY:-0}" != 1 ]]; then
  required_paths+=("${train_file}" "${val_file}")
fi
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'required path missing: %s\n' "${path}" >&2
    exit 2
  fi
done
mkdir -p "${checkpoint_dir}" "$(dirname "${log_file}")"
if [[ "${SITIAN_CONFIG_ONLY:-0}" != 1 ]]; then
  : > "${log_file}"
fi
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=true
export VLLM_LOGGING_LEVEL=INFO

common=(
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=False
  algorithm.use_kl_in_reward=False
  algorithm.filter_groups.enable=True
  algorithm.filter_groups.metric=outcome_composite
  algorithm.filter_groups.max_inflight_gen_batches=1
  algorithm.filter_groups.max_num_gen_batches="${max_gen_batches}"
  data.train_files="${train_file}"
  data.val_files="${val_file}"
  data.train_batch_size="${train_batch_size}"
  data.gen_batch_size=1
  data.max_prompt_length=4096
  data.max_response_length=28672
  data.return_raw_chat=True
  data.filter_overlong_prompts=True
  data.filter_overlong_prompts_workers=4
  data.truncation=error
  data.dataloader_num_workers=2
  actor_rollout_ref.model.path="${model_path}"
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.strategy="${SITIAN_FSDP_STRATEGY:-fsdp}"
  actor_rollout_ref.ref.strategy="${SITIAN_FSDP_STRATEGY:-fsdp}"
  actor_rollout_ref.actor.optim.lr=3e-6
  actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  +actor_rollout_ref.actor.checkpoint.save_lora_only=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_util}"
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.n="${rollout_n}"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.load_format=safetensors
  # Single-GPU FSDP runs as NO_SHARD; the layered summon path returns empty
  # and falls back to summon_full_params(offload_to_cpu=True), which torch
  # rejects for NO_SHARD.  The plain summon path is correct on one GPU.
  actor_rollout_ref.rollout.layered_summon="${layered_summon}"
  actor_rollout_ref.rollout.max_model_len=32768
  actor_rollout_ref.rollout.max_num_batched_tokens=32768
  actor_rollout_ref.rollout.max_num_seqs="${rollout_max_num_seqs}"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.format=hermes
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${tool_config}"
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=12
  actor_rollout_ref.rollout.multi_turn.max_user_turns=12
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=16000
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right
  actor_rollout_ref.rollout.agent.default_agent_loop=sitian_tool_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${agent_config}"
  actor_rollout_ref.rollout.agent.num_workers="${agent_workers}"
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  trainer.critic_warmup=0
  trainer.logger="${SITIAN_LOGGERS:-[\"console\"]}"
  trainer.project_name=sitian_agent_rl
  trainer.experiment_name="${experiment_name}"
  trainer.n_gpus_per_node="${n_gpus}"
  trainer.nnodes=1
  trainer.val_before_train=False
  trainer.test_freq=-1
  trainer.total_epochs=999
  trainer.default_local_dir="${checkpoint_dir}"
  trainer.max_actor_ckpt_to_keep=3
  ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --no-sync --frozen --all-packages --extra vllm --extra fsdp"
)

for variable in VLLM_WSL2_ENABLE_PIN_MEMORY VLLM_USE_FLASHINFER_SAMPLER SITIAN_LEGACY_PROJECT_ROOT; do
  if [[ -n "${!variable:-}" ]]; then
    common+=("+ray_kwargs.ray_init.runtime_env.env_vars.${variable}='${!variable}'")
  fi
done

if [[ "${SITIAN_SKILL_PILOT:-0}" == 1 ]]; then
  : "${SITIAN_VALIDATION_DIR:?pilot needs a validation output directory}"
  common+=(
    trainer.val_before_train="${SITIAN_VAL_BEFORE_TRAIN:-True}"
    trainer.test_freq=25
    trainer.validation_data_dir="${SITIAN_VALIDATION_DIR}"
    data.val_batch_size=8
    data.seed=20260909
    +data.apply_chat_template_kwargs.enable_thinking=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
  )
fi

run_train() {
  cd "${verl_root}"
  uv run --no-sync --frozen --all-packages --extra vllm --extra fsdp \
    python3 -m verl.trainer.main_ppo "${common[@]}" "$@"
}

if [[ "${SITIAN_CONFIG_ONLY:-0}" == 1 ]]; then
  # Hydra CLI flags must precede positional overrides.  Keep this path fully
  # GPU-free so upstream config drift can be caught before releasing the
  # frozen evaluation server.
  cd "${verl_root}"
  uv run --no-sync --frozen --all-packages --extra vllm --extra fsdp \
    python3 -m verl.trainer.main_ppo --cfg job --resolve "${common[@]}" \
    trainer.total_training_steps="${steps}" \
    trainer.save_freq="${steps}" trainer.resume_mode=disable
  exit 0
fi

"${verl_root}/.venv/bin/python" "${project_root}/scripts/check_verl_dataset_paths.py" \
  "${train_file}" "${val_file}"

if tmux has-session -t sitian_vllm_probe_v04 2>/dev/null; then
  printf 'frozen AWQ server still owns the GPU; finish the formal probe and stop it first\n' >&2
  exit 3
fi

# Respect CUDA_VISIBLE_DEVICES, including reordered indexes and GPU UUIDs.
free_mib="$("${verl_root}/.venv/bin/python" - "${n_gpus}" <<'PY'
import sys
import torch
count = int(sys.argv[1])
if torch.cuda.device_count() < count:
    raise SystemExit(f"need {count} visible GPUs; found {torch.cuda.device_count()}")
print(min(torch.cuda.mem_get_info(index)[0] // (1024 * 1024) for index in range(count)))
PY
)"
if [[ -z "${free_mib}" || "${free_mib}" -lt 21000 ]]; then
  printf 'need at least 21000 MiB free GPU memory, found %s\n' "${free_mib:-unknown}" >&2
  exit 4
fi

printf 'phase=train steps=%s model=%s rollout_n=%s train_batch_size=%s ppo_mini_batch_size=%s temperature=1.0 ppo_mini_batch_unit=prompts optimizer_trajectory_batch_size=%s\n' \
  "${steps}" "${model_path}" "${rollout_n}" "${train_batch_size}" \
  "${ppo_mini_batch_size}" "${optimizer_trajectory_batch_size}" | tee -a "${log_file}"
run_train \
  trainer.total_training_steps="${steps}" \
  trainer.save_freq="${SITIAN_SAVE_FREQ:-${steps}}" \
  trainer.resume_mode=disable 2>&1 | tee -a "${log_file}"

if [[ "${SITIAN_SKILL_PILOT:-0}" == 1 ]]; then
  test -d "${checkpoint_dir}/global_step_${steps}"
  exit 0
fi

checkpoint_50="${checkpoint_dir}/global_step_${steps}"
if [[ ! -d "${checkpoint_50}" ]]; then
  printf 'expected checkpoint missing: %s\n' "${checkpoint_50}" >&2
  exit 5
fi

printf 'phase=reload_and_one_step checkpoint=%s\n' "${checkpoint_50}" | tee -a "${log_file}"
run_train \
  trainer.total_training_steps="${resume_step}" \
  trainer.save_freq="${resume_step}" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${checkpoint_50}" 2>&1 | tee -a "${log_file}"

cd "${project_root}"
"${verl_root}/.venv/bin/python" scripts/audit_trajectory_mask.py \
  --tokenizer "${model_path}" --out "${mask_audit}"
"${verl_root}/.venv/bin/python" scripts/audit_training_smoke.py \
  --log "${log_file}" --checkpoint-dir "${checkpoint_dir}" \
  --expected-steps "${steps}" --model "${model_path}" --verl-root "${verl_root}" \
  --mask-audit "${mask_audit}" \
  --dataset-manifest "${dataset_manifest}" \
  --expected-rollout-n "${rollout_n}" \
  --out "${smoke_audit}"
