#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"
model_path="${FH_MODEL_PATH:-${project_root}/models/Qwen3-8B}"
python_bin="${SITIAN_VERL_ROOT}/.venv/bin/python"
[[ -x "${python_bin}" ]] || { printf 'Run bash scripts/setup_verl_stack.sh first\n' >&2; exit 2; }
export PATH="${SITIAN_VERL_ROOT}/.venv/bin:${PATH}"
exec "${python_bin}" -m vllm.entrypoints.cli.main serve "${model_path}" \
  --host "${FH_HOST:-127.0.0.1}" --port "${FH_PORT:-8000}" \
  --served-model-name "${FH_MODEL:-Qwen/Qwen3-8B}" \
  --dtype bfloat16 --max-model-len "${FH_MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${FH_GPU_MEMORY_UTILIZATION:-0.6}" \
  --enable-auto-tool-choice --tool-call-parser hermes "$@"
