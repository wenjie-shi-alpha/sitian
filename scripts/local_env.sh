#!/usr/bin/env bash
# Shared defaults; caller overrides always win. Source from Bash launchers.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${project_root}/configs/local.env" ]]; then
  source "${project_root}/configs/local.env"
fi
export SITIAN_VERL_ROOT="${SITIAN_VERL_ROOT:-${project_root}/.local/verl-upstream}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PATH="${HOME}/.local/bin:${PATH}"
# WSL workarounds are inappropriate defaults on native Linux.
if [[ "$(uname -r)" == *[Mm]icrosoft* ]]; then
  export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
  export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
fi
