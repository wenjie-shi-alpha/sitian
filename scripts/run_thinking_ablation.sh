#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

model="${FH_MODEL:-Qwen/Qwen3-8B-AWQ}"
base_url="${FH_BASE_URL:-http://127.0.0.1:8000/v1}"
run_tag="${FH_RUN_TAG:-v04_n50_g4_paired}"
expected_reward="${FH_EXPECTED_REWARD:-0.4.1}"
master_log="data/interim/probe_thinking_paired_${run_tag}.master.log"

actual_reward="$(PYTHONPATH=src python3 -c 'from sitian.scoring import REWARD_VERSION; print(REWARD_VERSION)')"
if [[ "$actual_reward" != "$expected_reward" ]]; then
  printf '%s reward mismatch: expected=%s actual=%s\n' \
    "$(date --iso-8601=seconds)" "$expected_reward" "$actual_reward" | tee -a "$master_log"
  exit 2
fi

common=(
  python3 scripts/probe_model.py
  --split val
  --n 50
  --stratified
  --group-size 4
  --temperature 0.6
  --seed 7
  --rollout-seed 7
  --policy-stage frozen_baseline
)

run_arm() {
  local thinking_flag="$1"
  local arm="$2"
  local output="data/interim/probe_thinking_${arm}_${run_tag}.json"
  local log="data/interim/probe_thinking_${arm}_${run_tag}.log"

  if [[ -e "$output" ]]; then
    printf '%s refusing to overwrite %s\n' "$(date --iso-8601=seconds)" "$output" | tee -a "$master_log"
    return 2
  fi

  printf '%s starting arm=%s output=%s\n' "$(date --iso-8601=seconds)" "$arm" "$output" | tee -a "$master_log"
  PYTHONUNBUFFERED=1 FH_MODEL="$model" FH_BASE_URL="$base_url" \
    "${common[@]}" "$thinking_flag" --out "$output" >"$log" 2>&1
  printf '%s completed arm=%s output=%s\n' "$(date --iso-8601=seconds)" "$arm" "$output" | tee -a "$master_log"
}

run_arm --no-thinking off
run_arm --thinking on
