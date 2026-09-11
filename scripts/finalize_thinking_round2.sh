#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tag="${FH_RUN_TAG:-v041_round2_n50_g4_paired}"
off="data/interim/probe_thinking_off_${tag}.json"
on="data/interim/probe_thinking_on_${tag}.json"
analysis="data/interim/analysis_thinking_${tag}.json"
preflight="data/interim/rl_preflight_${tag}.json"
log="data/interim/finalize_thinking_${tag}.log"

while [[ ! -s "$off" || ! -s "$on" ]]; do
  printf '%s waiting off=%s on=%s\n' "$(date --iso-8601=seconds)" \
    "$([[ -s "$off" ]] && printf ready || printf pending)" \
    "$([[ -s "$on" ]] && printf ready || printf pending)" >>"$log"
  sleep 60
done

printf '%s analyzing paired artifacts\n' "$(date --iso-8601=seconds)" >>"$log"
PYTHONPATH=src python3 scripts/analyze_thinking_ablation.py \
  --off "$off" --on "$on" --expected-reward 0.4.1 --bootstrap 10000 \
  --out "$analysis" >>"$log" 2>&1

selected="$(python3 - "$analysis" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["round2_preregistered_decision"]["selected_training_thinking"])
PY
)"
if [[ "$selected" == "pending_format_remediation" ]]; then
  printf '%s decision pending format remediation; preflight not run\n' \
    "$(date --iso-8601=seconds)" >>"$log"
  exit 3
fi
probe="$off"
[[ "$selected" == "on" ]] && probe="$on"

printf '%s selected=%s running preflight probe=%s\n' \
  "$(date --iso-8601=seconds)" "$selected" "$probe" >>"$log"
PYTHONPATH=src python3 scripts/rl_preflight.py \
  --probe "$probe" \
  --tabular-baseline data/interim/eval_tabular_open_evidence_v041.json \
  --out "$preflight" >>"$log" 2>&1 || true
printf '%s finalized analysis=%s preflight=%s\n' \
  "$(date --iso-8601=seconds)" "$analysis" "$preflight" >>"$log"
