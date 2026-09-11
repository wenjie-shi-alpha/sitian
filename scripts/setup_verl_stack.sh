#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"
verl_root="${SITIAN_VERL_ROOT}"
verl_repo="https://github.com/verl-project/verl.git"
verl_ref="${SITIAN_VERL_REF:-c2429f29a25d573f63d9bcc29e7ceb690817dce9}"
model_id="${SITIAN_TRAINABLE_MODEL_ID:-Qwen/Qwen3-1.7B}"

if [[ ! -d "${verl_root}/.git" ]]; then
  mkdir -p "$(dirname "${verl_root}")"
  git clone --filter=blob:none "${verl_repo}" "${verl_root}"
fi
git -C "${verl_root}" fetch origin "${verl_ref}" --depth 1
git -C "${verl_root}" checkout --detach "${verl_ref}"

cd "${verl_root}"
uv sync --python 3.12 --frozen --all-packages --extra vllm --extra fsdp
# Ignore upstream override-dependencies here, otherwise its numpy>=2 override
# replaces even our explicit pin when this command runs inside the checkout.
uv --no-config pip install --python .venv/bin/python -r "${project_root}/configs/verl/local-compatibility.txt"
uv pip check --python .venv/bin/python
export UV_NO_SYNC=1
uv run --no-sync --frozen --all-packages --extra vllm --extra fsdp python3 - <<'PY'
import peft, ray, torch, verl, vllm
print({
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "peft": peft.__version__,
    "ray": ray.__version__,
    "verl": getattr(verl, "__version__", "source"),
})
PY
if [[ "${SITIAN_SKIP_MODEL_DOWNLOAD:-0}" != 1 ]]; then
  uv run --no-sync --frozen --all-packages --extra vllm --extra fsdp \
    hf download "${model_id}"
fi

printf 'veRL stack ready\nproject=%s\ncommit=%s\nmodel=%s\n' \
  "${project_root}" "$(git rev-parse HEAD)" "${model_id}"
