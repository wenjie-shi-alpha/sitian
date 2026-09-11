#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"
cd "${project_root}"
command -v uv >/dev/null || { printf 'Install uv: https://docs.astral.sh/uv/\n' >&2; exit 2; }
uv sync --frozen --python 3.12 --extra dev --extra imaging --extra evidence --extra analysis
.venv/bin/python scripts/check_local_environment.py
printf 'Activate with: source %s/.venv/bin/activate\n' "${project_root}"
