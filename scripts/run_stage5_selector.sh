#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -x "${REPO_ROOT}/.venv_gpu/bin/python" ]]; then
  PYTHON_EXEC="${REPO_ROOT}/.venv_gpu/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_EXEC="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_EXEC="python3"
fi

export PYTHONPATH="${REPO_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/photo_score_project_mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

exec "${PYTHON_EXEC}" "${REPO_ROOT}/tools/run_stage5_selector.py" "$@"
