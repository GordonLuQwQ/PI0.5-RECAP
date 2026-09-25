#!/usr/bin/env bash

set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORKFLOW_ROOT}/.." && pwd)"

if [[ -f "${WORKFLOW_ROOT}/local.env" ]]; then
    source "${WORKFLOW_ROOT}/local.env"
fi

DATA_ENGINE_ROOT="${DATA_ENGINE_ROOT:-${PROJECT_ROOT}/piperx_data_engine}"
RLINF_ROOT="${RLINF_ROOT:-${PROJECT_ROOT}/third_party/RLinf}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_PYTHON="${DATA_PYTHON:-python}"
RLINF_PYTHON="${RLINF_PYTHON:-${RLINF_ROOT}/.venv/bin/python}"
UTILITY_PYTHON="${UTILITY_PYTHON:-python3}"

export WORKFLOW_ROOT PROJECT_ROOT DATA_ENGINE_ROOT RLINF_ROOT ARTIFACT_ROOT
export DATA_PYTHON RLINF_PYTHON UTILITY_PYTHON
export PYTHONPATH="${RLINF_ROOT}:${DATA_ENGINE_ROOT}:${PYTHONPATH:-}"

mkdir -p "${ARTIFACT_ROOT}" "${WORKFLOW_ROOT}/generated"

require_vars() {
    local name
    for name in "$@"; do
        if [[ -z "${!name:-}" ]]; then
            echo "Set ${name} before running this stage." >&2
            exit 2
        fi
    done
}

dataset_episodes() {
    "${UTILITY_PYTHON}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads((Path(sys.argv[1]) / "meta" / "info.json").read_text())
print(int(info["total_episodes"]))
PY
}

json_field() {
    "${UTILITY_PYTHON}" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text())
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

render_config() {
    "${UTILITY_PYTHON}" "${WORKFLOW_ROOT}/render_config.py" "$1" "$2"
}
