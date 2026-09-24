#!/usr/bin/env bash
set -euo pipefail

ENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENESIS_ROOT="${GENESIS_ROOT:-$(cd "${ENGINE_ROOT}/../.." && pwd)}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd "$(dirname "${GENESIS_ROOT}")" && pwd)/lerobot}"
ENV_NAME="${ENV_NAME:-piperx_data}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

CONDA_BIN="${CONDA_EXE:-}"
if [[ -z "${CONDA_BIN}" ]]; then
    CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "${CONDA_BIN}" ]]; then
    echo "conda was not found. Install Miniconda or add conda to PATH." >&2
    exit 1
fi
if [[ ! -f "${GENESIS_ROOT}/pyproject.toml" ]]; then
    echo "Genesis repository not found at: ${GENESIS_ROOT}" >&2
    echo "Set GENESIS_ROOT=/absolute/path/to/genesis-world and run again." >&2
    exit 1
fi
if [[ ! -f "${LEROBOT_ROOT}/pyproject.toml" ]]; then
    echo "LeRobot repository not found at: ${LEROBOT_ROOT}" >&2
    echo "Set LEROBOT_ROOT=/absolute/path/to/lerobot and run again." >&2
    exit 1
fi

if ! "${CONDA_BIN}" run -n "${ENV_NAME}" python -c "import sys" >/dev/null 2>&1; then
    "${CONDA_BIN}" create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -e "${GENESIS_ROOT}"
(
    cd "${LEROBOT_ROOT}"
    "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -e ".[dataset]"
)
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -e "${ENGINE_ROOT}"

"${CONDA_BIN}" run -n "${ENV_NAME}" python -c '
from piperx_data_engine.env import DEFAULT_URDF
import genesis
import lerobot
assert DEFAULT_URDF.is_file()
print(f"Genesis: {genesis.__version__}")
print(f"LeRobot: {lerobot.__version__}")
print(f"Bundled PiperX URDF: {DEFAULT_URDF}")
'

echo
echo "Installation complete. Activate the environment with:"
echo "  conda activate ${ENV_NAME}"
