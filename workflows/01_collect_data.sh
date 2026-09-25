#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DEMO_RUN="${DEMO_RUN:-${ARTIFACT_ROOT}/demonstrations_${RUN_ID}}"
DEMO_RAW="${DEMO_RAW:-${DEMO_RUN}_raw}"
DEMO_DATASET="${DEMO_DATASET:-${DEMO_RUN}_lerobot}"
EPISODES_PER_PAIR="${EPISODES_PER_PAIR:-100}"
COLLECT_ENVS="${COLLECT_ENVS:-6}"
MAX_ATTEMPTS_PER_PAIR="${MAX_ATTEMPTS_PER_PAIR:-1000}"
TRAIN_SEED_START="${TRAIN_SEED_START:-1000}"
ENCODER_THREADS="${ENCODER_THREADS:-4}"

PYTHONPATH="${DATA_ENGINE_ROOT}:${PYTHONPATH}" PYOPENGL_PLATFORM=egl \
"${DATA_PYTHON}" -u -m piperx_data_engine.collect \
  --backend gpu \
  --num-envs "${COLLECT_ENVS}" \
  --task-set all \
  --episodes-per-pair "${EPISODES_PER_PAIR}" \
  --max-attempts-per-pair "${MAX_ATTEMPTS_PER_PAIR}" \
  --seed-start "${TRAIN_SEED_START}" \
  --cameras third_person wrist \
  --output "${DEMO_RAW}"

PYTHONPATH="${DATA_ENGINE_ROOT}:${PYTHONPATH}" \
"${DATA_PYTHON}" -u -m piperx_data_engine.export_lerobot \
  --input "${DEMO_RAW}" \
  --output "${DEMO_DATASET}" \
  --repo-id local/piperx_stacking_demonstrations \
  --cameras third_person wrist \
  --encoder-threads "${ENCODER_THREADS}"

printf 'DEMO_RAW=%s\nDEMO_DATASET=%s\nCALIBRATION=%s\n' \
  "${DEMO_RAW}" "${DEMO_DATASET}" "${DEMO_RAW}/cameras.json"
