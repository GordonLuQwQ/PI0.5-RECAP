#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ROLLOUT_ROOT="${ROLLOUT_ROOT:-${SOURCE_ROLLOUT_ROOT:-}}"
require_vars ROLLOUT_ROOT CALIBRATION

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CORRECTION_RAW="${CORRECTION_RAW:-${ARTIFACT_ROOT}/ik_corrections_${RUN_ID}_raw}"
CORRECTION_DATASET="${CORRECTION_DATASET:-${ARTIFACT_ROOT}/ik_corrections_${RUN_ID}_lerobot}"
CORRECTION_QUOTAS="${CORRECTION_QUOTAS:-6,6,9,7,3,4}"
WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-3.0}"
ENCODER_THREADS="${ENCODER_THREADS:-4}"

PYTHONPATH="${DATA_ENGINE_ROOT}:${PYTHONPATH}" PYOPENGL_PLATFORM=egl \
"${DATA_PYTHON}" -u -m piperx_data_engine.recover_policy_failures \
  --source "${ROLLOUT_ROOT}/standard" \
  --calibration "${CALIBRATION}" \
  --output "${CORRECTION_RAW}" \
  --quotas "${CORRECTION_QUOTAS}" \
  --window-seconds "${WATCHDOG_SECONDS}" \
  --allow-partial \
  --backend gpu \
  --cameras third_person wrist

PYTHONPATH="${DATA_ENGINE_ROOT}:${PYTHONPATH}" \
"${DATA_PYTHON}" -u -m piperx_data_engine.export_recovery_preview \
  --input "${CORRECTION_RAW}"

PYTHONPATH="${DATA_ENGINE_ROOT}:${PYTHONPATH}" \
"${DATA_PYTHON}" -u -m piperx_data_engine.export_recovery_lerobot \
  --input "${CORRECTION_RAW}" \
  --output "${CORRECTION_DATASET}" \
  --repo-id local/piperx_ik_corrections \
  --allow-partial \
  --encoder-threads "${ENCODER_THREADS}"

printf 'CORRECTION_RAW=%s\nCORRECTION_DATASET=%s\nPREVIEW=%s\n' \
  "${CORRECTION_RAW}" "${CORRECTION_DATASET}" "${CORRECTION_RAW}/preview_videos/watch.html"
