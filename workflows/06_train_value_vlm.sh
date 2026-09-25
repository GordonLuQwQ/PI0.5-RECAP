#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_vars SFT_POLICY_CHECKPOINT NORM_STATS DEMO_DATASET

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
FAILURE_DATASET="${FAILURE_DATASET:-${ARTIFACT_ROOT}/policy_failures_${RUN_ID}_lerobot}"
VALUE_OUTPUT="${VALUE_OUTPUT:-${ARTIFACT_ROOT}/value_vlm_${RUN_ID}}"
VALUE_CONFIG="${VALUE_CONFIG:-${WORKFLOW_ROOT}/generated/value_vlm_${RUN_ID}.yaml}"
VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-8}"
VALUE_NUM_WORKERS="${VALUE_NUM_WORKERS:-4}"
VALUE_STEPS="${VALUE_STEPS:-3000}"
VALUE_CHECKPOINT_STEPS="${VALUE_CHECKPOINT_STEPS:-[]}"
VALUE_RESUME_FROM="${VALUE_RESUME_FROM:-null}"
ENCODER_THREADS="${ENCODER_THREADS:-4}"
export FAILURE_DATASET VALUE_OUTPUT VALUE_BATCH_SIZE VALUE_NUM_WORKERS VALUE_STEPS
export VALUE_CHECKPOINT_STEPS VALUE_RESUME_FROM SFT_POLICY_CHECKPOINT NORM_STATS DEMO_DATASET

if [[ ! -f "${FAILURE_DATASET}/meta/info.json" ]]; then
    ROLLOUT_ROOT="${ROLLOUT_ROOT:-${SOURCE_ROLLOUT_ROOT:-}}"
    require_vars ROLLOUT_ROOT CALIBRATION
    PYTHONPATH="${DATA_ENGINE_ROOT}:${PYTHONPATH}" PYOPENGL_PLATFORM=egl \
    "${DATA_PYTHON}" -u -m piperx_data_engine.export_policy_failures_lerobot \
      --rollouts "${ROLLOUT_ROOT}/standard" \
      --calibration "${CALIBRATION}" \
      --output "${FAILURE_DATASET}" \
      --repo-id local/piperx_policy_failures \
      --backend gpu \
      --encoder-threads "${ENCODER_THREADS}"
fi

render_config "${WORKFLOW_ROOT}/configs/value_vlm.yaml.in" "${VALUE_CONFIG}"

RECAP_DIR="${RLINF_ROOT}/examples/offline_rl/piperx_recap_edited"
cd "${RECAP_DIR}"
"${RLINF_PYTHON}" -u compute_returns_edited.py --config "${VALUE_CONFIG}"
"${RLINF_PYTHON}" -u train_value_edited.py --config "${VALUE_CONFIG}"

printf 'FAILURE_DATASET=%s\nVALUE_OUTPUT=%s\nVALUE_CHECKPOINT=%s\n' \
  "${FAILURE_DATASET}" "${VALUE_OUTPUT}" "${VALUE_OUTPUT}/pi05_value_final.pt"
