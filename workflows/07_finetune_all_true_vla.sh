#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_vars SFT_POLICY_CHECKPOINT NORM_STATS

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
POSITIVE_DATASET="${POSITIVE_DATASET:-${ARTIFACT_ROOT}/positive_rollout_corrections_${RUN_ID}_lerobot}"
POSITIVE_POLICY_OUTPUT="${POSITIVE_POLICY_OUTPUT:-${ARTIFACT_ROOT}/all_true_policy_${RUN_ID}}"
POSITIVE_CONFIG="${POSITIVE_CONFIG:-${WORKFLOW_ROOT}/generated/positive_vla_${RUN_ID}.yaml}"
POSITIVE_POLICY_STEPS="${POSITIVE_POLICY_STEPS:-3000}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-8}"
POLICY_NUM_WORKERS="${POLICY_NUM_WORKERS:-2}"
POLICY_WARMUP_STEPS="${POLICY_WARMUP_STEPS:-1000}"
REPLAY_ENVS="${REPLAY_ENVS:-12}"
ENCODER_THREADS="${ENCODER_THREADS:-4}"

if [[ ! -f "${POSITIVE_DATASET}/meta/info.json" ]]; then
    ROLLOUT_ROOT="${ROLLOUT_ROOT:-${SOURCE_ROLLOUT_ROOT:-}}"
    require_vars ROLLOUT_ROOT CORRECTION_RAW CALIBRATION
    PYTHONPATH="${DATA_ENGINE_ROOT}:${PYTHONPATH}" PYOPENGL_PLATFORM=egl \
    "${DATA_PYTHON}" -u -m piperx_data_engine.export_positive_recap_lerobot \
      --rollouts "${ROLLOUT_ROOT}/standard" \
      --calibration "${CALIBRATION}" \
      --corrections "${CORRECTION_RAW}" \
      --output "${POSITIVE_DATASET}" \
      --repo-id local/piperx_positive_rollout_and_corrections \
      --backend gpu \
      --num-envs "${REPLAY_ENVS}" \
      --encoder-threads "${ENCODER_THREADS}"
fi

POLICY_SUCCESS_EPISODES="$(json_field "${POSITIVE_DATASET}/positive_export.json" policy_success_episodes)"
CORRECTION_EPISODES="$(json_field "${POSITIVE_DATASET}/positive_export.json" expert_suffix_episodes)"
export POSITIVE_DATASET POSITIVE_POLICY_OUTPUT POSITIVE_POLICY_STEPS POLICY_BATCH_SIZE
export POLICY_NUM_WORKERS POLICY_WARMUP_STEPS POLICY_SUCCESS_EPISODES CORRECTION_EPISODES
export SFT_POLICY_CHECKPOINT NORM_STATS

render_config "${WORKFLOW_ROOT}/configs/positive_vla.yaml.in" "${POSITIVE_CONFIG}"

RECAP_DIR="${RLINF_ROOT}/examples/offline_rl/piperx_recap_edited"
cd "${RECAP_DIR}"
"${RLINF_PYTHON}" -u train_positive_policy_edited.py --config "${POSITIVE_CONFIG}"

printf 'POSITIVE_DATASET=%s\nPOSITIVE_POLICY_OUTPUT=%s\nPOSITIVE_POLICY_CHECKPOINT=%s\n' \
  "${POSITIVE_DATASET}" "${POSITIVE_POLICY_OUTPUT}" \
  "${POSITIVE_POLICY_OUTPUT}/checkpoints/global_step_${POSITIVE_POLICY_STEPS}/actor"
