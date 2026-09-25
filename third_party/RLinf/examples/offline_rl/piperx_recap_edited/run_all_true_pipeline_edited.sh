#!/usr/bin/env bash
# Preserve the existing value model and run advantage-conditioned Pi0.5 LoRA SFT.

set -Eeuo pipefail

RLINF_ROOT=/path/to/RLinf
GENESIS_ROOT=/path/to/genesis-world
RLINF_PY=/path/to/RLinf/.venv/bin/python
GENESIS_PY=/path/to/miniconda3/bin/python
EDITED_DIR=${RLINF_ROOT}/examples/offline_rl/piperx_recap_edited
POLICY_CONFIG=${EDITED_DIR}/positive_policy_edited.yaml

INITIAL_DATA=${GENESIS_ROOT}/vla/stacking/data/all6_100_20260920_223136_lerobot
ROLLOUTS=/path/to/rlinf-experiments/pi05_vlm_action_lora_2views_10k_plus5k_20260921/policy_rollouts_300_pi05_step5000/results_no_rtc_parallel25_50_per_task_timeout60s_fast/standard
CORRECTIONS=${GENESIS_ROOT}/vla/stacking/data/pi05_ik_corrections_35_fixed_20260923_204117_raw
FAILURE_DATA=${GENESIS_ROOT}/vla/stacking/data/pi05_policy_failures_73_lerobot
VALUE_DATA=${GENESIS_ROOT}/vla/stacking/data/pi05_value_600success_73failure_lerobot
POSITIVE_DATA=${GENESIS_ROOT}/vla/stacking/data/pi05_positive_227plus26_lerobot
VALUE_OUTPUT=/path/to/rlinf-experiments/piperx_recap_value_stage1_600plus73_edited
POLICY_OUTPUT=/path/to/rlinf-experiments/piperx_advantage_stage1_all_true_edited
VALUE_CHECKPOINT=${VALUE_OUTPUT}/checkpoints/step_003000/pi05_value.pt
POLICY_CHECKPOINT=${POLICY_OUTPUT}/checkpoints/global_step_3000/actor

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false

for required in \
  "${RLINF_PY}" \
  "${GENESIS_PY}" \
  "${POLICY_CONFIG}" \
  "${INITIAL_DATA}/meta/info.json" \
  "${ROLLOUTS}/manifest.json" \
  "${CORRECTIONS}/collection.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done

available_kib=$(df --output=avail /path/to | tail -1 | tr -d ' ')
if (( available_kib < 10 * 1024 * 1024 )); then
  echo "At least 10 GiB of free disk space is required; available KiB: ${available_kib}" >&2
  exit 2
fi

echo "[1/6] Replaying all 73 failed policy episodes into LeRobot v3 with both training cameras."
if [[ -f "${FAILURE_DATA}/failure_export.json" ]]; then
  echo "Failure dataset already complete; skipping export: ${FAILURE_DATA}"
elif [[ -e "${FAILURE_DATA}" ]]; then
  echo "Incomplete failure dataset already exists: ${FAILURE_DATA}" >&2
  exit 2
else
  cd "${GENESIS_ROOT}"
  env -u LD_PRELOAD \
    PYOPENGL_PLATFORM=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${GENESIS_PY}" -u -m vla.stacking.export_policy_failures_lerobot \
      --rollouts "${ROLLOUTS}" \
      --output "${FAILURE_DATA}" \
      --repo-id local/piperx_policy_failures_73 \
      --backend gpu \
      --encoder-threads 2
fi

echo "[2/6] Physically merging 600 successes and 73 failures into one 673-episode dataset."
if [[ -f "${VALUE_DATA}/merge_summary.json" ]]; then
  echo "Merged value dataset already complete; skipping merge: ${VALUE_DATA}"
elif [[ -e "${VALUE_DATA}" ]]; then
  echo "Incomplete merged value dataset already exists: ${VALUE_DATA}" >&2
  exit 2
else
  cd "${GENESIS_ROOT}"
  env -u LD_PRELOAD \
    "${GENESIS_PY}" -u -m vla.stacking.merge_value_lerobot \
      --success-data "${INITIAL_DATA}" \
      --failure-data "${FAILURE_DATA}" \
      --output "${VALUE_DATA}" \
      --repo-id local/piperx_value_600success_73failure
fi

echo "[3/6] Keeping the existing value function unchanged."
cd "${EDITED_DIR}"
if [[ -f "${VALUE_CHECKPOINT}" ]]; then
  echo "Existing value checkpoint found; it will not be fine-tuned: ${VALUE_CHECKPOINT}"
else
  echo "Existing value checkpoint is missing: ${VALUE_CHECKPOINT}" >&2
  exit 2
fi
test -f "${VALUE_CHECKPOINT}"

echo "[4/6] Replaying 227 successful policy episodes and adding 26 IK expert suffixes."
if [[ -f "${POSITIVE_DATA}/positive_export.json" ]]; then
  echo "Positive dataset already complete; skipping export: ${POSITIVE_DATA}"
elif [[ -e "${POSITIVE_DATA}" ]]; then
  echo "Incomplete positive dataset already exists: ${POSITIVE_DATA}" >&2
  exit 2
else
  cd "${GENESIS_ROOT}"
  env -u LD_PRELOAD \
    PYOPENGL_PLATFORM=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${GENESIS_PY}" -u -m vla.stacking.export_positive_recap_lerobot \
      --rollouts "${ROLLOUTS}" \
      --corrections "${CORRECTIONS}" \
      --output "${POSITIVE_DATA}" \
      --repo-id local/piperx_positive_227plus26 \
      --backend gpu \
      --num-envs 60 \
      --encoder-threads 8
fi

echo "[5/6] Verifying all 253 episodes and every forced-positive advantage label."
cd "${RLINF_ROOT}"
"${RLINF_PY}" -u "${EDITED_DIR}/train_positive_policy_edited.py" \
  --config "${POLICY_CONFIG}" \
  --check-only

echo "[6/6] Fine-tuning native Pi0.5 for 3000 all-true advantage steps."
if [[ -f "${POLICY_CHECKPOINT}/adapter_config.json" ]]; then
  echo "Policy adapter already complete; skipping policy training: ${POLICY_CHECKPOINT}"
elif [[ -e "${POLICY_OUTPUT}" ]]; then
  echo "Incomplete policy output already exists: ${POLICY_OUTPUT}" >&2
  exit 2
else
  "${RLINF_PY}" -u "${EDITED_DIR}/train_positive_policy_edited.py" \
    --config "${POLICY_CONFIG}"
fi
test -f "${POLICY_CHECKPOINT}/adapter_config.json"
test -f "${POLICY_CHECKPOINT}/adapter_weights.safetensors"

echo "Pipeline complete."
echo "Failure dataset: ${FAILURE_DATA}"
echo "Merged 673-episode value dataset: ${VALUE_DATA}"
echo "Value checkpoint: ${VALUE_CHECKPOINT}"
echo "Positive dataset: ${POSITIVE_DATA}"
echo "Pi0.5 adapter: ${POLICY_CHECKPOINT}"
