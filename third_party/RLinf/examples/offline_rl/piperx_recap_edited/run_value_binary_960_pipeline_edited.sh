#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT="/home/ajifang/RLinf"
PYTHON_BIN="${RLINF_ROOT}/.venv/bin/python"
PIPELINE_DIR="${RLINF_ROOT}/examples/offline_rl/piperx_recap_edited"
CONFIG="${PIPELINE_DIR}/value_binary_960_edited.yaml"

cd "${RLINF_ROOT}"
export PYTHONPATH="${RLINF_ROOT}:${PIPELINE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[1/3] Inferring value VLM scores and writing 0/1 advantage labels."
"${PYTHON_BIN}" -u "${PIPELINE_DIR}/label_value_advantages_edited.py" \
  --config "${CONFIG}"

echo "[2/3] Verifying the exact 600 + 300 + 60 training mixture."
"${PYTHON_BIN}" -u "${PIPELINE_DIR}/train_value_conditioned_policy_edited.py" \
  --config "${CONFIG}" \
  --check-only

echo "[3/3] Fine-tuning Pi0.5 VLM LoRA + action-expert LoRA for 5000 steps."
"${PYTHON_BIN}" -u "${PIPELINE_DIR}/train_value_conditioned_policy_edited.py" \
  --config "${CONFIG}"
