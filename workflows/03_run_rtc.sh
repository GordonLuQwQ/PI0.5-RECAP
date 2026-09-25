#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${SFT_POLICY_CHECKPOINT:-}}"
PIPERX_PYTHON="${PIPERX_PYTHON:-${DATA_PYTHON}}"
require_vars POLICY_CHECKPOINT NORM_STATS CALIBRATION PIPERX_PYTHON

RTC_OUTPUT="${RTC_OUTPUT:-${ARTIFACT_ROOT}/rtc_evaluation_$(date +%Y%m%d_%H%M%S)}"
RTC_EXEC_HORIZON="${RTC_EXEC_HORIZON:-10}"

export PIPERX_RESULT_ROOT="${RTC_OUTPUT}"
export PIPERX_PROJECT_ROOT="${DATA_ENGINE_ROOT}"
export PIPERX_POLICY_CHECKPOINT="${POLICY_CHECKPOINT}"
export PIPERX_NORM_STATS="${NORM_STATS}"
export PIPERX_CALIBRATION="${CALIBRATION}"
export PIPERX_PYTHON

cd "${RLINF_ROOT}"
bash evaluations/piperx/run.sh \
  "actor.model.model_path=${POLICY_CHECKPOINT}" \
  "actor.model.openpi_data.norm_stats_path=${NORM_STATS}" \
  "runner.rtc.min_exec_horizon=${RTC_EXEC_HORIZON}" \
  "env.eval.piperx.project_path=${DATA_ENGINE_ROOT}" \
  "env.eval.piperx.python_path=${PIPERX_PYTHON}" \
  "env.eval.piperx.bridge_script=${RLINF_ROOT}/evaluations/piperx/simulator.py" \
  "env.eval.piperx.calibration=${CALIBRATION}" \
  "$@"

echo "RTC report: ${RTC_OUTPUT}/watch.html"
