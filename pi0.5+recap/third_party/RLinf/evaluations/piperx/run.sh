#!/usr/bin/env bash
set -euo pipefail

export REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="${REPO_PATH}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_PATH}:/home/ajifang/genesis-world:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RLINF_NODE_RANK=0
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/piperx-${UID}}"
export PIPERX_RESULT_ROOT="${PIPERX_RESULT_ROOT:-/home/ajifang/rlinf-experiments/pi05_vlm_action_lora_2views_10k_plus5k_20260921/evaluation_official_rtc_exact_h50_exec10_delay14_results}"
mkdir -p "${RAY_TMPDIR}" "${PIPERX_RESULT_ROOT}"
exec 8>"${PIPERX_EVAL_LOCK:-/dev/shm/piperx-evaluation-${UID}.lock}"
flock -n 8 || { echo "Another PiperX evaluation is already running." >&2; exit 1; }
exec 9>"${PIPERX_RESULT_ROOT}/evaluation.lock"
flock -n 9 || { echo "This evaluation is already running." >&2; exit 1; }
cd "${REPO_PATH}"

for suite in standard reversal; do
    config_name=piperx_eval_pi05_RTC
    if [ "${suite}" = reversal ]; then
        config_name=piperx_reversal_eval_pi05_RTC
    fi
    remaining="$(python evaluations/piperx/prepare.py "${config_name}" "$@")"
    if ! [[ "${remaining}" =~ ^[0-9]+$ ]]; then
        echo "Invalid pending trial count: ${remaining}" >&2
        exit 1
    fi
    if [ "${remaining}" -eq 0 ]; then
        echo "${suite}: already completed"
        continue
    fi
    echo "${suite}: ${remaining} episodes; official RLinf evaluation entrypoint"
    bash evaluations/run_eval.sh piperx "${config_name}" "$@" \
        "env.eval.rollout_epoch=${remaining}" \
        "runner.logger.log_path=${PIPERX_RESULT_ROOT}/${suite}/logs"
done
echo "Results: ${PIPERX_RESULT_ROOT}/watch.html"
