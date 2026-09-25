#!/usr/bin/env bash
set -euo pipefail

export REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PIPERX_PROJECT_ROOT="${PIPERX_PROJECT_ROOT:-$(cd "${REPO_PATH}/../../piperx_data_engine" && pwd)}"
export PATH="$(dirname "${RLINF_PYTHON:-${REPO_PATH}/.venv/bin/python}"):${PATH}"
export PYTHONPATH="${REPO_PATH}:${PIPERX_PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RLINF_NODE_RANK=0
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

scene_seed="${PIPERX_SCENE_SEED:-10000}"
for required in PIPERX_POLICY_CHECKPOINT PIPERX_NORM_STATS PIPERX_PYTHON PIPERX_CALIBRATION; do
    if [ -z "${!required:-}" ]; then
        echo "Set ${required} before running the paired RTC comparison." >&2
        exit 2
    fi
done
configured_delay="$(
    awk '/initial_delay_steps:/ {print $2; exit}' \
        "${REPO_PATH}/evaluations/piperx/piperx_eval_pi05_RTC.yaml"
)"
rtc_delay="${PIPERX_RTC_INITIAL_DELAY_STEPS:-${configured_delay}}"
default_root="${REPO_PATH}/outputs/piperx_paired_rtc_vs_no_rtc_h50_exec10_delay${rtc_delay}_seed${scene_seed}"
paired_root="${PIPERX_PAIRED_ROOT:-${default_root}}"
mkdir -p "${paired_root}"

global_lock="${PIPERX_EVAL_LOCK:-/dev/shm/piperx-evaluation-${UID}.lock}"
exec 8>"${global_lock}"
flock -n 8 || {
    echo "Another PiperX evaluation is already running." >&2
    exit 1
}
exec 9>"${paired_root}/evaluation.lock"
flock -n 9 || {
    echo "This paired evaluation is already running." >&2
    exit 1
}

cd "${REPO_PATH}"

run_variant() {
    local mode="$1"
    local result_root="${paired_root}/${mode}"
    local ray_tmp="/dev/shm/piperx-paired-${UID}-${mode}"
    local overrides=(
        "env.eval.piperx.seeds=[${scene_seed}]"
        "actor.model.openpi.action_horizon=50"
    )
    if [ "${mode}" = "no_rtc" ]; then
        overrides+=(
            "runner.rtc.enabled=False"
            "actor.model.num_action_chunks=10"
        )
    else
        overrides+=(
            "runner.rtc.enabled=True"
            "runner.rtc.initial_delay_steps=${rtc_delay}"
            "actor.model.num_action_chunks=50"
        )
    fi

    mkdir -p "${ray_tmp}" "${result_root}"
    export RAY_TMPDIR="${ray_tmp}"
    export PIPERX_RESULT_ROOT="${result_root}"
    local remaining
    remaining="$(python evaluations/piperx/prepare.py piperx_eval_pi05_RTC "${overrides[@]}")"
    if ! [[ "${remaining}" =~ ^[0-9]+$ ]]; then
        echo "Invalid pending trial count for ${mode}: ${remaining}" >&2
        exit 1
    fi
    if [ "${remaining}" -eq 0 ]; then
        echo "${mode}: all six paired tasks are already complete"
        return
    fi
    echo "${mode}: running ${remaining} task(s), scene seed ${scene_seed}"
    bash evaluations/run_eval.sh piperx piperx_eval_pi05_RTC \
        "${overrides[@]}" \
        "env.eval.rollout_epoch=${remaining}" \
        "runner.logger.log_path=${result_root}/standard/logs"
}

run_variant no_rtc

plain_reset="${paired_root}/no_rtc/standard/reset_states/${scene_seed}.npz"
rtc_reset_dir="${paired_root}/rtc/standard/reset_states"
rtc_reset="${rtc_reset_dir}/${scene_seed}.npz"
if [ ! -f "${plain_reset}" ]; then
    echo "Missing canonical no-RTC reset state: ${plain_reset}" >&2
    exit 1
fi
mkdir -p "${rtc_reset_dir}"
if [ -f "${rtc_reset}" ] && ! cmp -s "${plain_reset}" "${rtc_reset}"; then
    echo "Existing RTC reset state differs; choose a new PIPERX_PAIRED_ROOT." >&2
    exit 1
fi
cp "${plain_reset}" "${rtc_reset}"

run_variant rtc

python evaluations/piperx/plot_paired_joint_curves.py "${paired_root}"
echo "Plots: ${paired_root}/joint_comparison/index.html"
