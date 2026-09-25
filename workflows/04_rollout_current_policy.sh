#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${SFT_POLICY_CHECKPOINT:-}}"
PIPERX_PYTHON="${PIPERX_PYTHON:-${DATA_PYTHON}}"
require_vars POLICY_CHECKPOINT NORM_STATS CALIBRATION PIPERX_PYTHON

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${ARTIFACT_ROOT}/policy_rollouts_${RUN_ID}}"
ROLLOUT_DATASET="${ROLLOUT_DATASET:-${ARTIFACT_ROOT}/policy_rollouts_${RUN_ID}_lerobot}"
ROLLOUT_EPISODES_PER_PAIR="${ROLLOUT_EPISODES_PER_PAIR:-50}"
ROLLOUT_SEED_START="${ROLLOUT_SEED_START:-30000}"
PARALLEL_ENVS="${PARALLEL_ENVS:-30}"
EXEC_STEPS="${EXEC_STEPS:-10}"
ENCODER_THREADS="${ENCODER_THREADS:-8}"
ROLLOUT_TOTAL=$((6 * ROLLOUT_EPISODES_PER_PAIR))

if [[ -z "${ROLLOUT_SEEDS:-}" ]]; then
    ROLLOUT_SEEDS="$(${UTILITY_PYTHON} - "${ROLLOUT_SEED_START}" "${ROLLOUT_EPISODES_PER_PAIR}" <<'PY'
import sys
start, count = map(int, sys.argv[1:])
print("[" + ",".join(str(seed) for seed in range(start, start + count)) + "]")
PY
)"
fi

export REPO_PATH="${RLINF_ROOT}"
export PATH="$(dirname "${RLINF_PYTHON}"):${PATH}"
export PYTHONUNBUFFERED=1
export RLINF_NODE_RANK=0
export JAX_PLATFORMS=cpu
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/piperx-${UID}}"
export PIPERX_RESULT_ROOT="${ROLLOUT_ROOT}"
export PIPERX_PROJECT_ROOT="${DATA_ENGINE_ROOT}"
export PIPERX_POLICY_CHECKPOINT="${POLICY_CHECKPOINT}"
export PIPERX_NORM_STATS="${NORM_STATS}"
export PIPERX_CALIBRATION="${CALIBRATION}"
export PIPERX_PYTHON
mkdir -p "${RAY_TMPDIR}" "${ROLLOUT_ROOT}"

OVERRIDES=(
  "actor.model.model_path=${POLICY_CHECKPOINT}"
  "actor.model.openpi_data.norm_stats_path=${NORM_STATS}"
  "actor.model.num_action_chunks=${EXEC_STEPS}"
  "runner.rtc.enabled=False"
  "env.eval.total_num_envs=${PARALLEL_ENVS}"
  "env.eval.group_size=1"
  "env.eval.piperx.parallel_envs=${PARALLEL_ENVS}"
  "env.eval.piperx.fast_chunk_step=True"
  "env.eval.piperx.seeds=${ROLLOUT_SEEDS}"
  "env.eval.piperx.video_stride=1"
  "env.eval.piperx.video_failure_only=False"
  "env.eval.piperx.project_path=${DATA_ENGINE_ROOT}"
  "env.eval.piperx.python_path=${PIPERX_PYTHON}"
  "env.eval.piperx.bridge_script=${RLINF_ROOT}/evaluations/piperx/simulator.py"
  "env.eval.piperx.calibration=${CALIBRATION}"
  "runner.logger.experiment_name=piperx_policy_rollout_${RUN_ID}"
)

cd "${RLINF_ROOT}"
remaining_batches="$("${RLINF_PYTHON}" evaluations/piperx/prepare.py piperx_eval_pi05_RTC "${OVERRIDES[@]}")"
if [[ "${remaining_batches}" -gt 0 ]]; then
    bash evaluations/run_eval.sh piperx piperx_eval_pi05_RTC \
      "${OVERRIDES[@]}" \
      "env.eval.rollout_epoch=${remaining_batches}" \
      "runner.logger.log_path=${ROLLOUT_ROOT}/standard/logs"
fi

PYTHONPATH="${DATA_ENGINE_ROOT}:${PYTHONPATH}" \
"${DATA_PYTHON}" -u -m piperx_data_engine.export_policy_rollouts_lerobot \
  --rollouts "${ROLLOUT_ROOT}/standard" \
  --output "${ROLLOUT_DATASET}" \
  --repo-id local/piperx_policy_rollouts \
  --expected-per-pair "${ROLLOUT_EPISODES_PER_PAIR}" \
  --encoder-threads "${ENCODER_THREADS}"

printf 'ROLLOUT_ROOT=%s\nROLLOUT_DATASET=%s\nROLLOUT_EPISODES=%s\n' \
  "${ROLLOUT_ROOT}" "${ROLLOUT_DATASET}" "${ROLLOUT_TOTAL}"
