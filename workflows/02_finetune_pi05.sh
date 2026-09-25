#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_vars BASE_PI05 DEMO_DATASET

SFT_LOG_ROOT="${SFT_LOG_ROOT:-${ARTIFACT_ROOT}/pi05_sft}"
SFT_EXPERIMENT="${SFT_EXPERIMENT:-piperx_self_collected}"
SFT_STEPS="${SFT_STEPS:-5000}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
SFT_NUM_WORKERS="${SFT_NUM_WORKERS:-2}"
SFT_SAVE_INTERVAL="${SFT_SAVE_INTERVAL:-${SFT_STEPS}}"
NORM_STATS="${NORM_STATS:-${DEMO_DATASET}/norm_stats.json}"

DATA_OVERRIDES=(
  "data.train_data_paths=${DEMO_DATASET}"
  "data.piper_specs=null"
)
if [[ -n "${SFT_SECOND_DATASET:-}" ]]; then
    if [[ -z "${SFT_PIPER_SPECS:-}" ]]; then
        SFT_PIPER_SPECS='[{source_episodes:600,episodes:500,frames:238789,tasks:5,per_task:100,exclude_tasks:["put the red cube on the red cylinder"]},{source_episodes:100,episodes:100,frames:47100,tasks:2,per_task:50}]'
    fi
    DATA_OVERRIDES=(
      "data.train_data_paths=[${DEMO_DATASET},${SFT_SECOND_DATASET}]"
      "data.piper_specs=${SFT_PIPER_SPECS}"
    )
fi

if [[ ! -f "${NORM_STATS}" ]]; then
    cd "${RLINF_ROOT}"
    "${RLINF_PYTHON}" -u toolkits/lerobot/calculate_norm_stats.py \
      --config-name pi05_piperx \
      --repo-id "${DEMO_DATASET}"
    GENERATED_NORM_STATS="${DEMO_DATASET}/norm_stats.json"
    if [[ "${NORM_STATS}" != "${GENERATED_NORM_STATS}" ]]; then
        mkdir -p "$(dirname "${NORM_STATS}")"
        cp "${GENERATED_NORM_STATS}" "${NORM_STATS}"
    fi
fi
export NORM_STATS
export PIPERX_BASE_PI05="${BASE_PI05}"
export PIPERX_DEMO_DATASET="${DEMO_DATASET}"
export PIPERX_NORM_STATS="${NORM_STATS}"
export PIPERX_SFT_LOG_ROOT="${SFT_LOG_ROOT}"
export PIPERX_SFT_EXPERIMENT="${SFT_EXPERIMENT}"

cd "${RLINF_ROOT}"
"${RLINF_PYTHON}" -u examples/sft/train_vla_sft.py \
  --config-name piperx_sft_openpi_pi05_rlinf \
  "${DATA_OVERRIDES[@]}" \
  "data.num_workers=${SFT_NUM_WORKERS}" \
  "actor.model.model_path=${BASE_PI05}" \
  "actor.model.openpi.assets_dir=${BASE_PI05}" \
  "actor.model.openpi_data.norm_stats_path=${NORM_STATS}" \
  "actor.micro_batch_size=${SFT_BATCH_SIZE}" \
  "actor.global_batch_size=${SFT_BATCH_SIZE}" \
  "actor.optim.total_training_steps=${SFT_STEPS}" \
  "runner.max_steps=${SFT_STEPS}" \
  "runner.save_interval=${SFT_SAVE_INTERVAL}" \
  "runner.logger.log_path=${SFT_LOG_ROOT}" \
  "runner.logger.experiment_name=${SFT_EXPERIMENT}"

echo "NORM_STATS=${NORM_STATS}"
echo "SFT_POLICY_CHECKPOINT=${SFT_LOG_ROOT}/${SFT_EXPERIMENT}/checkpoints/global_step_${SFT_STEPS}/actor"
