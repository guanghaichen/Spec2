#!/usr/bin/env bash
set -euo pipefail

# 采集无需 Draft 的 Target-only 上下文轨迹。
# 每个输出目录只属于一个任务分片，适合在多张 GPU 上并行执行。
#
# 示例：
#   CUDA_VISIBLE_DEVICES=1 TASK_IDS=0 NUM_TRIALS=1 \
#     bash openvla/specdecoding/decode-scripts/run_contextual_reference_trace.sh goal

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

SUITE_ARG="${1:-${TASK_SUITE_NAME:-goal}}"
case "${SUITE_ARG}" in
  goal|libero_goal) TASK_SUITE_NAME=libero_goal ;;
  object|libero_object) TASK_SUITE_NAME=libero_object ;;
  spatial|libero_spatial) TASK_SUITE_NAME=libero_spatial ;;
  10|long|libero_10) TASK_SUITE_NAME=libero_10 ;;
  *) echo "不支持的 suite: ${SUITE_ARG}" >&2; exit 1 ;;
esac

init_libero_eval_env "${TASK_SUITE_NAME}"

TASK_IDS="${TASK_IDS:?请用 TASK_IDS=0 或 TASK_IDS=0,1 指定任务分片}"
NUM_TRIALS="${NUM_TRIALS:-1}"
TRIAL_START_INDEX="${TRIAL_START_INDEX:-0}"
CONTEXT_TRACE_MAX_LAG="${CONTEXT_TRACE_MAX_LAG:-3}"
TRACE_TAG="${TRACE_TAG:-${TASK_SUITE_SLUG}-tasks-${TASK_IDS//,/-}-seed${SEED}}"
TRACE_ROOT="${TRACE_ROOT:-${LOG_DIR}/../calibration/context_traces}"
OUTPUT_DIR="${TRACE_ROOT}/${TASK_SUITE_SLUG}/${TRACE_TAG}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
echo "TASK_IDS=${TASK_IDS}"
echo "NUM_TRIALS=${NUM_TRIALS}"
echo "TRIAL_START_INDEX=${TRIAL_START_INDEX}"
echo "CONTEXT_TRACE_MAX_LAG=${CONTEXT_TRACE_MAX_LAG}"
echo "VLA_PATH=${VLA_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python openvla/specdecoding/evidence/run_recoverability_calibration.py \
  --pretrained_checkpoint "${VLA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --task_ids "${TASK_IDS}" \
  --trial_start_index "${TRIAL_START_INDEX}" \
  --num_trials "${NUM_TRIALS}" \
  --seed "${SEED}" \
  --reference_only True \
  --context_trace True \
  --context_trace_max_lag "${CONTEXT_TRACE_MAX_LAG}" \
  --run_id_note "${TRACE_TAG}"
