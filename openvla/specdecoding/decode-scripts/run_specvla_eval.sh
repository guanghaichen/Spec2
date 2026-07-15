#!/usr/bin/env bash
set -euo pipefail

# SpecVLA strict / relaxed 统一入口。
# 用法：bash .../run_specvla_eval.sh [goal|object|spatial|10] [strict|relaxed]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

SUITE_ARG="${1:-${TASK_SUITE_NAME:-goal}}"
MODE="${2:-${EVAL_MODE:-strict}}"
case "${SUITE_ARG}" in
  goal|libero_goal) TASK_SUITE_NAME=libero_goal ;;
  object|libero_object) TASK_SUITE_NAME=libero_object ;;
  spatial|libero_spatial) TASK_SUITE_NAME=libero_spatial ;;
  10|long|libero_10) TASK_SUITE_NAME=libero_10 ;;
  *) echo "不支持的 suite: ${SUITE_ARG}" >&2; exit 1 ;;
esac

case "${MODE}" in
  strict)
    DEFAULT_THRESHOLD=0
    EVAL_ENTRY=openvla/experiments/robot/libero/run_libero_goal_Spec.py
    ;;
  relaxed)
    if [[ "${TASK_SUITE_NAME}" == "libero_10" ]]; then
      DEFAULT_THRESHOLD=5
    else
      DEFAULT_THRESHOLD=9
    fi
    EVAL_ENTRY=openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py
    ;;
  *) echo "不支持的模式: ${MODE}; 应为 strict 或 relaxed。" >&2; exit 1 ;;
esac

init_libero_eval_env "${TASK_SUITE_NAME}"
resolve_specvla_checkpoint
ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-${DEFAULT_THRESHOLD}}"
RUN_ID_NOTE="${RUN_ID_NOTE:-specvla-${MODE}-${TASK_SUITE_SLUG}-r${ACCEPT_THRESHOLD}}"

print_common_eval_config
echo "METHOD=specvla_${MODE}"
echo "ACCEPT_THRESHOLD=${ACCEPT_THRESHOLD}"

if [[ "${DRY_RUN:-False}" == "True" ]]; then
  echo "DRY_RUN=True：路径和配置检查通过，不启动 LIBERO。"
  exit 0
fi

python "${EVAL_ENTRY}" \
  --model_family openvla \
  --pretrained_checkpoint "${VLA_PATH}" \
  --spec_checkpoint "${SPEC_CKPT}" \
  --draft_backend eagle \
  --use_spec True \
  --parallel_draft False \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --center_crop True \
  --accept_threshold "${ACCEPT_THRESHOLD}" \
  --local_log_dir "${LOG_DIR}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --sync_cuda_timing "${SYNC_CUDA_TIMING}" \
  --timing_scope "${TIMING_SCOPE}" \
  --use_wandb "${USE_WANDB}" \
  --seed "${SEED}"
