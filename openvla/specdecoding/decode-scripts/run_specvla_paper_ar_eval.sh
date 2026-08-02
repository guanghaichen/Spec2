#!/usr/bin/env bash
set -euo pipefail

# SpecVLA 论文使用的 wrapped AR 分母。它不是纯 OpenVLA generate 路径。
# 用法：bash .../run_specvla_paper_ar_eval.sh [goal|object|spatial|10]

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
resolve_specvla_checkpoint
RUN_ID_NOTE="${RUN_ID_NOTE:-specvla-paper-ar-${TASK_SUITE_SLUG}}"

print_common_eval_config
echo "METHOD=specvla_paper_wrapped_ar"
echo "USE_SPEC=True"
echo "PARALLEL_DRAFT=False"

if [[ "${DRY_RUN:-False}" == "True" ]]; then
  echo "DRY_RUN=True：路径和配置检查通过，不启动 LIBERO。"
  exit 0
fi

AR_EVIDENCE_ARGS=(
  --trial_start_index "${TRIAL_START_INDEX:-0}"
  --ar_evidence_trace "${AR_EVIDENCE_TRACE:-False}"
)
if [[ -n "${MAX_EVAL_TASKS:-}" ]]; then
  AR_EVIDENCE_ARGS+=(--max_eval_tasks "${MAX_EVAL_TASKS}")
fi

python openvla/experiments/robot/libero/run_libero_goal_AR.py \
  --model_family openvla \
  --pretrained_checkpoint "${VLA_PATH}" \
  --spec_checkpoint "${SPEC_CKPT}" \
  --use_spec True \
  --parallel_draft False \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  "${AR_EVIDENCE_ARGS[@]}" \
  --center_crop True \
  --local_log_dir "${LOG_DIR}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --sync_cuda_timing "${SYNC_CUDA_TIMING}" \
  --timing_scope "${TIMING_SCOPE}" \
  --use_wandb "${USE_WANDB}" \
  --seed "${SEED}"
