#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/oft_common.sh"

TEACHER_EARLY_EXIT_LAYER="${TEACHER_EARLY_EXIT_LAYER:-16}"
TEACHER_FEATURE_LIMIT="${TEACHER_FEATURE_LIMIT:-4096}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
FEATURE_FILE="${FEATURE_FILE:-${OFT_RUN_ROOT}/teacher_features/libero_goal_layer${TEACHER_EARLY_EXIT_LAYER}.h5}"

cd "${OFT_ROOT}"
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${OFT_GOAL_MODEL}" \
  --task_suite_name libero_goal \
  --center_crop True \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --local_log_dir "${OFT_RUN_ROOT}/teacher_collection_logs" \
  --run_id_note "oft-goal-teacher-layer${TEACHER_EARLY_EXIT_LAYER}" \
  --teacher_feature_output "${FEATURE_FILE}" \
  --teacher_feature_layer "${TEACHER_EARLY_EXIT_LAYER}" \
  --teacher_feature_limit "${TEACHER_FEATURE_LIMIT}" \
  --save_rollout_videos False \
  --use_wandb False
