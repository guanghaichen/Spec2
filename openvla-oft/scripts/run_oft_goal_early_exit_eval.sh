#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/oft_common.sh"

EARLY_EXIT_LAYER="${EARLY_EXIT_LAYER:-16}"
EARLY_EXIT_CHECKPOINT="${EARLY_EXIT_CHECKPOINT:-${OFT_RUN_ROOT}/checkpoints/layer_exit_goal_l${EARLY_EXIT_LAYER}/latest}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
RUN_ID_NOTE="${RUN_ID_NOTE:-oft-goal-early-exit-l${EARLY_EXIT_LAYER}}"

cd "${OFT_ROOT}"
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${OFT_GOAL_MODEL}" \
  --task_suite_name libero_goal \
  --center_crop True \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --local_log_dir "${OFT_RUN_ROOT}/eval_logs" \
  --run_id_note "${RUN_ID_NOTE}" \
  --early_exit_checkpoint "${EARLY_EXIT_CHECKPOINT}" \
  --early_exit_layer "${EARLY_EXIT_LAYER}" \
  --record_action_timing True \
  --sync_cuda_timing True \
  --use_wandb False
