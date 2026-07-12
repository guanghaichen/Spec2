#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/oft_common.sh"

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
RUN_ID_NOTE="${RUN_ID_NOTE:-oft-goal-baseline}"

cd "${OFT_ROOT}"
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${OFT_GOAL_MODEL}" \
  --task_suite_name libero_goal \
  --center_crop True \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --local_log_dir "${OFT_RUN_ROOT}/eval_logs" \
  --run_id_note "${RUN_ID_NOTE}" \
  --record_action_timing True \
  --sync_cuda_timing True \
  --use_wandb False
