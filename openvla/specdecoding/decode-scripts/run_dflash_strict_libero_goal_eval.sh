#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/libero_eval_common.sh"

init_libero_goal_eval_env
resolve_dflash_goal_checkpoint

ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-0}"
DFLASH_NUM_DRAFT_LAYERS="${DFLASH_NUM_DRAFT_LAYERS:-1}"
RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-strict-${EVAL_EPOCH}-r${ACCEPT_THRESHOLD}}"

print_common_eval_config
echo "DFLASH_OUTPUT_DIR=${DFLASH_OUTPUT_DIR}"
echo "EVAL_EPOCH=${EVAL_EPOCH}"
echo "ACCEPT_THRESHOLD=${ACCEPT_THRESHOLD}"
echo "DFLASH_NUM_DRAFT_LAYERS=${DFLASH_NUM_DRAFT_LAYERS}"

python openvla/experiments/robot/libero/run_libero_goal_Spec.py \
  --model_family openvla \
  --pretrained_checkpoint "${VLA_PATH}" \
  --spec_checkpoint "${SPEC_CKPT}" \
  --draft_backend dflash \
  --use_spec True \
  --parallel_draft False \
  --task_suite_name libero_goal \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --center_crop True \
  --accept_threshold "${ACCEPT_THRESHOLD}" \
  --dflash_block_size 7 \
  --dflash_num_draft_layers "${DFLASH_NUM_DRAFT_LAYERS}" \
  --local_log_dir "${LOG_DIR}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --use_wandb "${USE_WANDB}" \
  --seed "${SEED}"
