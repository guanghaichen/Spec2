#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/libero_eval_common.sh"

TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
init_libero_eval_env "${TASK_SUITE_NAME}"
resolve_dflash_checkpoint

# Keep relaxed DFlash comparable with Spec-VLA: Long/libero_10 uses r=5,
# while Goal/Object/Spatial use r=9. An explicit environment value wins.
if [[ "${TASK_SUITE_NAME}" == "libero_10" ]]; then
  DEFAULT_RELAXED_ACCEPT_THRESHOLD=5
else
  DEFAULT_RELAXED_ACCEPT_THRESHOLD=9
fi
ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-${DEFAULT_RELAXED_ACCEPT_THRESHOLD}}"
DFLASH_NUM_DRAFT_LAYERS="${DFLASH_NUM_DRAFT_LAYERS:-1}"
RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-relaxed-${TASK_SUITE_SLUG}-e${EVAL_EPOCH}-r${ACCEPT_THRESHOLD}}"
SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING="${DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING:-False}"
DFLASH_CONFIDENCE_THRESHOLD="${DFLASH_CONFIDENCE_THRESHOLD:-0.0}"
DFLASH_CONFIDENCE_MIN_TOKENS="${DFLASH_CONFIDENCE_MIN_TOKENS:-1}"
DFLASH_ACCEPTANCE_MODE="${DFLASH_ACCEPTANCE_MODE:-action_group}"
DFLASH_TREE_MODE="${DFLASH_TREE_MODE:-off}"
DFLASH_TREE_BRANCH_POSITION="${DFLASH_TREE_BRANCH_POSITION:-0}"
DFLASH_TREE_FIRST_ANCHOR_ONLY="${DFLASH_TREE_FIRST_ANCHOR_ONLY:-True}"

print_common_eval_config
echo "DFLASH_OUTPUT_DIR=${DFLASH_OUTPUT_DIR}"
echo "EVAL_EPOCH=${EVAL_EPOCH}"
echo "ACCEPT_THRESHOLD=${ACCEPT_THRESHOLD}"
echo "DFLASH_NUM_DRAFT_LAYERS=${DFLASH_NUM_DRAFT_LAYERS}"
echo "DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=${DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING}"
echo "DFLASH_CONFIDENCE_THRESHOLD=${DFLASH_CONFIDENCE_THRESHOLD}"
echo "DFLASH_CONFIDENCE_MIN_TOKENS=${DFLASH_CONFIDENCE_MIN_TOKENS}"
echo "DFLASH_ACCEPTANCE_MODE=${DFLASH_ACCEPTANCE_MODE}"
echo "DFLASH_TREE_MODE=${DFLASH_TREE_MODE}"
echo "DFLASH_TREE_BRANCH_POSITION=${DFLASH_TREE_BRANCH_POSITION}"
echo "DFLASH_TREE_FIRST_ANCHOR_ONLY=${DFLASH_TREE_FIRST_ANCHOR_ONLY}"
echo "SYNC_CUDA_TIMING=${SYNC_CUDA_TIMING}"
echo "TIMING_SCOPE=${TIMING_SCOPE}"

python openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py \
  --model_family openvla \
  --pretrained_checkpoint "${VLA_PATH}" \
  --spec_checkpoint "${SPEC_CKPT}" \
  --draft_backend dflash \
  --use_spec True \
  --parallel_draft False \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --center_crop True \
  --accept_threshold "${ACCEPT_THRESHOLD}" \
  --dflash_block_size 7 \
  --dflash_num_draft_layers "${DFLASH_NUM_DRAFT_LAYERS}" \
  --dflash_use_causal_residual_sampling "${DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING}" \
  --dflash_confidence_threshold "${DFLASH_CONFIDENCE_THRESHOLD}" \
  --dflash_confidence_min_tokens "${DFLASH_CONFIDENCE_MIN_TOKENS}" \
  --dflash_acceptance_mode "${DFLASH_ACCEPTANCE_MODE}" \
  --dflash_tree_mode "${DFLASH_TREE_MODE}" \
  --dflash_tree_branch_position "${DFLASH_TREE_BRANCH_POSITION}" \
  --dflash_tree_first_anchor_only "${DFLASH_TREE_FIRST_ANCHOR_ONLY}" \
  --local_log_dir "${LOG_DIR}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --sync_cuda_timing "${SYNC_CUDA_TIMING}" \
  --timing_scope "${TIMING_SCOPE}" \
  --use_wandb "${USE_WANDB}" \
  --seed "${SEED}"
