#!/usr/bin/env bash
set -euo pipefail

# 当前 DFlash 只在 LIBERO-Goal 上训练，因此本入口故意不接受其它 suite。
# 用法：EVAL_EPOCH=100 bash .../run_dflash_goal_eval.sh [strict|relaxed]

MODE="${1:-${EVAL_MODE:-strict}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

init_libero_eval_env libero_goal
resolve_dflash_checkpoint

case "${MODE}" in
  strict)
    DEFAULT_THRESHOLD=0
    DEFAULT_ACCEPTANCE_MODE=token
    EVAL_ENTRY=openvla/experiments/robot/libero/run_libero_goal_Spec.py
    ;;
  relaxed)
    DEFAULT_THRESHOLD=9
    DEFAULT_ACCEPTANCE_MODE=action_group
    EVAL_ENTRY=openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py
    ;;
  *) echo "不支持的模式: ${MODE}; 应为 strict 或 relaxed。" >&2; exit 1 ;;
esac

ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-${DEFAULT_THRESHOLD}}"
DFLASH_NUM_DRAFT_LAYERS="${DFLASH_NUM_DRAFT_LAYERS:-1}"
DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING="${DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING:-True}"
DFLASH_CONFIDENCE_THRESHOLD="${DFLASH_CONFIDENCE_THRESHOLD:-0.0}"
DFLASH_CONFIDENCE_MIN_TOKENS="${DFLASH_CONFIDENCE_MIN_TOKENS:-1}"
DFLASH_ACCEPTANCE_MODE="${DFLASH_ACCEPTANCE_MODE:-${DEFAULT_ACCEPTANCE_MODE}}"
DFLASH_TREE_MODE="${DFLASH_TREE_MODE:-off}"
DFLASH_TREE_BUDGET="${DFLASH_TREE_BUDGET:-0}"
RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-action-rnn-${MODE}-goal-e${EVAL_EPOCH}-r${ACCEPT_THRESHOLD}}"

print_common_eval_config
echo "METHOD=dflash_action_rnn_${MODE}"
echo "DFLASH_OUTPUT_DIR=${DFLASH_OUTPUT_DIR}"
echo "EVAL_EPOCH=${EVAL_EPOCH}"
echo "ACCEPT_THRESHOLD=${ACCEPT_THRESHOLD}"
echo "DFLASH_NUM_DRAFT_LAYERS=${DFLASH_NUM_DRAFT_LAYERS}"
echo "DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=${DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING}"
echo "DFLASH_ACCEPTANCE_MODE=${DFLASH_ACCEPTANCE_MODE}"
echo "DFLASH_TREE_MODE=${DFLASH_TREE_MODE}"
echo "DFLASH_TREE_BUDGET=${DFLASH_TREE_BUDGET} (0=每个块使用与线性校验相同的 q-1 个节点)"

if [[ "${DRY_RUN:-False}" == "True" ]]; then
  echo "DRY_RUN=True：路径、checkpoint 和配置检查通过，不启动 LIBERO。"
  exit 0
fi

python "${EVAL_ENTRY}" \
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
  --dflash_use_causal_residual_sampling "${DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING}" \
  --dflash_confidence_threshold "${DFLASH_CONFIDENCE_THRESHOLD}" \
  --dflash_confidence_min_tokens "${DFLASH_CONFIDENCE_MIN_TOKENS}" \
  --dflash_acceptance_mode "${DFLASH_ACCEPTANCE_MODE}" \
  --dflash_tree_mode "${DFLASH_TREE_MODE}" \
  --dflash_tree_budget "${DFLASH_TREE_BUDGET}" \
  --local_log_dir "${LOG_DIR}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --sync_cuda_timing "${SYNC_CUDA_TIMING}" \
  --timing_scope "${TIMING_SCOPE}" \
  --use_wandb "${USE_WANDB}" \
  --seed "${SEED}"
