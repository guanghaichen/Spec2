#!/usr/bin/env bash
set -euo pipefail

# 一键评测 Action-RNN checkpoint 的 Goal strict + relaxed；不跨 LIBERO 子集。
# 推荐主实验是一层 Draft；具体深度和均匀层索引最终以 checkpoint 的 dflash_config.json 为准。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"
init_libero_eval_env libero_goal

export TASK_SUITE_NAME=libero_goal
export DFLASH_OUTPUT_DIR
export LOG_DIR
export EVAL_EPOCH="${EVAL_EPOCH:-100}"
export DFLASH_NUM_DRAFT_LAYERS=1
export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=True
export SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
export DFLASH_TREE_MODE="${DFLASH_TREE_MODE:-ddtree}"
export DFLASH_TREE_BUDGET="${DFLASH_TREE_BUDGET:-0}"

STRICT_RUN_NOTE="${STRICT_RUN_NOTE:-dflash-action-rnn-ddtree-strict-goal-e${EVAL_EPOCH}}"
RELAXED_RUN_NOTE="${RELAXED_RUN_NOTE:-dflash-action-rnn-ddtree-group-relaxed-goal-e${EVAL_EPOCH}-r${RELAXED_ACCEPT_THRESHOLD:-9}}"

echo "[Action-RNN strict] suite=libero_goal epoch=${EVAL_EPOCH} r=0"
SPEC_CKPT="" ACCEPT_THRESHOLD=0 DFLASH_ACCEPTANCE_MODE=token \
RUN_ID_NOTE="${STRICT_RUN_NOTE}" \
  bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict

if [[ "${DRY_RUN:-False}" == "True" ]]; then
  echo "DRY_RUN=True：strict 已通过；relaxed 使用同一固定预算 DDTree 继续检查。"
  SPEC_CKPT="" ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD:-9}" \
  DFLASH_ACCEPTANCE_MODE=action_group RUN_ID_NOTE="${RELAXED_RUN_NOTE}" \
    bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" relaxed
  exit 0
fi

echo "[Action-RNN relaxed] suite=libero_goal epoch=${EVAL_EPOCH} r=${RELAXED_ACCEPT_THRESHOLD:-9}"
SPEC_CKPT="" ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD:-9}" \
DFLASH_ACCEPTANCE_MODE=action_group \
RUN_ID_NOTE="${RELAXED_RUN_NOTE}" \
  bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" relaxed
