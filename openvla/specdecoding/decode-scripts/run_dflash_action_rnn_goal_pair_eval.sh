#!/usr/bin/env bash
set -euo pipefail

# 一键评测 Action-RNN checkpoint 的 Goal strict + relaxed；不跨 LIBERO 子集。
# 推荐主实验是一层 Draft；具体深度和均匀层索引最终以 checkpoint 的 dflash_config.json 为准。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "/data/wulin" ]]; then
  DEFAULT_OUTPUT_DIR="/data/wulin/c/specvla-data/ckpt_goal_dflash_action_rnn_prefix_1layer_b8x2_4gpu"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  DEFAULT_OUTPUT_DIR="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_action_rnn_prefix_1layer_b8x2_4gpu"
elif [[ -d "/mnt/storage/cgh" ]]; then
  DEFAULT_OUTPUT_DIR="/mnt/storage/cgh/specvla-data/ckpt_goal_dflash_action_rnn_prefix_1layer_b8x2_4gpu"
else
  DEFAULT_OUTPUT_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_action_rnn_prefix_1layer_b8x2_4gpu"
fi

export TASK_SUITE_NAME=libero_goal
export DFLASH_OUTPUT_DIR="${DFLASH_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
export EVAL_EPOCH="${EVAL_EPOCH:-200}"
export DFLASH_NUM_DRAFT_LAYERS=1
export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=True
export SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"

echo "[Action-RNN strict] suite=libero_goal epoch=${EVAL_EPOCH} r=0"
SPEC_CKPT="" ACCEPT_THRESHOLD=0 \
RUN_ID_NOTE="dflash-action-rnn-strict-goal-e${EVAL_EPOCH}" \
  bash "${SCRIPT_DIR}/run_dflash_strict_libero_goal_eval.sh"

echo "[Action-RNN relaxed] suite=libero_goal epoch=${EVAL_EPOCH} r=${RELAXED_ACCEPT_THRESHOLD:-9}"
SPEC_CKPT="" ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD:-9}" \
RUN_ID_NOTE="dflash-action-rnn-relaxed-goal-e${EVAL_EPOCH}-r${RELAXED_ACCEPT_THRESHOLD:-9}" \
  bash "${SCRIPT_DIR}/run_dflash_libero_goal_eval.sh"
