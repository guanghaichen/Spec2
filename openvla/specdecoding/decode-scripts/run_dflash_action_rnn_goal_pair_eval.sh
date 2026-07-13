#!/usr/bin/env bash
set -euo pipefail

# 一键评测三层 Action-RNN checkpoint 的 Goal strict + relaxed。
# 默认不启用 confidence truncation；先得到无额外 CUDA 标量同步的可比基线。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "/data/wulin" ]]; then
  DEFAULT_OUTPUT_DIR="/data/wulin/c/specvla-data/ckpt_goal_dflash_action_rnn_prefix_3layer_b8x2_4gpu"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  DEFAULT_OUTPUT_DIR="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_action_rnn_prefix_3layer_b8x2_4gpu"
elif [[ -d "/mnt/storage/cgh" ]]; then
  DEFAULT_OUTPUT_DIR="/mnt/storage/cgh/specvla-data/ckpt_goal_dflash_action_rnn_prefix_3layer_b8x2_4gpu"
else
  DEFAULT_OUTPUT_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_action_rnn_prefix_3layer_b8x2_4gpu"
fi

export TASK_SUITE_NAME=libero_goal
export DFLASH_OUTPUT_DIR="${DFLASH_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
export EVAL_EPOCH="${EVAL_EPOCH:-200}"
export DFLASH_NUM_DRAFT_LAYERS=3
export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=True
export DFLASH_CONFIDENCE_THRESHOLD="${DFLASH_CONFIDENCE_THRESHOLD:-0.0}"
export DFLASH_CONFIDENCE_MIN_TOKENS="${DFLASH_CONFIDENCE_MIN_TOKENS:-1}"
export SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"

echo "[Action-RNN strict] epoch=${EVAL_EPOCH} confidence=${DFLASH_CONFIDENCE_THRESHOLD}"
SPEC_CKPT="" ACCEPT_THRESHOLD=0 \
RUN_ID_NOTE="dflash-action-rnn-strict-goal-e${EVAL_EPOCH}-c${DFLASH_CONFIDENCE_THRESHOLD}" \
  bash "${SCRIPT_DIR}/run_dflash_strict_libero_goal_eval.sh"

echo "[Action-RNN relaxed] epoch=${EVAL_EPOCH} confidence=${DFLASH_CONFIDENCE_THRESHOLD}"
SPEC_CKPT="" ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD:-9}" \
RUN_ID_NOTE="dflash-action-rnn-relaxed-goal-e${EVAL_EPOCH}-r${RELAXED_ACCEPT_THRESHOLD:-9}-c${DFLASH_CONFIDENCE_THRESHOLD}" \
  bash "${SCRIPT_DIR}/run_dflash_libero_goal_eval.sh"
