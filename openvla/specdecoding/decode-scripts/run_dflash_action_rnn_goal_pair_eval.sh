#!/usr/bin/env bash
set -euo pipefail

# 一键评测 Action-RNN checkpoint 的 Goal strict + relaxed；不跨 LIBERO 子集。
# 推荐主实验是一层 Draft；具体深度和均匀层索引最终以 checkpoint 的 dflash_config.json 为准。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "/data/wulin" ]]; then
  DEFAULT_OUTPUT_DIR="/data/wulin/c/specvla-data/ckpt_goal_dflash_action_rnn_prefix_1layer_b8x2_4gpu"
  DEFAULT_LOG_DIR="/data/wulin/c/specvla-data/eval_logs"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  DEFAULT_OUTPUT_DIR="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_action_rnn_prefix_1layer_b8x2_4gpu"
  DEFAULT_LOG_DIR="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs"
elif [[ -d "/mnt/storage/cgh" ]]; then
  DEFAULT_OUTPUT_DIR="/mnt/storage/cgh/specvla-data/ckpt_goal_dflash_action_rnn_prefix_1layer_b8x2_4gpu"
  DEFAULT_LOG_DIR="/mnt/storage/cgh/specvla-data/eval_logs"
else
  DEFAULT_OUTPUT_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_action_rnn_prefix_1layer_b8x2_4gpu"
  DEFAULT_LOG_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/eval_logs"
fi

export TASK_SUITE_NAME=libero_goal
export DFLASH_OUTPUT_DIR="${DFLASH_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
export LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"
export EVAL_EPOCH="${EVAL_EPOCH:-200}"
export DFLASH_NUM_DRAFT_LAYERS=1
export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=True
export SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
export DFLASH_TREE_MODE="${DFLASH_TREE_MODE:-single_fork}"
export DFLASH_TREE_FIRST_ANCHOR_ONLY="${DFLASH_TREE_FIRST_ANCHOR_ONLY:-True}"
export DFLASH_TREE_AUTO_CALIBRATE="${DFLASH_TREE_AUTO_CALIBRATE:-True}"
export DFLASH_TREE_CALIBRATION_STEPS="${DFLASH_TREE_CALIBRATION_STEPS:-64}"
export DFLASH_TREE_CALIBRATION_POSITIONS="${DFLASH_TREE_CALIBRATION_POSITIONS:-0,2,3,4,5}"

STRICT_RUN_NOTE="${STRICT_RUN_NOTE:-dflash-action-rnn-tree-strict-goal-e${EVAL_EPOCH}}"
RELAXED_RUN_NOTE="${RELAXED_RUN_NOTE:-dflash-action-rnn-tree-group-relaxed-goal-e${EVAL_EPOCH}-r${RELAXED_ACCEPT_THRESHOLD:-9}}"

echo "[Action-RNN strict] suite=libero_goal epoch=${EVAL_EPOCH} r=0"
SPEC_CKPT="" ACCEPT_THRESHOLD=0 DFLASH_ACCEPTANCE_MODE=token \
DFLASH_TREE_BRANCH_POSITION=0 DFLASH_TREE_AUTO_CALIBRATE="${DFLASH_TREE_AUTO_CALIBRATE}" \
RUN_ID_NOTE="${STRICT_RUN_NOTE}" \
  bash "${SCRIPT_DIR}/run_dflash_strict_libero_goal_eval.sh"

STRICT_SUMMARY="$(find "${LOG_DIR}/dflash_strict" -maxdepth 1 -type f \
  -name "*--${STRICT_RUN_NOTE}-dflash_strict_summary.json" -print | sort | tail -n 1)"
if [[ -z "${STRICT_SUMMARY}" ]]; then
  echo "Cannot find strict summary for run note: ${STRICT_RUN_NOTE}" >&2
  exit 1
fi
SELECTED_TREE_POSITION="$(python - "${STRICT_SUMMARY}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(int(payload.get("dflash_tree_branch_position") or 0))
PY
)"
echo "[Action-RNN tree] strict calibration selected position=${SELECTED_TREE_POSITION}"

echo "[Action-RNN relaxed] suite=libero_goal epoch=${EVAL_EPOCH} r=${RELAXED_ACCEPT_THRESHOLD:-9}"
SPEC_CKPT="" ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD:-9}" \
DFLASH_ACCEPTANCE_MODE=action_group DFLASH_TREE_BRANCH_POSITION="${SELECTED_TREE_POSITION}" \
RUN_ID_NOTE="${RELAXED_RUN_NOTE}" \
  bash "${SCRIPT_DIR}/run_dflash_libero_goal_eval.sh"
