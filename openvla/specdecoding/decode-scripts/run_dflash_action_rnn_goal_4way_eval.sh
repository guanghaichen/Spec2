#!/usr/bin/env bash
set -euo pipefail

# 在同一 Goal checkpoint、seed 和计时口径下依次执行四组机制消融：
#   1. Action-RNN + strict
#   2. Action-RNN + sparse tree + strict
#   3. Action-RNN + action-group acceptance
#   4. Action-RNN + sparse tree + action-group acceptance
# 树版 strict 负责从 off/p2/p3/p4/p5 中校准分叉位置；第四组复用该位置。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"
init_libero_eval_env libero_goal

export TASK_SUITE_NAME=libero_goal
export DFLASH_OUTPUT_DIR
export LOG_DIR
export EVAL_EPOCH="${EVAL_EPOCH:-200}"
export DFLASH_NUM_DRAFT_LAYERS="${DFLASH_NUM_DRAFT_LAYERS:-1}"
export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING="${DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING:-True}"
export SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
export DFLASH_TREE_FIRST_ANCHOR_ONLY="${DFLASH_TREE_FIRST_ANCHOR_ONLY:-True}"
export DFLASH_TREE_CALIBRATION_STEPS="${DFLASH_TREE_CALIBRATION_STEPS:-64}"
export DFLASH_TREE_CALIBRATION_POSITIONS="${DFLASH_TREE_CALIBRATION_POSITIONS:-0,2,3,4,5}"

RELAXED_ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD:-9}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-dflash-action-rnn-4way-goal-e${EVAL_EPOCH}}"
LINEAR_STRICT_NOTE="${RUN_ID_PREFIX}-linear-strict-r0"
TREE_STRICT_NOTE="${RUN_ID_PREFIX}-tree-strict-r0"
GROUP_NOTE="${RUN_ID_PREFIX}-linear-action-group-r${RELAXED_ACCEPT_THRESHOLD}"
TREE_GROUP_NOTE="${RUN_ID_PREFIX}-tree-action-group-r${RELAXED_ACCEPT_THRESHOLD}"

echo "[1/4] Action-RNN + strict (tree off)"
SPEC_CKPT="" ACCEPT_THRESHOLD=0 DFLASH_ACCEPTANCE_MODE=token \
DFLASH_TREE_MODE=off DFLASH_TREE_AUTO_CALIBRATE=False \
RUN_ID_NOTE="${LINEAR_STRICT_NOTE}" \
  bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict

echo "[2/4] Action-RNN + sparse tree + strict (calibrate once)"
SPEC_CKPT="" ACCEPT_THRESHOLD=0 DFLASH_ACCEPTANCE_MODE=token \
DFLASH_TREE_MODE=single_fork DFLASH_TREE_BRANCH_POSITION=0 \
DFLASH_TREE_AUTO_CALIBRATE=True RUN_ID_NOTE="${TREE_STRICT_NOTE}" \
  bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict

if [[ "${DRY_RUN:-False}" == "True" ]]; then
  SELECTED_TREE_POSITION=0
else
  TREE_STRICT_SUMMARY="$(find "${LOG_DIR}/dflash_strict" -maxdepth 1 -type f \
    -name "*--${TREE_STRICT_NOTE}-dflash_strict_summary.json" -print | sort | tail -n 1)"
  if [[ -z "${TREE_STRICT_SUMMARY}" ]]; then
    echo "找不到树版 strict summary: ${TREE_STRICT_NOTE}" >&2
    exit 1
  fi
  SELECTED_TREE_POSITION="$(python - "${TREE_STRICT_SUMMARY}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(int(payload.get("dflash_tree_branch_position") or 0))
PY
)"
fi
echo "树校准选中的分叉位置: ${SELECTED_TREE_POSITION}（0 表示关闭）"

echo "[3/4] Action-RNN + action-group acceptance (tree off)"
SPEC_CKPT="" ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD}" \
DFLASH_ACCEPTANCE_MODE=action_group DFLASH_TREE_MODE=off \
DFLASH_TREE_AUTO_CALIBRATE=False RUN_ID_NOTE="${GROUP_NOTE}" \
  bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" relaxed

echo "[4/4] Action-RNN + sparse tree + action-group acceptance"
SPEC_CKPT="" ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD}" \
DFLASH_ACCEPTANCE_MODE=action_group DFLASH_TREE_MODE=single_fork \
DFLASH_TREE_BRANCH_POSITION="${SELECTED_TREE_POSITION}" \
DFLASH_TREE_AUTO_CALIBRATE=False RUN_ID_NOTE="${TREE_GROUP_NOTE}" \
  bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" relaxed

if [[ "${DRY_RUN:-False}" == "True" ]]; then
  echo "DRY_RUN=True：四组配置和 checkpoint 路径均已通过检查。"
  exit 0
fi

python - "${LOG_DIR}" \
  "${LINEAR_STRICT_NOTE}" "${TREE_STRICT_NOTE}" "${GROUP_NOTE}" "${TREE_GROUP_NOTE}" <<'PY'
import json
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
notes = sys.argv[2:]
print("\n四组评测摘要（Speedup 请用同机 paper-AR 分母计算）")
print("method\tSR\tLength\tavg_accept\thit_rate\tmean_s\ttree_pos")
for note in notes:
    family = "dflash_relaxed" if "action-group" in note else "dflash_strict"
    matches = sorted((log_dir / family).glob(f"*--{note}-{family}_summary.json"))
    if not matches:
        raise SystemExit(f"Missing summary for {note}")
    payload = json.loads(matches[-1].read_text(encoding="utf-8"))
    generation = payload.get("generation") or {}
    timing = payload.get("timing") or {}
    values = [
        note,
        payload.get("success_rate"),
        generation.get("length"),
        generation.get("avg_accept_length"),
        generation.get("overall_hit_rate"),
        timing.get("mean"),
        payload.get("dflash_tree_branch_position"),
    ]
    print("\t".join("NA" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value) for value in values))
PY
