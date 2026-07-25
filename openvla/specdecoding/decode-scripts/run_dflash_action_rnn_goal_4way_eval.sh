#!/usr/bin/env bash
set -euo pipefail

# 在同一 Goal checkpoint、seed 和计时口径下依次执行四组机制消融：
#   1. Action-RNN + strict
#   2. Action-RNN + dynamic DDTree + strict
#   3. Action-RNN + action-group acceptance
#   4. Action-RNN + dynamic DDTree + action-group acceptance
# DDTree 使用固定节点预算，不做在线校准；预算 0 表示每块自动取 q-1，
# 与线性校验的目标节点数相同，便于直接衡量动态分配分支带来的净收益。

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
export DFLASH_TREE_BUDGET="${DFLASH_TREE_BUDGET:-0}"
START_GROUP="${START_GROUP:-1}"

if [[ ! "${START_GROUP}" =~ ^[1-4]$ ]]; then
  echo "START_GROUP 必须是 1、2、3 或 4；当前值为 ${START_GROUP}。" >&2
  exit 1
fi

RELAXED_ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD:-9}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-dflash-action-rnn-ddtree-4way-goal-e${EVAL_EPOCH}}"
LINEAR_STRICT_NOTE="${RUN_ID_PREFIX}-linear-strict-r0"
TREE_STRICT_NOTE="${RUN_ID_PREFIX}-tree-strict-r0"
GROUP_NOTE="${RUN_ID_PREFIX}-linear-action-group-r${RELAXED_ACCEPT_THRESHOLD}"
TREE_GROUP_NOTE="${RUN_ID_PREFIX}-tree-action-group-r${RELAXED_ACCEPT_THRESHOLD}"

if (( START_GROUP <= 1 )); then
  echo "[1/4] Action-RNN + strict (tree off)"
  SPEC_CKPT="" ACCEPT_THRESHOLD=0 DFLASH_ACCEPTANCE_MODE=token \
  DFLASH_TREE_MODE=off \
  RUN_ID_NOTE="${LINEAR_STRICT_NOTE}" \
    bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict
else
  echo "[1/4] 已按 START_GROUP=${START_GROUP} 跳过；复用已有结果。"
fi

if (( START_GROUP <= 2 )); then
  echo "[2/4] Action-RNN + dynamic DDTree + strict"
  SPEC_CKPT="" ACCEPT_THRESHOLD=0 DFLASH_ACCEPTANCE_MODE=token \
  DFLASH_TREE_MODE=ddtree RUN_ID_NOTE="${TREE_STRICT_NOTE}" \
    bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict
else
  echo "[2/4] 已按 START_GROUP=${START_GROUP} 跳过；复用已有结果。"
fi

if (( START_GROUP <= 3 )); then
  echo "[3/4] Action-RNN + action-group acceptance (tree off)"
  SPEC_CKPT="" ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD}" \
  DFLASH_ACCEPTANCE_MODE=action_group DFLASH_TREE_MODE=off \
  RUN_ID_NOTE="${GROUP_NOTE}" \
    bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" relaxed
else
  echo "[3/4] 已按 START_GROUP=${START_GROUP} 跳过；复用已有结果。"
fi

echo "[4/4] Action-RNN + dynamic DDTree + action-group acceptance"
SPEC_CKPT="" ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD}" \
DFLASH_ACCEPTANCE_MODE=action_group DFLASH_TREE_MODE=ddtree \
RUN_ID_NOTE="${TREE_GROUP_NOTE}" \
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
print("method\tSR\tLength\tavg_accept\thit_rate\tmean_s\ttree_budget\ttree_nodes\ttree_depth")
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
        payload.get("dflash_tree_budget"),
        generation.get("tree_average_verified_nodes"),
        generation.get("tree_average_max_depth"),
    ]
    print("\t".join("NA" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value) for value in values))
PY
