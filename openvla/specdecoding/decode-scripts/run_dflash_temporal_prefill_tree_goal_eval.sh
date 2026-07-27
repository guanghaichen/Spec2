#!/usr/bin/env bash
set -euo pipefail

# VTPF 时序多候选 Prefill 树（仅 LIBERO-Goal）。
#
# 在当前图像必须执行的 target prefill 中，同时严格/宽松校验最多三条完整动作：
#   1. 保持上一条已执行动作；
#   2. 最近两条动作在连续动作空间中的恒速度外推；
#   3. 上上条已执行动作。
# 三条路径先合并公共前缀，再用祖先 mask 一次前向验证。没有额外 target
# forward；未命中的位置继续走原 DFlash。旧 VTPF 脚本与默认行为不受影响。
#
# 用法：
#   bash "$0" strict
#   bash "$0" relaxed

MODE="${1:-relaxed}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${MODE}" in
  strict)
    export ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-0}"
    export DFLASH_ACCEPTANCE_MODE=token
    ;;
  relaxed)
    export ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-9}"
    export DFLASH_ACCEPTANCE_MODE=action_group
    ;;
  *)
    echo "不支持的模式: ${MODE}; 应为 strict 或 relaxed。" >&2
    exit 1
    ;;
esac

export DFLASH_TARGET_LOGITS_MODE="${DFLASH_TARGET_LOGITS_MODE:-action_only}"
export DFLASH_TREE_MODE=off
export DFLASH_VERIFY_SKIP_MODE=route
export DFLASH_TEMPORAL_ROUTE_MIN_COSINE="${DFLASH_TEMPORAL_ROUTE_MIN_COSINE:-0.990}"
export DFLASH_TEMPORAL_ROUTE_STOP_ON_REJECT=True
export DFLASH_TEMPORAL_FUSE_VERIFY=True
export DFLASH_TEMPORAL_PREFILL_FUSION=False
export DFLASH_TEMPORAL_PREFILL_TREE=True
export DFLASH_TEMPORAL_PREFILL_TREE_MAX_CANDIDATES="${DFLASH_TEMPORAL_PREFILL_TREE_MAX_CANDIDATES:-3}"
export DFLASH_TEMPORAL_PREFILL_TREE_MIN_HISTORY="${DFLASH_TEMPORAL_PREFILL_TREE_MIN_HISTORY:-2}"
export DFLASH_PROFILE_STAGES="${DFLASH_PROFILE_STAGES:-False}"
export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-vtpf-temporal-tree-${MODE}-goal-e${EVAL_EPOCH:-200}-r${ACCEPT_THRESHOLD}}"

bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" "${MODE}"
