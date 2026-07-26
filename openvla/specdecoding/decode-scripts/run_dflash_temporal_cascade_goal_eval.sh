#!/usr/bin/env bash
set -euo pipefail

# DFlash 时序级联推理：
#   shadow  只采集门控标签，所有动作仍完整验证；建议先运行此模式。
#   route   复用上一条已验证动作，融合 anchor/verify；拒绝后切回 DFlash。
#   prefill 把上一动作候选并入当前多模态 prefill，并在一次 target forward 中严格校验。
#   cascade 在严格 route 上，仅对稳定段跳过动作尾部校验。
#
# 示例：
#   MAX_EVAL_TASKS=10 NUM_TRIALS_PER_TASK=1 bash "$0" shadow
#   NUM_TRIALS_PER_TASK=50 bash "$0" prefill
#   NUM_TRIALS_PER_TASK=50 bash "$0" cascade  # approximate 消融

MODE="${1:-cascade}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DFLASH_TARGET_LOGITS_MODE="${DFLASH_TARGET_LOGITS_MODE:-action_only}"
export DFLASH_TREE_MODE=off
export DFLASH_ACCEPTANCE_MODE=token
export ACCEPT_THRESHOLD=0
export DFLASH_TEMPORAL_ROUTE_MIN_COSINE="${DFLASH_TEMPORAL_ROUTE_MIN_COSINE:-0.990}"
export DFLASH_TEMPORAL_ROUTE_STOP_ON_REJECT="${DFLASH_TEMPORAL_ROUTE_STOP_ON_REJECT:-True}"
export DFLASH_TEMPORAL_FUSE_VERIFY="${DFLASH_TEMPORAL_FUSE_VERIFY:-True}"
export DFLASH_TEMPORAL_PREFILL_FUSION="${DFLASH_TEMPORAL_PREFILL_FUSION:-False}"
export DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS="${DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS:-3}"
export DFLASH_VERIFY_SKIP_MIN_STABLE_ACTIONS="${DFLASH_VERIFY_SKIP_MIN_STABLE_ACTIONS:-4}"
export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE="${DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE:-1}"
export DFLASH_PROFILE_STAGES="${DFLASH_PROFILE_STAGES:-False}"
export DFLASH_DEBUG_COMPARE_TARGET_AR="${DFLASH_DEBUG_COMPARE_TARGET_AR:-False}"

case "${MODE}" in
  shadow)
    export DFLASH_VERIFY_SKIP_MODE=shadow
    export DFLASH_VERIFY_SKIP_MIN_TEMPORAL_COSINE="${DFLASH_VERIFY_SKIP_MIN_TEMPORAL_COSINE:-0.998}"
    export TIMING_SCOPE="${TIMING_SCOPE:-full_suite}"
    DEFAULT_NOTE="dflash-temporal-cascade-shadow-goal-e${EVAL_EPOCH:-200}"
    ;;
  route)
    export DFLASH_VERIFY_SKIP_MODE=route
    export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
    DEFAULT_NOTE="dflash-temporal-route-goal-e${EVAL_EPOCH:-200}"
    ;;
  prefill)
    export DFLASH_VERIFY_SKIP_MODE=route
    export DFLASH_TEMPORAL_PREFILL_FUSION=True
    export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
    DEFAULT_NOTE="dflash-temporal-prefill-fusion-goal-e${EVAL_EPOCH:-200}"
    ;;
  cascade)
    export DFLASH_VERIFY_SKIP_MODE=active
    export DFLASH_VERIFY_SKIP_MIN_TEMPORAL_COSINE="${DFLASH_VERIFY_SKIP_MIN_TEMPORAL_COSINE:-0.998}"
    export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
    DEFAULT_NOTE="dflash-temporal-cascade-goal-e${EVAL_EPOCH:-200}"
    ;;
  *)
    echo "不支持的模式: ${MODE}; 应为 shadow、route、prefill 或 cascade。" >&2
    exit 1
    ;;
esac

export RUN_ID_NOTE="${RUN_ID_NOTE:-${DEFAULT_NOTE}}"

bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict
