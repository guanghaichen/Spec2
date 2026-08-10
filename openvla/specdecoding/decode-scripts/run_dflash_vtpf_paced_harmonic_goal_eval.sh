#!/usr/bin/env bash
set -euo pipefail

# VTPF-PacedHarmonic：在严格 target prefill 校验不变的前提下，组合两项
# 互补优化：
#   1. 每个 target 周期都允许上一条动作进入 prefill（k=1）；候选仍逐
#      token 接受，错误候选由 target 立即纠正。
#   2. 延长节拍为 T-H-H, T-H；第二个 hold 的 6 个连续控制维度按
#      1 / hold_depth 缩放为 0.5，gripper 不缩放，随后强制 target。
# 该入口不启用视觉特征缓存、动作组接受或树式 relaxed token 校验。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DFLASH_TEMPORAL_HOLD_ACTION_DECAY=inverse_age
export DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS=1
export DFLASH_TEMPORAL_ROUTE_LABEL="${DFLASH_TEMPORAL_ROUTE_LABEL:-PacedHarmonic}"
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-vtpf-paced-harmonic-e${EVAL_EPOCH:-latest}-p${DFLASH_TEMPORAL_VISUAL_BUDGET:-0.15}}"

bash "${SCRIPT_DIR}/run_dflash_vtpf_paced_budget_goal_eval.sh"
