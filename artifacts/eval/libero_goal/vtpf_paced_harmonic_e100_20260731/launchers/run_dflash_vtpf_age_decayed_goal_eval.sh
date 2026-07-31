#!/usr/bin/env bash
set -euo pipefail

# VTPF-TD Age-Decayed Hold：延长到第二个 hold 时，按命令年龄将
# 6 个连续控制维度缩放为 1 / hold_depth；首个 hold 保持 1.0，
# gripper 始终不做数值缩放。此处只改变执行的连续动作，不更改
# draft token、target 校验或下一帧强制 target 的边界。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DFLASH_TEMPORAL_HOLD_ACTION_DECAY=inverse_age
export DFLASH_TEMPORAL_ROUTE_LABEL=AgeDecayed
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-vtpf-td-age-decayed-e${EVAL_EPOCH:-latest}-p${DFLASH_TEMPORAL_VISUAL_BUDGET:-0.15}}"

bash "${SCRIPT_DIR}/run_dflash_vtpf_visual_budget_goal_eval.sh"
