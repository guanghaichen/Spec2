#!/usr/bin/env bash
set -euo pipefail

# VTPF-TD（Target-Anchored Temporal Decimation）：
# target 关键帧与一次动作保持交替执行，即 T -> hold -> T。hold 帧复用
# 最近一条 target-verified 动作并跳过完整 VLA prefill；由于默认最多保持
# 一帧，任何近似动作都会立刻被下一关键帧截断，不会形成连续漂移。
#
# 正式评测示例：
#   CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
#     bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_temporal_decimation_goal_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DFLASH_TEMPORAL_BYPASS_MAX_PIXEL_RELATIVE_L2="${DFLASH_TEMPORAL_BYPASS_MAX_PIXEL_RELATIVE_L2:-1.0}"
export DFLASH_TEMPORAL_BYPASS_USE_PIXEL_GUARD=False
export DFLASH_VERIFY_SKIP_MIN_STABLE_ACTIONS="${DFLASH_VERIFY_SKIP_MIN_STABLE_ACTIONS:-1}"
export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE="${DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE:-1}"
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-vtpf-temporal-decimation-goal-e${EVAL_EPOCH:-latest}-h${DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE}}"

bash "${SCRIPT_DIR}/run_dflash_vtpf_guarded_bypass_goal_eval.sh"
