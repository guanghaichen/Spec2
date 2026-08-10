#!/usr/bin/env bash
set -euo pipefail

# 最终 PacedHarmonic 入口：H1 与 H2 都由最近一次 Target 关键帧管控。
#   - 候选动作始终来自最近一次 Target 输出；
#   - 单位深度预算 beta 默认为 0.075；
#   - 第 d 个 Hold 使用 d * beta 的累计视觉界；
#   - Pace 债务、H2 的 1/2 连续动作缩放和 gripper 原值保持不变。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UNIT_VISUAL_BUDGET="${DFLASH_TEMPORAL_UNIT_VISUAL_BUDGET:-0.075}"
H1_VISUAL_BUDGET="${UNIT_VISUAL_BUDGET}"
H2_VISUAL_BUDGET="$(awk -v beta="${UNIT_VISUAL_BUDGET}" 'BEGIN { printf "%.12g", 2 * beta }')"
export DFLASH_TEMPORAL_BYPASS_MAX_PIXEL_RELATIVE_L2=1.0
export DFLASH_TEMPORAL_BYPASS_USE_PIXEL_GUARD=False
export DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2="${H2_VISUAL_BUDGET}"
export DFLASH_TEMPORAL_DEPTH_VISUAL_BOUNDS="${H1_VISUAL_BUDGET},${H2_VISUAL_BUDGET}"
export DFLASH_TEMPORAL_ROUTE_LABEL=PacedHarmonicDualAnchor
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-vtpf-paced-harmonic-dual-anchor-e${EVAL_EPOCH:-latest}-h1p${H1_VISUAL_BUDGET}-h2p${H2_VISUAL_BUDGET}}"

echo "[PacedHarmonicDualAnchor] H1_target_l2<=${H1_VISUAL_BUDGET} H2_target_l2<=${H2_VISUAL_BUDGET}"
bash "${SCRIPT_DIR}/run_dflash_vtpf_paced_harmonic_goal_eval.sh"
