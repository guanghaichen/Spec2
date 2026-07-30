#!/usr/bin/env bash
set -euo pipefail

# VTPF-TD Visual Budget：第一帧保持沿用 TD-Fast；第二帧保持仅由当前图像
# 相对最近 target 锚点的累计 relative-L2 预算控制，随后强制 target。
# 这是与原 Adaptive（exact target run + visual）隔离的速度候选，不改权重。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SPEC_CKPT:-}" ]]; then
  echo "请显式设置 SPEC_CKPT=/absolute/path/to/checkpoint。" >&2
  exit 1
fi

export DFLASH_TEMPORAL_HOLD_POLICY=visual_budget
export DFLASH_TEMPORAL_ADAPTIVE_MIN_VERIFIED_RUN=1
# 0.15 是 2026-07-30 正式评测的 aggressive 速度点（3.679x / SR 0.672），
# 不是无精度退化的 balanced 默认方案。复现实验可通过环境变量显式固定；
# 做阈值消融时不得修改脚本本身或隐去对应 SR。
export DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2="${DFLASH_TEMPORAL_VISUAL_BUDGET:-0.15}"
export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE=2
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-vtpf-td-visual-budget-e${EVAL_EPOCH:-latest}-p${DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2}}"

# 复用 Adaptive launcher 的模型、VTPF、strict fallback 与日志隔离设置；
# 该 launcher 尊重上面显式给出的 hold policy。
export DFLASH_TEMPORAL_HOLD_POLICY_OVERRIDE=True
echo "[VisualBudget] aggressive p=${DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2}; formal p=0.15 requires reporting SR together with Speedup"
bash "${SCRIPT_DIR}/run_dflash_vtpf_adaptive_decimation_goal_eval.sh"
