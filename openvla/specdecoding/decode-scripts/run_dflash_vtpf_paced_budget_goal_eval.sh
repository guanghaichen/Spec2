#!/usr/bin/env bash
set -euo pipefail

# VTPF-TD Paced Budget：第一跳保持 TD-Fast，第二跳继续受视觉预算约束；
# 一旦使用第二跳，下一个 target 周期最多只允许一跳，形成 T-H-H, T-H
# 的硬节拍。该入口不修改 checkpoint，也不启用 Action-RNN 或 relaxed token。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SPEC_CKPT:-}" ]]; then
  echo "请显式设置 SPEC_CKPT=/absolute/path/to/checkpoint。" >&2
  exit 1
fi

export DFLASH_TEMPORAL_HOLD_POLICY_OVERRIDE=True
export DFLASH_TEMPORAL_HOLD_POLICY=paced_budget
export DFLASH_TEMPORAL_ADAPTIVE_MIN_VERIFIED_RUN=1
export DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2="${DFLASH_TEMPORAL_VISUAL_BUDGET:-0.15}"
export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE=2
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-vtpf-td-paced-budget-e${EVAL_EPOCH:-latest}-p${DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2}}"

echo "[PacedBudget] cadence=T-H-H,T-H visual_budget=${DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2}"
bash "${SCRIPT_DIR}/run_dflash_vtpf_adaptive_decimation_goal_eval.sh"
