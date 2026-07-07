#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EVAL_EPOCH="${EVAL_EPOCH:-200}"
export ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-0}"
export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING="${DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING:-True}"
export SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-residual-strict-e${EVAL_EPOCH}-r${ACCEPT_THRESHOLD}}"

echo "SYNC_CUDA_TIMING=${SYNC_CUDA_TIMING}"
echo "TIMING_SCOPE=${TIMING_SCOPE}"

bash "${SCRIPT_DIR}/run_dflash_strict_libero_goal_eval.sh"
