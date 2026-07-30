#!/usr/bin/env bash
set -euo pipefail

# 使用同一个 Minimal Draft checkpoint 串行评测三条 Goal 路径：
#   1. 线性 DFlash strict
#   2. DFlash + VTPF strict
#   3. DFlash + VTPF-TD-Fast relaxed
#
# Minimal checkpoint 不含 Action-RNN，因此三条路径都强制 RNN-off。每组日志
# 写入独立的、带 epoch 标识的目录。中断后可用 START_CASE=2 或 3 续跑。
#
# 示例：
#   SPEC_CKPT=/absolute/path/epoch_100_step_044800 EVAL_EPOCH=100 \
#     bash openvla/specdecoding/decode-scripts/run_dflash_minimal_goal_3way_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

USER_DFLASH_OUTPUT_DIR="${DFLASH_OUTPUT_DIR:-}"
init_libero_eval_env libero_goal

CHECKPOINT="${SPEC_CKPT:-${1:-}}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "请通过 SPEC_CKPT 或第一个位置参数提供 Minimal checkpoint 目录。" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}/pytorch_model.bin" || ! -f "${CHECKPOINT}/dflash_config.json" ]]; then
  echo "Minimal checkpoint 不完整: ${CHECKPOINT}" >&2
  echo "需要 pytorch_model.bin 和 dflash_config.json。" >&2
  exit 1
fi

export SPEC_CKPT="${CHECKPOINT}"
export DFLASH_OUTPUT_DIR="${USER_DFLASH_OUTPUT_DIR:-$(dirname "${CHECKPOINT}")}"
export EVAL_EPOCH="${EVAL_EPOCH:-100}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
export SEED="${SEED:-7}"
export SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
export DFLASH_NUM_DRAFT_LAYERS=1
export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=False
export DFLASH_TARGET_LOGITS_MODE=action_only
export DFLASH_TREE_MODE=off
export DFLASH_TREE_BUDGET=0
export DFLASH_ACCEPTANCE_MODE=token
export ACCEPT_THRESHOLD=0
export DFLASH_CONFIDENCE_THRESHOLD=0.0
export DFLASH_CONFIDENCE_MIN_TOKENS=1
export DFLASH_TEMPORAL_PREFIX_CERT_TOKENS=0
export DFLASH_TEMPORAL_PREFILL_TREE=False
export DFLASH_PROFILE_STAGES=False
export DFLASH_DEBUG_COMPARE_TARGET_AR=False

START_CASE="${START_CASE:-1}"
if [[ ! "${START_CASE}" =~ ^[123]$ ]]; then
  echo "START_CASE 必须为 1、2 或 3，当前值为 ${START_CASE}。" >&2
  exit 1
fi

LOG_ROOT="${LOG_ROOT:-${LOG_DIR}}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-dflash-minimal-goal-e${EVAL_EPOCH}-s${SEED}}"

echo "Minimal checkpoint: ${SPEC_CKPT}"
echo "Protocol: trials=${NUM_TRIALS_PER_TASK}, seed=${SEED}, timing=${TIMING_SCOPE}, sync=${SYNC_CUDA_TIMING}"
echo "START_CASE=${START_CASE}"

if (( START_CASE <= 1 )); then
  echo "[1/3] Minimal DFlash linear strict"
  (
    export LOG_DIR="${LOG_ROOT}/dflash_strict/简化版Draft-e${EVAL_EPOCH}"
    export RUN_ID_NOTE="${RUN_ID_PREFIX}-linear-strict"
    export DFLASH_VERIFY_SKIP_MODE=off
    export DFLASH_TEMPORAL_ROUTE_MIN_COSINE=1.0
    export DFLASH_TEMPORAL_PREFILL_FUSION=False
    export DFLASH_TEMPORAL_BYPASS_MAX_PIXEL_RELATIVE_L2=0.0
    export DFLASH_TEMPORAL_BYPASS_USE_PIXEL_GUARD=True
    bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict
  )
fi

if (( START_CASE <= 2 )); then
  echo "[2/3] Minimal DFlash + VTPF strict"
  (
    export LOG_DIR="${LOG_ROOT}/dflash_strict/简化版Draft+VTPF-e${EVAL_EPOCH}"
    export RUN_ID_NOTE="${RUN_ID_PREFIX}-vtpf-strict"
    export DFLASH_VERIFY_SKIP_MODE=route
    export DFLASH_TEMPORAL_PREFILL_FUSION=True
    export DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS=3
    export DFLASH_TEMPORAL_ROUTE_MIN_COSINE=0.990
    export DFLASH_TEMPORAL_ROUTE_STOP_ON_REJECT=True
    export DFLASH_TEMPORAL_FUSE_VERIFY=True
    export DFLASH_TEMPORAL_BYPASS_MAX_PIXEL_RELATIVE_L2=0.0
    export DFLASH_TEMPORAL_BYPASS_USE_PIXEL_GUARD=True
    bash "${SCRIPT_DIR}/run_dflash_temporal_cascade_goal_eval.sh" prefill
  )
fi

if (( START_CASE <= 3 )); then
  echo "[3/3] Minimal DFlash + VTPF-TD-Fast"
  (
    export LOG_DIR="${LOG_ROOT}/dflash_relaxed/简化版Draft+VTPF-TD-e${EVAL_EPOCH}"
    export RUN_ID_NOTE="${RUN_ID_PREFIX}-vtpf-td-fast"
    export DFLASH_VERIFY_SKIP_MIN_STABLE_ACTIONS=1
    export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE=1
    bash "${SCRIPT_DIR}/run_dflash_vtpf_temporal_decimation_goal_eval.sh"
  )
fi

echo "三路评测完成。日志根目录: ${LOG_ROOT}"
