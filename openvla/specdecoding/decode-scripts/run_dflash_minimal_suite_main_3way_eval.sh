#!/usr/bin/env bash
set -euo pipefail

# 使用同一个 suite-specific Minimal Draft checkpoint 串行评测论文主表三条路径：
#   1. DFlash strict
#   2. DFlash + VTPF strict
#   3. DFlash + VTPF + PacedHarmonic relaxed
#
# 默认仍是 libero_goal。其它 suite 必须显式设置 TASK_SUITE_NAME，且
# SPEC_CKPT 必须指向该 suite 自己训练的 Draft，严禁跨 suite 混用。
# 三组结果写入 LOG_ROOT/<suite>/ 下的独立目录；中断后可用 START_CASE=2/3 续跑。
#
# Spatial 示例：
#   TASK_SUITE_NAME=libero_spatial \
#   SPEC_CKPT=/absolute/path/spatial/epoch_100_step_062200 \
#   EVAL_EPOCH=100 NUM_TRIALS_PER_TASK=50 \
#     bash openvla/specdecoding/decode-scripts/run_dflash_minimal_suite_main_3way_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

REQUESTED_TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
init_libero_eval_env "${REQUESTED_TASK_SUITE_NAME}"
export TASK_SUITE_NAME TASK_SUITE_SLUG

CHECKPOINT="${SPEC_CKPT:-${1:-}}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "请通过 SPEC_CKPT 或第一个位置参数提供 ${TASK_SUITE_SLUG} Minimal checkpoint。" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}/pytorch_model.bin" || ! -f "${CHECKPOINT}/dflash_config.json" ]]; then
  echo "Minimal checkpoint 不完整: ${CHECKPOINT}" >&2
  echo "需要 pytorch_model.bin 和 dflash_config.json。" >&2
  exit 1
fi

export SPEC_CKPT="${CHECKPOINT}"
export DFLASH_OUTPUT_DIR="$(dirname "${CHECKPOINT}")"
export EVAL_EPOCH="${EVAL_EPOCH:-100}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
export TRIAL_START_INDEX="${TRIAL_START_INDEX:-0}"
export SEED="${SEED:-7}"
export SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"

# 三路共同的 Minimal Draft 与精确 token 校验配置。
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

LOG_ROOT="${LOG_ROOT:-${LOG_DIR}/${TASK_SUITE_SLUG}}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-dflash-minimal-${TASK_SUITE_SLUG}-e${EVAL_EPOCH}-s${SEED}}"

echo "Task suite: ${TASK_SUITE_NAME}"
echo "Target model: ${VLA_PATH}"
echo "Minimal checkpoint: ${SPEC_CKPT}"
echo "Protocol: trials=${NUM_TRIALS_PER_TASK}, start=${TRIAL_START_INDEX}, seed=${SEED}, timing=${TIMING_SCOPE}, sync=${SYNC_CUDA_TIMING}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "START_CASE=${START_CASE}"

if (( START_CASE <= 1 )); then
  echo "[1/3] DFlash strict"
  (
    export LOG_DIR="${LOG_ROOT}/dflash_strict/DFlash-e${EVAL_EPOCH}"
    export RUN_ID_NOTE="${RUN_ID_PREFIX}-strict"
    export DFLASH_VERIFY_SKIP_MODE=off
    export DFLASH_TEMPORAL_ROUTE_MIN_COSINE=1.0
    export DFLASH_TEMPORAL_PREFILL_FUSION=False
    export DFLASH_TEMPORAL_BYPASS_MAX_PIXEL_RELATIVE_L2=0.0
    export DFLASH_TEMPORAL_BYPASS_USE_PIXEL_GUARD=True
    bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict
  )
fi

if (( START_CASE <= 2 )); then
  echo "[2/3] DFlash + VTPF strict"
  (
    export LOG_DIR="${LOG_ROOT}/dflash_strict/DFlash+VTPF-e${EVAL_EPOCH}"
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
  echo "[3/3] DFlash + VTPF + PacedHarmonic"
  (
    # PacedHarmonic launcher 自己追加方法目录，故这里只传 suite 日志根目录。
    export LOG_DIR="${LOG_ROOT}"
    export RUN_ID_NOTE="${RUN_ID_PREFIX}-vtpf-paced-harmonic"
    export DFLASH_TEMPORAL_VISUAL_BUDGET="${DFLASH_TEMPORAL_VISUAL_BUDGET:-0.15}"
    bash "${SCRIPT_DIR}/run_dflash_vtpf_paced_harmonic_goal_eval.sh"
  )
fi

echo "三路主表评测完成。日志根目录: ${LOG_ROOT}"
