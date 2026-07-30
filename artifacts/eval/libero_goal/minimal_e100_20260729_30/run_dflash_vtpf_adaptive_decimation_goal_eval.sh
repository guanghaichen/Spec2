#!/usr/bin/env bash
set -euo pipefail

# VTPF-TD-Adaptive（单组 Goal 评测）：
#   1. 第一帧 hold 完全复用已验证的 TD-Fast 行为；
#   2. 第二帧 hold 仅在两个 target 关键帧给出完全相同动作，且当前图像
#      相对最后 target 锚点的累计变化足够小时放行；
#   3. 第二帧 hold 后无条件强制 target，绝不允许第三次连续免校验。
#
# 必须显式提供要评测的 checkpoint，避免误用旧实验目录：
#   SPEC_CKPT=/absolute/path/epoch_100_step_044800 EVAL_EPOCH=100 \
#   NUM_TRIALS_PER_TASK=50 CUDA_VISIBLE_DEVICES=0 \
#     bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_adaptive_decimation_goal_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

USER_LOG_DIR="${LOG_DIR:-}"
init_libero_eval_env libero_goal

if [[ -z "${SPEC_CKPT:-}" ]]; then
  echo "请显式设置 SPEC_CKPT=/absolute/path/to/epoch_xxx_step_xxxxxx。" >&2
  exit 1
fi
if [[ ! -f "${SPEC_CKPT}/pytorch_model.bin" || ! -f "${SPEC_CKPT}/dflash_config.json" ]]; then
  echo "DFlash checkpoint 不完整: ${SPEC_CKPT}" >&2
  echo "需要 pytorch_model.bin 和 dflash_config.json。" >&2
  exit 1
fi
# SPEC_CKPT 是本入口唯一可信的权重来源；同步展示其父目录，避免日志中
# 出现虽然不会被使用、但容易误解为实际权重的历史默认目录。
export DFLASH_OUTPUT_DIR="$(dirname "${SPEC_CKPT}")"

export EVAL_EPOCH="${EVAL_EPOCH:-100}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
export SEED="${SEED:-7}"
export SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
export LOG_DIR="${USER_LOG_DIR:-${LOG_DIR}}/dflash_relaxed/简化版Draft+VTPF-TD-Adaptive-e${EVAL_EPOCH}"

# 模型与校验路径保持和 TD-Fast 相同，仅替换 hold 调度策略。
export DFLASH_NUM_DRAFT_LAYERS=1
export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=False
export DFLASH_TARGET_LOGITS_MODE=action_only
export DFLASH_ACCEPTANCE_MODE=token
export ACCEPT_THRESHOLD=0
export DFLASH_TREE_MODE=off
export DFLASH_TREE_BUDGET=0
export DFLASH_CONFIDENCE_THRESHOLD=0.0
export DFLASH_CONFIDENCE_MIN_TOKENS=1

export DFLASH_VERIFY_SKIP_MODE=active
export DFLASH_TEMPORAL_PREFILL_FUSION=True
export DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS=3
export DFLASH_TEMPORAL_PREFIX_CERT_TOKENS=0
export DFLASH_TEMPORAL_PREFILL_TREE=False
export DFLASH_TEMPORAL_ROUTE_MIN_COSINE=0.99
export DFLASH_TEMPORAL_ROUTE_STOP_ON_REJECT=True
export DFLASH_TEMPORAL_FUSE_VERIFY=True

# 第一帧不加视觉门；视觉信号只控制是否将预算从一次扩展到两次。
export DFLASH_TEMPORAL_BYPASS_MAX_PIXEL_RELATIVE_L2=1.0
export DFLASH_TEMPORAL_BYPASS_USE_PIXEL_GUARD=False
export DFLASH_VERIFY_SKIP_MIN_STABLE_ACTIONS=1
export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE=2
export DFLASH_TEMPORAL_HOLD_POLICY=adaptive
export DFLASH_TEMPORAL_ADAPTIVE_MIN_VERIFIED_RUN="${DFLASH_TEMPORAL_ADAPTIVE_MIN_VERIFIED_RUN:-2}"
export DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2="${DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2:-0.03}"

export DFLASH_PROFILE_STAGES=False
export DFLASH_DEBUG_COMPARE_TARGET_AR=False
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-vtpf-td-adaptive-goal-e${EVAL_EPOCH}-s${SEED}-vr${DFLASH_TEMPORAL_ADAPTIVE_MIN_VERIFIED_RUN}-p${DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2}}"

bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" relaxed
