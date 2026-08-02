#!/usr/bin/env bash
set -euo pipefail

# P0 机制证据入口。三组短评测使用相同 suite、task、初始状态与 seed：
#   profile: 同步 CUDA 的 DFlash 逐阶段成本；
#   temporal: 论文 wrapped AR 分母路径，只记录时序冗余；
#   parity: strict VTPF fused prefill 与串行 target AR 的因果前缀等价性。
# 输出同时保留完整原始日志、SHA-256 manifest、CSV 和 ICLR 尺寸 PDF/PNG。

SUITE_ARG="${1:-goal}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

case "${SUITE_ARG}" in
  goal|libero_goal) TASK_SUITE_NAME=libero_goal ;;
  spatial|libero_spatial) TASK_SUITE_NAME=libero_spatial ;;
  *)
    echo "P0 当前只接受 goal 或 spatial（两者已有 Draft 权重）。" >&2
    exit 1
    ;;
esac

init_libero_eval_env "${TASK_SUITE_NAME}"
export TASK_SUITE_NAME TASK_SUITE_SLUG

if [[ -z "${SPEC_CKPT:-}" ]]; then
  echo "请显式设置 SPEC_CKPT=/absolute/path/to/epoch_100_step_xxxxxx。" >&2
  exit 1
fi
if [[ ! -f "${SPEC_CKPT}/pytorch_model.bin" || ! -f "${SPEC_CKPT}/dflash_config.json" ]]; then
  echo "不完整的 DFlash checkpoint: ${SPEC_CKPT}" >&2
  exit 1
fi

RUN_STAMP="${P0_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
P0_TRIALS="${P0_TRIALS:-3}"
P0_TASKS="${P0_TASKS:-1}"
P0_DATA_ROOT="${P0_DATA_ROOT:-$(dirname "${DEFAULT_LOG_DIR}")/evidence/p0}"
RAW_RUN_ROOT="${P0_DATA_ROOT}/${TASK_SUITE_SLUG}/${RUN_STAMP}/raw_runs"
OUTPUT_ROOT="${P0_OUTPUT_ROOT:-${REPO_ROOT}/artifacts/evidence/p0/${TASK_SUITE_SLUG}/${RUN_STAMP}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EVAL_EPOCH="${EVAL_EPOCH:-100}"
export NUM_TRIALS_PER_TASK="${P0_TRIALS}"
export MAX_EVAL_TASKS="${P0_TASKS}"
export TRIAL_START_INDEX="${TRIAL_START_INDEX:-0}"
export SEED="${SEED:-7}"
export TIMING_SCOPE=full_suite
export SYNC_CUDA_TIMING=False
export USE_WANDB=False
export DFLASH_NUM_DRAFT_LAYERS=1
export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=False
export DFLASH_TARGET_LOGITS_MODE=action_only
export DFLASH_ACCEPTANCE_MODE=token
export ACCEPT_THRESHOLD=0
export DFLASH_TREE_MODE=off
export DFLASH_TREE_BUDGET=0

latest_summary() {
  local directory="$1"
  mapfile -t summaries < <(find "${directory}" -type f -name '*_summary.json' | sort)
  if [[ "${#summaries[@]}" -ne 1 ]]; then
    echo "${directory} 应只生成一个 summary，实际为 ${#summaries[@]}。" >&2
    printf '%s\n' "${summaries[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${summaries[0]}"
}

if [[ -n "${P0_REUSE_PROFILE_SUMMARY:-}" ]]; then
  PROFILE_SUMMARY="${P0_REUSE_PROFILE_SUMMARY}"
  echo "[P0 1/3] reuse synchronized profile: ${PROFILE_SUMMARY}"
else
  echo "[P0 1/3] synchronized DFlash stage profile"
  (
    export LOG_DIR="${RAW_RUN_ROOT}/profile"
    export RUN_ID_NOTE="p0-${TASK_SUITE_SLUG}-${RUN_STAMP}-profile"
    export DFLASH_VERIFY_SKIP_MODE=off
    export DFLASH_TEMPORAL_PREFILL_FUSION=False
    export DFLASH_PROFILE_STAGES=True
    export DFLASH_DEBUG_COMPARE_TARGET_AR=False
    export DFLASH_EVIDENCE_TRACE=False
    bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict
  )
  PROFILE_SUMMARY="$(latest_summary "${RAW_RUN_ROOT}/profile")"
fi

echo "[P0 2/3] paper-compatible wrapped-AR temporal trace"
(
  export LOG_DIR="${RAW_RUN_ROOT}/shadow"
  export RUN_ID_NOTE="p0-${TASK_SUITE_SLUG}-${RUN_STAMP}-wrapped-ar-trace"
  export AR_EVIDENCE_TRACE=True
  bash "${SCRIPT_DIR}/run_specvla_paper_ar_eval.sh" "${TASK_SUITE_SLUG}"
)
SHADOW_SUMMARY="$(latest_summary "${RAW_RUN_ROOT}/shadow")"

if [[ -n "${P0_REUSE_PARITY_SUMMARY:-}" ]]; then
  PARITY_SUMMARY="${P0_REUSE_PARITY_SUMMARY}"
  echo "[P0 3/3] reuse strict VTPF parity: ${PARITY_SUMMARY}"
else
  echo "[P0 3/3] strict VTPF fused-vs-serial parity"
  (
    export LOG_DIR="${RAW_RUN_ROOT}/parity"
    export RUN_ID_NOTE="p0-${TASK_SUITE_SLUG}-${RUN_STAMP}-vtpf-parity"
    export DFLASH_VERIFY_SKIP_MODE=route
    export DFLASH_TEMPORAL_PREFILL_FUSION=True
    export DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS=1
    export DFLASH_TEMPORAL_ROUTE_MIN_COSINE=0.0
    export DFLASH_TEMPORAL_ROUTE_STOP_ON_REJECT=True
    export DFLASH_TEMPORAL_FUSE_VERIFY=True
    export DFLASH_PROFILE_STAGES=True
    export DFLASH_DEBUG_COMPARE_TARGET_AR=True
    export DFLASH_EVIDENCE_TRACE=False
    bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict
  )
  PARITY_SUMMARY="$(latest_summary "${RAW_RUN_ROOT}/parity")"
fi

ANALYSIS_ARGS=(
  --suite "${TASK_SUITE_NAME}"
  --profile-summary "${PROFILE_SUMMARY}"
  --shadow-summary "${SHADOW_SUMMARY}"
  --parity-summary "${PARITY_SUMMARY}"
  --raw-run-root "${RAW_RUN_ROOT}"
  --output "${OUTPUT_ROOT}"
  --repo-root "${REPO_ROOT}"
)
if [[ -n "${P0_AR_SUMMARY:-}" ]]; then
  ANALYSIS_ARGS+=(--ar-summary "${P0_AR_SUMMARY}")
fi
for index in 1 2 3 4 5 6; do
  variable="P0_METHOD_SUMMARY_${index}"
  if [[ -n "${!variable:-}" ]]; then
    ANALYSIS_ARGS+=(--method-summary "${!variable}")
  fi
done

python openvla/specdecoding/evidence/build_p0_evidence.py "${ANALYSIS_ARGS[@]}"

echo "P0 evidence complete"
echo "Raw runs: ${RAW_RUN_ROOT}"
echo "Curated pack: ${OUTPUT_ROOT}"
