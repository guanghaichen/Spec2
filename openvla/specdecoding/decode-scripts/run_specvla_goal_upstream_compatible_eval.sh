#!/usr/bin/env bash
set -euo pipefail

# Reproduce the upstream SpecVLA Goal paper protocol in one serial run:
# paper wrapped AR (use_spec=True) -> strict (r=0) -> relaxed (r=9).
# The individual launchers write their normal JSON/TXT artifacts; this wrapper
# additionally writes one compact comparison TXT with SR, Length and Speedup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

init_libero_goal_eval_env

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
RUN_TAG="${RUN_TAG:-specvla-goal-paper-protocol-$(date +%Y%m%d-%H%M%S)}"
export NUM_TRIALS_PER_TASK SYNC_CUDA_TIMING TIMING_SCOPE SEED

AR_NOTE="${RUN_TAG}-paper-wrapped-ar"
STRICT_NOTE="${RUN_TAG}-strict-r0"
RELAXED_NOTE="${RUN_TAG}-relaxed-r9"

find_summary() {
  local directory="$1"
  local note="$2"
  local family="$3"
  local matches=()

  mapfile -t matches < <(find "${directory}" -maxdepth 1 -type f -name "*--${note}-${family}_summary.json" | sort)
  if [[ "${#matches[@]}" -ne 1 ]]; then
    echo "Expected exactly one ${family} summary for RUN_TAG=${RUN_TAG}, found ${#matches[@]}." >&2
    printf '%s\n' "${matches[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

echo "============================================================"
echo "SpecVLA Goal paper-compatible protocol"
echo "AR_BASELINE=specvla_paper_wrapped_ar (use_spec=True)"
echo "RUN_TAG=${RUN_TAG}"
echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "SYNC_CUDA_TIMING=${SYNC_CUDA_TIMING}"
echo "TIMING_SCOPE=${TIMING_SCOPE}"
echo "============================================================"

RUN_ID_NOTE="${AR_NOTE}" bash "${SCRIPT_DIR}/run_openvla_ar_libero_goal_eval.sh"
RUN_ID_NOTE="${STRICT_NOTE}" bash "${SCRIPT_DIR}/run_specvla_libero_goal_eval.sh"
RUN_ID_NOTE="${RELAXED_NOTE}" bash "${SCRIPT_DIR}/run_specvla_relaxed_libero_goal_eval.sh"

AR_SUMMARY="$(find_summary "${LOG_DIR}/openvla_ar" "${AR_NOTE}" "openvla_ar")"
STRICT_SUMMARY="$(find_summary "${LOG_DIR}/specvla_strict" "${STRICT_NOTE}" "specvla_strict")"
RELAXED_SUMMARY="$(find_summary "${LOG_DIR}/specvla_relaxed" "${RELAXED_NOTE}" "specvla_relaxed")"

OUTPUT_DIR="${LOG_DIR}/reproduction"
OUTPUT_FILE="${OUTPUT_DIR}/${RUN_TAG}_comparison.txt"
mkdir -p "${OUTPUT_DIR}"

python "${REPO_ROOT}/openvla/specdecoding/test-speed/summarize_eval_summaries.py" \
  --ar-summary "${AR_SUMMARY}" \
  "${STRICT_SUMMARY}" "${RELAXED_SUMMARY}" | tee "${OUTPUT_FILE}"

echo "Saved Goal baseline comparison: ${OUTPUT_FILE}"
echo "AR summary: ${AR_SUMMARY}"
echo "Strict summary: ${STRICT_SUMMARY}"
echo "Relaxed summary: ${RELAXED_SUMMARY}"
