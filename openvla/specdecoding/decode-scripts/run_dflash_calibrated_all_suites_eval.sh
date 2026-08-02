#!/usr/bin/env bash
set -euo pipefail

# 在每个子集的独立测试分割上运行其冻结 profile；不允许跨 suite 共享 Draft 或 profile。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

IFS=',' read -r -a SUITES <<< "${EVAL_SUITES:-libero_goal,libero_object,libero_spatial,libero_10}"
if [[ "${#SUITES[@]}" -eq 0 ]]; then
  echo "EVAL_SUITES 不能为空。" >&2
  exit 1
fi
if [[ -z "${CALIBRATION_RUN_STAMP:-}" ]]; then
  echo "请设置 CALIBRATION_RUN_STAMP，与四子集校准输出一致。" >&2
  exit 1
fi

TEST_START_INDEX="${TRIAL_START_INDEX:-5}"
TEST_TRIALS="${NUM_TRIALS_PER_TASK:-45}"
init_libero_eval_env "${SUITES[0]}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$(dirname "${DEFAULT_LOG_DIR}")/calibration}"
DRAFT_ROOT="${DFLASH_DRAFT_ROOT:-$(dirname "${DEFAULT_LOG_DIR}")/Draft_checkpoint}"

draft_override() {
  case "$1" in
    goal) echo "${DFLASH_DRAFT_GOAL:-}" ;;
    object) echo "${DFLASH_DRAFT_OBJECT:-}" ;;
    spatial) echo "${DFLASH_DRAFT_SPATIAL:-}" ;;
    10) echo "${DFLASH_DRAFT_10:-}" ;;
    *) return 1 ;;
  esac
}

for SUITE in "${SUITES[@]}"; do
  SLUG="$(libero_suite_slug "${SUITE}")"
  PROFILE_DIR="${CALIBRATION_ROOT}/${SLUG}/${CALIBRATION_RUN_STAMP}/profile"
  mapfile -t MATCHED_PROFILES < <(
    find "${PROFILE_DIR}" -maxdepth 1 -type f -name "${SUITE}-*.json" | sort
  )
  if [[ "${#MATCHED_PROFILES[@]}" -ne 1 ]]; then
    echo "${SUITE} 需要一个冻结 profile，实际找到 ${#MATCHED_PROFILES[@]}。" >&2
    exit 1
  fi

  SPEC_CKPT="$(draft_override "${SLUG}")"
  if [[ -z "${SPEC_CKPT}" ]]; then
    mapfile -t MATCHED_CKPTS < <(
      find "${DRAFT_ROOT}/${SLUG}" -maxdepth 1 -type d \
        -name "epoch_100_step_*" | sort -V
    )
    if [[ "${#MATCHED_CKPTS[@]}" -ne 1 ]]; then
      echo "${SUITE} 需要一个 epoch-100 Draft，实际找到 ${#MATCHED_CKPTS[@]}。" >&2
      echo "可显式设置 DFLASH_DRAFT_${SLUG^^}=/absolute/checkpoint/path。" >&2
      exit 1
    fi
    SPEC_CKPT="${MATCHED_CKPTS[0]}"
  fi

  echo "[$SUITE] frozen test states ${TEST_START_INDEX}..$((TEST_START_INDEX + TEST_TRIALS - 1))"
  TASK_SUITE_NAME="${SUITE}" \
  VLA_PATH_OVERRIDE="" \
  SPEC_CKPT="${SPEC_CKPT}" \
  CALIBRATED_PROFILE="${MATCHED_PROFILES[0]}" \
  TRIAL_START_INDEX="${TEST_START_INDEX}" \
  NUM_TRIALS_PER_TASK="${TEST_TRIALS}" \
  RUN_ID_NOTE="raes-${SLUG}-${CALIBRATION_RUN_STAMP}-frozen-test" \
    bash "${SCRIPT_DIR}/run_dflash_calibrated_suite_eval.sh"
done

echo "All frozen-profile tests complete."
