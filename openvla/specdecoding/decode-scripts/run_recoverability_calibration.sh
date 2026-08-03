#!/usr/bin/env bash
set -euo pipefail

# 对一个 LIBERO 子集执行配对校准并生成可直接用于正式评测的冻结配置。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

REQUESTED_SUITE="${TASK_SUITE_NAME:-libero_spatial}"
init_libero_eval_env "${REQUESTED_SUITE}"
export TASK_SUITE_NAME TASK_SUITE_SLUG
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/openvla:${PYTHONPATH:-}"

RUN_STAMP="${CALIBRATION_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${CALIBRATION_OUTPUT_ROOT:-$(dirname "${DEFAULT_LOG_DIR}")/calibration/${TASK_SUITE_SLUG}/${RUN_STAMP}}"
RUN_NOTE="cal-${TASK_SUITE_SLUG}-${RUN_STAMP}"
RESUME_ARGS=()
if [[ "${CALIBRATION_RESUME:-False}" == "True" || "${CALIBRATION_RESUME:-False}" == "true" ]]; then
  RESUME_ARGS+=(--resume True)
fi

python openvla/specdecoding/evidence/run_recoverability_calibration.py \
  --pretrained_checkpoint "${VLA_PATH}" \
  --output_dir "${OUTPUT_ROOT}/raw" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --task_ids "${CALIBRATION_TASK_IDS:-0,1,2,3,4,5,6,7,8,9}" \
  --trial_start_index "${TRIAL_START_INDEX:-0}" \
  --num_trials "${CALIBRATION_TRIALS:-5}" \
  --seed "${SEED:-7}" \
  --center_crop True \
  --schedule_resolution "${CALIBRATION_SCHEDULE_RESOLUTION:-10}" \
  --max_hold_depth "${CALIBRATION_MAX_HOLD_DEPTH:-2}" \
  --min_target_density "${CALIBRATION_MIN_TARGET_DENSITY:-0.30}" \
  --max_target_density "${CALIBRATION_MAX_TARGET_DENSITY:-0.50}" \
  --min_authority_exponent "${CALIBRATION_MIN_AUTHORITY_EXPONENT:-0.0}" \
  --max_authority_exponent "${CALIBRATION_MAX_AUTHORITY_EXPONENT:-1.0}" \
  --num_authority_exponents "${CALIBRATION_NUM_AUTHORITY_EXPONENTS:-3}" \
  --run_id_note "${RUN_NOTE}" \
  "${RESUME_ARGS[@]}"

RECORDS="${OUTPUT_ROOT}/raw/recoverability-calibration-${RUN_NOTE}.jsonl"
MANIFEST="${OUTPUT_ROOT}/raw/recoverability-calibration-${RUN_NOTE}-manifest.json"
python openvla/specdecoding/evidence/build_recoverability_profile.py \
  --records "${RECORDS}" \
  --run_manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_ROOT}/profile" \
  --risk_budget "${CALIBRATION_RISK_BUDGET:-0.10}" \
  --alpha "${CALIBRATION_ALPHA:-0.05}"

echo "Calibration complete: ${OUTPUT_ROOT}"
