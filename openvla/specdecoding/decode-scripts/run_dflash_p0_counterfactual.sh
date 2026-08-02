#!/usr/bin/env bash
set -euo pipefail

# Same-state recoverability experiment. This loads only the frozen target VLA;
# no Draft checkpoint is required. Defaults are a small Goal task-0 smoke.

SUITE_ARG="${1:-goal}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

case "${SUITE_ARG}" in
  goal|libero_goal) TASK_SUITE_NAME=libero_goal ;;
  spatial|libero_spatial) TASK_SUITE_NAME=libero_spatial ;;
  *) echo "当前反事实入口只支持 goal 或 spatial。" >&2; exit 1 ;;
esac
init_libero_eval_env "${TASK_SUITE_NAME}"

RUN_STAMP="${P0_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${P0_COUNTERFACTUAL_OUTPUT_DIR:-$(dirname "${DEFAULT_LOG_DIR}")/evidence/p0/${TASK_SUITE_SLUG}/${RUN_STAMP}/counterfactual}"
CURATED_DIR="${P0_COUNTERFACTUAL_CURATED_DIR:-${REPO_ROOT}/artifacts/evidence/p0/${TASK_SUITE_SLUG}/${RUN_STAMP}/counterfactual}"

echo "suite=${TASK_SUITE_NAME} target=${VLA_PATH} output=${OUTPUT_DIR}"
python openvla/specdecoding/evidence/run_counterfactual_recovery.py \
  --pretrained_checkpoint "${VLA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --task_id "${P0_TASK_ID:-0}" \
  --trial_start_index "${TRIAL_START_INDEX:-0}" \
  --num_reference_episodes "${P0_REFERENCE_EPISODES:-1}" \
  --max_reference_attempts "${P0_MAX_REFERENCE_ATTEMPTS:-10}" \
  --require_reference_success True \
  --num_forks_per_episode "${P0_FORKS_PER_EPISODE:-2}" \
  --max_recovery_steps "${P0_MAX_RECOVERY_STEPS:-0}" \
  --seed "${SEED:-7}" \
  --center_crop True \
  --run_id_note "p0-${TASK_SUITE_SLUG}-${RUN_STAMP}"

RECORDS="${OUTPUT_DIR}/counterfactual-p0-${TASK_SUITE_SLUG}-${RUN_STAMP}.jsonl"
MANIFEST="${OUTPUT_DIR}/counterfactual-p0-${TASK_SUITE_SLUG}-${RUN_STAMP}-manifest.json"
python openvla/specdecoding/evidence/build_counterfactual_evidence.py \
  --records "${RECORDS}" \
  --run-manifest "${MANIFEST}" \
  --output "${CURATED_DIR}" \
  --repo-root "${REPO_ROOT}"

echo "Counterfactual evidence complete: ${CURATED_DIR}"
