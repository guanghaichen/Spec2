#!/usr/bin/env bash
set -euo pipefail

# Evaluation-only Chapter 5 queue. It reuses the frozen Goal Target and PAD
# checkpoint. No model parameters or datasets are modified.

ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
REPO="${ROOT}/SpecVLA-DFLASH"
SCRIPT_DIR="${REPO}/openvla/specdecoding/decode-scripts"
SPEC_CKPT="${ROOT}/specvla-data/Draft_checkpoint/goal/epoch_100_step_044800"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/specvla-data/evidence/chapter5/no_retrain_sensitivity}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_STAMP}"
STATUS_LOG="${RUN_ROOT}/queue_status.tsv"
AUDIT_SCREEN_PATTERN="[.]chapter5_audit"

mkdir -p "${RUN_ROOT}"
printf "case\tstatus\tutc_time\tconfiguration\n" > "${STATUS_LOG}"

if [[ "${WAIT_FOR_CHAPTER5_AUDIT:-True}" == "True" ]]; then
  while screen -list 2>/dev/null | grep -q "${AUDIT_SCREEN_PATTERN}"; do
    echo "[queue] Waiting for chapter5_audit to finish..."
    sleep 120
  done
fi

if [[ ! -f "${SPEC_CKPT}/pytorch_model.bin" || ! -f "${SPEC_CKPT}/dflash_config.json" ]]; then
  echo "Incomplete Goal PAD checkpoint: ${SPEC_CKPT}" >&2
  exit 1
fi

run_case() {
  local case_name="$1"
  local depth_bounds="$2"
  local h2_fallback_bound="$3"
  local max_depth="$4"
  local authority_exponent="$5"
  local case_dir="${RUN_ROOT}/${case_name}"
  local config="bounds=${depth_bounds};h2_fallback=${h2_fallback_bound};D=${max_depth};p=${authority_exponent}"

  mkdir -p "${case_dir}"
  printf "%s\tstarted\t%s\t%s\n" "${case_name}" "$(date -u +%FT%TZ)" "${config}" >> "${STATUS_LOG}"
  echo "[queue] Starting ${case_name}: ${config}"

  (
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export TASK_SUITE_NAME=libero_goal
    export SPEC_CKPT="${SPEC_CKPT}"
    export EVAL_EPOCH=100
    export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
    export MAX_EVAL_TASKS=10
    export TRIAL_START_INDEX=0
    export SEED=7
    export TIMING_SCOPE=last_task
    export SYNC_CUDA_TIMING=False
    export USE_WANDB=False
    export LOG_DIR="${case_dir}"
    export RUN_ID_NOTE="chapter5-${case_name}-goal-e100-s7"

    export DFLASH_TEMPORAL_HOLD_POLICY_OVERRIDE=True
    export DFLASH_TEMPORAL_HOLD_POLICY=paced_budget
    export DFLASH_TEMPORAL_ADAPTIVE_MIN_VERIFIED_RUN=1
    export DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS=1
    export DFLASH_TEMPORAL_BYPASS_MAX_PIXEL_RELATIVE_L2=1.0
    export DFLASH_TEMPORAL_BYPASS_USE_PIXEL_GUARD=False
    export DFLASH_TEMPORAL_DEPTH_VISUAL_BOUNDS="${depth_bounds}"
    export DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2="${h2_fallback_bound}"
    export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE="${max_depth}"
    export DFLASH_TEMPORAL_HOLD_ACTION_DECAY=power_law
    export DFLASH_TEMPORAL_AUTHORITY_EXPONENT="${authority_exponent}"
    export DFLASH_TEMPORAL_ROUTE_LABEL="Chapter5_${case_name}"
    export DFLASH_EVIDENCE_TRACE=True
    export DFLASH_PROFILE_STAGES=False
    export DFLASH_DEBUG_COMPARE_TARGET_AR=False

    bash "${SCRIPT_DIR}/run_dflash_vtpf_adaptive_decimation_goal_eval.sh"
  ) 2>&1 | tee "${case_dir}/queue_console.log"

  printf "%s\tcomplete\t%s\t%s\n" "${case_name}" "$(date -u +%FT%TZ)" "${config}" >> "${STATUS_LOG}"
  echo "[queue] Completed ${case_name}"
}

# Existing formal runs provide beta=0.075, D=0/2, p=0/1 and the full method.
# These five runs fill the remaining no-retraining sensitivity conditions.
run_case beta_005 "0.05,0.10" "0.10" 2 1.0
run_case beta_010 "0.10,0.20" "0.20" 2 1.0
run_case depth_1 "0.075" "0.075" 1 1.0
run_case authority_p05 "0.075,0.15" "0.15" 2 0.5
run_case no_state_bound "inf,inf" "1000000" 2 1.0

echo "Chapter 5 no-retraining sensitivity queue complete: ${RUN_ROOT}"
