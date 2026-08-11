#!/usr/bin/env bash
set -uo pipefail

# Chapter 5 evaluation-only evidence queue for the RTX 4090 host.
# It never changes checkpoints or training data. Every case writes to an
# isolated directory and records its own status so one failure cannot erase
# the rest of the overnight queue.

ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
REPO="${ROOT}/SpecVLA-DFLASH"
SCRIPT_DIR="${REPO}/openvla/specdecoding/decode-scripts"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/specvla-data/evidence/chapter5/hard_evidence_queue}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_STAMP}"
STATUS_LOG="${RUN_ROOT}/queue_status.tsv"

declare -A CHECKPOINTS=(
  [goal]="${ROOT}/specvla-data/Draft_checkpoint/goal/epoch_100_step_044800"
  [spatial]="${ROOT}/specvla-data/Draft_checkpoint/spatial/epoch_100_step_062200"
  [object]="${ROOT}/specvla-data/Draft_checkpoint/object/epoch_060_step_058440"
  [long]="${ROOT}/specvla-data/Draft_checkpoint/10/epoch_100_step_138200"
)
declare -A SUITE_NAMES=(
  [goal]=libero_goal
  [spatial]=libero_spatial
  [object]=libero_object
  [long]=libero_10
)
declare -A EPOCHS=(
  [goal]=100
  [spatial]=100
  [object]=60
  [long]=100
)
SUITES=(goal spatial object long)

mkdir -p "${RUN_ROOT}"
if [[ ! -f "${STATUS_LOG}" ]]; then
  printf "case\tstatus\tutc_time\texit_code\tconfiguration\n" > "${STATUS_LOG}"
fi

source /home/asus/miniconda3/etc/profile.d/conda.sh
conda activate specvla
cd "${REPO}"
git rev-parse HEAD > "${RUN_ROOT}/git_commit.txt"
nvidia-smi -q > "${RUN_ROOT}/nvidia_smi.txt"

for suite in "${SUITES[@]}"; do
  checkpoint="${CHECKPOINTS[${suite}]}"
  if [[ ! -f "${checkpoint}/pytorch_model.bin" || ! -f "${checkpoint}/dflash_config.json" ]]; then
    echo "Incomplete checkpoint for ${suite}: ${checkpoint}" >&2
    exit 1
  fi
done

record_status() {
  local name="$1"
  local status="$2"
  local code="$3"
  local config="$4"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${name}" "${status}" "$(date -u +%FT%TZ)" "${code}" "${config}" >> "${STATUS_LOG}"
}

run_logged_case() {
  local name="$1"
  local config="$2"
  shift 2
  local case_dir="${RUN_ROOT}/${name}"
  local code

  if awk -F '\t' -v name="${name}" \
      '$1 == name && $2 == "complete" { found=1 } END { exit !found }' "${STATUS_LOG}"; then
    echo "[queue] Skipping completed case ${name}"
    return 0
  fi

  mkdir -p "${case_dir}"
  record_status "${name}" started - "${config}"
  echo "[queue] Starting ${name}: ${config}"
  set +e
  ("$@") 2>&1 | tee "${case_dir}/queue_console.log"
  code=${PIPESTATUS[0]}
  set -e
  if [[ "${code}" -eq 0 ]]; then
    record_status "${name}" complete 0 "${config}"
    echo "[queue] Completed ${name}"
  else
    record_status "${name}" failed "${code}" "${config}"
    echo "[queue] FAILED ${name} with exit code ${code}; continuing." >&2
  fi
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export USE_WANDB=False
export SYNC_CUDA_TIMING=False
export SEED=7

set_common_dflash_env() {
  export DFLASH_NUM_DRAFT_LAYERS=1
  export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=False
  export DFLASH_TARGET_LOGITS_MODE=action_only
  export DFLASH_ACCEPTANCE_MODE=token
  export ACCEPT_THRESHOLD=0
  export DFLASH_TREE_MODE=off
  export DFLASH_TREE_BUDGET=0
  export DFLASH_CONFIDENCE_THRESHOLD=0.0
  export DFLASH_CONFIDENCE_MIN_TOKENS=1
}

run_tpv_parity() {
  local suite="$1"
  local case_dir="$2"
  set_common_dflash_env
  export TASK_SUITE_NAME="${SUITE_NAMES[${suite}]}"
  export SPEC_CKPT="${CHECKPOINTS[${suite}]}"
  export EVAL_EPOCH="${EPOCHS[${suite}]}"
  export NUM_TRIALS_PER_TASK=2
  export TASK_START_INDEX=0
  export MAX_EVAL_TASKS=10
  export TRIAL_START_INDEX=0
  export TIMING_SCOPE=full_suite
  export LOG_DIR="${case_dir}"
  export RUN_ID_NOTE="chapter5-tpv-parity-${suite}-n2-s7"
  export DFLASH_VERIFY_SKIP_MODE=route
  export DFLASH_TEMPORAL_PREFILL_FUSION=True
  export DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS=1
  export DFLASH_TEMPORAL_ROUTE_MIN_COSINE=0.0
  export DFLASH_TEMPORAL_ROUTE_STOP_ON_REJECT=True
  export DFLASH_TEMPORAL_FUSE_VERIFY=True
  export DFLASH_PROFILE_STAGES=True
  export DFLASH_DEBUG_COMPARE_TARGET_AR=True
  export DFLASH_EVIDENCE_TRACE=True
  bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict
}

run_stage_profile() {
  local suite="$1"
  local variant="$2"
  local case_dir="$3"
  set_common_dflash_env
  export TASK_SUITE_NAME="${SUITE_NAMES[${suite}]}"
  export SPEC_CKPT="${CHECKPOINTS[${suite}]}"
  export EVAL_EPOCH="${EPOCHS[${suite}]}"
  export NUM_TRIALS_PER_TASK=3
  export TASK_START_INDEX=9
  export MAX_EVAL_TASKS=1
  export TRIAL_START_INDEX=0
  export TIMING_SCOPE=full_suite
  export LOG_DIR="${case_dir}"
  export RUN_ID_NOTE="chapter5-stage-${suite}-${variant}-n3-s7"
  export DFLASH_PROFILE_STAGES=True
  export DFLASH_DEBUG_COMPARE_TARGET_AR=False
  export DFLASH_EVIDENCE_TRACE=True

  case "${variant}" in
    pad)
      export DFLASH_VERIFY_SKIP_MODE=off
      export DFLASH_TEMPORAL_PREFILL_FUSION=False
      bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict
      ;;
    tpv)
      export DFLASH_VERIFY_SKIP_MODE=route
      export DFLASH_TEMPORAL_PREFILL_FUSION=True
      export DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS=1
      export DFLASH_TEMPORAL_ROUTE_MIN_COSINE=0.0
      export DFLASH_TEMPORAL_ROUTE_STOP_ON_REJECT=True
      export DFLASH_TEMPORAL_FUSE_VERIFY=True
      bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict
      ;;
    full)
      export DFLASH_PROFILE_STAGES=True
      export DFLASH_DEBUG_COMPARE_TARGET_AR=False
      export DFLASH_EVIDENCE_TRACE=True
      export DFLASH_TEMPORAL_UNIT_VISUAL_BUDGET=0.075
      bash "${SCRIPT_DIR}/run_dflash_vtpf_paced_harmonic_dual_anchor_eval.sh"
      ;;
    *)
      echo "Unknown profile variant: ${variant}" >&2
      return 2
      ;;
  esac
}

run_full_suite_mechanism() {
  local suite="$1"
  local case_dir="$2"
  set_common_dflash_env
  export TASK_SUITE_NAME="${SUITE_NAMES[${suite}]}"
  export SPEC_CKPT="${CHECKPOINTS[${suite}]}"
  export EVAL_EPOCH="${EPOCHS[${suite}]}"
  export NUM_TRIALS_PER_TASK=10
  export TASK_START_INDEX=0
  export MAX_EVAL_TASKS=10
  export TRIAL_START_INDEX=0
  export TIMING_SCOPE=full_suite
  export LOG_DIR="${case_dir}"
  export RUN_ID_NOTE="chapter5-mechanism-${suite}-n10-s7"
  export DFLASH_PROFILE_STAGES=False
  export DFLASH_DEBUG_COMPARE_TARGET_AR=False
  export DFLASH_EVIDENCE_TRACE=True
  export DFLASH_TEMPORAL_UNIT_VISUAL_BUDGET=0.075
  bash "${SCRIPT_DIR}/run_dflash_vtpf_paced_harmonic_dual_anchor_eval.sh"
}

run_btc_intervention() {
  local suite="$1"
  local variant="$2"
  local case_dir="$3"
  set_common_dflash_env
  export TASK_SUITE_NAME="${SUITE_NAMES[${suite}]}"
  export SPEC_CKPT="${CHECKPOINTS[${suite}]}"
  export EVAL_EPOCH="${EPOCHS[${suite}]}"
  export NUM_TRIALS_PER_TASK=50
  export TASK_START_INDEX=0
  export MAX_EVAL_TASKS=10
  export TRIAL_START_INDEX=0
  export TIMING_SCOPE=last_task
  export LOG_DIR="${case_dir}"
  export RUN_ID_NOTE="chapter5-btc-${suite}-${variant}-500-s7"
  export DFLASH_TEMPORAL_HOLD_POLICY_OVERRIDE=True
  export DFLASH_TEMPORAL_ADAPTIVE_MIN_VERIFIED_RUN=1
  export DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS=1
  export DFLASH_TEMPORAL_BYPASS_MAX_PIXEL_RELATIVE_L2=1.0
  export DFLASH_TEMPORAL_BYPASS_USE_PIXEL_GUARD=False
  export DFLASH_TEMPORAL_DEPTH_VISUAL_BOUNDS="0.075,0.15"
  export DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2=0.15
  export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE=2
  export DFLASH_PROFILE_STAGES=False
  export DFLASH_DEBUG_COMPARE_TARGET_AR=False
  export DFLASH_EVIDENCE_TRACE=True

  case "${variant}" in
    no_renewal_no_authority)
      export DFLASH_TEMPORAL_HOLD_POLICY=visual_budget
      export DFLASH_TEMPORAL_HOLD_ACTION_DECAY=none
      export DFLASH_TEMPORAL_AUTHORITY_EXPONENT=0.0
      ;;
    renewal_only)
      export DFLASH_TEMPORAL_HOLD_POLICY=paced_budget
      export DFLASH_TEMPORAL_HOLD_ACTION_DECAY=none
      export DFLASH_TEMPORAL_AUTHORITY_EXPONENT=0.0
      ;;
    authority_only)
      export DFLASH_TEMPORAL_HOLD_POLICY=visual_budget
      export DFLASH_TEMPORAL_HOLD_ACTION_DECAY=power_law
      export DFLASH_TEMPORAL_AUTHORITY_EXPONENT=1.0
      ;;
    depth_1)
      export DFLASH_TEMPORAL_HOLD_POLICY=paced_budget
      export DFLASH_TEMPORAL_HOLD_ACTION_DECAY=power_law
      export DFLASH_TEMPORAL_AUTHORITY_EXPONENT=1.0
      export DFLASH_TEMPORAL_DEPTH_VISUAL_BOUNDS=0.075
      export DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2=0.075
      export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE=1
      ;;
    no_state_bound)
      export DFLASH_TEMPORAL_HOLD_POLICY=paced_budget
      export DFLASH_TEMPORAL_HOLD_ACTION_DECAY=power_law
      export DFLASH_TEMPORAL_AUTHORITY_EXPONENT=1.0
      export DFLASH_TEMPORAL_DEPTH_VISUAL_BOUNDS="inf,inf"
      export DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2=1000000
      ;;
    *)
      echo "Unknown BTC intervention: ${variant}" >&2
      return 2
      ;;
  esac
  export DFLASH_TEMPORAL_ROUTE_LABEL="Chapter5_${suite}_${variant}"
  bash "${SCRIPT_DIR}/run_dflash_vtpf_adaptive_decimation_goal_eval.sh"
}

# 1. Strict TPV causal-prefix parity on every task in every suite.
for suite in "${SUITES[@]}"; do
  name="01_tpv_parity_${suite}"
  run_logged_case "${name}" "10 tasks x 2 episodes; fused TPV vs serial Target AR" \
    run_tpv_parity "${suite}" "${RUN_ROOT}/${name}"
done

# 2. Matched stage profiles for PAD, PAD+TPV and the complete method.
for suite in "${SUITES[@]}"; do
  for variant in pad tpv full; do
    name="02_stage_${suite}_${variant}"
    run_logged_case "${name}" "last task x 3 episodes; synchronized stage profiler" \
      run_stage_profile "${suite}" "${variant}" "${RUN_ROOT}/${name}"
  done
done

# 3. Denser cross-task mechanism audit: 100 episodes per suite.
for suite in "${SUITES[@]}"; do
  name="03_mechanism_${suite}_n10"
  run_logged_case "${name}" "10 tasks x 10 episodes; full-suite mechanism trace" \
    run_full_suite_mechanism "${suite}" "${RUN_ROOT}/${name}"
done

# 4. Spatial is the hardest closed-loop suite: isolate depth and state bounds.
for variant in depth_1 no_state_bound; do
  name="04_spatial_${variant}_500"
  run_logged_case "${name}" "Spatial 500 episodes; ${variant}" \
    run_btc_intervention spatial "${variant}" "${RUN_ROOT}/${name}"
done

# 5. Long-horizon 2x2 factor completion. The full condition already exists.
for variant in no_renewal_no_authority renewal_only authority_only; do
  name="05_long_${variant}_500"
  run_logged_case "${name}" "Long 500 episodes; ${variant}" \
    run_btc_intervention long "${variant}" "${RUN_ROOT}/${name}"
done

echo "Chapter 5 hard-evidence queue finished: ${RUN_ROOT}"
echo "Status: ${STATUS_LOG}"
