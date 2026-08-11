#!/usr/bin/env bash
set -euo pipefail

# Capture one known-success initial state per LIBERO suite for qualitative
# evidence. This launcher never changes model weights or main-table logs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/specvla-data/evidence/chapter5/qualitative_rollouts}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

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
declare -A OUTPUT_KEYS=(
  [goal]=goal
  [spatial]=spatial
  [object]=object
  [long]=long_success
)
declare -A TASK_INDEX=(
  [goal]=3
  [spatial]=4
  [object]=0
  [long]=9
)
declare -A TRIAL_INDEX=(
  [goal]=2
  [spatial]=0
  [object]=0
  [long]=0
)

START_SUITE="${START_SUITE:-goal}"
SUITES=(goal spatial object long)
started=False
for suite in "${SUITES[@]}"; do
  if [[ "${suite}" == "${START_SUITE}" ]]; then
    started=True
  fi
  if [[ "${started}" != True ]]; then
    continue
  fi

  checkpoint="${CHECKPOINTS[${suite}]}"
  if [[ ! -f "${checkpoint}/pytorch_model.bin" || ! -f "${checkpoint}/dflash_config.json" ]]; then
    echo "Incomplete checkpoint for ${suite}: ${checkpoint}" >&2
    exit 1
  fi

  case_dir="${OUTPUT_ROOT}/${RUN_STAMP}/${OUTPUT_KEYS[${suite}]}"
  echo "[Qualitative capture] suite=${suite} task=${TASK_INDEX[${suite}]} trial=${TRIAL_INDEX[${suite}]}"
  (
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export TASK_SUITE_NAME="${SUITE_NAMES[${suite}]}"
    export SPEC_CKPT="${checkpoint}"
    export EVAL_EPOCH="$( [[ "${suite}" == object ]] && echo 60 || echo 100 )"
    export TASK_START_INDEX="${TASK_INDEX[${suite}]}"
    export MAX_EVAL_TASKS=1
    export TRIAL_START_INDEX="${TRIAL_INDEX[${suite}]}"
    export NUM_TRIALS_PER_TASK=1
    export SEED=7
    export TIMING_SCOPE=full_suite
    export SYNC_CUDA_TIMING=False
    export USE_WANDB=False
    export LOG_DIR="${case_dir}/logs"
    export SAVE_ROLLOUT_VIDEOS=True
    export ROLLOUT_VIDEO_DIR="${case_dir}/videos"
    export DFLASH_EVIDENCE_TRACE=True
    export RUN_ID_NOTE="chapter5-qualitative-${suite}-task${TASK_INDEX[${suite}]}-trial${TRIAL_INDEX[${suite}]}-s7"

    export DFLASH_NUM_DRAFT_LAYERS=1
    export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=False
    export DFLASH_TARGET_LOGITS_MODE=action_only
    export DFLASH_ACCEPTANCE_MODE=token
    export ACCEPT_THRESHOLD=0
    export DFLASH_TREE_MODE=off
    export DFLASH_TREE_BUDGET=0
    export DFLASH_CONFIDENCE_THRESHOLD=0.0
    export DFLASH_CONFIDENCE_MIN_TOKENS=1
    export DFLASH_PROFILE_STAGES=False
    export DFLASH_DEBUG_COMPARE_TARGET_AR=False
    export DFLASH_TEMPORAL_UNIT_VISUAL_BUDGET=0.075

    bash "${SCRIPT_DIR}/run_dflash_vtpf_paced_harmonic_dual_anchor_eval.sh"
  )
done

echo "Qualitative rollout capture complete: ${OUTPUT_ROOT}/${RUN_STAMP}"
