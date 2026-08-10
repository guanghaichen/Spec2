#!/usr/bin/env bash
set -euo pipefail

# Chapter 5 mechanism audit. This is an evaluation-only job: it reuses the
# frozen suite-specific Target and PAD checkpoints and records all ten tasks.
# Main-table SR/Speedup still come from the formal 500-episode runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

if [[ ! -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  echo "This launcher currently targets the RTX 4090 evaluation host." >&2
  exit 1
fi

ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
AUDIT_TRIALS="${AUDIT_TRIALS:-5}"
START_SUITE="${START_SUITE:-goal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/specvla-data/evidence/chapter5/full_suite_mechanism}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

declare -A CHECKPOINTS=(
  [goal]="${ROOT}/specvla-data/Draft_checkpoint/goal/epoch_100_step_044800"
  [spatial]="${ROOT}/specvla-data/Draft_checkpoint/spatial/epoch_100_step_062200"
  [object]="${ROOT}/specvla-data/Draft_checkpoint/object/epoch_060_step_058440"
  [10]="${ROOT}/specvla-data/Draft_checkpoint/10/epoch_100_step_138200"
)
declare -A SUITE_NAMES=(
  [goal]=libero_goal
  [spatial]=libero_spatial
  [object]=libero_object
  [10]=libero_10
)
SUITES=(goal spatial object 10)

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

  echo "[Chapter 5 audit] suite=${suite} trials/task=${AUDIT_TRIALS}"
  (
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export TASK_SUITE_NAME="${SUITE_NAMES[${suite}]}"
    export SPEC_CKPT="${checkpoint}"
    export EVAL_EPOCH="$( [[ "${suite}" == object ]] && echo 60 || echo 100 )"
    export NUM_TRIALS_PER_TASK="${AUDIT_TRIALS}"
    export MAX_EVAL_TASKS=10
    export TRIAL_START_INDEX=0
    export SEED=7
    export TIMING_SCOPE=full_suite
    export SYNC_CUDA_TIMING=False
    export USE_WANDB=False
    export LOG_DIR="${OUTPUT_ROOT}/${RUN_STAMP}/${suite}"
    export RUN_ID_NOTE="chapter5-full-suite-mechanism-${suite}-n${AUDIT_TRIALS}-s7"

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
    export DFLASH_EVIDENCE_TRACE=True
    export DFLASH_TEMPORAL_UNIT_VISUAL_BUDGET=0.075

    bash "${SCRIPT_DIR}/run_dflash_vtpf_paced_harmonic_dual_anchor_eval.sh"
  )
done

echo "Chapter 5 full-suite mechanism audit complete: ${OUTPUT_ROOT}/${RUN_STAMP}"
