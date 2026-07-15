#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/libero_eval_common.sh"

init_libero_eval_env libero_spatial
resolve_specvla_checkpoint
RUN_ID_NOTE="${RUN_ID_NOTE:-specvla-paper-wrapped-ar-spatial}"
SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
TIMING_SCOPE="${TIMING_SCOPE:-last_task}"

print_common_eval_config
echo "AR_BASELINE=specvla_paper_wrapped_ar"
echo "USE_SPEC=True"
echo "SYNC_CUDA_TIMING=${SYNC_CUDA_TIMING}"
echo "TIMING_SCOPE=${TIMING_SCOPE}"

# Historical filename; this launches the SpecVLA paper's wrapped AR baseline.
python openvla/experiments/robot/libero/run_libero_goal_AR.py \
  --model_family openvla \
  --pretrained_checkpoint "${VLA_PATH}" \
  --spec_checkpoint "${SPEC_CKPT}" \
  --use_spec True \
  --parallel_draft False \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --center_crop True \
  --local_log_dir "${LOG_DIR}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --sync_cuda_timing "${SYNC_CUDA_TIMING}" \
  --timing_scope "${TIMING_SCOPE}" \
  --use_wandb "${USE_WANDB}" \
  --seed "${SEED}"
