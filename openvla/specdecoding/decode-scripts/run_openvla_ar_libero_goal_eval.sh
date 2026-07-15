#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/libero_eval_common.sh"

init_libero_goal_eval_env
resolve_specvla_goal_checkpoint
RUN_ID_NOTE="${RUN_ID_NOTE:-specvla-paper-wrapped-ar}"
SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
TIMING_SCOPE="${TIMING_SCOPE:-last_task}"

print_common_eval_config
echo "AR_BASELINE=specvla_paper_wrapped_ar"
echo "USE_SPEC=True"
echo "SYNC_CUDA_TIMING=${SYNC_CUDA_TIMING}"
echo "TIMING_SCOPE=${TIMING_SCOPE}"

# The historical filename is retained for compatibility. This is the AR
# baseline used by the SpecVLA authors: it loads the draft wrapper and runs
# ea_forward autoregressively with use_spec=True. It is not pure OpenVLA AR.
python openvla/experiments/robot/libero/run_libero_goal_AR.py \
  --model_family openvla \
  --pretrained_checkpoint "${VLA_PATH}" \
  --spec_checkpoint "${SPEC_CKPT}" \
  --use_spec True \
  --parallel_draft False \
  --task_suite_name libero_goal \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --center_crop True \
  --local_log_dir "${LOG_DIR}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --sync_cuda_timing "${SYNC_CUDA_TIMING}" \
  --timing_scope "${TIMING_SCOPE}" \
  --use_wandb "${USE_WANDB}" \
  --seed "${SEED}"
