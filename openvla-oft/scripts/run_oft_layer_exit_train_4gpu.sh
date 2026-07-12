#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/oft_common.sh"

EARLY_EXIT_LAYER="${EARLY_EXIT_LAYER:-16}"
FEATURE_FILE="${FEATURE_FILE:-${OFT_RUN_ROOT}/teacher_features/libero_goal_layer${EARLY_EXIT_LAYER}.h5}"
ACTION_HEAD_CHECKPOINT="${ACTION_HEAD_CHECKPOINT:-${OFT_GOAL_MODEL}/action_head--50000_checkpoint.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${OFT_RUN_ROOT}/checkpoints/layer_exit_goal_l${EARLY_EXIT_LAYER}}"
GPUS="${GPUS:-0,1,2,3}"

cd "${OFT_ROOT}"
CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node=4 \
  experiments/early_exit/train_layer_exit_adapter.py \
  --feature-file "${FEATURE_FILE}" \
  --action-head-checkpoint "${ACTION_HEAD_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS:-40}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-1}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --bottleneck-size "${BOTTLENECK_SIZE:-512}" \
  --mixer-layers "${MIXER_LAYERS:-2}" \
  --attention-heads "${ATTENTION_HEADS:-8}"
