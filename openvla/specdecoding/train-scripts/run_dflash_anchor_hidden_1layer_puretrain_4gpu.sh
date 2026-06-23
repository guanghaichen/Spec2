#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
VLA_PATH="${VLA_PATH:-/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal}"
DATAPATH="${DATAPATH:-/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_puretrain_4gpu}"

torchrun --standalone --nnodes 1 --nproc_per_node 4 \
  openvla/specdecoding/train-scripts/train_dflash_libero_goal.py \
  --run_name dflash-anchor-hidden-1layer-puretrain-4gpu \
  --vla_path "${VLA_PATH}" \
  --datapath "${DATAPATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_draft_layers 1 \
  --target_layer_ids 1 8 15 22 29 \
  --include_anchor_hidden \
  --anchor_consistency_w 0 \
  --soft_w 0 \
  --hidden_w 1.0 \
  --cos_w 0.05 \
  --hidden_noise 0.03 \
  --weight_decay 0.05 \
  --lr 5e-5 \
  --batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_epochs 200 \
  --warmup_steps 2000 \
  --save_every 10 \
  --val_split 0 \
  --block_size 7
