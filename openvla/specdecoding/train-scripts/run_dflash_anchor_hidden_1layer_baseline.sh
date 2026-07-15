#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
SEED="${SEED:-7}"

python openvla/specdecoding/train-scripts/train_dflash_libero_goal.py \
  --run_name dflash-anchor-hidden-1layer-baseline \
  --seed "${SEED}" \
  --output_dir /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_baseline \
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
  --patience 3 \
  --block_size 7
