#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ -d "/data/wulin" ]]; then
  DEFAULT_VLA_PATH="/data/wulin/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/data/wulin/c/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_puretrain_4gpu"
elif [[ -d "/mnt/storage/cgh" ]]; then
  DEFAULT_VLA_PATH="/mnt/storage/cgh/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/mnt/storage/cgh/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/mnt/storage/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_puretrain_4gpu"
else
  DEFAULT_VLA_PATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_puretrain_4gpu"
fi

VLA_PATH="${VLA_PATH:-${DEFAULT_VLA_PATH}}"
DATAPATH="${DATAPATH:-${DEFAULT_DATAPATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

echo "VLA_PATH=${VLA_PATH}"
echo "DATAPATH=${DATAPATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

torchrun --standalone --nnodes 1 --nproc_per_node 4 \
  openvla/specdecoding/train-scripts/train_dflash_libero_goal.py \
  --run_name dflash-anchor-hidden-1layer-finalhidden-puretrain-4gpu \
  --vla_path "${VLA_PATH}" \
  --datapath "${DATAPATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_draft_layers 1 \
  --target_layer_ids 1 8 15 22 29 \
  --selected_hidden_variant replace_22_with_final \
  --include_anchor_hidden \
  --anchor_consistency_w 0 \
  --soft_w 0 \
  --hidden_w 1.0 \
  --cos_w 0.05 \
  --hidden_noise 0.05 \
  --weight_decay 0.05 \
  --lr 5e-5 \
  --batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_epochs 200 \
  --warmup_steps 2000 \
  --save_every 10 \
  --val_split 0 \
  --block_size 7
