#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ -d "/data/wulin" ]]; then
  DEFAULT_VLA_PATH="/data/wulin/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/data/wulin/c/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_tokence_soft01_b16_4gpu"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  DEFAULT_VLA_PATH="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_tokence_soft01_b16_4gpu"
elif [[ -d "/mnt/storage/cgh" ]]; then
  DEFAULT_VLA_PATH="/mnt/storage/cgh/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/mnt/storage/cgh/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/mnt/storage/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_tokence_soft01_b16_4gpu"
else
  DEFAULT_VLA_PATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_tokence_soft01_b16_4gpu"
fi

VLA_PATH="${VLA_PATH:-${DEFAULT_VLA_PATH}}"
DATAPATH="${DATAPATH:-${DEFAULT_DATAPATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RESIDUAL_CAD_W="${RESIDUAL_CAD_W:-0.10}"
RESIDUAL_CAD_TYPE="${RESIDUAL_CAD_TYPE:-cosine}"
RESIDUAL_CAD_WARMUP_STEPS="${RESIDUAL_CAD_WARMUP_STEPS:-4000}"
RESIDUAL_TOKEN_CE_W="${RESIDUAL_TOKEN_CE_W:-0.10}"
LOGIT_MARKOV_TYPE="${LOGIT_MARKOV_TYPE:-bias}"
LOGIT_MARKOV_RANK="${LOGIT_MARKOV_RANK:-256}"
LOGIT_MARKOV_SCALE="${LOGIT_MARKOV_SCALE:-1.0}"
ANCHOR_LOGIT_DISTILL_W="${ANCHOR_LOGIT_DISTILL_W:-0.10}"
ANCHOR_LOGIT_DISTILL_TEMPERATURE="${ANCHOR_LOGIT_DISTILL_TEMPERATURE:-2.0}"
SOFT_W="${SOFT_W:-0.10}"
REFINED_HIDDEN_W="${REFINED_HIDDEN_W:-0.30}"
REFINED_HIDDEN_TYPE="${REFINED_HIDDEN_TYPE:-smooth_l1}"
WEAK_FAR_SLOT_BOOST="${WEAK_FAR_SLOT_BOOST:-2.0}"
ANCHOR0_P2_BOOST="${ANCHOR0_P2_BOOST:-4.0}"
HIDDEN_NOISE="${HIDDEN_NOISE:-0.03}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "VLA_PATH=${VLA_PATH}"
echo "DATAPATH=${DATAPATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LR=${LR}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "RESIDUAL_CAD_W=${RESIDUAL_CAD_W}"
echo "RESIDUAL_CAD_TYPE=${RESIDUAL_CAD_TYPE}"
echo "RESIDUAL_CAD_WARMUP_STEPS=${RESIDUAL_CAD_WARMUP_STEPS}"
echo "REFINED_HIDDEN_W=${REFINED_HIDDEN_W}"
echo "RESIDUAL_TOKEN_CE_W=${RESIDUAL_TOKEN_CE_W}"
echo "SOFT_W=${SOFT_W}"
echo "LOGIT_MARKOV_TYPE=${LOGIT_MARKOV_TYPE}"
echo "LOGIT_MARKOV_RANK=${LOGIT_MARKOV_RANK}"
echo "LOGIT_MARKOV_SCALE=${LOGIT_MARKOV_SCALE}"
echo "ANCHOR_LOGIT_DISTILL_W=${ANCHOR_LOGIT_DISTILL_W}"
echo "ANCHOR_LOGIT_DISTILL_TEMPERATURE=${ANCHOR_LOGIT_DISTILL_TEMPERATURE}"
echo "WEAK_FAR_SLOT_BOOST=${WEAK_FAR_SLOT_BOOST}"
echo "ANCHOR0_P2_BOOST=${ANCHOR0_P2_BOOST}"

torchrun --standalone --nnodes 1 --nproc_per_node 4 \
  openvla/specdecoding/train-scripts/train_dflash_libero_goal.py \
  --run_name dflash-anchor-hidden-1layer-finalhidden-markov-acd-tokence-soft01-b16-4gpu \
  --vla_path "${VLA_PATH}" \
  --datapath "${DATAPATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_draft_layers 1 \
  --target_layer_ids 1 8 15 22 29 \
  --selected_hidden_variant replace_22_with_final \
  --include_anchor_hidden \
  --anchor_consistency_w 0 \
  --causal_residual_type hidden \
  --causal_residual_rank 256 \
  --causal_residual_scale 1.0 \
  --causal_residual_start_index 1 \
  --causal_residual_cad_w "${RESIDUAL_CAD_W}" \
  --causal_residual_cad_type "${RESIDUAL_CAD_TYPE}" \
  --causal_residual_cad_warmup_steps "${RESIDUAL_CAD_WARMUP_STEPS}" \
  --causal_residual_cad_correct_teacher_only \
  --causal_residual_min_position 2 \
  --causal_residual_max_position 5 \
  --refined_hidden_w "${REFINED_HIDDEN_W}" \
  --refined_hidden_loss_type "${REFINED_HIDDEN_TYPE}" \
  --refined_hidden_min_position 2 \
  --refined_hidden_max_position 5 \
  --residual_token_ce_w "${RESIDUAL_TOKEN_CE_W}" \
  --residual_token_ce_min_position 2 \
  --residual_token_ce_max_position 5 \
  --residual_token_ce_label_smoothing 0 \
  --logit_markov_type "${LOGIT_MARKOV_TYPE}" \
  --logit_markov_rank "${LOGIT_MARKOV_RANK}" \
  --logit_markov_scale "${LOGIT_MARKOV_SCALE}" \
  --anchor_logit_distill_w "${ANCHOR_LOGIT_DISTILL_W}" \
  --anchor_logit_distill_temperature "${ANCHOR_LOGIT_DISTILL_TEMPERATURE}" \
  --anchor_logit_distill_min_position 2 \
  --anchor_logit_distill_max_position 5 \
  --anchor_logit_distill_correct_teacher_only \
  --weak_far_slot_boost "${WEAK_FAR_SLOT_BOOST}" \
  --anchor0_p2_boost "${ANCHOR0_P2_BOOST}" \
  --soft_w "${SOFT_W}" \
  --hidden_w 1.0 \
  --cos_w 0.05 \
  --slot_decay 1.0 \
  --hidden_noise "${HIDDEN_NOISE}" \
  --weight_decay 0.05 \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps 1 \
  --num_epochs "${NUM_EPOCHS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --val_split 0 \
  --block_size 7
