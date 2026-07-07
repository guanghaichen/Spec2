#!/usr/bin/env bash
set -euo pipefail

# Markov-ACD / DFlash Goal 训练入口。
# 默认用于 3090 四卡训练；所有常改超参数都可以用环境变量覆盖，例如：
#   CUDA_VISIBLE_DEVICES=4,5,6,7 BATCH_SIZE=12 SOFT_W=0.05 \
#     bash openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_residual_cad_4gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# -----------------------------------------------------------------------------
# 1. 机器路径默认值
# -----------------------------------------------------------------------------
# 注意：这里的 VLA_PATH 是 Goal 训练专用路径，不是评测脚本里的全局 VLA_PATH。
# 多子集评测请交给 decode-scripts/libero_eval_common.sh 自动选择 checkpoint。
OUTPUT_NAME="ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_firststep_tokence_soft01_b16_4gpu"
if [[ -d "/data/wulin" ]]; then
  DEFAULT_VLA_PATH="/data/wulin/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/data/wulin/c/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/data/wulin/c/specvla-data/${OUTPUT_NAME}"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  DEFAULT_VLA_PATH="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/${OUTPUT_NAME}"
elif [[ -d "/mnt/storage/cgh" ]]; then
  DEFAULT_VLA_PATH="/mnt/storage/cgh/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/mnt/storage/cgh/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/mnt/storage/cgh/specvla-data/${OUTPUT_NAME}"
else
  DEFAULT_VLA_PATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/${OUTPUT_NAME}"
fi

VLA_PATH="${VLA_PATH:-${DEFAULT_VLA_PATH}}"
DATAPATH="${DATAPATH:-${DEFAULT_DATAPATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

# -----------------------------------------------------------------------------
# 2. 训练规模与优化器
# -----------------------------------------------------------------------------
# BATCH_SIZE 是每张卡的 micro batch；4 卡默认 global batch = 16 * 4 = 64。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
SAVE_EVERY="${SAVE_EVERY:-10}"
HIDDEN_NOISE="${HIDDEN_NOISE:-0.03}"

# -----------------------------------------------------------------------------
# 3. 主损失权重
# -----------------------------------------------------------------------------
# hidden_w / cos_w 在 torchrun 参数区固定；这里保留经常要调的 soft_w 和 refined_hidden_w。
SOFT_W="${SOFT_W:-0.10}"                     # teacher soft distribution 蒸馏，低权重辅助 token 分布对齐。
REFINED_HIDDEN_W="${REFINED_HIDDEN_W:-0.30}" # 残差修正后的 weak-path hidden 直接追目标 hidden。
REFINED_HIDDEN_TYPE="${REFINED_HIDDEN_TYPE:-smooth_l1}"

# -----------------------------------------------------------------------------
# 4. Markov-ACD / 弱路径增强
# -----------------------------------------------------------------------------
CAUSAL_RESIDUAL_START_INDEX="${CAUSAL_RESIDUAL_START_INDEX:-0}"  # 0=连 local slot0/第一跳也用前序 token 修正。
RESIDUAL_CAD_W="${RESIDUAL_CAD_W:-0.10}"                         # refined weak hidden 追 strong anchor hidden。
RESIDUAL_CAD_TYPE="${RESIDUAL_CAD_TYPE:-cosine}"
RESIDUAL_CAD_WARMUP_STEPS="${RESIDUAL_CAD_WARMUP_STEPS:-4000}"   # 避免训练初期强路径尚未稳定时 CAD 过早压制 weak path。
RESIDUAL_TOKEN_CE_W="${RESIDUAL_TOKEN_CE_W:-0.10}"               # 只监督残差修正后的 logits，下一版覆盖 p1-p5。
RESIDUAL_TOKEN_CE_MIN_POSITION="${RESIDUAL_TOKEN_CE_MIN_POSITION:-1}"
RESIDUAL_TOKEN_CE_MAX_POSITION="${RESIDUAL_TOKEN_CE_MAX_POSITION:-5}"
LOGIT_MARKOV_TYPE="${LOGIT_MARKOV_TYPE:-bias}"                   # logits 级轻量 Markov bias；none 可关闭。
LOGIT_MARKOV_RANK="${LOGIT_MARKOV_RANK:-256}"
LOGIT_MARKOV_SCALE="${LOGIT_MARKOV_SCALE:-1.0}"
REFINED_HIDDEN_MIN_POSITION="${REFINED_HIDDEN_MIN_POSITION:-1}"
REFINED_HIDDEN_MAX_POSITION="${REFINED_HIDDEN_MAX_POSITION:-5}"
ANCHOR_LOGIT_DISTILL_W="${ANCHOR_LOGIT_DISTILL_W:-0.10}"         # weak anchor logits 追 strong anchor logits。
ANCHOR_LOGIT_DISTILL_TEMPERATURE="${ANCHOR_LOGIT_DISTILL_TEMPERATURE:-2.0}"
WEAK_FAR_SLOT_BOOST="${WEAK_FAR_SLOT_BOOST:-2.0}"                # p2-p5 等弱路径位置主损失加权。
FIRST_STEP_BOOST="${FIRST_STEP_BOOST:-2.0}"                      # local slot0/第一跳单独加权，补强 t1 与各 anchor 一步预测。
FIRST_STEP_BOOST_MIN_POSITION="${FIRST_STEP_BOOST_MIN_POSITION:-1}"
FIRST_STEP_BOOST_MAX_POSITION="${FIRST_STEP_BOOST_MAX_POSITION:-5}"
ANCHOR0_P2_BOOST="${ANCHOR0_P2_BOOST:-4.0}"                      # 最瓶颈的 anchor0->p2 额外加权。

print_config() {
  cat <<EOF
========== DFlash Markov-ACD 训练配置 ==========
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
NPROC_PER_NODE=${NPROC_PER_NODE}
VLA_PATH=${VLA_PATH}
DATAPATH=${DATAPATH}
OUTPUT_DIR=${OUTPUT_DIR}

[训练规模]
BATCH_SIZE=${BATCH_SIZE}
LR=${LR}
WARMUP_STEPS=${WARMUP_STEPS}
NUM_EPOCHS=${NUM_EPOCHS}
SAVE_EVERY=${SAVE_EVERY}
HIDDEN_NOISE=${HIDDEN_NOISE}

[损失权重]
SOFT_W=${SOFT_W}
REFINED_HIDDEN_W=${REFINED_HIDDEN_W}
REFINED_HIDDEN_TYPE=${REFINED_HIDDEN_TYPE}
RESIDUAL_CAD_W=${RESIDUAL_CAD_W}
RESIDUAL_CAD_TYPE=${RESIDUAL_CAD_TYPE}
RESIDUAL_CAD_WARMUP_STEPS=${RESIDUAL_CAD_WARMUP_STEPS}
RESIDUAL_TOKEN_CE_W=${RESIDUAL_TOKEN_CE_W}
ANCHOR_LOGIT_DISTILL_W=${ANCHOR_LOGIT_DISTILL_W}
ANCHOR_LOGIT_DISTILL_TEMPERATURE=${ANCHOR_LOGIT_DISTILL_TEMPERATURE}

[Markov / 弱路径]
CAUSAL_RESIDUAL_START_INDEX=${CAUSAL_RESIDUAL_START_INDEX}
LOGIT_MARKOV_TYPE=${LOGIT_MARKOV_TYPE}
LOGIT_MARKOV_RANK=${LOGIT_MARKOV_RANK}
LOGIT_MARKOV_SCALE=${LOGIT_MARKOV_SCALE}
REFINED_HIDDEN_MIN_POSITION=${REFINED_HIDDEN_MIN_POSITION}
REFINED_HIDDEN_MAX_POSITION=${REFINED_HIDDEN_MAX_POSITION}
RESIDUAL_TOKEN_CE_MIN_POSITION=${RESIDUAL_TOKEN_CE_MIN_POSITION}
RESIDUAL_TOKEN_CE_MAX_POSITION=${RESIDUAL_TOKEN_CE_MAX_POSITION}
WEAK_FAR_SLOT_BOOST=${WEAK_FAR_SLOT_BOOST}
FIRST_STEP_BOOST=${FIRST_STEP_BOOST}
FIRST_STEP_BOOST_MIN_POSITION=${FIRST_STEP_BOOST_MIN_POSITION}
FIRST_STEP_BOOST_MAX_POSITION=${FIRST_STEP_BOOST_MAX_POSITION}
ANCHOR0_P2_BOOST=${ANCHOR0_P2_BOOST}
================================================
EOF
}

print_config

export CUDA_VISIBLE_DEVICES

torchrun --standalone --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" \
  openvla/specdecoding/train-scripts/train_dflash_libero_goal.py \
  --run_name dflash-anchor-hidden-1layer-finalhidden-markov-acd-firststep-tokence-soft01-b16-4gpu \
  --vla_path "${VLA_PATH}" \
  --datapath "${DATAPATH}" \
  --output_dir "${OUTPUT_DIR}" \
  \
  --num_draft_layers 1 \
  --block_size 7 \
  --target_layer_ids 1 8 15 22 29 \
  --selected_hidden_variant replace_22_with_final \
  --include_anchor_hidden \
  \
  --hidden_w 1.0 \
  --cos_w 0.05 \
  --soft_w "${SOFT_W}" \
  --slot_decay 1.0 \
  --hidden_noise "${HIDDEN_NOISE}" \
  \
  --anchor_consistency_w 0 \
  --causal_residual_type hidden \
  --causal_residual_rank 256 \
  --causal_residual_scale 1.0 \
  --causal_residual_start_index "${CAUSAL_RESIDUAL_START_INDEX}" \
  --causal_residual_cad_w "${RESIDUAL_CAD_W}" \
  --causal_residual_cad_type "${RESIDUAL_CAD_TYPE}" \
  --causal_residual_cad_warmup_steps "${RESIDUAL_CAD_WARMUP_STEPS}" \
  --causal_residual_cad_correct_teacher_only \
  --causal_residual_min_position 2 \
  --causal_residual_max_position 5 \
  --refined_hidden_w "${REFINED_HIDDEN_W}" \
  --refined_hidden_loss_type "${REFINED_HIDDEN_TYPE}" \
  --refined_hidden_min_position "${REFINED_HIDDEN_MIN_POSITION}" \
  --refined_hidden_max_position "${REFINED_HIDDEN_MAX_POSITION}" \
  --residual_token_ce_w "${RESIDUAL_TOKEN_CE_W}" \
  --residual_token_ce_min_position "${RESIDUAL_TOKEN_CE_MIN_POSITION}" \
  --residual_token_ce_max_position "${RESIDUAL_TOKEN_CE_MAX_POSITION}" \
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
  --first_step_boost "${FIRST_STEP_BOOST}" \
  --first_step_boost_min_position "${FIRST_STEP_BOOST_MIN_POSITION}" \
  --first_step_boost_max_position "${FIRST_STEP_BOOST_MAX_POSITION}" \
  --anchor0_p2_boost "${ANCHOR0_P2_BOOST}" \
  \
  --weight_decay 0.05 \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps 1 \
  --num_epochs "${NUM_EPOCHS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --val_split 0
