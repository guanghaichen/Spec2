#!/usr/bin/env bash
set -euo pipefail

# 三层 DFlash + 动作子词表顺序头训练入口。
# 核心路径：DFlash 主干一次并行生成 hidden，低秩前缀状态只在 256 个动作 bin 上递推；
# 训练目标直接覆盖 token 命中、teacher 动作分布、连续前缀存活和置信度校准。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_NAME="ckpt_goal_dflash_action_rnn_prefix_3layer_b8x2_4gpu"
if [[ -d "/data/wulin" ]]; then
  DEFAULT_VLA_PATH="/data/wulin/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/data/wulin/c/specvla-data/dflash_goal_dataset.h5"
  [[ -f "${DEFAULT_DATAPATH}" ]] || DEFAULT_DATAPATH="/data/wulin/c/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/data/wulin/c/specvla-data/${OUTPUT_NAME}"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
  DEFAULT_VLA_PATH="${ROOT}/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="${ROOT}/specvla-data/dflash_goal_dataset.h5"
  [[ -f "${DEFAULT_DATAPATH}" ]] || DEFAULT_DATAPATH="${ROOT}/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="${ROOT}/specvla-data/${OUTPUT_NAME}"
elif [[ -d "/mnt/storage/cgh" ]]; then
  DEFAULT_VLA_PATH="/mnt/storage/cgh/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/mnt/storage/cgh/specvla-data/dflash_goal_dataset.h5"
  [[ -f "${DEFAULT_DATAPATH}" ]] || DEFAULT_DATAPATH="/mnt/storage/cgh/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/mnt/storage/cgh/specvla-data/${OUTPUT_NAME}"
else
  ROOT="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh"
  DEFAULT_VLA_PATH="${ROOT}/data/models--openvla--openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="${ROOT}/specvla-data/dflash_goal_dataset.h5"
  [[ -f "${DEFAULT_DATAPATH}" ]] || DEFAULT_DATAPATH="${ROOT}/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="${ROOT}/specvla-data/${OUTPUT_NAME}"
fi

VLA_PATH="${VLA_PATH:-${DEFAULT_VLA_PATH}}"
DATAPATH="${DATAPATH:-${DEFAULT_DATAPATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# 训练规模：8 * 2 * 4 = global batch 64，与上一版 b16 四卡实验一致。
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
LR="${LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-1}"
DATASET_FORMAT="${DATASET_FORMAT:-auto}"

# 新版损失：前缀存活本身已对早期错误施加连锁惩罚，因此不再叠加 slot_decay 人工偏置。
HIDDEN_W="${HIDDEN_W:-0.30}"
COS_W="${COS_W:-0.02}"
ACTION_TOKEN_CE_W="${ACTION_TOKEN_CE_W:-0.10}"
ACTION_DISTILL_L1_W="${ACTION_DISTILL_L1_W:-0.90}"
PREFIX_SURVIVAL_W="${PREFIX_SURVIVAL_W:-0.50}"
ACTION_CONFIDENCE_W="${ACTION_CONFIDENCE_W:-0.10}"
ANCHOR_LOGIT_DISTILL_W="${ANCHOR_LOGIT_DISTILL_W:-0.10}"

cat <<EOF
========== DFlash Action-RNN Prefix 3-layer ==========
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
VLA_PATH=${VLA_PATH}
DATAPATH=${DATAPATH}
OUTPUT_DIR=${OUTPUT_DIR}
global_batch=${BATCH_SIZE} * ${GRAD_ACCUM_STEPS} * ${NPROC_PER_NODE}
epochs=${NUM_EPOCHS} lr=${LR} warmup=${WARMUP_STEPS}
hidden=${HIDDEN_W} cos=${COS_W}
action_ce=${ACTION_TOKEN_CE_W} action_l1=${ACTION_DISTILL_L1_W}
prefix_survival=${PREFIX_SURVIVAL_W} confidence=${ACTION_CONFIDENCE_W}
anchor_logit_distill=${ANCHOR_LOGIT_DISTILL_W}
=======================================================
EOF

export CUDA_VISIBLE_DEVICES
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

torchrun --standalone --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" \
  openvla/specdecoding/train-scripts/train_dflash_libero_goal.py \
  --run_name dflash-action-rnn-prefix-3layer-b8x2-4gpu \
  --vla_path "${VLA_PATH}" \
  --datapath "${DATAPATH}" \
  --dataset_format "${DATASET_FORMAT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_draft_layers 3 \
  --block_size 7 \
  --target_layer_ids 1 8 15 22 29 \
  --selected_hidden_variant replace_22_with_final \
  --include_anchor_hidden \
  --action_head_type slot_rnn \
  --action_head_rank 256 \
  --action_vocab_size 256 \
  --action_confidence_enabled \
  --hidden_w "${HIDDEN_W}" \
  --cos_w "${COS_W}" \
  --soft_w 0 \
  --action_token_ce_w "${ACTION_TOKEN_CE_W}" \
  --action_distill_l1_w "${ACTION_DISTILL_L1_W}" \
  --action_distill_temperature 1.0 \
  --prefix_survival_w "${PREFIX_SURVIVAL_W}" \
  --action_confidence_w "${ACTION_CONFIDENCE_W}" \
  --slot_decay 1.0 \
  --position_balance \
  --hidden_noise 0.03 \
  --anchor_consistency_w 0 \
  --causal_residual_type none \
  --causal_residual_cad_w 0 \
  --refined_hidden_w 0 \
  --residual_token_ce_w 0 \
  --logit_markov_type none \
  --anchor_logit_distill_w "${ANCHOR_LOGIT_DISTILL_W}" \
  --anchor_logit_distill_temperature 2.0 \
  --anchor_logit_distill_min_position 2 \
  --anchor_logit_distill_max_position 6 \
  --anchor_logit_distill_correct_teacher_only \
  --weight_decay 0.05 \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --num_epochs "${NUM_EPOCHS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --no-save_training_state \
  --no-save_latest_root_copy \
  --num_workers "${NUM_WORKERS}" \
  --dataloader_prefetch_factor 1 \
  --no-pin_memory \
  --no-persistent_workers \
  --swanlab_log_every_steps 200 \
  --swanlab_detail_every_steps 1000 \
  --val_split 0
