#!/usr/bin/env bash
set -euo pipefail

# 当前 DFlash 训练统一入口。
#
#   bash .../run_dflash_train.sh main        # 推荐：一层 Action-RNN Prefix Survival
#   bash .../run_dflash_train.sh three_layer # 容量消融：三层主干
#   bash .../run_dflash_train.sh no_prefix   # 去掉 Prefix Survival
#   bash .../run_dflash_train.sh no_anchor   # 去掉跨 anchor logits 蒸馏
#
# 所有默认值均可用同名环境变量覆盖；启动时会完整打印最终配置。

PROFILE="${1:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

case "${PROFILE}" in
  main)
    PROFILE_LAYERS=1
    PROFILE_PREFIX_W=0.20
    PROFILE_ANCHOR_W=0.05
    PROFILE_OUTPUT_NAME=ckpt_goal_dflash_action_rnn_decoupled_1layer_b8x2_4gpu
    ;;
  three_layer)
    PROFILE_LAYERS=3
    PROFILE_PREFIX_W=0.20
    PROFILE_ANCHOR_W=0.05
    PROFILE_OUTPUT_NAME=ckpt_goal_dflash_action_rnn_decoupled_3layer_b8x2_4gpu
    ;;
  no_prefix)
    PROFILE_LAYERS=1
    PROFILE_PREFIX_W=0
    PROFILE_ANCHOR_W=0.05
    PROFILE_OUTPUT_NAME=ckpt_goal_dflash_action_rnn_decoupled_no_prefix_1layer_b8x2_4gpu
    ;;
  no_anchor)
    PROFILE_LAYERS=1
    PROFILE_PREFIX_W=0.20
    PROFILE_ANCHOR_W=0
    PROFILE_OUTPUT_NAME=ckpt_goal_dflash_action_rnn_decoupled_no_anchor_1layer_b8x2_4gpu
    ;;
  *)
    echo "用法: bash $0 [main|three_layer|no_prefix|no_anchor]" >&2
    exit 1
    ;;
esac

NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-${PROFILE_LAYERS}}"
PREFIX_SURVIVAL_W="${PREFIX_SURVIVAL_W:-${PROFILE_PREFIX_W}}"
ANCHOR_LOGIT_DISTILL_W="${ANCHOR_LOGIT_DISTILL_W:-${PROFILE_ANCHOR_W}}"
OUTPUT_NAME="${OUTPUT_NAME:-${PROFILE_OUTPUT_NAME}}"

if [[ -d "/data/wulin" ]]; then
  DEFAULT_VLA_PATH="/data/wulin/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/data/wulin/c/specvla-data/dflash_goal_dataset_envfix_20260714.h5"
  DEFAULT_OUTPUT_DIR="/data/wulin/c/specvla-data/${OUTPUT_NAME}"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  MACHINE_ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
  DEFAULT_VLA_PATH="${MACHINE_ROOT}/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="${MACHINE_ROOT}/specvla-data/dflash_goal_dataset_envfix_20260714.h5"
  DEFAULT_OUTPUT_DIR="${MACHINE_ROOT}/specvla-data/${OUTPUT_NAME}"
elif [[ -d "/mnt/storage/cgh" ]]; then
  DEFAULT_VLA_PATH="/mnt/storage/cgh/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/mnt/storage/cgh/specvla-data/dflash_goal_dataset_envfix_20260714.h5"
  DEFAULT_OUTPUT_DIR="/mnt/storage/cgh/specvla-data/${OUTPUT_NAME}"
else
  echo "无法识别机器路径；请显式设置 VLA_PATH、DATAPATH 和 OUTPUT_DIR。" >&2
  exit 1
fi

VLA_PATH="${VLA_PATH:-${DEFAULT_VLA_PATH}}"
DATAPATH="${DATAPATH:-${DEFAULT_DATAPATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# 训练规模：默认 global batch = 8 micro-batch * 2 累积 * 4 GPU = 64。
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
LR="${LR:-2e-5}"
ACTION_HEAD_LR="${ACTION_HEAD_LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SEED="${SEED:-7}"

# 稳定版反传：hidden/cos 只训练 Draft 主干；动作损失只训练 Action-RNN。
HIDDEN_W="${HIDDEN_W:-1.00}"
COS_W="${COS_W:-0.05}"
ACTION_TOKEN_CE_W="${ACTION_TOKEN_CE_W:-0.10}"
ACTION_DISTILL_L1_W="${ACTION_DISTILL_L1_W:-0.40}"

# 共享服务器 IO 与 SwanLab 设置。
NUM_WORKERS="${NUM_WORKERS:-1}"
SWANLAB_LOG_EVERY_STEPS="${SWANLAB_LOG_EVERY_STEPS:-20}"
SWANLAB_DETAIL_EVERY_STEPS="${SWANLAB_DETAIL_EVERY_STEPS:-200}"

for path in "${VLA_PATH}" "${DATAPATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "所需路径不存在: ${path}" >&2
    exit 1
  fi
done

cat <<EOF
========== DFlash Goal 训练 ==========
PROFILE=${PROFILE}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
NPROC_PER_NODE=${NPROC_PER_NODE}
VLA_PATH=${VLA_PATH}
DATAPATH=${DATAPATH}
OUTPUT_DIR=${OUTPUT_DIR}

[模型]
NUM_DRAFT_LAYERS=${NUM_DRAFT_LAYERS}
ACTION_HEAD=slot_rnn(rank=256)
TARGET_FEATURE_LAYERS=5_evenly_spaced

[优化]
GLOBAL_BATCH=${BATCH_SIZE} * ${GRAD_ACCUM_STEPS} * ${NPROC_PER_NODE}
NUM_EPOCHS=${NUM_EPOCHS}
LR=${LR}
ACTION_HEAD_LR=${ACTION_HEAD_LR}
WARMUP_STEPS=${WARMUP_STEPS}
SAVE_EVERY=${SAVE_EVERY}
SEED=${SEED}

[损失权重]
HIDDEN_W=${HIDDEN_W}
COS_W=${COS_W}
ACTION_TOKEN_CE_W=${ACTION_TOKEN_CE_W}
ACTION_DISTILL_L1_W=${ACTION_DISTILL_L1_W}
PREFIX_SURVIVAL_W=${PREFIX_SURVIVAL_W}
ANCHOR_LOGIT_DISTILL_W=${ANCHOR_LOGIT_DISTILL_W}

[IO / 日志]
NUM_WORKERS=${NUM_WORKERS}
SWANLAB_LOG_EVERY_STEPS=${SWANLAB_LOG_EVERY_STEPS}
SWANLAB_DETAIL_EVERY_STEPS=${SWANLAB_DETAIL_EVERY_STEPS}
======================================
EOF

export CUDA_VISIBLE_DEVICES
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

if [[ "${DRY_RUN:-False}" == "True" ]]; then
  echo "DRY_RUN=True：路径和配置检查通过，不启动 torchrun。"
  exit 0
fi

torchrun --standalone --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" \
  openvla/specdecoding/train-scripts/train_dflash_libero_goal.py \
  --run_name "dflash-action-rnn-${PROFILE}-${NUM_DRAFT_LAYERS}layer-b8x2-4gpu" \
  --vla_path "${VLA_PATH}" \
  --datapath "${DATAPATH}" \
  --dataset_format auto \
  --output_dir "${OUTPUT_DIR}" \
  --num_draft_layers "${NUM_DRAFT_LAYERS}" \
  --block_size 7 \
  --num_target_feature_layers 5 \
  --selected_hidden_variant target_layers \
  --include_anchor_hidden \
  --action_head_type slot_rnn \
  --action_head_rank 256 \
  --detach_action_head_inputs \
  --action_vocab_size 256 \
  --no-action_confidence_enabled \
  --hidden_w "${HIDDEN_W}" \
  --cos_w "${COS_W}" \
  --soft_w 0 \
  --action_token_ce_w "${ACTION_TOKEN_CE_W}" \
  --action_distill_l1_w "${ACTION_DISTILL_L1_W}" \
  --action_distill_temperature 1.0 \
  --prefix_survival_w "${PREFIX_SURVIVAL_W}" \
  --action_confidence_w 0 \
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
  --action_head_lr "${ACTION_HEAD_LR}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --num_epochs "${NUM_EPOCHS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --seed "${SEED}" \
  --no-save_training_state \
  --no-save_latest_root_copy \
  --num_workers "${NUM_WORKERS}" \
  --dataloader_prefetch_factor 1 \
  --no-pin_memory \
  --no-persistent_workers \
  --swanlab_log_every_steps "${SWANLAB_LOG_EVERY_STEPS}" \
  --swanlab_detail_every_steps "${SWANLAB_DETAIL_EVERY_STEPS}" \
  --val_split 0
