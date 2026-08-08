#!/usr/bin/env bash
set -euo pipefail

# DFlash 主训练的唯一入口。
#
#   bash openvla/specdecoding/train-scripts/run_dflash_train.sh minimal
#   bash openvla/specdecoding/train-scripts/run_dflash_train.sh joint
#
# minimal 是当前正式训练协议：只保留一层并行 Draft、完整目标上下文、
# multi-anchor 覆盖、Hidden/Cos 与轻量 Soft-KL；不创建 Action-RNN，也不启用跨
# Anchor 蒸馏、Domino 交接、hard CE、L1 或 Prefix Survival。
# 当前 Minimal 正式协议固定为 100 epoch、2 卡、每卡 batch 32、不做梯度累积，
# 即 global batch=64，与 Goal 正式实验保持一致；TASK_SUITE_NAME 选择四个 LIBERO 子集。
# joint、stage1、stage2 仅保留用于复现实验历程，不再是推荐主线。

PHASE="${1:-minimal}"
case "${PHASE}" in
  joint)
    TRAINING_PHASE=joint
    ;;
  minimal)
    TRAINING_PHASE=joint
    ;;
  stage1|representation)
    TRAINING_PHASE=representation
    ;;
  stage2|refinement)
    TRAINING_PHASE=refinement
    ;;
  *)
    echo "用法: bash $0 [minimal|joint|stage1|stage2]" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -d "/data/wulin" ]]; then
  MACHINE_DATA_ROOT="/data/wulin/c/specvla-data"
  DEFAULT_HF_ROOT="/data/wulin/hf_files"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  MACHINE_ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
  MACHINE_DATA_ROOT="${MACHINE_ROOT}/specvla-data"
  DEFAULT_HF_ROOT="${MACHINE_ROOT}/hf_files"
elif [[ -d "/mnt/storage/cgh" ]]; then
  MACHINE_DATA_ROOT="/mnt/storage/cgh/specvla-data"
  DEFAULT_HF_ROOT="/mnt/storage/cgh/hf_files"
else
  echo "无法识别机器路径；请显式设置 VLA_PATH、DATAPATH 和 OUTPUT_DIR。" >&2
  exit 1
fi

TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
case "${TASK_SUITE_NAME}" in
  libero_goal) SUITE_SLUG="goal" ;;
  libero_object) SUITE_SLUG="object" ;;
  libero_spatial) SUITE_SLUG="spatial" ;;
  libero_10) SUITE_SLUG="10" ;;
  *)
    echo "不支持的 TASK_SUITE_NAME=${TASK_SUITE_NAME}；应为 libero_goal、libero_object、libero_spatial 或 libero_10。" >&2
    exit 1
    ;;
esac

DEFAULT_VLA_PATH="${DEFAULT_HF_ROOT}/openvla-7b-finetuned-libero-${SUITE_SLUG}"
if [[ "${TASK_SUITE_NAME}" == "libero_goal" ]]; then
  DEFAULT_DATAPATH="${MACHINE_DATA_ROOT}/dflash_goal_dataset_envfix_20260714_packed_v2.h5"
else
  DEFAULT_DATAPATH="${MACHINE_DATA_ROOT}/dflash_${SUITE_SLUG}_dataset_packed_v2.h5"
fi
VLA_PATH="${VLA_PATH:-${DEFAULT_VLA_PATH}}"
DATAPATH="${DATAPATH:-${DEFAULT_DATAPATH}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-1}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SEED="${SEED:-7}"
NUM_WORKERS="${NUM_WORKERS:-1}"
# 每张卡连续读取64个样本；两卡在同一物理超级块内读取相邻切片。
HDF5_BLOCK_SIZE="${HDF5_BLOCK_SIZE:-64}"
SWANLAB_LOG_EVERY_STEPS="${SWANLAB_LOG_EVERY_STEPS:-20}"
SWANLAB_DETAIL_EVERY_STEPS="${SWANLAB_DETAIL_EVERY_STEPS:-200}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
IO_NICE_CLASS="${IO_NICE_CLASS:-2}"
IO_NICE_LEVEL="${IO_NICE_LEVEL:-7}"
CPU_NICE_LEVEL="${CPU_NICE_LEVEL:-10}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-dflash-libero-${SUITE_SLUG}}"

IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_GPU_IDS[@]}" -ne "${NPROC_PER_NODE}" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} 含 ${#VISIBLE_GPU_IDS[@]} 张卡，但 NPROC_PER_NODE=${NPROC_PER_NODE}。" >&2
  exit 1
fi

TRAIN_SHAPE="b${BATCH_SIZE}x${GRAD_ACCUM_STEPS}_${NPROC_PER_NODE}gpu"
TWO_STAGE_ROOT="${TWO_STAGE_ROOT:-${MACHINE_DATA_ROOT}/ckpt_${SUITE_SLUG}_dflash_two_stage_1layer_${TRAIN_SHAPE}}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${TWO_STAGE_ROOT}/stage1_representation}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${TWO_STAGE_ROOT}/stage2_refinement}"
JOINT_OUTPUT_DIR="${JOINT_OUTPUT_DIR:-${MACHINE_DATA_ROOT}/ckpt_${SUITE_SLUG}_dflash_joint_domino_1layer_${TRAIN_SHAPE}_packedv2}"
MINIMAL_OUTPUT_DIR="${MINIMAL_OUTPUT_DIR:-${MACHINE_DATA_ROOT}/ckpt_${SUITE_SLUG}_dflash_minimal_1layer_hidden_soft_${TRAIN_SHAPE}_packedv2}"

# 两阶段共同的高维表征约束。
HIDDEN_W="${HIDDEN_W:-1.00}"
COS_W="${COS_W:-0.05}"
SLOT_DECAY="${SLOT_DECAY:-0.90}"
HIDDEN_NOISE="${HIDDEN_NOISE:-0.03}"

INIT_ARGS=()
CURRICULUM_ARGS=(--no-staged_training --no-unified_cosine_curriculum --no-domino_linear_curriculum)
ACTION_HEAD_TYPE="slot_rnn"
MINIMAL_RECIPE_ARGS=(--no-minimal_draft_training)
if [[ "${PHASE}" == "minimal" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-${MINIMAL_OUTPUT_DIR}}"
  NUM_EPOCHS="${NUM_EPOCHS:-100}"
  LR="${LR:-2e-5}"
  ACTION_HEAD_LR="${ACTION_HEAD_LR:-${LR}}"
  WARMUP_STEPS="${WARMUP_STEPS:-1000}"
  ACTION_HEAD_WARMUP_STEPS=0
  SOFT_W="${SOFT_W:-0.05}"
  ACTION_TOKEN_CE_W=0
  BACKBONE_ANCHOR_LOGIT_DISTILL_W=0
  ACTION_DISTILL_L1_W=0
  PREFIX_SURVIVAL_W=0
  LOW_DIM_BUDGET_RATIO=0
  LOW_DIM_SCALE=1
  LOSS_SCALE_CALIBRATION_STEPS=1
  ACTION_HEAD_TYPE="action_only"
  MINIMAL_RECIPE_ARGS=(--minimal_draft_training)
  ACTION_HEAD_STATUS="disabled; frozen lm_head action rows only"
  RUN_NAME="${RUN_NAME:-dflash-${SUITE_SLUG}-minimal-hidden-soft-packedv2-${NUM_DRAFT_LAYERS}layer-b${BATCH_SIZE}x${GRAD_ACCUM_STEPS}-${NPROC_PER_NODE}gpu}"
elif [[ "${TRAINING_PHASE}" == "joint" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-${JOINT_OUTPUT_DIR}}"
  NUM_EPOCHS="${NUM_EPOCHS:-200}"
  LR="${LR:-2e-5}"
  ACTION_HEAD_LR="${ACTION_HEAD_LR:-5e-5}"
  WARMUP_STEPS="${WARMUP_STEPS:-1000}"
  ACTION_HEAD_WARMUP_STEPS="${ACTION_HEAD_WARMUP_STEPS:-500}"
  SOFT_W="${SOFT_W:-0.05}"
  ACTION_TOKEN_CE_W="${ACTION_TOKEN_CE_W:-0.01}"
  BACKBONE_ANCHOR_LOGIT_DISTILL_W="${BACKBONE_ANCHOR_LOGIT_DISTILL_W:-0.05}"
  ACTION_DISTILL_L1_W="${ACTION_DISTILL_L1_W:-0.05}"
  PREFIX_SURVIVAL_W="${PREFIX_SURVIVAL_W:-0.05}"
  LOW_DIM_BUDGET_RATIO="${LOW_DIM_BUDGET_RATIO:-0.10}"
  LOW_DIM_SCALE="${LOW_DIM_SCALE:--1}"
  LOSS_SCALE_CALIBRATION_STEPS="${LOSS_SCALE_CALIBRATION_STEPS:-8}"
  ACTION_HEAD_STATUS="joint_trainable"
  CURRICULUM_ARGS=(--no-staged_training --no-unified_cosine_curriculum --domino_linear_curriculum)
  RUN_NAME="${RUN_NAME:-dflash-${SUITE_SLUG}-joint-domino-packedv2-${NUM_DRAFT_LAYERS}layer-b${BATCH_SIZE}x${GRAD_ACCUM_STEPS}-${NPROC_PER_NODE}gpu}"
elif [[ "${TRAINING_PHASE}" == "representation" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-${STAGE1_OUTPUT_DIR}}"
  NUM_EPOCHS="${NUM_EPOCHS:-100}"
  LR="${LR:-2e-5}"
  ACTION_HEAD_LR="${ACTION_HEAD_LR:-5e-5}"
  WARMUP_STEPS="${WARMUP_STEPS:-1000}"
  ACTION_HEAD_WARMUP_STEPS="${ACTION_HEAD_WARMUP_STEPS:-500}"
  SOFT_W=0
  ACTION_TOKEN_CE_W=0
  BACKBONE_ANCHOR_LOGIT_DISTILL_W=0
  ACTION_DISTILL_L1_W=0
  PREFIX_SURVIVAL_W=0
  LOW_DIM_BUDGET_RATIO=0.10
  LOW_DIM_SCALE=1
  LOSS_SCALE_CALIBRATION_STEPS=8
  ACTION_HEAD_STATUS="frozen_and_skipped"
  RUN_NAME="${RUN_NAME:-dflash-${SUITE_SLUG}-stage1-representation-${NUM_DRAFT_LAYERS}layer-b${BATCH_SIZE}x${GRAD_ACCUM_STEPS}-${NPROC_PER_NODE}gpu}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${STAGE2_OUTPUT_DIR}}"
  NUM_EPOCHS="${NUM_EPOCHS:-100}"
  # 阶段二保护已成熟的 Draft，同时让新训练的 Action-RNN 使用独立较大学习率。
  LR="${LR:-5e-6}"
  ACTION_HEAD_LR="${ACTION_HEAD_LR:-5e-5}"
  WARMUP_STEPS="${WARMUP_STEPS:-500}"
  ACTION_HEAD_WARMUP_STEPS="${ACTION_HEAD_WARMUP_STEPS:-500}"
  SOFT_W="${SOFT_W:-0.05}"
  ACTION_TOKEN_CE_W="${ACTION_TOKEN_CE_W:-0.01}"
  BACKBONE_ANCHOR_LOGIT_DISTILL_W="${BACKBONE_ANCHOR_LOGIT_DISTILL_W:-0.05}"
  ACTION_DISTILL_L1_W="${ACTION_DISTILL_L1_W:-0.05}"
  PREFIX_SURVIVAL_W="${PREFIX_SURVIVAL_W:-0.05}"
  LOW_DIM_BUDGET_RATIO=0.10
  LOW_DIM_SCALE=1
  LOSS_SCALE_CALIBRATION_STEPS=8
  ACTION_HEAD_STATUS="trainable"
  STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_OUTPUT_DIR}}"
  if [[ ! -d "${STAGE1_CKPT}" ]]; then
    echo "阶段一 checkpoint 路径不存在: ${STAGE1_CKPT}" >&2
    exit 1
  fi
  if [[ ! -f "${STAGE1_CKPT}/pytorch_model.bin" ]] && \
     [[ ! -f "${STAGE1_CKPT}/latest_checkpoint.txt" ]]; then
    echo "阶段一目录缺少 pytorch_model.bin/latest_checkpoint.txt: ${STAGE1_CKPT}" >&2
    exit 1
  fi
  INIT_ARGS=(--init_model_checkpoint "${STAGE1_CKPT}")
  RUN_NAME="${RUN_NAME:-dflash-${SUITE_SLUG}-stage2-final-refinement-${NUM_DRAFT_LAYERS}layer-b${BATCH_SIZE}x${GRAD_ACCUM_STEPS}-${NPROC_PER_NODE}gpu}"
fi

for path in "${VLA_PATH}" "${DATAPATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "所需路径不存在: ${path}" >&2
    exit 1
  fi
done

if [[ -e "${OUTPUT_DIR}/run_config.json" ]] || \
   [[ -e "${OUTPUT_DIR}/metrics.jsonl" ]] || \
   [[ -d "${OUTPUT_DIR}/swanlog" ]]; then
  echo "输出目录已有训练日志: ${OUTPUT_DIR}" >&2
  echo "请归档旧目录，或设置新的 OUTPUT_DIR；不要把两次 SwanLab run 写进同一目录。" >&2
  exit 1
fi

cat <<EOF
========== DFlash 训练 ==========
PHASE=${PHASE} (${TRAINING_PHASE})
TASK_SUITE_NAME=${TASK_SUITE_NAME}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
GLOBAL_BATCH=${BATCH_SIZE} * ${GRAD_ACCUM_STEPS} * ${NPROC_PER_NODE}
VLA_PATH=${VLA_PATH}
DATAPATH=${DATAPATH}
OUTPUT_DIR=${OUTPUT_DIR}
TWO_STAGE_ROOT=${TWO_STAGE_ROOT}
NUM_EPOCHS=${NUM_EPOCHS}
LR=${LR}
ACTION_HEAD_LR=${ACTION_HEAD_LR} (${ACTION_HEAD_STATUS})
ACTION_HEAD_TYPE=${ACTION_HEAD_TYPE}
HIDDEN_W=${HIDDEN_W}
COS_W=${COS_W}
SOFT_KL_W=${SOFT_W}
FINAL_HARD_CE_W=${ACTION_TOKEN_CE_W}
CROSS_ANCHOR_KL_W=${BACKBONE_ANCHOR_LOGIT_DISTILL_W}
ACTION_RNN_L1_W=${ACTION_DISTILL_L1_W}
PREFIX_W=${PREFIX_SURVIVAL_W}
LOW_DIM_BUDGET_RATIO=${LOW_DIM_BUDGET_RATIO}
LOW_DIM_SCALE=${LOW_DIM_SCALE} (-1 表示启动前自动标定)
HDF5_BLOCK_SIZE=${HDF5_BLOCK_SIZE}
NUM_WORKERS=${NUM_WORKERS}
PROCESS_PRIORITY=ionice(${IO_NICE_CLASS},${IO_NICE_LEVEL}) nice(${CPU_NICE_LEVEL})
STAGE1_CKPT=${STAGE1_CKPT:-N/A}
SWANLAB_RUN=${RUN_NAME}
SWANLAB_PROJECT=${SWANLAB_PROJECT}
SWANLAB_MODE=${SWANLAB_MODE}
========================================
EOF

export CUDA_VISIBLE_DEVICES
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

if [[ "${DRY_RUN:-False}" == "True" ]]; then
  echo "DRY_RUN=True：路径、阶段依赖和输出隔离检查通过。"
  exit 0
fi

ionice -c "${IO_NICE_CLASS}" -n "${IO_NICE_LEVEL}" \
  nice -n "${CPU_NICE_LEVEL}" \
  torchrun --standalone --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" \
  openvla/specdecoding/train-scripts/train_dflash_libero_goal.py \
  --training_phase "${TRAINING_PHASE}" \
  "${MINIMAL_RECIPE_ARGS[@]}" \
  "${INIT_ARGS[@]}" \
  --run_name "${RUN_NAME}" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --vla_path "${VLA_PATH}" \
  --datapath "${DATAPATH}" \
  --dataset_format auto \
  --output_dir "${OUTPUT_DIR}" \
  --num_draft_layers "${NUM_DRAFT_LAYERS}" \
  --block_size 7 \
  --num_target_feature_layers 5 \
  --selected_hidden_variant target_layers \
  --include_anchor_hidden \
  --action_head_type "${ACTION_HEAD_TYPE}" \
  --action_head_rank 256 \
  --no-detach_action_head_inputs \
  --action_vocab_size 256 \
  --no-action_confidence_enabled \
  --hidden_w "${HIDDEN_W}" \
  --cos_w "${COS_W}" \
  --soft_w "${SOFT_W}" \
  --soft_loss_type kl \
  --soft_temperature 2.0 \
  --action_token_ce_w "${ACTION_TOKEN_CE_W}" \
  --action_distill_l1_w "${ACTION_DISTILL_L1_W}" \
  --action_distill_temperature 1.0 \
  --prefix_survival_w "${PREFIX_SURVIVAL_W}" \
  --action_confidence_w 0 \
  --slot_decay "${SLOT_DECAY}" \
  --position_balance \
  --hidden_noise "${HIDDEN_NOISE}" \
  --anchor_consistency_w 0 \
  --causal_residual_type none \
  --causal_residual_cad_w 0 \
  --refined_hidden_w 0 \
  --residual_token_ce_w 0 \
  --logit_markov_type none \
  --anchor_logit_distill_w 0 \
  --backbone_anchor_logit_distill_w "${BACKBONE_ANCHOR_LOGIT_DISTILL_W}" \
  --anchor_logit_distill_temperature 2.0 \
  --anchor_logit_distill_min_position 2 \
  --anchor_logit_distill_max_position 6 \
  --anchor_logit_distill_correct_teacher_only \
  --weight_decay 0.05 \
  --lr "${LR}" \
  --action_head_lr "${ACTION_HEAD_LR}" \
  "${CURRICULUM_ARGS[@]}" \
  --low_dim_budget_ratio "${LOW_DIM_BUDGET_RATIO}" \
  --low_dim_scale "${LOW_DIM_SCALE}" \
  --loss_scale_calibration_steps "${LOSS_SCALE_CALIBRATION_STEPS}" \
  --action_head_warmup_steps "${ACTION_HEAD_WARMUP_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --num_epochs "${NUM_EPOCHS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --seed "${SEED}" \
  --no-save_training_state \
  --no-save_latest_root_copy \
  --num_workers "${NUM_WORKERS}" \
  --hdf5_block_size "${HDF5_BLOCK_SIZE}" \
  --dataloader_prefetch_factor 1 \
  --no-pin_memory \
  --persistent_workers \
  --swanlab_log_every_steps "${SWANLAB_LOG_EVERY_STEPS}" \
  --swanlab_detail_every_steps "${SWANLAB_DETAIL_EVERY_STEPS}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_mode "${SWANLAB_MODE}" \
  --val_split 0
