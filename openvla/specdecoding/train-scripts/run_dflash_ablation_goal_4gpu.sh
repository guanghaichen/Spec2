#!/usr/bin/env bash
set -euo pipefail

# 三档核心消融统一入口。默认四卡 3090 训练，目标是形成清楚的论文故事线：
#   1) pure_hidden : 纯 DFlash 块并行 draft，只有 hidden/cos 几何监督，无 soft、无 CAD、无残差头。
#   2) anchor_cad  : 在 pure_hidden 上加入跨 anchor 的 Hidden/Logit CAD，但不启用 Markov 残差修正头。
#   3) markov_acd  : 当前完整方案，加入 Markov-aware residual/logit 修正、CAD、低权重 soft loss。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

ABLATION_STAGE="${ABLATION_STAGE:-markov_acd}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SEED="${SEED:-7}"
HIDDEN_NOISE="${HIDDEN_NOISE:-0.03}"
SLOT_DECAY="${SLOT_DECAY:-0.90}"              # 三档共用，保持位置训练权重一致，消融只比较结构信号。
HIDDEN_W="${HIDDEN_W:-1.0}"
COS_W="${COS_W:-0.05}"                       # hidden 几何辅助项，不引入 token/soft 标签。

# 共享服务器 IO 控制：优先使用 sharded 数据；默认不保存 optimizer state，减少硬盘写入。
DATASET_FORMAT="${DATASET_FORMAT:-auto}"
NUM_WORKERS="${NUM_WORKERS:-1}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
SWANLAB_LOG_EVERY_STEPS="${SWANLAB_LOG_EVERY_STEPS:-200}"
SWANLAB_DETAIL_EVERY_STEPS="${SWANLAB_DETAIL_EVERY_STEPS:-1000}"
SAVE_TRAINING_STATE="${SAVE_TRAINING_STATE:-False}"
SAVE_LATEST_ROOT_COPY="${SAVE_LATEST_ROOT_COPY:-False}"

case "${ABLATION_STAGE}" in
  pure_hidden)
    RUN_NAME="${RUN_NAME:-dflash-ablation-1-pure-hidden-1layer-finalhidden-b16-4gpu}"
    OUTPUT_NAME="${OUTPUT_NAME:-ckpt_goal_dflash_ablation_1_pure_hidden_1layer_finalhidden_slotdecay090_b16_4gpu}"
    SOFT_W="${SOFT_W:-0}"
    CAUSAL_RESIDUAL_TYPE="${CAUSAL_RESIDUAL_TYPE:-none}"
    RESIDUAL_CAD_W="${RESIDUAL_CAD_W:-0}"
    REFINED_HIDDEN_W="${REFINED_HIDDEN_W:-0}"
    RESIDUAL_TOKEN_CE_W="${RESIDUAL_TOKEN_CE_W:-0}"
    LOGIT_MARKOV_TYPE="${LOGIT_MARKOV_TYPE:-none}"
    ANCHOR_LOGIT_DISTILL_W="${ANCHOR_LOGIT_DISTILL_W:-0}"
    ;;
  anchor_cad)
    RUN_NAME="${RUN_NAME:-dflash-ablation-2-anchor-cad-1layer-finalhidden-b16-4gpu}"
    OUTPUT_NAME="${OUTPUT_NAME:-ckpt_goal_dflash_ablation_2_anchor_cad_1layer_finalhidden_slotdecay090_b16_4gpu}"
    SOFT_W="${SOFT_W:-0}"
    CAUSAL_RESIDUAL_TYPE="${CAUSAL_RESIDUAL_TYPE:-none}"
    RESIDUAL_CAD_W="${RESIDUAL_CAD_W:-0.10}"
    REFINED_HIDDEN_W="${REFINED_HIDDEN_W:-0}"
    RESIDUAL_TOKEN_CE_W="${RESIDUAL_TOKEN_CE_W:-0}"
    LOGIT_MARKOV_TYPE="${LOGIT_MARKOV_TYPE:-none}"
    ANCHOR_LOGIT_DISTILL_W="${ANCHOR_LOGIT_DISTILL_W:-0.10}"
    ;;
  markov_acd)
    RUN_NAME="${RUN_NAME:-dflash-anchor-hidden-1layer-finalhidden-markov-acd-start0-slotdecay090-tokence-soft01-b16-4gpu}"
    OUTPUT_NAME="${OUTPUT_NAME:-ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu}"
    SOFT_W="${SOFT_W:-0.10}"
    CAUSAL_RESIDUAL_TYPE="${CAUSAL_RESIDUAL_TYPE:-hidden}"
    RESIDUAL_CAD_W="${RESIDUAL_CAD_W:-0.10}"
    REFINED_HIDDEN_W="${REFINED_HIDDEN_W:-0.30}"
    RESIDUAL_TOKEN_CE_W="${RESIDUAL_TOKEN_CE_W:-0.10}"
    LOGIT_MARKOV_TYPE="${LOGIT_MARKOV_TYPE:-bias}"
    ANCHOR_LOGIT_DISTILL_W="${ANCHOR_LOGIT_DISTILL_W:-0.10}"
    ;;
  *)
    echo "Unsupported ABLATION_STAGE=${ABLATION_STAGE}" >&2
    echo "Expected one of: pure_hidden, anchor_cad, markov_acd" >&2
    exit 1
    ;;
esac

if [[ -d "/data/wulin" ]]; then
  DEFAULT_VLA_PATH="/data/wulin/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/data/wulin/c/specvla-data/dflash_goal_dataset.h5"
  [[ -f "${DEFAULT_DATAPATH}" ]] || DEFAULT_DATAPATH="/data/wulin/c/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/data/wulin/c/specvla-data/${OUTPUT_NAME}"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  DEFAULT_ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
  DEFAULT_VLA_PATH="${DEFAULT_ROOT}/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="${DEFAULT_ROOT}/specvla-data/dflash_goal_dataset.h5"
  [[ -f "${DEFAULT_DATAPATH}" ]] || DEFAULT_DATAPATH="${DEFAULT_ROOT}/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="${DEFAULT_ROOT}/specvla-data/${OUTPUT_NAME}"
elif [[ -d "/mnt/storage/cgh" ]]; then
  DEFAULT_VLA_PATH="/mnt/storage/cgh/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/mnt/storage/cgh/specvla-data/dflash_goal_dataset.h5"
  [[ -f "${DEFAULT_DATAPATH}" ]] || DEFAULT_DATAPATH="/mnt/storage/cgh/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/mnt/storage/cgh/specvla-data/${OUTPUT_NAME}"
else
  DEFAULT_VLA_PATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal"
  DEFAULT_DATAPATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset.h5"
  [[ -f "${DEFAULT_DATAPATH}" ]] || DEFAULT_DATAPATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset"
  DEFAULT_OUTPUT_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/${OUTPUT_NAME}"
fi

VLA_PATH="${VLA_PATH:-${DEFAULT_VLA_PATH}}"
DATAPATH="${DATAPATH:-${DEFAULT_DATAPATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

CAUSAL_RESIDUAL_START_INDEX="${CAUSAL_RESIDUAL_START_INDEX:-0}"
RESIDUAL_CAD_TYPE="${RESIDUAL_CAD_TYPE:-cosine}"
RESIDUAL_CAD_WARMUP_STEPS="${RESIDUAL_CAD_WARMUP_STEPS:-4000}"
REFINED_HIDDEN_TYPE="${REFINED_HIDDEN_TYPE:-smooth_l1}"
REFINED_HIDDEN_MIN_POSITION="${REFINED_HIDDEN_MIN_POSITION:-1}"
REFINED_HIDDEN_MAX_POSITION="${REFINED_HIDDEN_MAX_POSITION:-5}"
RESIDUAL_TOKEN_CE_MIN_POSITION="${RESIDUAL_TOKEN_CE_MIN_POSITION:-1}"
RESIDUAL_TOKEN_CE_MAX_POSITION="${RESIDUAL_TOKEN_CE_MAX_POSITION:-5}"
LOGIT_MARKOV_RANK="${LOGIT_MARKOV_RANK:-256}"
LOGIT_MARKOV_SCALE="${LOGIT_MARKOV_SCALE:-1.0}"
ANCHOR_LOGIT_DISTILL_TEMPERATURE="${ANCHOR_LOGIT_DISTILL_TEMPERATURE:-2.0}"

cat <<EOF
========== DFlash Goal 消融训练配置 ==========
ABLATION_STAGE=${ABLATION_STAGE}
RUN_NAME=${RUN_NAME}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
NPROC_PER_NODE=${NPROC_PER_NODE}
VLA_PATH=${VLA_PATH}
DATAPATH=${DATAPATH}
DATASET_FORMAT=${DATASET_FORMAT}
OUTPUT_DIR=${OUTPUT_DIR}

[共同训练设置]
BATCH_SIZE=${BATCH_SIZE}
LR=${LR}
WARMUP_STEPS=${WARMUP_STEPS}
NUM_EPOCHS=${NUM_EPOCHS}
SAVE_EVERY=${SAVE_EVERY}
SEED=${SEED}
HIDDEN_NOISE=${HIDDEN_NOISE}
SLOT_DECAY=${SLOT_DECAY}
HIDDEN_W=${HIDDEN_W}
COS_W=${COS_W}
SOFT_W=${SOFT_W}

[结构 / CAD / 修正头]
CAUSAL_RESIDUAL_TYPE=${CAUSAL_RESIDUAL_TYPE}
CAUSAL_RESIDUAL_START_INDEX=${CAUSAL_RESIDUAL_START_INDEX}
RESIDUAL_CAD_W=${RESIDUAL_CAD_W}
REFINED_HIDDEN_W=${REFINED_HIDDEN_W}
RESIDUAL_TOKEN_CE_W=${RESIDUAL_TOKEN_CE_W}
LOGIT_MARKOV_TYPE=${LOGIT_MARKOV_TYPE}
ANCHOR_LOGIT_DISTILL_W=${ANCHOR_LOGIT_DISTILL_W}
=============================================
EOF

bool_arg() {
  local name="$1"
  local value="$2"
  if [[ "${value}" == "True" || "${value}" == "true" || "${value}" == "1" ]]; then
    printf -- "--%s" "${name}"
  else
    printf -- "--no-%s" "${name}"
  fi
}
SAVE_TRAINING_STATE_ARG="$(bool_arg save_training_state "${SAVE_TRAINING_STATE}")"
SAVE_LATEST_ROOT_COPY_ARG="$(bool_arg save_latest_root_copy "${SAVE_LATEST_ROOT_COPY}")"

export CUDA_VISIBLE_DEVICES
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

torchrun --standalone --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" \
  openvla/specdecoding/train-scripts/train_dflash_libero_goal.py \
  --run_name "${RUN_NAME}" \
  --vla_path "${VLA_PATH}" \
  --datapath "${DATAPATH}" \
  --dataset_format "${DATASET_FORMAT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_draft_layers 1 \
  --block_size 7 \
  --target_layer_ids 1 8 15 22 29 \
  --selected_hidden_variant replace_22_with_final \
  --include_anchor_hidden \
  --hidden_w "${HIDDEN_W}" \
  --cos_w "${COS_W}" \
  --soft_w "${SOFT_W}" \
  --slot_decay "${SLOT_DECAY}" \
  --hidden_noise "${HIDDEN_NOISE}" \
  --anchor_consistency_w 0 \
  --causal_residual_type "${CAUSAL_RESIDUAL_TYPE}" \
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
  --weight_decay 0.05 \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps 1 \
  --num_epochs "${NUM_EPOCHS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --seed "${SEED}" \
  "${SAVE_TRAINING_STATE_ARG}" \
  "${SAVE_LATEST_ROOT_COPY_ARG}" \
  --num_workers "${NUM_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  --no-pin_memory \
  --no-persistent_workers \
  --swanlab_log_every_steps "${SWANLAB_LOG_EVERY_STEPS}" \
  --swanlab_detail_every_steps "${SWANLAB_DETAIL_EVERY_STEPS}" \
  --val_split 0
