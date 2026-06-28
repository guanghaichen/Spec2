#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

if [[ -d "/data/wulin" ]]; then
  DEFAULT_VLA_PATH="/data/wulin/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_OUTPUT_DIR="/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_puretrain_4gpu"
  DEFAULT_LOG_DIR="/data/wulin/c/specvla-data/eval_logs"
  DEFAULT_LIBERO_PATH="/data/wulin/c/LIBERO"
else
  DEFAULT_VLA_PATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal"
  DEFAULT_OUTPUT_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_puretrain_4gpu"
  DEFAULT_LOG_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/eval_logs"
  DEFAULT_LIBERO_PATH=""
fi

VLA_PATH="${VLA_PATH:-${DEFAULT_VLA_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"
LIBERO_PATH="${LIBERO_PATH:-${DEFAULT_LIBERO_PATH}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-9}"
RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-finalhidden-latest-r${ACCEPT_THRESHOLD}}"
USE_WANDB="${USE_WANDB:-False}"

if [[ -z "${SPEC_CKPT:-}" ]]; then
  LATEST_FILE="${OUTPUT_DIR}/latest_checkpoint.txt"
  if [[ ! -f "${LATEST_FILE}" ]]; then
    echo "Missing latest checkpoint file: ${LATEST_FILE}" >&2
    echo "Set OUTPUT_DIR to a DFLASH run directory or set SPEC_CKPT directly." >&2
    exit 1
  fi
  SPEC_CKPT="$(cat "${LATEST_FILE}")"
fi

if [[ ! -d "${SPEC_CKPT}" ]]; then
  echo "SPEC_CKPT does not exist or is not a directory: ${SPEC_CKPT}" >&2
  exit 1
fi

if [[ ! -d "${VLA_PATH}" ]]; then
  echo "VLA_PATH does not exist or is not a directory: ${VLA_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/openvla:${PYTHONPATH:-}"
if [[ -n "${LIBERO_PATH}" && -d "${LIBERO_PATH}" ]]; then
  export PYTHONPATH="${LIBERO_PATH}:${PYTHONPATH}"
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "VLA_PATH=${VLA_PATH}"
echo "SPEC_CKPT=${SPEC_CKPT}"
echo "LOG_DIR=${LOG_DIR}"
echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "ACCEPT_THRESHOLD=${ACCEPT_THRESHOLD}"
echo "RUN_ID_NOTE=${RUN_ID_NOTE}"

python openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py \
  --pretrained_checkpoint "${VLA_PATH}" \
  --spec_checkpoint "${SPEC_CKPT}" \
  --draft_backend dflash \
  --use_spec True \
  --parallel_draft False \
  --task_suite_name libero_goal \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --center_crop True \
  --accept_threshold "${ACCEPT_THRESHOLD}" \
  --dflash_block_size 7 \
  --dflash_num_draft_layers 1 \
  --local_log_dir "${LOG_DIR}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --use_wandb "${USE_WANDB}"
