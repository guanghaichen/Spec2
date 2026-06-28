#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ -d "/data/wulin" ]]; then
  DEFAULT_VLA_PATH="/data/wulin/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_OUTPUT_DIR="/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_puretrain_4gpu"
  DEFAULT_LOG_DIR="/data/wulin/c/specvla-data/eval_logs"
  DEFAULT_LIBERO_PATH="/data/wulin/c/LIBERO"
  DEFAULT_NVIDIA_EGL_SHIM_DIR="/data/wulin/c/nvidia-egl-570.133.07/slim-lib"
  DEFAULT_NVIDIA_EGL_VENDOR_JSON="/data/wulin/c/nvidia-egl-570.133.07/egl_vendor.d/10_nvidia_570.json"
else
  DEFAULT_VLA_PATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal"
  DEFAULT_OUTPUT_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_puretrain_4gpu"
  DEFAULT_LOG_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/eval_logs"
  DEFAULT_LIBERO_PATH=""
  DEFAULT_NVIDIA_EGL_SHIM_DIR=""
  DEFAULT_NVIDIA_EGL_VENDOR_JSON=""
fi

VLA_PATH="${VLA_PATH:-${DEFAULT_VLA_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"
LIBERO_PATH="${LIBERO_PATH:-${DEFAULT_LIBERO_PATH}}"
NVIDIA_EGL_SHIM_DIR="${NVIDIA_EGL_SHIM_DIR:-${DEFAULT_NVIDIA_EGL_SHIM_DIR}}"
NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-${DEFAULT_NVIDIA_EGL_VENDOR_JSON}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-9}"
EVAL_EPOCH="${EVAL_EPOCH:-latest}"
RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-finalhidden-${EVAL_EPOCH}-r${ACCEPT_THRESHOLD}}"
USE_WANDB="${USE_WANDB:-False}"

if [[ -z "${SPEC_CKPT:-}" ]]; then
  if [[ "${EVAL_EPOCH}" == "latest" ]]; then
    LATEST_FILE="${OUTPUT_DIR}/latest_checkpoint.txt"
    if [[ ! -f "${LATEST_FILE}" ]]; then
      echo "Missing latest checkpoint file: ${LATEST_FILE}" >&2
      echo "Set OUTPUT_DIR to a DFLASH run directory, set EVAL_EPOCH, or set SPEC_CKPT directly." >&2
      exit 1
    fi
    SPEC_CKPT="$(cat "${LATEST_FILE}")"
  elif [[ "${EVAL_EPOCH}" == epoch_* ]]; then
    SPEC_CKPT="${OUTPUT_DIR}/${EVAL_EPOCH}"
  else
    if [[ ! "${EVAL_EPOCH}" =~ ^[0-9]+$ ]]; then
      echo "EVAL_EPOCH must be an integer epoch number, 'latest', or a checkpoint directory name like epoch_180_step_160740." >&2
      exit 1
    fi
    EPOCH_TAG="$(printf "%03d" "${EVAL_EPOCH}")"
    mapfile -t MATCHED_CKPTS < <(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name "epoch_${EPOCH_TAG}_step_*" | sort -V)
    if [[ "${#MATCHED_CKPTS[@]}" -ne 1 ]]; then
      echo "Expected exactly one checkpoint matching ${OUTPUT_DIR}/epoch_${EPOCH_TAG}_step_*, found ${#MATCHED_CKPTS[@]}." >&2
      printf '%s\n' "${MATCHED_CKPTS[@]}" >&2
      exit 1
    fi
    SPEC_CKPT="${MATCHED_CKPTS[0]}"
  fi
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

if [[ -n "${NVIDIA_EGL_SHIM_DIR}" && -d "${NVIDIA_EGL_SHIM_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_EGL_SHIM_DIR}:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${NVIDIA_EGL_VENDOR_JSON}" && -f "${NVIDIA_EGL_VENDOR_JSON}" ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-}"
echo "NVIDIA_EGL_SHIM_DIR=${NVIDIA_EGL_SHIM_DIR}"
echo "__EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-}"
echo "VLA_PATH=${VLA_PATH}"
echo "SPEC_CKPT=${SPEC_CKPT}"
echo "LOG_DIR=${LOG_DIR}"
echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "ACCEPT_THRESHOLD=${ACCEPT_THRESHOLD}"
echo "EVAL_EPOCH=${EVAL_EPOCH}"
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
