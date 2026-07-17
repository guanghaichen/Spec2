#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

libero_suite_slug() {
  case "$1" in
    libero_goal) echo "goal" ;;
    libero_object) echo "object" ;;
    libero_spatial) echo "spatial" ;;
    libero_10) echo "10" ;;
    libero_90) echo "90" ;;
    *)
      echo "Unsupported LIBERO task suite: $1" >&2
      echo "Expected one of: libero_goal, libero_object, libero_spatial, libero_10, libero_90." >&2
      return 1
      ;;
  esac
}

init_libero_eval_env() {
  TASK_SUITE_NAME="${1:-${TASK_SUITE_NAME:-libero_goal}}"
  TASK_SUITE_SLUG="$(libero_suite_slug "${TASK_SUITE_NAME}")"

  cd "${REPO_ROOT}"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"

  if [[ -d "/data/wulin" ]]; then
    DEFAULT_VLA_PATH="/data/wulin/hf_files/openvla-7b-finetuned-libero-${TASK_SUITE_SLUG}"
    DEFAULT_DFLASH_OUTPUT_DIR="/data/wulin/c/specvla-data/ckpt_goal_dflash_two_stage_1layer_b16x1_4gpu/stage2_refinement"
    DEFAULT_SPECVLA_CKPT="/data/wulin/c/specvla-data/specvla_checkpoint/${TASK_SUITE_SLUG}"
    DEFAULT_LOG_DIR="/data/wulin/c/specvla-data/eval_logs"
    DEFAULT_LIBERO_PATH="/data/wulin/c/LIBERO"
    DEFAULT_NVIDIA_EGL_SHIM_DIR="/data/wulin/c/nvidia-egl-570.133.07/slim-lib"
    DEFAULT_NVIDIA_EGL_VENDOR_JSON="/data/wulin/c/nvidia-egl-570.133.07/egl_vendor.d/10_nvidia_570.json"
  elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
    DEFAULT_ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
    DEFAULT_VLA_PATH="${DEFAULT_ROOT}/hf_files/openvla-7b-finetuned-libero-${TASK_SUITE_SLUG}"
    DEFAULT_DFLASH_OUTPUT_DIR="${DEFAULT_ROOT}/specvla-data/ckpt_goal_dflash_two_stage_1layer_b16x1_4gpu/stage2_refinement"
    DEFAULT_SPECVLA_CKPT="${DEFAULT_ROOT}/specvla-data/specvla_checkpoint/${TASK_SUITE_SLUG}"
    DEFAULT_LOG_DIR="${DEFAULT_ROOT}/specvla-data/eval_logs"
    DEFAULT_LIBERO_PATH="${DEFAULT_ROOT}/LIBERO"
    DEFAULT_NVIDIA_EGL_SHIM_DIR=""
    DEFAULT_NVIDIA_EGL_VENDOR_JSON=""
  elif [[ -d "/mnt/storage/cgh" ]]; then
    DEFAULT_VLA_PATH="/mnt/storage/cgh/hf_files/openvla-7b-finetuned-libero-${TASK_SUITE_SLUG}"
    DEFAULT_DFLASH_OUTPUT_DIR="/mnt/storage/cgh/specvla-data/ckpt_goal_dflash_two_stage_1layer_b16x1_4gpu/stage2_refinement"
    DEFAULT_SPECVLA_CKPT="/mnt/storage/cgh/specvla-data/specvla_checkpoint/${TASK_SUITE_SLUG}"
    DEFAULT_LOG_DIR="/mnt/storage/cgh/specvla-data/eval_logs"
    DEFAULT_LIBERO_PATH="/mnt/storage/cgh/LIBERO"
    DEFAULT_NVIDIA_EGL_SHIM_DIR=""
    DEFAULT_NVIDIA_EGL_VENDOR_JSON=""
  else
    DEFAULT_VLA_PATH="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-${TASK_SUITE_SLUG}"
    DEFAULT_DFLASH_OUTPUT_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_two_stage_1layer_b16x1_4gpu/stage2_refinement"
    DEFAULT_SPECVLA_CKPT="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_libero_${TASK_SUITE_SLUG}_debug_ckpt"
    DEFAULT_LOG_DIR="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/eval_logs"
    DEFAULT_LIBERO_PATH=""
    DEFAULT_NVIDIA_EGL_SHIM_DIR=""
    DEFAULT_NVIDIA_EGL_VENDOR_JSON=""
  fi

  # Do not let a global VLA_PATH from ~/.bashrc leak across suites. The default
  # must follow TASK_SUITE_NAME; use VLA_PATH_OVERRIDE for an intentional override.
  VLA_PATH="${VLA_PATH_OVERRIDE:-${DEFAULT_VLA_PATH}}"
  DFLASH_OUTPUT_DIR="${DFLASH_OUTPUT_DIR:-${DEFAULT_DFLASH_OUTPUT_DIR}}"
  SPECVLA_CKPT="${SPECVLA_CKPT:-}"
  SPECVLA_GOAL_CKPT="${SPECVLA_GOAL_CKPT:-}"
  SPECVLA_CKPT_ROOT="${SPECVLA_CKPT_ROOT:-}"
  LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"
  LIBERO_PATH="${LIBERO_PATH:-${DEFAULT_LIBERO_PATH}}"
  NVIDIA_EGL_SHIM_DIR="${NVIDIA_EGL_SHIM_DIR:-${DEFAULT_NVIDIA_EGL_SHIM_DIR}}"
  NVIDIA_EGL_VENDOR_JSON="${NVIDIA_EGL_VENDOR_JSON:-${DEFAULT_NVIDIA_EGL_VENDOR_JSON}}"
  NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
  USE_WANDB="${USE_WANDB:-False}"
  SEED="${SEED:-7}"
  SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
  TIMING_SCOPE="${TIMING_SCOPE:-last_task}"

  if [[ ! -d "${VLA_PATH}" ]]; then
    echo "VLA_PATH does not exist or is not a directory: ${VLA_PATH}" >&2
    echo "Default model path is selected from TASK_SUITE_NAME=${TASK_SUITE_NAME}." >&2
    echo "For an intentional override, set VLA_PATH_OVERRIDE=/path/to/openvla-suite-model." >&2
    exit 1
  fi

  mkdir -p "${LOG_DIR}"

  export PYTHONPATH="${REPO_ROOT}/openvla:${PYTHONPATH:-}"
  if [[ -n "${LIBERO_PATH}" && -d "${LIBERO_PATH}" ]]; then
    export PYTHONPATH="${LIBERO_PATH}:${PYTHONPATH}"
  fi

  if [[ -n "${NVIDIA_EGL_SHIM_DIR}" && -d "${NVIDIA_EGL_SHIM_DIR}" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_EGL_SHIM_DIR}:${LD_LIBRARY_PATH:-}"
  fi
  if [[ -n "${NVIDIA_EGL_VENDOR_JSON}" && -f "${NVIDIA_EGL_VENDOR_JSON}" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_JSON}"
  elif [[ -d "/data/wulin" ]]; then
    echo "WARNING: NVIDIA EGL shim was not found." >&2
    echo "If LIBERO fails with an EGL device-display error, run:" >&2
    echo "  bash openvla/specdecoding/decode-scripts/setup_3090_nvidia_egl_shim.sh" >&2
  fi
}

init_libero_goal_eval_env() {
  init_libero_eval_env libero_goal
  DEFAULT_SPECVLA_GOAL_CKPT="${DEFAULT_SPECVLA_CKPT}"
}

resolve_specvla_checkpoint() {
  # SpecVLA paper baselines deliberately ignore the generic SPEC_CKPT variable:
  # that variable belongs to DFlash launchers and may still point at a trained
  # DFlash checkpoint from the previous command in the same shell.
  if [[ -n "${SPECVLA_CKPT}" ]]; then
    SPEC_CKPT="${SPECVLA_CKPT}"
  elif [[ -n "${SPECVLA_GOAL_CKPT}" && "${TASK_SUITE_NAME:-libero_goal}" == "libero_goal" ]]; then
    SPEC_CKPT="${SPECVLA_GOAL_CKPT}"
  elif [[ -n "${SPECVLA_CKPT_ROOT}" ]]; then
    SPEC_CKPT="${SPECVLA_CKPT_ROOT}/${TASK_SUITE_SLUG}"
  else
    SPEC_CKPT="${DEFAULT_SPECVLA_CKPT}"
  fi

  if [[ ! -d "${SPEC_CKPT}" ]]; then
    echo "SpecVLA SPEC_CKPT does not exist or is not a directory: ${SPEC_CKPT}" >&2
    echo "Set SPEC_CKPT=/path/to/${TASK_SUITE_SLUG}_ckpt or SPECVLA_CKPT_ROOT=/path/to/specvla_checkpoint_root." >&2
    exit 1
  fi
}

resolve_specvla_goal_checkpoint() {
  # See resolve_specvla_checkpoint: never inherit a DFlash SPEC_CKPT here.
  if [[ -n "${SPECVLA_GOAL_CKPT}" ]]; then
    SPEC_CKPT="${SPECVLA_GOAL_CKPT}"
  elif [[ -n "${SPECVLA_CKPT}" ]]; then
    SPEC_CKPT="${SPECVLA_CKPT}"
  elif [[ -n "${SPECVLA_CKPT_ROOT}" ]]; then
    SPEC_CKPT="${SPECVLA_CKPT_ROOT}/goal"
  else
    SPEC_CKPT="${DEFAULT_SPECVLA_GOAL_CKPT:-${DEFAULT_SPECVLA_CKPT}}"
  fi

  if [[ ! -d "${SPEC_CKPT}" ]]; then
    echo "SpecVLA SPEC_CKPT does not exist or is not a directory: ${SPEC_CKPT}" >&2
    echo "Set SPEC_CKPT=/path/to/goal_ckpt or SPECVLA_GOAL_CKPT=/path/to/goal_ckpt." >&2
    exit 1
  fi
}

resolve_dflash_checkpoint() {
  EVAL_EPOCH="${EVAL_EPOCH:-latest}"

  if [[ -z "${SPEC_CKPT:-}" ]]; then
    if [[ "${EVAL_EPOCH}" == "latest" ]]; then
      LATEST_FILE="${DFLASH_OUTPUT_DIR}/latest_checkpoint.txt"
      if [[ ! -f "${LATEST_FILE}" ]]; then
        echo "Missing latest checkpoint file: ${LATEST_FILE}" >&2
        echo "Set DFLASH_OUTPUT_DIR, EVAL_EPOCH, or SPEC_CKPT directly." >&2
        exit 1
      fi
      SPEC_CKPT="$(cat "${LATEST_FILE}")"
    elif [[ "${EVAL_EPOCH}" == epoch_* ]]; then
      SPEC_CKPT="${DFLASH_OUTPUT_DIR}/${EVAL_EPOCH}"
    else
      if [[ ! "${EVAL_EPOCH}" =~ ^[0-9]+$ ]]; then
        echo "EVAL_EPOCH must be an integer epoch number, 'latest', or a checkpoint directory name like epoch_180_step_160740." >&2
        exit 1
      fi
      EPOCH_TAG="$(printf "%03d" "${EVAL_EPOCH}")"
      mapfile -t MATCHED_CKPTS < <(find "${DFLASH_OUTPUT_DIR}" -maxdepth 1 -type d -name "epoch_${EPOCH_TAG}_step_*" | sort -V)
      if [[ "${#MATCHED_CKPTS[@]}" -ne 1 ]]; then
        echo "Expected exactly one checkpoint matching ${DFLASH_OUTPUT_DIR}/epoch_${EPOCH_TAG}_step_*, found ${#MATCHED_CKPTS[@]}." >&2
        printf '%s\n' "${MATCHED_CKPTS[@]}" >&2
        exit 1
      fi
      SPEC_CKPT="${MATCHED_CKPTS[0]}"
    fi
  else
    :
  fi

  if [[ ! -d "${SPEC_CKPT}" ]]; then
    echo "DFLASH SPEC_CKPT does not exist or is not a directory: ${SPEC_CKPT}" >&2
    exit 1
  fi
}

resolve_dflash_goal_checkpoint() {
  resolve_dflash_checkpoint
}

print_common_eval_config() {
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "MUJOCO_GL=${MUJOCO_GL}"
  echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-}"
  echo "NVIDIA_EGL_SHIM_DIR=${NVIDIA_EGL_SHIM_DIR}"
  echo "__EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-}"
  echo "TASK_SUITE_NAME=${TASK_SUITE_NAME:-}"
  echo "VLA_PATH=${VLA_PATH}"
  echo "VLA_PATH_OVERRIDE=${VLA_PATH_OVERRIDE:-}"
  echo "SPEC_CKPT=${SPEC_CKPT:-}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
  echo "RUN_ID_NOTE=${RUN_ID_NOTE:-}"
  echo "USE_WANDB=${USE_WANDB}"
  echo "SEED=${SEED}"
  echo "SYNC_CUDA_TIMING=${SYNC_CUDA_TIMING}"
  echo "TIMING_SCOPE=${TIMING_SCOPE}"
}
