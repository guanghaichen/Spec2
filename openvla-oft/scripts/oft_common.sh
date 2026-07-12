#!/usr/bin/env bash

# Shared environment bootstrap for the vendored OpenVLA-OFT experiment.
# This file is sourced by all OFT scripts; do not run it as a standalone job.

set -euo pipefail

OFT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPECVLA_ROOT="$(cd "${OFT_ROOT}/.." && pwd)"

if [[ -d "/data/wulin" ]]; then
  WORK_ROOT="/data/wulin/c"
  DATA_ROOT="/data/wulin/c/specvla-data"
  HF_ROOT="/data/wulin/hf_files"
  LIBERO_PATH="/data/wulin/c/LIBERO"
  CONDA_SH="/data/wulin/miniconda3/etc/profile.d/conda.sh"
  EGL_ROOT="/data/wulin/c/nvidia-egl-570.133.07"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  WORK_ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
  DATA_ROOT="${WORK_ROOT}/specvla-data"
  HF_ROOT="${WORK_ROOT}/hf_files"
  LIBERO_PATH="${WORK_ROOT}/LIBERO"
  CONDA_SH="/home/asus/miniconda3/etc/profile.d/conda.sh"
  EGL_ROOT=""
else
  echo "Unsupported host layout. Set OFT_ROOT/WORK_ROOT manually." >&2
  return 1
fi

OFT_ENV_NAME="${OFT_ENV_NAME:-oft}"
OFT_TRANSFORMERS_DIR="${OFT_TRANSFORMERS_DIR:-${WORK_ROOT}/openvla-oft-third_party/transformers-openvla-oft}"
OFT_GOAL_MODEL="${OFT_GOAL_MODEL:-${HF_ROOT}/openvla-7b-oft-finetuned-libero-goal}"
OFT_RUN_ROOT="${OFT_RUN_ROOT:-${DATA_ROOT}/oft_runs}"
export OFT_ROOT SPECVLA_ROOT WORK_ROOT DATA_ROOT HF_ROOT LIBERO_PATH OFT_TRANSFORMERS_DIR OFT_GOAL_MODEL OFT_RUN_ROOT

source "${CONDA_SH}"
conda activate "${OFT_ENV_NAME}"

export PYTHONPATH="${OFT_ROOT}:${LIBERO_PATH}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HF_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_ROOT}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_ROOT}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ -d "${EGL_ROOT}/slim-lib" ]]; then
  export LD_LIBRARY_PATH="${EGL_ROOT}/slim-lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -f "${EGL_ROOT}/egl_vendor.d/10_nvidia_570.json" ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${EGL_ROOT}/egl_vendor.d/10_nvidia_570.json"
fi

mkdir -p "${OFT_RUN_ROOT}"
