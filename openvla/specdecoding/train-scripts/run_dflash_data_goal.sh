#!/usr/bin/env bash
set -euo pipefail

# LIBERO-Goal DFlash 离线数据统一入口。
#
#   bash .../run_dflash_data_goal.sh smoke   # 默认 32 条，先验证环境和格式
#   bash .../run_dflash_data_goal.sh full    # 正式生成完整单文件 HDF5
#
# 常用覆盖：GPU_ID、VLA_PATH、RLDS_ROOT、OUT_FILE、SEED、MAX_SAMPLES。

MODE="${1:-smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -d "/data/wulin" ]]; then
  DEFAULT_VLA_PATH="/data/wulin/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_RLDS_ROOT="/data/wulin/c/datasets/modified_libero_rlds"
  DEFAULT_DATA_ROOT="/data/wulin/c/specvla-data"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  MACHINE_ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
  DEFAULT_VLA_PATH="${MACHINE_ROOT}/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_RLDS_ROOT="${MACHINE_ROOT}/datasets/modified_libero_rlds"
  DEFAULT_DATA_ROOT="${MACHINE_ROOT}/specvla-data"
elif [[ -d "/mnt/storage/cgh" ]]; then
  DEFAULT_VLA_PATH="/mnt/storage/cgh/hf_files/openvla-7b-finetuned-libero-goal"
  DEFAULT_RLDS_ROOT="/mnt/storage/cgh/datasets/modified_libero_rlds"
  DEFAULT_DATA_ROOT="/mnt/storage/cgh/specvla-data"
else
  echo "无法识别机器路径；请显式设置 VLA_PATH、RLDS_ROOT 和 OUT_FILE。" >&2
  exit 1
fi

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
SEED="${SEED:-7}"
VLA_PATH="${VLA_PATH:-${DEFAULT_VLA_PATH}}"
RLDS_ROOT="${RLDS_ROOT:-${DEFAULT_RLDS_ROOT}}"
DATASET_NAME="${DATASET_NAME:-libero_goal_no_noops}"

case "${MODE}" in
  smoke)
    OUT_FILE="${OUT_FILE:-${DEFAULT_DATA_ROOT}/dflash_goal_smoke.h5}"
    MAX_SAMPLES="${MAX_SAMPLES:-32}"
    ;;
  full)
    OUT_FILE="${OUT_FILE:-${DEFAULT_DATA_ROOT}/dflash_goal_dataset_envfix_20260714.h5}"
    MAX_SAMPLES="${MAX_SAMPLES:-}"
    ;;
  *)
    echo "用法: bash $0 [smoke|full]" >&2
    exit 1
    ;;
esac

for path in "${VLA_PATH}" "${RLDS_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "所需路径不存在: ${path}" >&2
    exit 1
  fi
done
mkdir -p "$(dirname "${OUT_FILE}")"

echo "========== DFlash Goal 数据生成 =========="
echo "MODE=${MODE}"
echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "VLA_PATH=${VLA_PATH}"
echo "RLDS_ROOT=${RLDS_ROOT}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "OUT_FILE=${OUT_FILE}"
echo "SEED=${SEED}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-all}"
echo "OUTPUT_FORMAT=hdf5"
echo "=========================================="

command=(
  python openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py
  --vla_path "${VLA_PATH}"
  --data_root_dir "${RLDS_ROOT}"
  --dataset_name "${DATASET_NAME}"
  --outdir "${OUT_FILE}"
  --output_format hdf5
  --seed "${SEED}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  command+=(--max_samples "${MAX_SAMPLES}")
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
if [[ "${DRY_RUN:-False}" == "True" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi
"${command[@]}"
