#!/usr/bin/env bash
set -euo pipefail

# LIBERO DFlash 离线数据统一入口。文件名保留 goal 以兼容旧命令，但可用
# TASK_SUITE_NAME 选择 goal/object/spatial/10，并自动绑定同 suite 的模型、
# RLDS 数据与输出文件。
#
#   bash .../run_dflash_data_goal.sh smoke   # 默认 32 条，先验证环境和格式
#   bash .../run_dflash_data_goal.sh full    # 生成原始 HDF5，再无损打包为训练用 packed v2
#
# 常用覆盖：TASK_SUITE_NAME、GPU_ID、VLA_PATH、RLDS_ROOT、RAW_OUT_FILE、
# OUT_FILE、SEED、MAX_SAMPLES。
# OUT_FILE 始终表示最终训练文件；默认在打包成功后删除中间 v1 文件，最终只保留
# 一个 packed v2。调试数据格式时可显式设置 KEEP_RAW=True 保留中间文件。

MODE="${1:-smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -d "/data/wulin" ]]; then
  DEFAULT_HF_ROOT="/data/wulin/hf_files"
  DEFAULT_RLDS_ROOT="/data/wulin/c/datasets/modified_libero_rlds"
  DEFAULT_DATA_ROOT="/data/wulin/c/specvla-data"
elif [[ -d "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh" ]]; then
  MACHINE_ROOT="/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh"
  DEFAULT_HF_ROOT="${MACHINE_ROOT}/hf_files"
  DEFAULT_RLDS_ROOT="${MACHINE_ROOT}/datasets/modified_libero_rlds"
  DEFAULT_DATA_ROOT="${MACHINE_ROOT}/specvla-data"
elif [[ -d "/mnt/storage/cgh" ]]; then
  DEFAULT_HF_ROOT="/mnt/storage/cgh/hf_files"
  DEFAULT_RLDS_ROOT="/mnt/storage/cgh/datasets/modified_libero_rlds"
  DEFAULT_DATA_ROOT="/mnt/storage/cgh/specvla-data"
else
  echo "无法识别机器路径；请显式设置 VLA_PATH、RLDS_ROOT 和 OUT_FILE。" >&2
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

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
SEED="${SEED:-7}"
DEFAULT_VLA_PATH="${DEFAULT_HF_ROOT}/openvla-7b-finetuned-libero-${SUITE_SLUG}"
VLA_PATH="${VLA_PATH:-${DEFAULT_VLA_PATH}}"
RLDS_ROOT="${RLDS_ROOT:-${DEFAULT_RLDS_ROOT}}"
DATASET_NAME="${DATASET_NAME:-${TASK_SUITE_NAME}_no_noops}"
OUTPUT_STEM="dflash_${SUITE_SLUG}_dataset"

case "${MODE}" in
  smoke)
    RAW_OUT_FILE="${RAW_OUT_FILE:-${DEFAULT_DATA_ROOT}/.dflash_work/${OUTPUT_STEM}_smoke_raw_v1.h5}"
    OUT_FILE="${OUT_FILE:-${DEFAULT_DATA_ROOT}/${OUTPUT_STEM}_smoke_packed_v2.h5}"
    MAX_SAMPLES="${MAX_SAMPLES:-32}"
    ;;
  full)
    if [[ "${TASK_SUITE_NAME}" == "libero_goal" ]]; then
      RAW_OUT_FILE="${RAW_OUT_FILE:-${DEFAULT_DATA_ROOT}/.dflash_work/dflash_goal_dataset_envfix_20260714_raw_v1.h5}"
      OUT_FILE="${OUT_FILE:-${DEFAULT_DATA_ROOT}/dflash_goal_dataset_envfix_20260714_packed_v2.h5}"
    else
      RAW_OUT_FILE="${RAW_OUT_FILE:-${DEFAULT_DATA_ROOT}/.dflash_work/${OUTPUT_STEM}_raw_v1.h5}"
      OUT_FILE="${OUT_FILE:-${DEFAULT_DATA_ROOT}/${OUTPUT_STEM}_packed_v2.h5}"
    fi
    MAX_SAMPLES="${MAX_SAMPLES:-}"
    ;;
  *)
    echo "用法: bash $0 [smoke|full]" >&2
    exit 1
    ;;
esac

for path in "${VLA_PATH}" "${RLDS_ROOT}" "${RLDS_ROOT}/${DATASET_NAME}"; do
  if [[ ! -e "${path}" ]]; then
    echo "所需路径不存在: ${path}" >&2
    exit 1
  fi
done
mkdir -p "$(dirname "${RAW_OUT_FILE}")" "$(dirname "${OUT_FILE}")"
KEEP_RAW="${KEEP_RAW:-False}"
KEEP_FAILED_OUTPUTS="${KEEP_FAILED_OUTPUTS:-False}"
PACK_COPY_BATCH_SIZE="${PACK_COPY_BATCH_SIZE:-16}"
SERIALIZE_PACKING="${SERIALIZE_PACKING:-True}"
PACK_LOCK_FILE="${PACK_LOCK_FILE:-${DEFAULT_DATA_ROOT}/.dflash_hdf5_pack.lock}"

# A failed generator can leave a valid HDF5 header with zero samples. Such a
# file is not resumable and used to look like a real dataset in the data root.
RAW_OUT_EXISTED_BEFORE=False
OUT_EXISTED_BEFORE=False
[[ -e "${RAW_OUT_FILE}" ]] && RAW_OUT_EXISTED_BEFORE=True
[[ -e "${OUT_FILE}" ]] && OUT_EXISTED_BEFORE=True
cleanup_failed_outputs() {
  local exit_code=$?
  trap - EXIT
  if [[ "${exit_code}" -ne 0 && "${KEEP_FAILED_OUTPUTS}" != "True" ]]; then
    if [[ "${RAW_OUT_EXISTED_BEFORE}" == "False" ]]; then
      rm -f "${RAW_OUT_FILE}"
    fi
    if [[ "${OUT_EXISTED_BEFORE}" == "False" ]]; then
      rm -f "${OUT_FILE}"
    fi
    echo "数据生成失败；已清理本次产生的不完整文件。" >&2
  fi
  exit "${exit_code}"
}
trap cleanup_failed_outputs EXIT

echo "========== DFlash ${TASK_SUITE_NAME} 数据生成 =========="
echo "MODE=${MODE}"
echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "VLA_PATH=${VLA_PATH}"
echo "RLDS_ROOT=${RLDS_ROOT}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "RAW_OUT_FILE=${RAW_OUT_FILE}"
echo "OUT_FILE=${OUT_FILE} (packed v2，训练读取此文件)"
echo "KEEP_RAW=${KEEP_RAW}"
echo "KEEP_FAILED_OUTPUTS=${KEEP_FAILED_OUTPUTS}"
echo "SERIALIZE_PACKING=${SERIALIZE_PACKING}"
echo "SEED=${SEED}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-all}"
echo "OUTPUT_FORMAT=hdf5_packed_v2"
echo "=========================================="

command=(
  python openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py
  --vla_path "${VLA_PATH}"
  --data_root_dir "${RLDS_ROOT}"
  --dataset_name "${DATASET_NAME}"
  --task_suite_name "${TASK_SUITE_NAME}"
  --outdir "${RAW_OUT_FILE}"
  --output_format hdf5
  --seed "${SEED}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  command+=(--max_samples "${MAX_SAMPLES}")
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
if [[ "${DRY_RUN:-False}" == "True" ]]; then
  printf 'DRY_RUN generate:'
  printf ' %q' "${command[@]}"
  printf '\n'
  printf 'DRY_RUN pack:'
  printf ' %q' python openvla/specdecoding/train-scripts/pack_dflash_hdf5.py \
    --input "${RAW_OUT_FILE}" --output "${OUT_FILE}" --copy_batch_size "${PACK_COPY_BATCH_SIZE}"
  printf '\n'
  exit 0
fi
"${command[@]}"
pack_command=(
  python openvla/specdecoding/train-scripts/pack_dflash_hdf5.py
  --input "${RAW_OUT_FILE}"
  --output "${OUT_FILE}"
  --copy_batch_size "${PACK_COPY_BATCH_SIZE}"
)
if [[ "${SERIALIZE_PACKING}" == "True" ]] && command -v flock >/dev/null 2>&1; then
  echo "等待 HDF5 打包锁: ${PACK_LOCK_FILE}"
  flock "${PACK_LOCK_FILE}" "${pack_command[@]}"
else
  "${pack_command[@]}"
fi

if [[ "${KEEP_RAW}" == "False" ]]; then
  rm -f "${RAW_OUT_FILE}"
  echo "已在 packed v2 完成后删除中间文件: ${RAW_OUT_FILE}"
fi
