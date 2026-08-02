#!/usr/bin/env bash
set -euo pipefail

# 使用完全相同的候选族和统计准则依次校准多个 LIBERO 子集，并生成跨子集恢复地形。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

IFS=',' read -r -a SUITES <<< "${CALIBRATION_SUITES:-libero_goal,libero_object,libero_spatial,libero_10}"
if [[ "${#SUITES[@]}" -eq 0 ]]; then
  echo "CALIBRATION_SUITES 不能为空。" >&2
  exit 1
fi

# 只借助 common 脚本确定机器数据根目录；每个子进程仍会按自己的 suite 选择目标权重。
init_libero_eval_env "${SUITES[0]}"
RUN_STAMP="${CALIBRATION_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$(dirname "${DEFAULT_LOG_DIR}")/calibration}"
PROFILE_PATHS=()

for SUITE in "${SUITES[@]}"; do
  SLUG="$(libero_suite_slug "${SUITE}")"
  OUTPUT_ROOT="${CALIBRATION_ROOT}/${SLUG}/${RUN_STAMP}"
  echo "[$SUITE] paired calibration -> ${OUTPUT_ROOT}"
  TASK_SUITE_NAME="${SUITE}" \
  VLA_PATH_OVERRIDE="" \
  CALIBRATION_RUN_STAMP="${RUN_STAMP}" \
  CALIBRATION_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    bash "${SCRIPT_DIR}/run_recoverability_calibration.sh"

  mapfile -t MATCHED_PROFILES < <(
    find "${OUTPUT_ROOT}/profile" -maxdepth 1 -type f \
      -name "${SUITE}-*.json" | sort
  )
  if [[ "${#MATCHED_PROFILES[@]}" -ne 1 ]]; then
    echo "${SUITE} 应产生且只产生一个 profile，实际为 ${#MATCHED_PROFILES[@]}。" >&2
    exit 1
  fi
  PROFILE_PATHS+=("${MATCHED_PROFILES[0]}")
done

LANDSCAPE_DIR="${CALIBRATION_ROOT}/cross_suite/${RUN_STAMP}"
python "${REPO_ROOT}/openvla/specdecoding/evidence/build_recoverability_landscape.py" \
  --profiles "${PROFILE_PATHS[@]}" \
  --output_dir "${LANDSCAPE_DIR}"

echo "All-suite calibration complete."
echo "RUN_STAMP=${RUN_STAMP}"
echo "LANDSCAPE_DIR=${LANDSCAPE_DIR}"
printf 'PROFILE=%s\n' "${PROFILE_PATHS[@]}"
