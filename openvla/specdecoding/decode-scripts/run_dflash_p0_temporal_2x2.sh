#!/usr/bin/env bash
set -euo pipefail

# 配对的目标模型控制实验。两个正交因子均由统一数学定义得到：
#   时间正则性：最小前缀偏差排列 vs 最大长间隔聚集排列；
#   控制权衰减：线性累计控制权 vs 临界幂律累计控制权。
# 四组条件共享目标调用预算、间隔多重集、初始状态与随机种子。

SUITE_ARG="${1:-spatial}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

case "${SUITE_ARG}" in
  goal|libero_goal) TASK_SUITE_NAME=libero_goal ;;
  spatial|libero_spatial) TASK_SUITE_NAME=libero_spatial ;;
  *) echo "当前 2x2 入口只支持 goal 或 spatial。" >&2; exit 1 ;;
esac
init_libero_eval_env "${TASK_SUITE_NAME}"

RUN_STAMP="${P0_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${P0_TEMPORAL_OUTPUT_DIR:-$(dirname "${DEFAULT_LOG_DIR}")/evidence/p0/${TASK_SUITE_SLUG}/${RUN_STAMP}/temporal_2x2}"
CURATED_DIR="${P0_TEMPORAL_CURATED_DIR:-${REPO_ROOT}/artifacts/evidence/p0/${TASK_SUITE_SLUG}/${RUN_STAMP}/temporal_2x2}"

echo "suite=${TASK_SUITE_NAME} target=${VLA_PATH} paired_trials=${P0_TRIALS:-10}"
python openvla/specdecoding/evidence/run_temporal_schedule_p0.py \
  --pretrained_checkpoint "${VLA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --task_id "${P0_TASK_ID:-0}" \
  --trial_start_index "${TRIAL_START_INDEX:-0}" \
  --num_trials "${P0_TRIALS:-10}" \
  --seed "${SEED:-7}" \
  --center_crop True \
  --run_id_note "p0-${TASK_SUITE_SLUG}-${RUN_STAMP}"

RECORDS="${OUTPUT_DIR}/temporal-2x2-p0-${TASK_SUITE_SLUG}-${RUN_STAMP}.jsonl"
MANIFEST="${OUTPUT_DIR}/temporal-2x2-p0-${TASK_SUITE_SLUG}-${RUN_STAMP}-manifest.json"
python openvla/specdecoding/evidence/build_temporal_schedule_evidence.py \
  --records "${RECORDS}" \
  --run-manifest "${MANIFEST}" \
  --output "${CURATED_DIR}" \
  --repo-root "${REPO_ROOT}"

echo "Temporal 2x2 evidence complete: ${CURATED_DIR}"
