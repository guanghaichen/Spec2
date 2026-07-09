#!/usr/bin/env bash
set -euo pipefail

# 一键评测 SpecVLA strict / relaxed 四个 LIBERO suite，并跳过已跑过的项。
# 默认 suite: Goal / Object / Spatial / Long(libero_10)
# 默认跳过条件：同 method/suite 已有 summary，或已有 txt/timing/summary 任一输出文件。
#
# 常用命令：
#   CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
#     bash openvla/specdecoding/decode-scripts/run_specvla_main_table_eval.sh
#
# 如果某个旧 run 失败但留下 txt，需要强制重跑：
#   FORCE_RERUN=True CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

TASK_SUITES="${TASK_SUITES:-libero_goal libero_object libero_spatial libero_10}"
RUN_STRICT="${RUN_STRICT:-True}"
RUN_RELAXED="${RUN_RELAXED:-True}"
FORCE_RERUN="${FORCE_RERUN:-False}"
DRY_RUN="${DRY_RUN:-False}"
SKIP_EXISTING_ANY_ARTIFACT="${SKIP_EXISTING_ANY_ARTIFACT:-True}"

# 初始化一次以获得当前机器默认 LOG_DIR；真正评测时各 suite 的 launcher 会重新 init。
FIRST_SUITE="${TASK_SUITES%% *}"
init_libero_eval_env "${FIRST_SUITE}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-${LOG_DIR}/main_table_specvla_baselines}"

script_for() {
  local method="$1"
  local suite="$2"
  case "${method}:${suite}" in
    strict:libero_goal) echo "${SCRIPT_DIR}/run_specvla_libero_goal_eval.sh" ;;
    strict:libero_object) echo "${SCRIPT_DIR}/run_specvla_libero_object_eval.sh" ;;
    strict:libero_spatial) echo "${SCRIPT_DIR}/run_specvla_libero_spatial_eval.sh" ;;
    strict:libero_10) echo "${SCRIPT_DIR}/run_specvla_libero_10_eval.sh" ;;
    relaxed:libero_goal) echo "${SCRIPT_DIR}/run_specvla_relaxed_libero_goal_eval.sh" ;;
    relaxed:libero_object) echo "${SCRIPT_DIR}/run_specvla_relaxed_libero_object_eval.sh" ;;
    relaxed:libero_spatial) echo "${SCRIPT_DIR}/run_specvla_relaxed_libero_spatial_eval.sh" ;;
    relaxed:libero_10) echo "${SCRIPT_DIR}/run_specvla_relaxed_libero_10_eval.sh" ;;
    *)
      echo "Unsupported method/suite: ${method}/${suite}" >&2
      return 1
      ;;
  esac
}

has_existing_run() {
  local method="$1"
  local suite="$2"
  local family="specvla_${method}"
  python3 - "${LOG_DIR}" "${family}" "${suite}" "${SKIP_EXISTING_ANY_ARTIFACT}" <<'PY'
import json
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
family = sys.argv[2]
suite = sys.argv[3]
skip_any = sys.argv[4].lower() == "true"
family_dir = log_dir / family
if not family_dir.exists():
    raise SystemExit(1)

# 完整 summary 是最可靠的完成标志。
for path in family_dir.glob("*_summary.json"):
    try:
        payload = json.load(open(path))
    except Exception:
        continue
    if payload.get("task_suite_name") == suite and payload.get("eval_family") == family:
        print(path)
        raise SystemExit(0)

# 长 run 正在跑或已经跑过但还没写 summary 时，会先留下 txt/timing 文件。
# 为了避免重复跑，默认把任一输出 artifact 也视为“已存在”。
if skip_any:
    prefix = f"EVAL-{suite}-"
    for pattern in ("*.txt", "*_timing.json", "*_summary.json"):
        matches = sorted(p for p in family_dir.glob(pattern) if p.name.startswith(prefix))
        if matches:
            print(matches[-1])
            raise SystemExit(0)

raise SystemExit(1)
PY
}

maybe_run() {
  local method="$1"
  local suite="$2"
  local launcher
  launcher="$(script_for "${method}" "${suite}")"

  if [[ "${FORCE_RERUN}" != "True" ]] && existing="$(has_existing_run "${method}" "${suite}" 2>/dev/null)"; then
    echo "[skip] SpecVLA ${method} ${suite}: existing artifact -> ${existing}"
    return 0
  fi

  if [[ "${DRY_RUN}" == "True" ]]; then
    echo "[dry-run] SpecVLA ${method} ${suite}: ${launcher}"
    return 0
  fi

  echo "[run] SpecVLA ${method} ${suite}: ${launcher}"
  bash "${launcher}"
}

cat <<EOF
========== SpecVLA baseline 主表评测链 ==========
TASK_SUITES=${TASK_SUITES}
RUN_STRICT=${RUN_STRICT}
RUN_RELAXED=${RUN_RELAXED}
FORCE_RERUN=${FORCE_RERUN}
DRY_RUN=${DRY_RUN}
SKIP_EXISTING_ANY_ARTIFACT=${SKIP_EXISTING_ANY_ARTIFACT}
LOG_DIR=${LOG_DIR}
SUMMARY_PREFIX=${SUMMARY_PREFIX}
SYNC_CUDA_TIMING=${SYNC_CUDA_TIMING}
TIMING_SCOPE=${TIMING_SCOPE}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
===============================================
EOF

for suite in ${TASK_SUITES}; do
  if [[ "${RUN_STRICT}" == "True" ]]; then
    maybe_run strict "${suite}"
  fi
  if [[ "${RUN_RELAXED}" == "True" ]]; then
    maybe_run relaxed "${suite}"
  fi
done

if [[ "${DRY_RUN}" == "True" ]]; then
  echo "DRY_RUN=True: skip summary generation."
  exit 0
fi

python openvla/specdecoding/test-speed/summarize_main_table_eval.py \
  --log-dir "${LOG_DIR}" \
  --ar-dir "${LOG_DIR}/openvla_ar" \
  --output-csv "${SUMMARY_PREFIX}.csv" \
  --output-md "${SUMMARY_PREFIX}.md"

cat <<EOF

汇总完成：
  ${SUMMARY_PREFIX}.csv
  ${SUMMARY_PREFIX}.md
EOF
