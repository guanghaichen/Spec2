#!/usr/bin/env bash
set -euo pipefail

# 一键评测 DFlash CAD-head strict / relaxed，并汇总成主表草稿。
# 默认扫四个 LIBERO suite 与 120/150/180/200 四个 checkpoint：
#   CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
#     bash openvla/specdecoding/decode-scripts/run_dflash_cad_head_main_table_eval.sh
# 常用覆盖：
#   TASK_SUITES="libero_goal" EVAL_EPOCHS="200" bash ...
#   DFLASH_OUTPUT_DIR=/path/to/ckpt_root EVAL_EPOCHS="180 200" bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

TASK_SUITES="${TASK_SUITES:-libero_goal libero_object libero_spatial libero_10}"
EVAL_EPOCHS="${EVAL_EPOCHS:-120 150 180 200}"
RUN_STRICT="${RUN_STRICT:-True}"
RUN_RELAXED="${RUN_RELAXED:-True}"
STRICT_ACCEPT_THRESHOLD="${STRICT_ACCEPT_THRESHOLD:-0}"
RELAXED_ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD:-9}"

# 初始化一次只是为了拿到当前机器的 LOG_DIR 默认值；真正评测时每个 suite 会重新 init。
FIRST_SUITE="${TASK_SUITES%% *}"
init_libero_eval_env "${FIRST_SUITE}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-${LOG_DIR}/main_table_dflash_cad_head}"

cat <<EOF
========== DFlash CAD-head 主表评测链 ==========
TASK_SUITES=${TASK_SUITES}
EVAL_EPOCHS=${EVAL_EPOCHS}
RUN_STRICT=${RUN_STRICT}
RUN_RELAXED=${RUN_RELAXED}
LOG_DIR=${LOG_DIR}
SUMMARY_PREFIX=${SUMMARY_PREFIX}
SYNC_CUDA_TIMING=${SYNC_CUDA_TIMING}
TIMING_SCOPE=${TIMING_SCOPE}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
==============================================
EOF

for suite in ${TASK_SUITES}; do
  suite_slug="${suite#libero_}"
  for epoch in ${EVAL_EPOCHS}; do
    if [[ "${RUN_STRICT}" == "True" ]]; then
      echo "[CAD strict] suite=${suite} epoch=${epoch}"
      SPEC_CKPT="" TASK_SUITE_NAME="${suite}" EVAL_EPOCH="${epoch}" \
        ACCEPT_THRESHOLD="${STRICT_ACCEPT_THRESHOLD}" \
        RUN_ID_NOTE="dflash-cad-strict-${suite_slug}-e${epoch}-r${STRICT_ACCEPT_THRESHOLD}" \
        bash "${SCRIPT_DIR}/run_dflash_residual_strict_libero_goal_eval.sh"
    fi

    if [[ "${RUN_RELAXED}" == "True" ]]; then
      echo "[CAD relaxed] suite=${suite} epoch=${epoch}"
      SPEC_CKPT="" TASK_SUITE_NAME="${suite}" EVAL_EPOCH="${epoch}" \
        ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD}" \
        RUN_ID_NOTE="dflash-cad-relaxed-${suite_slug}-e${epoch}-r${RELAXED_ACCEPT_THRESHOLD}" \
        bash "${SCRIPT_DIR}/run_dflash_residual_libero_goal_eval.sh"
    fi
  done
done

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
