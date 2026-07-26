#!/usr/bin/env bash
set -euo pipefail

# 只做诊断，不改变任何动作输出：
# 1. 完整目标词表与动作子词表 argmax 等价性；
# 2. 免校验门控信号对“整块严格正确”的预测能力；
# 3. 同步 CUDA 后的逐阶段真实耗时。
# 默认只跑 Goal 的第一个 task、一个 episode；正式门槛不能据此直接确定。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DFLASH_TARGET_LOGITS_MODE="${DFLASH_TARGET_LOGITS_MODE:-shadow}"
export DFLASH_VERIFY_SKIP_MODE="${DFLASH_VERIFY_SKIP_MODE:-shadow}"
export DFLASH_TREE_MODE=off
export DFLASH_ACCEPTANCE_MODE=token
export ACCEPT_THRESHOLD=0
export DFLASH_PROFILE_STAGES="${DFLASH_PROFILE_STAGES:-True}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
export MAX_EVAL_TASKS="${MAX_EVAL_TASKS:-1}"
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-verify-skip-shadow-goal-e${EVAL_EPOCH:-200}}"

bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" strict
