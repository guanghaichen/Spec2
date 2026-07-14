#!/usr/bin/env bash
set -euo pipefail

# 推荐主实验：一层并行 DFlash 主干。
# 复用三层容量消融脚本中的同一套数据、loss 和 IO 配置，只覆盖 Draft 深度和输出目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_DRAFT_LAYERS=1 \
OUTPUT_NAME="${OUTPUT_NAME:-ckpt_goal_dflash_action_rnn_prefix_1layer_b8x2_4gpu}" \
  bash "${SCRIPT_DIR}/run_dflash_action_rnn_prefix_3layer_4gpu.sh"
