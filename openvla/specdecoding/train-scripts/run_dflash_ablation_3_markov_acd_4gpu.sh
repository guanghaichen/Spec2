#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_STAGE=markov_acd bash "${SCRIPT_DIR}/run_dflash_ablation_goal_4gpu.sh"
