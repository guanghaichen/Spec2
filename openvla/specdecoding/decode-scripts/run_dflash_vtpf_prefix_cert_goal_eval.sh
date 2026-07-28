#!/usr/bin/env bash
set -euo pipefail

# VTPF-PrefixCert：用目标模型严格一致的短因果前缀认证时序候选，
# 再信任该动作的剩余后缀。认证失败立即回退到严格 DFlash；默认每次
# 信任后强制下一动作完整校验，因此不会形成连续无校验漂移。
#
# 用法：
#   EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=10 \
#     bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_prefix_cert_goal_eval.sh
#
# 可覆盖的主要风险旋钮：
#   DFLASH_TEMPORAL_PREFIX_CERT_TOKENS=4     # 严格认证前 4 个 token
#   DFLASH_VERIFY_SKIP_MIN_STABLE_ACTIONS=3 # 至少 3 个已验证的相同动作
#   DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE=1    # 最多连续信任 1 次

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=False
export DFLASH_ACCEPTANCE_MODE=token
export ACCEPT_THRESHOLD=0
export DFLASH_TREE_MODE=off
export DFLASH_TREE_BUDGET=0
export DFLASH_TARGET_LOGITS_MODE=action_only
export DFLASH_VERIFY_SKIP_MODE=active
export DFLASH_TEMPORAL_PREFILL_FUSION=True
export DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS="${DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS:-3}"
export DFLASH_TEMPORAL_PREFIX_CERT_TOKENS="${DFLASH_TEMPORAL_PREFIX_CERT_TOKENS:-4}"
export DFLASH_VERIFY_SKIP_MIN_STABLE_ACTIONS="${DFLASH_VERIFY_SKIP_MIN_STABLE_ACTIONS:-3}"
export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE="${DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE:-1}"
export DFLASH_TEMPORAL_ROUTE_MIN_COSINE="${DFLASH_TEMPORAL_ROUTE_MIN_COSINE:-0.99}"
export DFLASH_TEMPORAL_ROUTE_STOP_ON_REJECT=True
export DFLASH_TEMPORAL_FUSE_VERIFY=True
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-vtpf-prefixcert-goal-e${EVAL_EPOCH:-latest}-m${DFLASH_TEMPORAL_PREFIX_CERT_TOKENS}-k${DFLASH_VERIFY_SKIP_MIN_STABLE_ACTIONS}}"

# 使用 relaxed 评测入口只是为了把日志归档到 dflash_relaxed；实际动作
# 前缀认证始终是 exact token，绝不启用旧的 token-distance/action-group。
bash "${SCRIPT_DIR}/run_dflash_goal_eval.sh" relaxed
