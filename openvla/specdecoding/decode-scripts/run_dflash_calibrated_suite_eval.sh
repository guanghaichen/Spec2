#!/usr/bin/env bash
set -euo pipefail

# 使用独立校准阶段生成的冻结配置评测任意 LIBERO 子集。
# 在线代码只读取一般策略参数，不包含 suite-specific 数值。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE_PATH="${CALIBRATED_PROFILE:-${1:-}}"
if [[ -z "${PROFILE_PATH}" || ! -f "${PROFILE_PATH}" ]]; then
  echo "请通过 CALIBRATED_PROFILE 或第一个位置参数提供冻结配置 JSON。" >&2
  exit 1
fi
if [[ -z "${SPEC_CKPT:-}" ]]; then
  echo "请显式设置 SPEC_CKPT=/absolute/path/to/suite/checkpoint。" >&2
  exit 1
fi

mapfile -t PROFILE_VALUES < <("${PYTHON_BIN:-python3}" - "${PROFILE_PATH}" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
selected = payload.get("selected_configuration") or payload.get("selected")
if not isinstance(selected, dict):
    raise SystemExit("Profile lacks selected_configuration.")

bounds = selected.get("depth_visual_bounds", [])
encoded_bounds = ",".join(
    "inf" if value is None or math.isinf(float(value)) else str(float(value))
    for value in bounds
)
values = (
    payload.get("task_suite_name", ""),
    payload.get("profile_id", path.stem),
    selected["schedule_period"],
    selected["schedule_target_count"],
    ",".join(str(int(value)) for value in selected.get("schedule_offsets", [])),
    selected.get("schedule_phase", -1.0),
    selected["authority_exponent"],
    selected["max_consecutive_holds"],
    encoded_bounds,
)
for value in values:
    print(value)
PY
)

if [[ "${#PROFILE_VALUES[@]}" -ne 9 ]]; then
  echo "冻结配置字段不完整: ${PROFILE_PATH}" >&2
  exit 1
fi

PROFILE_SUITE="${PROFILE_VALUES[0]}"
PROFILE_ID="${PROFILE_VALUES[1]}"
export TASK_SUITE_NAME="${TASK_SUITE_NAME:-${PROFILE_SUITE}}"
if [[ -z "${TASK_SUITE_NAME}" || "${TASK_SUITE_NAME}" != "${PROFILE_SUITE}" ]]; then
  echo "配置属于 ${PROFILE_SUITE}，当前请求 ${TASK_SUITE_NAME}；拒绝跨子集复用。" >&2
  exit 1
fi

export DFLASH_TEMPORAL_HOLD_POLICY_OVERRIDE=True
export DFLASH_TEMPORAL_HOLD_POLICY=calibrated
export DFLASH_TEMPORAL_SCHEDULE_PERIOD="${PROFILE_VALUES[2]}"
export DFLASH_TEMPORAL_SCHEDULE_TARGET_COUNT="${PROFILE_VALUES[3]}"
export DFLASH_TEMPORAL_SCHEDULE_OFFSETS="${PROFILE_VALUES[4]}"
export DFLASH_TEMPORAL_SCHEDULE_PHASE="${PROFILE_VALUES[5]}"
export DFLASH_TEMPORAL_HOLD_ACTION_DECAY=power_law
export DFLASH_TEMPORAL_AUTHORITY_EXPONENT="${PROFILE_VALUES[6]}"
if (( PROFILE_VALUES[7] < 1 )); then
  export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE=1
else
  export DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE="${PROFILE_VALUES[7]}"
fi
export DFLASH_TEMPORAL_DEPTH_VISUAL_BOUNDS="${PROFILE_VALUES[8]}"
export DFLASH_TEMPORAL_ADAPTIVE_MIN_VERIFIED_RUN=1
export DFLASH_TEMPORAL_PREFILL_MIN_STABLE_ACTIONS=1
export DFLASH_TEMPORAL_ROUTE_LABEL="Calibrated-${PROFILE_ID}"
export RUN_ID_NOTE="${RUN_ID_NOTE:-dflash-calibrated-${PROFILE_ID}-e${EVAL_EPOCH:-100}}"

echo "CALIBRATED_PROFILE=${PROFILE_PATH}"
echo "PROFILE_ID=${PROFILE_ID} suite=${PROFILE_SUITE}"
echo "schedule=${DFLASH_TEMPORAL_SCHEDULE_TARGET_COUNT}/${DFLASH_TEMPORAL_SCHEDULE_PERIOD} offsets=${DFLASH_TEMPORAL_SCHEDULE_OFFSETS:-mechanical} authority_p=${DFLASH_TEMPORAL_AUTHORITY_EXPONENT} max_depth=${DFLASH_VERIFY_SKIP_MAX_CONSECUTIVE} visual_bounds=${DFLASH_TEMPORAL_DEPTH_VISUAL_BOUNDS:-unbounded}"

bash "${SCRIPT_DIR}/run_dflash_vtpf_adaptive_decimation_goal_eval.sh"
