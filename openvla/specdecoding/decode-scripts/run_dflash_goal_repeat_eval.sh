#!/usr/bin/env bash
set -euo pipefail

# Repeat Goal-only DFlash CAD-head evaluation several times to check whether
# the observed Length/Speedup gap vs SpecVLA is stable rather than a single-run
# artifact. This script is intentionally Goal-only because the current DFlash
# checkpoint was trained on LIBERO Goal data.
#
# Default:
#   CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
#     bash openvla/specdecoding/decode-scripts/run_dflash_goal_repeat_eval.sh
#
# Useful overrides:
#   REPEAT_SEEDS="7 8 9 10 11" EVAL_EPOCH=200 bash ...
#   RUN_STRICT=False RUN_RELAXED=True bash ...
#   DFLASH_OUTPUT_DIR=/path/to/ckpt_root SPEC_CKPT=/path/to/epoch_ckpt bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/libero_eval_common.sh"

export TASK_SUITE_NAME="libero_goal"
export EVAL_EPOCH="${EVAL_EPOCH:-200}"
export RUN_STRICT="${RUN_STRICT:-True}"
export RUN_RELAXED="${RUN_RELAXED:-True}"
export STRICT_ACCEPT_THRESHOLD="${STRICT_ACCEPT_THRESHOLD:-0}"
export RELAXED_ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD:-9}"
export DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING="${DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING:-True}"
export DFLASH_NUM_DRAFT_LAYERS="${DFLASH_NUM_DRAFT_LAYERS:-1}"
export SYNC_CUDA_TIMING="${SYNC_CUDA_TIMING:-False}"
export TIMING_SCOPE="${TIMING_SCOPE:-last_task}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
export USE_WANDB="${USE_WANDB:-False}"

REPEAT_SEEDS="${REPEAT_SEEDS:-7 8 9 10 11}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-dflash-goal-repeat-e${EVAL_EPOCH}-$(date +%Y%m%d-%H%M%S)}"

# Initialize paths and resolve checkpoint once for printing/summary defaults.
init_libero_eval_env "${TASK_SUITE_NAME}"
resolve_dflash_checkpoint
SUMMARY_PREFIX="${SUMMARY_PREFIX:-${LOG_DIR}/repeat_${RUN_ID_PREFIX}}"

cat <<EOF
========== DFlash Goal repeated evaluation ==========
TASK_SUITE_NAME=${TASK_SUITE_NAME}
EVAL_EPOCH=${EVAL_EPOCH}
SPEC_CKPT=${SPEC_CKPT}
RUN_STRICT=${RUN_STRICT}
RUN_RELAXED=${RUN_RELAXED}
STRICT_ACCEPT_THRESHOLD=${STRICT_ACCEPT_THRESHOLD}
RELAXED_ACCEPT_THRESHOLD=${RELAXED_ACCEPT_THRESHOLD}
REPEAT_SEEDS=${REPEAT_SEEDS}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}
SYNC_CUDA_TIMING=${SYNC_CUDA_TIMING}
TIMING_SCOPE=${TIMING_SCOPE}
RUN_ID_PREFIX=${RUN_ID_PREFIX}
SUMMARY_PREFIX=${SUMMARY_PREFIX}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
====================================================
EOF

for seed in ${REPEAT_SEEDS}; do
  if [[ "${RUN_STRICT}" == "True" ]]; then
    echo "[DFlash Goal strict repeat] seed=${seed}"
    SEED="${seed}" \
    TASK_SUITE_NAME="libero_goal" \
    EVAL_EPOCH="${EVAL_EPOCH}" \
    ACCEPT_THRESHOLD="${STRICT_ACCEPT_THRESHOLD}" \
    DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING="${DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING}" \
    DFLASH_NUM_DRAFT_LAYERS="${DFLASH_NUM_DRAFT_LAYERS}" \
    RUN_ID_NOTE="${RUN_ID_PREFIX}-strict-seed${seed}-r${STRICT_ACCEPT_THRESHOLD}" \
      bash "${SCRIPT_DIR}/run_dflash_residual_strict_libero_goal_eval.sh"
  fi

  if [[ "${RUN_RELAXED}" == "True" ]]; then
    echo "[DFlash Goal relaxed repeat] seed=${seed}"
    SEED="${seed}" \
    TASK_SUITE_NAME="libero_goal" \
    EVAL_EPOCH="${EVAL_EPOCH}" \
    ACCEPT_THRESHOLD="${RELAXED_ACCEPT_THRESHOLD}" \
    DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING="${DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING}" \
    DFLASH_NUM_DRAFT_LAYERS="${DFLASH_NUM_DRAFT_LAYERS}" \
    RUN_ID_NOTE="${RUN_ID_PREFIX}-relaxed-seed${seed}-r${RELAXED_ACCEPT_THRESHOLD}" \
      bash "${SCRIPT_DIR}/run_dflash_residual_libero_goal_eval.sh"
  fi
done

python3 - "${LOG_DIR}" "${RUN_ID_PREFIX}" "${SUMMARY_PREFIX}" <<\PY
import csv
import json
import statistics
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
run_id_prefix = sys.argv[2]
summary_prefix = Path(sys.argv[3])

ar_candidates = sorted(
    (log_dir / "openvla_ar").glob("*libero_goal*openvla_ar_summary.json"),
    key=lambda path: path.stat().st_mtime,
)
if not ar_candidates:
    raise SystemExit(f"No Goal AR summary found under {log_dir / 'openvla_ar'}")
ar_summary = json.loads(ar_candidates[-1].read_text())
ar_mean = ar_summary.get("timing", {}).get("mean")

rows = []
for subdir, mode in [("dflash_strict", "strict"), ("dflash_relaxed", "relaxed")]:
    for path in sorted((log_dir / subdir).glob(f"*{run_id_prefix}*summary.json")):
        payload = json.loads(path.read_text())
        gen = payload.get("generation") or {}
        mean_time = payload.get("timing", {}).get("mean")
        speedup = (float(ar_mean) / float(mean_time)) if ar_mean and mean_time else None
        row = {
            "mode": mode,
            "seed": payload.get("seed"),
            "accept_threshold": payload.get("accept_threshold"),
            "success_rate": payload.get("success_rate"),
            "length": gen.get("length"),
            "avg_accept_length": gen.get("avg_accept_length"),
            "overall_hit_rate": gen.get("overall_hit_rate"),
            "mean_step_time": mean_time,
            "speedup_vs_ar": speedup,
            "summary_path": str(path),
        }
        for item in gen.get("per_position") or []:
            row[f"p{item.get('position')}_hit_rate"] = item.get("hit_rate")
        rows.append(row)

if not rows:
    raise SystemExit(f"No repeated DFlash summaries found for prefix {run_id_prefix!r}")

summary_prefix.parent.mkdir(parents=True, exist_ok=True)
csv_path = summary_prefix.with_suffix(".csv")
md_path = summary_prefix.with_suffix(".md")
fields = [
    "mode",
    "seed",
    "accept_threshold",
    "success_rate",
    "length",
    "avg_accept_length",
    "overall_hit_rate",
    "mean_step_time",
    "speedup_vs_ar",
    "p1_hit_rate",
    "p2_hit_rate",
    "p3_hit_rate",
    "p4_hit_rate",
    "p5_hit_rate",
    "p6_hit_rate",
    "summary_path",
]
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

def fmt(value, digits=3):
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"

lines = [
    f"# DFlash Goal repeated eval: {run_id_prefix}",
    "",
    f"AR baseline summary: `{ar_candidates[-1]}`",
    f"AR mean step time: {fmt(ar_mean, 6)} s",
    "",
    "| Mode | Seed | SR | Length | Avg Accept | Hit Rate | Mean Step | Speedup | p1 | p2 | p3 | p4 | p5 | p6 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        "| {mode} | {seed} | {sr} | {length} | {accept} | {hit} | {mean} | {speedup}x | {p1} | {p2} | {p3} | {p4} | {p5} | {p6} |".format(
            mode=row["mode"],
            seed=row["seed"],
            sr=fmt(row["success_rate"]),
            length=fmt(row["length"]),
            accept=fmt(row["avg_accept_length"]),
            hit=fmt(row["overall_hit_rate"]),
            mean=fmt(row["mean_step_time"], 4),
            speedup=fmt(row["speedup_vs_ar"]),
            p1=fmt(row.get("p1_hit_rate")),
            p2=fmt(row.get("p2_hit_rate")),
            p3=fmt(row.get("p3_hit_rate")),
            p4=fmt(row.get("p4_hit_rate")),
            p5=fmt(row.get("p5_hit_rate")),
            p6=fmt(row.get("p6_hit_rate")),
        )
    )

lines.extend([
    "",
    "## Aggregate",
    "",
    "| Mode | N | SR mean±std | Length mean±std | Speedup mean±std |",
    "| --- | ---: | ---: | ---: | ---: |",
])
for mode in ["strict", "relaxed"]:
    group = [row for row in rows if row["mode"] == mode]
    if not group:
        continue
    def mean_std(key):
        vals = [float(row[key]) for row in group if row.get(key) is not None]
        if not vals:
            return ""
        if len(vals) == 1:
            return fmt(vals[0]) + "±0.000"
        return f"{statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}"
    lines.append(f"| {mode} | {len(group)} | {mean_std('success_rate')} | {mean_std('length')} | {mean_std('speedup_vs_ar')} |")

md_text = "\n".join(lines) + "\n"
md_path.write_text(md_text)
print(md_text)
print(f"CSV: {csv_path}")
print(f"Markdown: {md_path}")
PY
