# VTPF-TD-VisualBudget e100 evaluation (2026-07-30)

This directory freezes the two non-overlapping LIBERO Goal pilot batches used
to select the Visual Budget working point, followed by the complete
50-trial-per-task evaluation at `0.15`.

## Fixed setup

- Target: local OpenVLA Goal checkpoint on the RTX 4090 evaluation machine.
- Draft: `Draft_checkpoint/goal/epoch_100_step_044800` (Minimal, one layer,
  Action-RNN off). `pytorch_model.bin` SHA-256:
  `f9975d3776f5c7e5f84b7caca5efdf53650e7a97524d7c9e9f463dc82f73f41a`.
- Policy: `visual_budget`; first hold follows TD-Fast, a second hold is allowed
  when cumulative target-anchor pixel relative L2 is at most `0.15`, then the
  next action is forced through target.
- Acceptance: token strict fallback, action-group relaxation off, tree off.
- Timing: `SYNC_CUDA_TIMING=False`, `TIMING_SCOPE=last_task`.
- AR denominator: `0.182718 s/action` on the same machine and paper wrapper.

## Results

| Initial-state indices per task | Episodes | Success | Mean step | Speedup | Target rate | Extension rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-2 | 30 | 23/30 | 0.056270 s | 3.247x | 34.7% | 89.3% |
| 3-5 | 30 | 22/30 | 0.053744 s | 3.400x | 34.2% | 93.8% |
| pooled | 60 | 45/60 | 0.054848 s | 3.331x | 34.4% | 91.8% |
| **formal 0-49** | **500** | **336/500** | **0.049670 s** | **3.679x** | **34.3%** | **92.3%** |

All six last-task trajectories individually exceeded 3x. The two successful
trajectories in the first batch were 3.249x and 3.128x; the successful
trajectory in the second batch was 3.361x. This check prevents a failed,
long-horizon rollout from being the sole source of an apparently high speedup.

On the same 60 fixed initial states, the previously frozen Minimal TD-Fast log
contains 43 successes and the old exact-action Adaptive log contains 44. This
pilot did not expose the eventual full-suite regression. In the formal paired
evaluation, VisualBudget succeeded on 336/500 versus TD-Fast's 377/500. The
paired outcomes were 280 both-success, 56 VisualBudget-only, 97 TD-Fast-only,
and 67 both-failure; the exact two-sided McNemar p-value is `0.00115`.

The speedup is real but the `0.15` operating point is too aggressive for the
default method. Last-task successful trajectories alone averaged `0.057676 s`
or `3.168x`, while failure trajectories averaged `0.044138 s` or `4.140x`.
Thus failure-length bias amplifies the aggregate `3.679x`, but does not create
the entire gain. Keep `0.15` as an aggressive Pareto point and report its SR;
do not present it as a no-regression replacement for TD-Fast.

## Post-formal threshold check

Two non-overlapping `0.10` batches produced 46/60 successes and a pooled
`3.251x`, but that aggregate is not sufficient evidence for a balanced 3x
point: all three successful last-task trajectories were in the slower batch
(`2.875x`), while all three trajectories in the `3.475x` batch failed. A
`0.12` diagnostic was stopped after 30/60 episodes because it had only 22
successes versus 25 for the corresponding `0.15` states. Closed-loop success
is not monotonic in this global visual threshold, so further threshold tuning
was rejected as a remedy for stale-action risk.

## Reproduction

```bash
SPEC_CKPT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/Draft_checkpoint/goal/epoch_100_step_044800 \
EVAL_EPOCH=100 CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=3 \
TRIAL_START_INDEX=0 SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
DFLASH_TEMPORAL_VISUAL_BUDGET=0.15 \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_visual_budget_goal_eval.sh

# Independent fixed initial states 3, 4, and 5 for every task.
TRIAL_START_INDEX=3  # keep all other arguments identical
```

The repository copy of the launcher is the authoritative configuration. The
launcher snapshot here is included only to make this artifact self-contained.
