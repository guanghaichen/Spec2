# VTPF-TD-VisualBudget e100 pilots (2026-07-30)

This directory freezes the two non-overlapping LIBERO Goal pilot batches used
to select the Visual Budget working point. These are pilot results, not the
final 50-trial-per-task paper result.

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

All six last-task trajectories individually exceeded 3x. The two successful
trajectories in the first batch were 3.249x and 3.128x; the successful
trajectory in the second batch was 3.361x. This check prevents a failed,
long-horizon rollout from being the sole source of an apparently high speedup.

On the same 60 fixed initial states, the previously frozen Minimal TD-Fast log
contains 43 successes and the old exact-action Adaptive log contains 44. No
success-rate loss from Visual Budget was observed in this pilot, but the final
500-episode evaluation is still required.

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
