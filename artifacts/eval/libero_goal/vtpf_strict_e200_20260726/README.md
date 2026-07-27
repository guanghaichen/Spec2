# VTPF strict Goal formal evaluation

This directory is an immutable evidence snapshot for the first formal
Verified Temporal Prefill Fusion (VTPF) result. Keep the raw files unchanged;
put later reruns in a new timestamped directory.

## Provenance

- Code commit used for evaluation: `d60c555`
- Evaluation start time: `2026-07-26 19:45:10` (Asia/Shanghai)
- Suite: `libero_goal`
- Trials: 50 per task, 500 episodes total
- Seed: 7
- Checkpoint: `epoch_200_step_089600`
- Timing: `last_task`, `SYNC_CUDA_TIMING=False`
- Hardware: single RTX 4090

Command:

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_temporal_cascade_goal_eval.sh prefill
```

Strictness configuration:

```text
acceptance_mode=token
accept_threshold=0
tree_mode=off
verify_skip_mode=route
verify_skipped_blocks=0
temporal_prefill_min_stable_actions=3
```

## Headline result

| Metric | Value |
| --- | ---: |
| Success rate | 0.790 (395/500) |
| Mean action time | 0.142036 s |
| Paper-wrapped AR speedup | 1.286x |
| Length | 2.422 |
| Average accepted length | 1.466 |

The AR denominator is `0.182718 s/action` from the upstream-compatible Goal
baseline. See the repository README for the comparison table, uncertainty
analysis, and success/failure-conditioned diagnosis.

## Files

- `summary.json`: complete configuration and aggregated metrics.
- `timing.json`: raw per-action `(end_time, start_time)` pairs for the last task.
- `eval.txt`: all 500 episode outcomes and per-episode speculative statistics.
- `dflash_config.json`: the evaluated Draft checkpoint configuration.

The 550 MiB `pytorch_model.bin` exceeds GitHub's normal file limit and is not
stored in this directory. Its immutable identity is:

```text
e10127daa030ab5d7fbe639090078d3380c91a6d98b9302b31cf4d2f9dc5dac8  pytorch_model.bin
0b9026527183971e68c0199b1a9067dfa34a1307fc7863b8c52c2805e4915a18  dflash_config.json
```

Archive the weight separately with Git LFS or a private model registry before
removing it from the evaluation server.

SHA-256:

```text
6a10483860e0ef184b0dfe194bd3a9ebc9b8dc524b1ee4d2a7a5056705860e28  summary.json
36d299739209df1f9f344ea019e0f0a47bc7799b520732deb47cc48168e94cc3  timing.json
a6fa1f0ec26da2e56b745e09a632ad899187d77a5cefd68767f8d6ab0cc06201  eval.txt
```
