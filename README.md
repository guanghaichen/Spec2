# Spec²: Speculative Inference Beyond Token Space for Low-Latency VLA Control

Spec² is a dual-axis speculative inference framework for accelerating frozen autoregressive Vision-Language-Action
policies. It extends speculation beyond the action-token sequence to the control-time axis, reducing both the work performed
inside a target frame and the number of control frames that require a full multimodal target-model invocation.

This repository contains the OpenVLA implementation and LIBERO evaluation pipeline. It is built on
[OpenVLA](https://github.com/openvla/openvla) and [SpecVLA](https://github.com/PineTreeWss/SpecVLA); its block-parallel
drafting path is inspired by [DFlash](https://arxiv.org/abs/2602.06036) and the
[SpecForge implementation](https://github.com/sgl-project/SpecForge).

## Why Speculation Must Go Beyond Tokens

OpenVLA represents one robot action with only seven autoregressive tokens. Existing VLA speculative decoders reduce the
serial cost of producing and verifying these tokens, but still invoke the large multimodal target policy once per control
step. Our stage-level profiling exposes the resulting bottleneck inversion: parallel draft generation accounts for only
about **3%** of end-to-end action-model latency, while multimodal target prefill and verification dominate the remaining
cost. Even a zero-cost draft therefore cannot break the per-step target-grounding lower bound.

Embodied control provides a second source of redundancy. Consecutive observations and locally valid actions are correlated
along a physical trajectory, making the latest target-grounded action a zero-generation-cost temporal proposal. Unlike a
rejected language token, however, an executed robot action changes the world and cannot be rolled back. Exploiting temporal
redundancy consequently requires both causal validation and explicit bounds on target-free execution.

Spec² addresses two coupled speculative axes while keeping the OpenVLA policy frozen:

- the **action-token axis**, where candidates are generated and verified within one control frame;
- the **control-time axis**, where recently grounded actions become proposals for subsequent observations.

## Method Overview

Spec² organizes inference as three nested operators, each removing a different latency term:

Let `C_P`, `C_D`, `C_V`, and `C_H` denote multimodal target prefill, draft proposal, target tail verification, and
target-free Hold cost; let `p_V` be the VTPF full-match probability on target frames and `rho` the target-frame ratio. The
execution cost is organized as `E[C] = rho * [C_P + (1 - p_V) * (C_D + C_V)] + (1 - rho) * C_H`. Parallel Action Drafting
reduces `C_D`, VTPF increases `p_V`, and PacedHarmonic reduces `rho` while bounding the physical effect of the `C_H` path.

1. **Parallel Action Drafting** conditions a one-layer DFlash draft on the complete multimodal prompt and verified action
   prefix, then proposes the remaining action-token block in one non-autoregressive forward pass. Every proposed token is
   still checked by the frozen target policy.
2. **Temporal Prefill Verification (VTPF)** appends the latest target-grounded action to the current multimodal prompt. The
   mandatory causal target prefill then both encodes the new observation and verifies that temporal proposal. A full match
   skips the draft and tail verifier; a partial match becomes a verified anchor for the parallel draft.
3. **PacedHarmonic** operates before the target path. It permits at most two target-free Holds of the latest grounded action,
   with both depths measured from the same target-image anchor. A depth-`d` Hold must remain within the cumulative observation
   drift bound `d * beta`, where `beta=0.075`. Extension debt bounds the long-run open-loop cadence, while continuous action
   authority decays as `1 / d`; the discrete gripper state is unchanged.

The operators are nested in this order: PacedHarmonic first decides whether the current control frame can bypass the target;
otherwise VTPF runs inside the mandatory target prefill, and Parallel Action Drafting is invoked only when VTPF does not
complete the action. Parallel Action Drafting and VTPF are strict target-token-equivalent paths. PacedHarmonic introduces
bounded target-free execution, so the complete method must be evaluated jointly by closed-loop success rate and latency.

## Results

LIBERO evaluation uses one RTX 4090, BF16, batch size 1, seed 7, 50 trials for each of 10 tasks, and an identical timing
scope for every local baseline. Each entry is **success rate / speedup over the suite-specific OpenVLA AR baseline**.

### Strict target-equivalent inference

| Method | Goal | Spatial | Object | LIBERO-10 | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenVLA AR | 74.2 / 1.00x | 87.0 / 1.00x | 88.4 / 1.00x | 51.4 / 1.00x | 75.3 / 1.00x |
| SpecVLA strict | 76.8 / 1.02x | 85.0 / 1.00x | 87.6 / 1.10x | 54.4 / 0.99x | 76.0 / 1.03x |
| Parallel Draft strict | 79.2 / 1.14x | 85.8 / 1.15x | 88.2 / 1.18x | 52.8 / 1.18x | 76.5 / 1.16x |
| Parallel Draft + VTPF strict | 77.6 / 1.30x | 86.4 / 1.17x | 87.2 / 1.19x | 50.0 / 1.29x | 75.3 / 1.24x |

### Approximate / target-free inference

| Method | Goal | Spatial | Object | LIBERO-10 | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| SpecVLA relaxed | 73.4 / 1.29x | 86.2 / 1.15x | 85.0 / 1.31x | 49.8 / 1.11x | 73.6 / 1.21x |
| **Spec²** | **74.8 / 3.26x** | **83.4 / 2.36x** | **84.8 / 2.36x** | **50.8 / 3.02x** | **73.5 / 2.75x** |

Spec² reaches **2.75x average speedup** across the four suites and exceeds **3x** on LIBERO-Goal and LIBERO-10, while
retaining 73.5% average success rate compared with 75.3% for the same-protocol OpenVLA AR baseline.

Goal, Spatial, and LIBERO-10 use 100-epoch draft checkpoints. The currently frozen Object result uses the independently
trained 60-epoch checkpoint. Compact run metadata, source paths, and checksums are frozen in
[`artifacts/eval/dual_anchor_main_20260809/`](artifacts/eval/dual_anchor_main_20260809/README.md).

## Installation

The tested environment uses:

```text
Python 3.10
PyTorch 2.2.0 + CUDA 12.1
Transformers 4.40.1
MuJoCo 3.3.2
Robosuite 1.4.1
NumPy 1.26.4
```

Create an isolated environment and install the repository requirements:

```bash
conda create -n specvla python=3.10 -y
conda activate specvla
pip install -r requirements-min.txt
```

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) separately and add it to `PYTHONPATH`, or place it at
one of the locations recognized by `openvla/specdecoding/decode-scripts/libero_eval_common.sh`.

## Data and Checkpoints

Prepare the suite-specific OpenVLA checkpoints and modified LIBERO RLDS data:

```text
hf_files/
├── openvla-7b-finetuned-libero-goal
├── openvla-7b-finetuned-libero-spatial
├── openvla-7b-finetuned-libero-object
└── openvla-7b-finetuned-libero-10

datasets/modified_libero_rlds/
├── libero_goal_no_noops
├── libero_spatial_no_noops
├── libero_object_no_noops
└── libero_10_no_noops
```

Large datasets, model weights, rollout videos, and raw evaluation logs are intentionally excluded from Git.

## Training

### 1. Generate a packed training dataset

Run a smoke test first, then generate the complete dataset. The full pipeline packs samples into one HDF5 file to avoid
high-frequency small-file I/O.

```bash
TASK_SUITE_NAME=libero_goal GPU_ID=0 \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh smoke

TASK_SUITE_NAME=libero_goal GPU_ID=0 \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh full
```

Set `TASK_SUITE_NAME` to `libero_goal`, `libero_spatial`, `libero_object`, or `libero_10`.

### 2. Train the minimal draft

The released recipe uses one draft layer, two GPUs, one data worker, and 100 epochs:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
TASK_SUITE_NAME=libero_goal NUM_WORKERS=1 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh minimal
```

The minimal objective combines hidden Smooth-L1, cosine alignment, and soft target-distribution distillation. Checkpoints
contain `pytorch_model.bin` and `dflash_config.json`.

## Evaluation

Evaluation scripts select the target model from `TASK_SUITE_NAME`, but the draft checkpoint must be provided explicitly and
must belong to the same suite.

### Reproduce the three main variants

```bash
CUDA_VISIBLE_DEVICES=0 \
TASK_SUITE_NAME=libero_goal \
SPEC_CKPT=/absolute/path/to/goal/epoch_100_step_xxxxxx \
EVAL_EPOCH=100 NUM_TRIALS_PER_TASK=50 SEED=7 \
SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_minimal_suite_main_3way_eval.sh
```

The launcher runs:

1. Parallel Draft strict;
2. Parallel Draft + VTPF strict;
3. the complete Spec² path with the shared target-anchor PacedHarmonic rule.

Resume an interrupted sequence with `START_CASE=2` or `START_CASE=3`.

### Evaluate only the complete method

```bash
CUDA_VISIBLE_DEVICES=0 \
TASK_SUITE_NAME=libero_goal \
SPEC_CKPT=/absolute/path/to/goal/checkpoint \
EVAL_EPOCH=100 NUM_TRIALS_PER_TASK=50 SEED=7 \
SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_paced_harmonic_dual_anchor_eval.sh
```

Override the unit drift budget only for controlled sensitivity studies:

```bash
DFLASH_TEMPORAL_UNIT_VISUAL_BUDGET=0.075 ...
```

### Capture frame-aligned qualitative evidence

The qualitative launcher records one fixed successful initial state per suite. It saves the rollout MP4 together with the
frame-aligned Target/H1/H2 decision trace and action/state records used to produce the paper's robot-frame figures:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT=/absolute/path/to/qualitative_rollouts \
  bash openvla/specdecoding/decode-scripts/run_chapter5_qualitative_rollouts.sh
```

This path is disabled during ordinary evaluation. `TASK_START_INDEX`, `TRIAL_START_INDEX`, `SAVE_ROLLOUT_VIDEOS`, and
`ROLLOUT_VIDEO_DIR` can also be set on the generic suite launcher for a targeted audit. Video encoding requires the optional
`imageio-ffmpeg` package; model inference and quantitative evaluation do not.

## Output Metrics

Each evaluation produces a text log, a timing JSON, and a summary JSON. Important fields are:

| Field | Meaning |
| --- | --- |
| `success_rate` | closed-loop task success rate |
| `timing.mean` | mean action-model latency under the configured timing scope |
| `generation.length` | average progress length per speculative block |
| `generation.avg_accept_length` | average accepted draft tokens per block |
| `generation.per_position` | online hit rate at each proposal position |
| `generation.temporal_hold.target_prefill_rate` | fraction of control steps that invoke the full target prefill |
| `base_holds` / `extended_holds` | accepted H1 / H2 decisions |

Speedup must be computed against OpenVLA AR on the same GPU, suite, software environment, and timing protocol.

## Repository Structure

```text
openvla/specdecoding/model/                 draft and temporal-control modules
openvla/specdecoding/train-scripts/         data generation and draft training
openvla/specdecoding/decode-scripts/        evaluation launchers
openvla/specdecoding/tests/                 unit tests
openvla/experiments/robot/libero/           LIBERO rollouts and metric summaries
openvla/prismatic/extern/hf/                OpenVLA speculative inference path
artifacts/                                   compact, versioned evidence
docs/                                        additional documentation
```

## Reproducibility Notes

- Official local results use `NUM_TRIALS_PER_TASK=50`, `SEED=7`, `SYNC_CUDA_TIMING=False`, and
  `TIMING_SCOPE=last_task`.
- `Length` measures speculative token progress; it is not the temporal Hold depth and does not by itself determine latency.
- PacedHarmonic is not strict token-equivalent because held actions are physically executed before the next target call.
- The drift budget is a deployment hyperparameter, not a formal robot-safety certificate.
- Current evidence is limited to OpenVLA-7B and LIBERO.

## Citation

The paper and BibTeX entry will be added after public release.

## License

This project follows the licenses in [LICENSE](LICENSE) and [LICENSE.txt](LICENSE.txt). Third-party components retain their
original licenses.

## Acknowledgements

We thank the authors of OpenVLA, SpecVLA, DFlash, SpecForge, and LIBERO for releasing their code and models.
