# Spec²: Dual-Axis Speculative Inference for OpenVLA

Spec² accelerates autoregressive Vision-Language-Action inference without modifying the frozen OpenVLA policy. It combines
block-parallel action drafting, verification inside the target prefill, and bounded temporal action reuse.

This repository is research code built on [OpenVLA](https://github.com/openvla/openvla) and
[SpecVLA](https://github.com/PineTreeWss/SpecVLA). The parallel draft design is inspired by
[DFlash](https://arxiv.org/abs/2602.06036) and its
[SpecForge implementation](https://github.com/sgl-project/SpecForge).

## Overview

OpenVLA encodes each action as seven autoregressive tokens. Spec² reduces latency along two complementary axes:

1. **Parallel Action Drafting** proposes the remaining action-token block in one lightweight forward pass.
2. **Verified Temporal Prefill Fusion (VTPF)** verifies the previous target-confirmed action while the current multimodal
   prompt is being prefetched, avoiding a separate tail-verification pass when the prefix matches.
3. **PacedHarmonic** reuses the latest target-confirmed action for at most two control steps. Both hold depths are checked
   against the same target image anchor. With unit drift budget `beta=0.075`, hold depth `d` is allowed only when
   `relative_visual_drift <= d * beta`. Continuous action authority is scaled by `1 / d`; the gripper state is unchanged.

The first two components preserve strict target-token verification. PacedHarmonic executes bounded target-free actions and
must therefore be evaluated jointly by task success rate and latency.

## Results

LIBERO evaluation uses one RTX 4090, BF16, batch size 1, seed 7, 50 trials for each of 10 tasks, and the same timing scope
for every local baseline. Each entry is **success rate / speedup over the suite-specific OpenVLA AR baseline**.

| Method | Goal | Spatial | Object | LIBERO-10 | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenVLA AR | 74.2 / 1.00x | 87.0 / 1.00x | 88.4 / 1.00x | 51.4 / 1.00x | 75.3 / 1.00x |
| SpecVLA strict | 76.8 / 1.02x | 85.0 / 1.00x | 87.6 / 1.10x | 54.4 / 0.99x | 76.0 / 1.03x |
| SpecVLA relaxed | 73.4 / 1.29x | 86.2 / 1.15x | 85.0 / 1.31x | 49.8 / 1.11x | 73.6 / 1.21x |
| Parallel Draft strict | 79.2 / 1.14x | 85.8 / 1.15x | 88.2 / 1.18x | 52.8 / 1.18x | 76.5 / 1.16x |
| Parallel Draft + VTPF strict | 77.6 / 1.30x | 86.4 / 1.17x | 87.2 / 1.19x | 50.0 / 1.29x | 75.3 / 1.24x |
| **Spec²** | **74.8 / 3.26x** | **83.4 / 2.36x** | **84.8 / 2.36x** | **50.8 / 3.02x** | **73.5 / 2.75x** |

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
