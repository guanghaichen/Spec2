# SpecVLA-DFLASH

> **Repository role.** This is an experimental fork of
> [PineTreeWss/SpecVLA](https://github.com/PineTreeWss/SpecVLA), itself built on
> [OpenVLA](https://github.com/openvla/openvla).  It keeps the OpenVLA LIBERO
> action interface and the Spec-VLA verification setting, while replacing the
> autoregressive EAGLE-style draft with a DFlash-inspired, block-parallel draft
> model.
>
> **Status on 2026-06-26.** This is an active research codebase, not a released
> reproduction.  The current DFLASH design, data format, and commands below are
> the source of truth for new experiments.  Do not report a speedup or policy
> success improvement until it has been measured in the LIBERO simulator.

## Read This First: Project Map

The research question is deliberately narrow:

> Can an OpenVLA policy on LIBERO-Goal obtain useful speculative acceptance with
> a lightweight **non-autoregressive block draft**, conditioned on target-model
> multi-layer hidden states, while target-model verification preserves the
> output policy?

The main code paths are:

| Role | Current file | Notes |
| --- | --- | --- |
| Offline data generation | `openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py` | Runs OpenVLA greedily on RLDS demonstrations and saves full prefix/action hidden context. |
| Offline DFLASH training | `openvla/specdecoding/train-scripts/train_dflash_libero_goal.py` | Dataset loading, multi-anchor supervision, losses, checkpoints, SwanLab, and DDP. |
| DFLASH architecture | `openvla/specdecoding/model/dflash.py` | Context projection, action-dimension embeddings, non-causal block attention, RoPE. |
| Online draft and verification | `openvla/prismatic/extern/hf/modeling_speculation.py` | Loads a DFLASH checkpoint, drafts a block, then accepts/corrects it using OpenVLA. |
| LIBERO DFLASH evaluation | `openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py` | Executes rollouts and writes success/time/acceptance statistics. |

`openvla/specdecoding/train-scripts/train_deepspeed_libero_goal.py` and the
original `run_libero_goal_Spec.py` remain the upstream Spec-VLA/EAGLE reference
path.  They are useful for comparison, but are not the current DFLASH training
entry points.

## Literature Baselines and Motivation

### Spec-VLA baseline

[Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed
Acceptance](https://aclanthology.org/2025.emnlp-main.1367.pdf) (EMNLP 2025)
adapts speculative decoding to OpenVLA.  Its draft generator is
autoregressive: target-model prefill hidden states and prior action tokens are
used to predict the next action token repeatedly.  It verifies draft tokens
with OpenVLA and introduces relaxed acceptance based on action-token distance.

The following are **the paper's Table 1 numbers**, not results reproduced by
this repository.  `Length` is the average number of tokens generated per
forward pass and speed is relative to autoregressive OpenVLA.

| LIBERO suite | OpenVLA AR success rate | Spec-VLA success / Length / speedup | Spec-VLA relaxed success / Length / speedup |
| --- | ---: | --- | --- |
| Goal | 78.0% | 74.2% / 2.04 / 1.09x | 74.4% / 2.94 / 1.42x |
| Object | 89.0% | 89.0% / 1.75 / 1.15x | 85.0% / 2.38 / 1.38x |
| Spatial | 85.0% | 83.8% / 1.59 / 1.08x | 85.8% / 2.14 / 1.28x |
| Long | 52.0% | 50.8% / 1.67 / 1.13x | 55.0% / 2.10 / 1.22x |

The paper reports a 44% acceptance-length increase and 1.42x speedup at its
best reported setting, without a success-rate loss under that setting.  The
relevant lesson for this fork is not that these values should be expected here;
it is that acceptance length and simulator success must be measured together.

### DFlash inspiration

[DFlash: Block Diffusion for Flash Speculative
Decoding](https://arxiv.org/abs/2602.06036) (ICML 2026) replaces sequential
drafting with a lightweight block-diffusion draft.  It conditions on target
context features and predicts a whole token block in one forward pass; the
target model still verifies the proposal.  Its authors report over 6x lossless
LLM acceleration and up to 2.5x higher speedup than EAGLE-3 in their LLM
experiments.  Those figures are **not** transferable claims for OpenVLA.

This implementation takes only the core mechanism: a small non-causal block
draft conditioned on target hidden states.  It does not claim to reproduce the
original DFlash training pipeline.  The closest public implementation reference
is [SpecForge's `dflash.py`](https://github.com/sgl-project/SpecForge/blob/main/specforge/modeling/draft/dflash.py).

## Current DFLASH Design

### What is generated in parallel

OpenVLA has a 7-token action representation.  At an action anchor `a`, the
target model has already decoded the anchor token and produced its hidden state.
The DFLASH draft receives:

1. the complete OpenVLA prefill/prefix multi-layer hidden sequence;
2. multi-layer hidden states for all target-verified action tokens through the
   current anchor, including the anchor itself; and
3. a length-`q` input block `[token_a, MASK, ..., MASK]` with RoPE positions and
   a learned action-dimension embedding for each action position.

In **one** DFLASH forward pass, the draft emits hidden states/logits for the
`q <= 6` future positions `token_(a+1) ... token_(a+q)`.  Its block attention is
non-causal (`is_causal=False`): this is parallel block drafting, not an internal
autoregressive loop.  The target model then obtains posterior tokens for the
proposal in parallel.  It accepts the longest valid prefix and writes its own
posterior token at the first rejection, so partial acceptance and correction are
implemented.

The current evaluator uses `accept_threshold=9`, which means relaxed
token-distance acceptance rather than strict equality.  Set it to `None` only
when a strict-acceptance ablation is explicitly desired.  Simulator success and
accepted-length statistics must be reported with the threshold.

### Context, layers, and position invariants

The training and inference paths must agree on all of the following:

- **Full prefix context:** do not reduce context to only the final prefill
  hidden state.  The full prompt sequence is present in both the offline data
  and online draft context.
- **Anchor context:** the target model decodes each current anchor before the
  block is drafted.  This contributes the true anchor hidden state and makes
  the target-side action history available to later anchors.
- **Source layers:** offline data stores selected OpenVLA layers
  `[1, 8, 15, 22, 29]` plus the final layer separately.  The current
  `replace_22_with_final` variant constructs `[1, 8, 15, 29, final]` at load
  time, preserving the five-layer feature width without regenerating data.
- **RoPE positions:** prefix positions are `0 ... prefix_len-1`; action-context
  and block positions follow immediately after the prefix.  The same rule is
  used offline and online.
- **Action identity:** `action_dim_embed` identifies the seven distinct action
  dimensions.  It is additional learned information; it does not replace or
  modify RoPE.

The checkpoint's `dflash_config.json` is read at inference time and overrides
the evaluator defaults for block size, draft depth, target layers, anchor-hidden
mode, mask token, and selected-hidden variant.  Always point evaluation at a
checkpoint directory that contains this file.

### Current loss and training policy

The current pure-training recipe intentionally uses hidden-state distillation,
not token-level CE:

```text
total = 1.0 * hidden_loss + 0.05 * cosine_hidden_loss
soft_w = 0
anchor_consistency_w = 0
```

Token accuracy is retained as a diagnostic metric, but hard CE is not part of
the optimization objective.  Earlier token soft-distribution experiments showed
early validation deterioration in this offline block setting; they are retained
as optional code (`--soft_w`, `--soft_temperature`) rather than the current
default.  Hidden-context noise is `0.05` in the checked-in recipe.  Per-anchor
and per-position diagnostic metrics are logged to SwanLab and local JSONL.

The intended long-run control signal for the current recipe is simulator
behavior, not early stopping on an offline validation split.  Accordingly the
pure-training launcher uses `--val_split 0`, disables validation/early stopping
by construction, and saves a checkpoint every 10 epochs.

## Offline Data: Current 4090 Artifact

The data generator uses `openvla/modified_libero_rlds` data through the
`libero_goal_no_noops` split.  For every RLDS sample it runs greedy OpenVLA,
then writes one `data_*.ckpt` tensor dictionary only when the returned action
hidden-state sequence and the 7 action tokens are structurally compatible.

The current 4090 directory is:

```text
/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset
```

Audited on 2026-06-26: `419G`, `28,639` saved `.ckpt` samples.  The generation
log reported `52,042` enumerated samples and `28,639` valid samples.  The large
size is expected because each valid sample carries complete prefix hidden
sequences and selected/final hidden states for action decoding, rather than only
discrete action tokens.

The essential per-sample fields are:

```text
input_ids                 tokenized vision-language prompt
pixel_values              preprocessed image tensor
loss_mask                 prompt attention mask
predicted_tokens          greedy OpenVLA action-token sequence (7 tokens)
hidden_state.prompt_selected  complete prefix, concatenated selected layers
hidden_state.prompt_last      complete prefix, final layer
hidden_state.action_selected  action hidden states, selected layers
hidden_state.action_last      action hidden states, final layer
dflash_data_format        full_prefix_plus_action_hidden_v4
```

Do not mix files generated by older data formats with this directory.  The
trainer checks expected fields/shapes, but an explicit dataset version and count
check is still required before an experiment.

## Experiment Record

This section is intentionally concise and chronological.  It records design
decisions, not a claim that an experimental question has been settled.

1. **Initial migration:** a DFlash-style draft was inserted into the Spec-VLA /
   OpenVLA speculative path.  Early drafts that used insufficient context had
   essentially no useful acceptance.
2. **Context correction:** the data and runtime were changed to preserve full
   prefill hidden sequences and target-verified action history.  The current
   `include_anchor_hidden` path decodes the anchor with the target before each
   parallel tail proposal.
3. **Offline supervision correction:** multi-anchor supervision, action-
   dimension embeddings, position balancing, hidden loss, cosine loss, and
   diagnostics were added.  The earlier hard-token-CE objective was removed
   from the active recipe.
4. **Soft-loss and consistency ablations:** soft token-distribution and
   cross-anchor consistency experiments were run as diagnostics.  They are
   available behind flags, but neither is in the current pure-training recipe.
5. **Current experiment:** train a 1-layer draft from the full 28,639-sample
   dataset with five context features `[1, 8, 15, 29, final]`, `soft_w=0`,
   `anchor_consistency_w=0`, no offline validation split, then compare
   checkpoints by LIBERO simulator success, acceptance length, hit rate, and
   wall-clock time.

Known limitations to keep visible:

- The block draft has non-causal intra-block inputs, so future slots do not
  receive ground-truth causal prefixes during a single draft forward.  This is
  the central modeling risk of the parallel design, not a bug to hide with an
  offline token metric.
- A low hidden loss alone does not establish useful speculative speedup.  The
  online acceptance distribution and target-call count determine speed.
- Relaxed acceptance can preserve practical actions while diverging from strict
  token equality.  It must be ablated and reported honestly.

## 4090 Reproduction Commands

The active development machine is **4090**.  Activate its environment and work
from the repository root:

```bash
ssh 4090
source /home/pc/miniconda3/bin/activate specvla
cd /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main
export PYTHONPATH="$PWD"
```

Paths can be overridden through `VLA_PATH`, `LIBERO_RLDS_ROOT`, and
`DFLASH_DATA_OUTDIR`; the generator also has explicit `--vla_path`,
`--data_root_dir`, and `--outdir` arguments.  This avoids accidental Hugging
Face downloads: local processor loading uses `trust_remote_code=False`.

### 1. Generate raw DFLASH data

Entry point: `openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py`

```bash
CUDA_VISIBLE_DEVICES=0 python openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py \
  --gpu_index 0 \
  --vla_path /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal \
  --data_root_dir /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/dataset/modified_libero_rlds \
  --dataset_name libero_goal_no_noops \
  --outdir /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset
```

Before training, confirm both size and count:

```bash
du -sh /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset
find /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/dflash_goal_dataset \
  -maxdepth 1 -name 'data_*.ckpt' | wc -l
```

### 2. Train the current 1-layer pure-training recipe

Recommended launcher:

```text
openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_puretrain_4gpu.sh
```

On 4090, launch it with four selected GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_puretrain_4gpu.sh
```

This launcher uses `torchrun --nproc_per_node 4`, one DFLASH layer, selected
hidden variant `replace_22_with_final`, batch size 8 per GPU (effective batch
32), 200 epochs, warmup 2000 optimizer steps, `save_every=10`, `val_split=0`,
and SwanLab's normal configured mode.  The output directory is printed by the
launcher.  Its important artifacts are:

```text
<output>/epoch_XXX_step_XXXXXX/pytorch_model.bin
<output>/epoch_XXX_step_XXXXXX/training_state.pt
<output>/epoch_XXX_step_XXXXXX/dflash_config.json
<output>/latest_checkpoint.txt
<output>/metrics.jsonl
<output>/swanlog/
```

`latest_checkpoint.txt` points to the checkpoint directory to use for
evaluation.  To continue an interrupted run, invoke the Python trainer with
`--resume_from_checkpoint latest` and the same `--output_dir`; do not silently
change world size, effective batch, or scheduler settings when comparing runs.

Two older diagnostic launchers are kept for controlled ablations, not as the
default recipe:

```text
openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_baseline.sh
openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_consistency.sh
```

### 3. Evaluate in LIBERO-Goal

Entry point: `openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py`

```bash
OUT=/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_puretrain_4gpu
SPEC_CKPT="$(cat "$OUT/latest_checkpoint.txt")"
VLA_PATH=/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
python openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py \
  --pretrained_checkpoint "$VLA_PATH" \
  --spec_checkpoint "$SPEC_CKPT" \
  --draft_backend dflash \
  --use_spec True \
  --parallel_draft False \
  --task_suite_name libero_goal \
  --num_trials_per_task 50 \
  --center_crop True \
  --accept_threshold 9 \
  --local_log_dir /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/eval_logs \
  --run_id_note dflash-finalhidden-latest \
  --use_wandb False
```

The evaluator logs task success, action-generation timing, average accepted
length, total hit rate, and per-position hit/reject rate.  Compare each DFLASH
checkpoint with the autoregressive reference under the same LIBERO task suite,
trial count, seed, model checkpoint, and crop setting.

## Git and Server Workflow

The development workflow is intentionally one-way:

```text
4090 (primary edit/test machine) -> commit -> GitHub main -> 3090 (pull/sync only when requested)
```

Therefore:

1. Make and verify code/documentation changes on 4090.
2. Commit only files relevant to that change and push `main` to
   [guanghaichen/SpecVLA-DFLASH](https://github.com/guanghaichen/SpecVLA-DFLASH).
3. Do not copy uncommitted 4090 changes to 3090.
4. Only after GitHub contains the intended commit, synchronize 3090 when a
   training or generation task requires it.

Before every experiment, record the Git commit, data directory/count, launcher,
checkpoint path, selected-hidden variant, acceptance threshold, and evaluation
seed.  These six lines are the minimum metadata needed to make an outcome
comparable rather than anecdotal.

## Environment Notes

- Python 3.10, PyTorch 2.2.0 with CUDA 12.1, and LIBERO 0.1.0 are the original
  tested environment notes inherited from Spec-VLA.
- Install the repository package with `cd openvla && pip install -e .` after
  dependency setup.
- The model and RLDS data are intentionally local paths on 4090.  Do not allow
  a training/data-generation run to fall back to an unintended remote download.
- This repository may contain historical scripts and comments from Spec-VLA.
  For DFLASH behavior, use the files in the project map above as authoritative.

## References

```bibtex
@inproceedings{wang2025specvla,
  title={Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance},
  author={Wang, Songsheng and others},
  booktitle={EMNLP},
  year={2025}
}

@inproceedings{chen2026dflash,
  title={DFlash: Block Diffusion for Flash Speculative Decoding},
  author={Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle={ICML},
  year={2026}
}
```
