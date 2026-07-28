# 2026-07-20 至 2026-07-28 正式评测归档

本目录是 4090 单卡 LIBERO 正式评测的不可覆盖证据包。它保留用户整理后的原始目录、逐 episode 文本日志、
逐动作 timing 和机器可读 summary；`results.csv` 是从原始 JSON 提炼的索引，不替代原始文件。
`baseline/main_table_specvla_baselines.csv` 是整理目录前生成的原始表，内部绝对 `summary_path` 仍指向旧位置；
数值有效，但定位文件应使用本目录结构或新的 `results.csv`。该原始表刻意不改写，以保持证据原貌。

## 目录

```text
baseline/
  openvla_ar/          四个 suite 的 paper-wrapped AR
  specvla_strict/      四个 suite 的 SpecVLA strict
  specvla_relaxed/     四个 suite 的 SpecVLA relaxed
dflash_strict/
  复杂版Draft/                     Golden e200 + Action-RNN + DDTree strict
  复杂版Draft+VTPF/                Golden e200 + Action-RNN + VTPF strict（正式）
  复杂版去掉RNN的Draft+VTPF/       Golden e200 + VTPF、RNN-off（100-episode pilot）
dflash_relaxed/
  复杂版去掉RNN的Draft+VTPF-TD/    Golden e200 + VTPF-TD-Fast（正式）
checkpoint/dflash_config.json       Golden e200 训练配置快照
launchers/                          归档时的训练/评测入口快照
environment.txt                     模型、代码、硬件与依赖身份
results.csv                         统一结果索引
SHA256SUMS                          本目录所有证据文件的完整性校验
```

## 核心结果

所有 Speedup 都用同机、同 Goal、同计时协议的 paper-wrapped AR
`0.18271759773649937 s/action` 作分母。

| 方法 | 性质 | Episodes | SR | last-task mean | Speedup | Length | avg accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper-wrapped AR | 精确基线 | 500 | 0.742 | 0.182718s | 1.000x | - | - |
| SpecVLA strict | target-verified | 500 | 0.768 | 0.178764s | 1.022x | 1.631 | 0.631 |
| SpecVLA relaxed (`r=9`) | 近似接受 | 500 | 0.734 | 0.141228s | 1.294x | 2.361 | 1.361 |
| Golden + DDTree strict | target-verified | 500 | 0.776 | 0.172794s | 1.057x | 2.221 | 1.024 |
| **Golden + VTPF strict** | **target-verified** | **500** | **0.790** | **0.142036s** | **1.286x** | **2.422** | **1.466** |
| Golden + VTPF、RNN-off | target-verified pilot | 100 | 0.770 | 0.133986s | 1.364x | 2.593 | 1.675 |
| **Golden + VTPF-TD-Fast** | **单步时序近似** | **500** | **0.754** | **0.070050s** | **2.608x** | **3.653** | **1.119** |

VTPF-TD 相比同一权重、同一 seed 的 VTPF：SR 下降 `3.6` 个百分点，动作延迟下降 `50.7%`，系统速度提高
`2.028x`。配对结果为共同成功 319、仅 VTPF 成功 76、仅 TD 成功 58、共同失败 47；McNemar 精确检验
`p=0.142`，单 seed 下不能声称成功率差异具有统计显著性。

## Golden Draft 身份

VTPF 与 VTPF-TD 都使用同一个复杂版 Golden checkpoint：

```text
训练配方：run_dflash_train.sh joint
训练数据：dflash_goal_dataset_envfix_20260714_packed_v2.h5
数据语义：28,501 samples；full_prefix_plus_action_hidden_v4；layers=[1,9,16,24,31]
训练：1-layer Draft，block=7，global batch=64，200 epochs
权重：ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2/epoch_200_step_089600
权重 SHA-256：e10127daa030ab5d7fbe639090078d3380c91a6d98b9302b31cf4d2f9dc5dac8
配置 SHA-256：0b9026527183971e68c0199b1a9067dfa34a1307fc7863b8c52c2805e4915a18
冻结复现 tag：golden-vtpf-e200-20260726（ea7bcbb）
```

完整 loss、层选择、优化器和课程参数见 `checkpoint/dflash_config.json`。模型二进制不提交 GitHub；SHA 用于
核对外部 checkpoint 是否与本归档一致。

## VTPF 正式运行

VTPF 把上一条已被 target 连续确认至少 3 次的动作 `c0..c5` 附到当前多模态 prompt 后。当前本来必做的
target prefill 同时给出 `c0..c6` 的 posterior；只提交连续精确命中的前缀，首个错误由 target 纠正，错误
候选的 KV 被裁掉，余下位置回退 DFlash。它没有跳过 target，也没有放宽 token 接受条件。

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_temporal_cascade_goal_eval.sh prefill
```

运行身份：机制代码 `d60c555`；冻结复现 tag `ea7bcbb`；RNN 开；`tree=off`；`acceptance=token`；
`threshold=0`；`target_logits=action_only`；`verify_skip_mode=route`；实际跳过数为 0。

## VTPF-TD-Fast 正式运行

VTPF-TD 在正常 target 关键帧后，最多用一次最近的 target-verified 七维动作作为保持帧。保持帧完全跳过当前
OpenVLA prefill、Draft 与 target verify；它不增加 verified history，下一控制步必须重新调用 target，形成
`target -> hold -> target`。这不是 strict speculative acceptance，而是有界的时序降采样。

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_temporal_decimation_goal_eval.sh
```

运行身份：实现 commit `5626cb6`，正式证据 commit `208482a`；RNN 关；`tree=off`；动作组关；fallback
仍为 `token/threshold=0`；视觉门关；`max_consecutive=1`。最后计时 task 共 11,165 个动作状态、5,575 次
prefill bypass，跳过率约 50.0%。保持帧计入执行推进 `Length=7`，但不计为 target 接受 token，因此
`Length` 与 `avg_accept_length` 必须分栏解释。

## 统一环境

```text
GPU=NVIDIA GeForce RTX 4090 (24 GB)
seed=7
num_trials_per_task=50
sync_cuda_timing=False
timing_scope=last_task
Python=3.10.20
torch=2.2.0+cu121
transformers=4.40.1
mujoco=3.3.2
robosuite=1.4.1
numpy=1.26.4
LIBERO=8f1084e3132a39270c3a13ebe37270a43ece2a01
```

更完整的模型 SHA 与环境身份见 `environment.txt`。校验归档：

```bash
cd artifacts/eval/curated_20260720_20260728
sha256sum -c SHA256SUMS
```

## 解释边界

- `Length` 是每个 speculative/时序块平均推进 token 数；`avg_accept_length` 只统计经 target 验证接受的 draft
  token。VTPF-TD 的 hold 是执行推进，不是 target 接受。
- `timing.mean`、Length 与生成统计均按仓库固定协议解释；论文主表不得把不同 GPU、不同 suite 或不同计时
  范围混成 Speedup。
- 当前 DFlash checkpoint 只在 LIBERO-Goal 上训练，不能用它外推 Object、Spatial 或 Long。
- VTPF-TD 目前只有 Goal、seed 7 的正式 500-episode 结果；2.608x 是已复现实验点，不是跨 seed 泛化结论。
