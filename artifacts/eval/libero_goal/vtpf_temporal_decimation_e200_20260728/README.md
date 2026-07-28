# VTPF-TD relaxed 研发证据（2026-07-28）

该目录固化 Goal、golden epoch 200 上的目标锚定时序降采样（VTPF-TD）原始结果。实现基于仓库
`34b09a2` 之后的同一次研发提交；checkpoint 为：

```text
ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2/epoch_200_step_089600
```

共同配置：RTX 4090、seed 7、Action-RNN off、tree off、exact-token fallback、`max_consecutive=1`。
任何保持帧之后都强制执行 target 关键帧；保持帧不会增加 target-verified history。

## Fast 确认（5 trials/task，共 50 条轨迹）

`summary.json`、`timing.json`、`eval.txt` 是 VTPF-TD-Fast 确认结果。该次运行的旧 launcher 仍计算了未参与
决策的图像签名；后续 1x10 优化复测证明去掉它不会改变动作轨迹。

| 指标 | 数值 |
| --- | ---: |
| 成功数 / 轨迹数 | 36 / 50 |
| SR | 0.720 |
| full-suite mean | 0.076206s |
| last-task mean | 0.079781s |
| paper AR mean | 0.182718s |
| last-task AR-relative Speedup | 2.290x |
| Length / corrected verified avg accept | 3.473 / 0.987 |
| prefill bypass | 4,225 |
| timing steps | 8,432 |

Paper AR 的正式 SR 为 0.742，点估计相差 -2.2 个百分点；相同 seed 下 golden VTPF strict 的前 5 个
初始状态为 41/50=0.82，差值为 -10 个百分点。前者不是 50 条轨迹的配对比较，后者才是同状态小样本；
两种参照都必须披露。正式论文结论仍需 50 trials/task（500 episodes）。

## Guard 确认（3 trials/task，共 30 条轨迹）

`guard_p03_k1_3x10_*` 使用 processor 图像的 `16x16` 池化相对 L2 门：阈值 `0.03`、target history 至少 1、
最多保持 1 帧。

| 指标 | 数值 |
| --- | ---: |
| 成功数 / 轨迹数 | 24 / 30 |
| SR | 0.800 |
| full-suite / last-task mean | 0.112370s / 0.117607s |
| last-task AR-relative Speedup | 1.554x |
| Length / corrected verified avg accept | 2.786 / 1.139 |
| prefill bypass | 1,311 |

相同初始状态的 golden VTPF strict 为 23/30。该结果只支持“当前小样本未观察到 SR 退化”，不支持 Guard
提高成功率的结论。

## 去视觉门复测（1 trial/task，共 10 条轨迹）

`fast_no_pixel_guard_1x10_*` 完全不构造图像签名，固定执行 `target -> hold -> target`。它与旧 Fast pilot
逐项一致：8/10、1,518 steps、761 bypass、`Length=3.399745`；mean 为 `0.077829s`。这证明决策轨迹不依赖
被移除的图像计算，主要收益确实来自整次 target prefill 的省略。

## 复现命令

```bash
# Fast
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_temporal_decimation_goal_eval.sh

# Guard
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_guarded_bypass_goal_eval.sh
```

`Length=7` 的 bypass 块表示一次调用推进完整动作，不表示 target 接受了 7 个 draft token。本目录的原始
summary 生成于统计修复前，`avg_accept_length` 曾把 bypass 误记为 7；表中 corrected 值按
`verified_accepted_tokens / num_blocks` 重算。当前代码已经把 bypass 的 `accept_lengths/accepted_tokens`
置零，同时保留 `progress_length=7`，并用 `temporal_prefill_bypassed_actions` 单独报告。
