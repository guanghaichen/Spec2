# SpecVLA-DFLASH

> **仓库定位。** 本仓库是基于
> [PineTreeWss/SpecVLA](https://github.com/PineTreeWss/SpecVLA) 的实验性分支；
> SpecVLA 本身构建在 [OpenVLA](https://github.com/openvla/openvla) 之上。
> 当前代码保留 OpenVLA 的 LIBERO 动作接口和 SpecVLA 的 target-model 校验框架，
> 但把原本 EAGLE 风格的自回归 draft 替换为受 DFlash 启发的块并行 draft。
>
> **状态记录。** 这是一个正在迭代中的研究代码库，还不是公开复现实验包。
> 当前 DFLASH 设计、数据格式和 README 中的命令是新实验的主要上下文来源。
> 在 LIBERO 仿真中实际测量之前，不要声称已经获得稳定加速或成功率提升。

## 先看这里：项目地图

当前研究问题可以概括为：

> 在 LIBERO-Goal 上，OpenVLA 能否使用一个轻量的、非自回归的块并行 draft，
> 在 target-model 多层 hidden state 条件下获得可用的 speculative acceptance，
> 同时仍由 target model 校验来保持输出策略可靠性？

主要代码路径如下：

| 作用 | 当前文件 | 说明 |
| --- | --- | --- |
| 离线数据生成 | `openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py` | 在 RLDS demonstration 上贪心运行 OpenVLA，并保存完整 prefix/action hidden context。 |
| 离线 DFLASH 训练 | `openvla/specdecoding/train-scripts/train_dflash_libero_goal.py` | Dataset、multi-anchor 监督、loss、checkpoint、SwanLab、DDP。 |
| DFLASH 模型结构 | `openvla/specdecoding/model/dflash.py` | Context projection、action-dimension embedding、非因果 block attention、RoPE。 |
| 在线 draft 和 target 校验 | `openvla/prismatic/extern/hf/modeling_speculation.py` | 加载 DFLASH checkpoint，一次 draft 一个 block，再用 OpenVLA 接受/纠正。 |
| LIBERO 推理评测 | `openvla/experiments/robot/libero/run_libero_goal_AR.py`、`run_libero_goal_Spec.py`、`run_libero_goal_Spec_Relaxed.py` | 执行 rollout，并记录成功率、耗时、Length、acceptance 统计。 |

`openvla/specdecoding/train-scripts/train_deepspeed_libero_goal.py` 和原始
`run_libero_goal_Spec.py` 仍然是上游 SpecVLA/EAGLE 的参考路径。它们适合用来做 baseline 对比，
但不是当前 DFLASH 训练的入口。

## 文献 baseline 和动机

### SpecVLA baseline

[Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance](https://aclanthology.org/2025.emnlp-main.1367.pdf)
发表于 EMNLP 2025。它把 speculative decoding 迁移到 OpenVLA 上：draft generator 是自回归的，
使用 target model 的 prefill hidden state 和历史 action token 反复预测下一个 action token；
然后用 OpenVLA 校验 draft token，并提出基于 action-token distance 的 relaxed acceptance。

下面是 **SpecVLA 论文 Table 1 的数值**，不是本仓库复现结果。`Length` 表示每次 forward
平均生成 token 数，speedup 是相对 OpenVLA 自回归推理的速度。

| LIBERO suite | OpenVLA AR 成功率 | SpecVLA 成功率 / Length / speedup | SpecVLA relaxed 成功率 / Length / speedup |
| --- | ---: | --- | --- |
| Goal | 78.0% | 74.2% / 2.04 / 1.09x | 74.4% / 2.94 / 1.42x |
| Object | 89.0% | 89.0% / 1.75 / 1.15x | 85.0% / 2.38 / 1.38x |
| Spatial | 85.0% | 83.8% / 1.59 / 1.08x | 85.8% / 2.14 / 1.28x |
| Long | 52.0% | 50.8% / 1.67 / 1.13x | 55.0% / 2.10 / 1.22x |

论文报告的最好设置中，acceptance length 提升 44%，速度达到 1.42x，且成功率没有下降。
对本项目最重要的启发不是“我们也应该直接得到这些数值”，而是：**acceptance length、推理耗时、
仿真成功率必须放在一起报告**。

### DFlash 灵感来源

[DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
发表于 ICML 2026。它用轻量 block-diffusion draft 替代逐 token draft，条件是 target context feature，
一次 forward 预测整块 token，然后仍由 target model 校验 proposal。论文在 LLM 实验中报告了
超过 6x 的 lossless acceleration，并比 EAGLE-3 最高快 2.5x。

这些 LLM 数值 **不能直接迁移为 OpenVLA 结论**。本仓库只迁移其核心机制：一个小型的、
非因果 block draft，条件来自 target hidden state。本仓库不声称复现 DFlash 原论文完整训练流程。
最近的公开代码参考是 [SpecForge 的 `dflash.py`](https://github.com/sgl-project/SpecForge/blob/main/specforge/modeling/draft/dflash.py)。

## 当前 DFLASH 设计

### 并行生成的到底是什么

OpenVLA 的动作由 7 个 action token 表示。给定 action anchor `a` 时，target model 已经解码出当前
anchor token，并产生了该位置 hidden state。DFLASH draft 接收：

1. 完整 OpenVLA prefill/prefix 多层 hidden 序列；
2. 从第一个 action token 到当前 anchor 为止，所有已经被 target 验证过的 action token 多层 hidden；
3. 一个长度为 `q` 的输入块 `[token_a, MASK, ..., MASK]`，带 RoPE position 和每个 action 维度的
   learned action-dimension embedding。

在 **一次** DFLASH forward 中，draft 会为未来 `q <= 6` 个位置
`token_(a+1) ... token_(a+q)` 输出 hidden/logits。它的 block attention 是非因果的
（`is_causal=False`），所以这是块并行 draft，不是内部自回归循环。随后 target model 并行得到
proposal 的 posterior token，接受最长合法前缀；如果遇到拒绝位置，就写入 target model 自己的
posterior token，因此支持 partial acceptance 和 partial correction。

当前主评测使用 `accept_threshold=9`，也就是基于 token distance 的 relaxed acceptance，而不是严格相等。
只有在做 strict-acceptance 消融时才把阈值设成 `0` 或显式 strict 脚本。报告实验时必须同时写清楚阈值、
仿真成功率和 accepted length。

### Context、层选择和 position invariant

训练和推理必须保持下面这些不变量一致：

- **完整 prefix context：** 不要把 context 压缩成只有最后一个 prefill hidden。离线数据和在线推理都保留完整 prompt sequence。
- **Anchor context：** 每次 draft block 前，target model 会先解码当前 anchor。这一步提供真实 anchor hidden，
  也让后续 anchor 能看到 target-side action history。
- **Source layers：** 离线数据保存 OpenVLA 选定层 `[1, 8, 15, 22, 29]`，并额外保存 final layer。
  当前 `replace_22_with_final` 变体会在加载时构造 `[1, 8, 15, 29, final]`，
  保持五层特征宽度，不需要重新生成数据。
- **RoPE positions：** prefix positions 是 `0 ... prefix_len-1`；action context 和 block positions 紧接在 prefix 后面。
  训练和推理必须使用同一规则。
- **Action identity：** `action_dim_embed` 标识 7 个不同动作维度。这是额外学习到的信息，不替代 RoPE，也不修改 RoPE。

推理时会读取 checkpoint 目录里的 `dflash_config.json`，并用它覆盖 evaluator 默认值，包括 block size、
draft depth、target layers、anchor-hidden mode、mask token 和 selected-hidden variant。评测 DFLASH 时，
必须让 `SPEC_CKPT` 指向包含 `dflash_config.json` 的 checkpoint 目录。

### 当前方案：Action-RNN Prefix Survival

当前推荐训练入口已经更新为：

```text
openvla/specdecoding/train-scripts/run_dflash_action_rnn_prefix_3layer_4gpu.sh
```

上一版 **Markov-ACD** 仍完整保留用于历史 checkpoint 和消融，但不再是下一轮主训练。原因是其推理路径会对
每个 slot 重复执行 hidden residual、完整 `lm_head` 和全词表 logit bias；即使 Length 提高，这些串行大词表
投影也可能吃掉投机解码节省的时间。新版结构统一称为 **Action-RNN Prefix Survival**：

1. **三层并行 DFlash 主干。** 完整 prompt/action target hidden 仍只经过一次块并行 draft forward，模型深度
   从 1 层增至 3 层，提高 base hidden 对视觉、指令和动作结构的表达能力。
2. **动作子词表前缀状态头。** 只取冻结 `lm_head` 中对应 256 个 OpenVLA 动作 token 的权重行，对整块做一次
   小投影；随后递推一个 256 维前缀状态。第 i 步输入当前并行 hidden、前一个动作 token、动作维度 embedding
   和上一状态，输出 256 类动作 bias。训练时使用 teacher 真实前缀，推理时使用刚刚生成的 draft 前缀。
3. **跨 Anchor Logit Distillation。** 对同一目标位置，短前缀弱路径的最终动作分布追一步强路径，并对 strong
   path `stop_gradient`。这保留了本项目“不同 anchor 是同一目标的强弱因果视角”的核心设计。
4. **Expected Prefix Survival。** 不再靠指定 p2/p3 等位置的 boost。先用 teacher/student 动作分布的
   `1 - total_variation` 估计逐位置可接受概率，再对块内概率做累乘，直接优化连续可接受前缀。
   因为前面一处错误会同时破坏后面所有前缀，早期位置会自然得到更重要的梯度。
5. **Confidence Truncation。** 小置信度头学习逐 slot 可接受概率。推理可在低置信位置前缩短 proposal，减少
   明知大概率会失败的验证开销。默认阈值为 `0.0`，先测无截断基线，再单独扫阈值，避免未经校准就改变结果。

新版明确关闭旧 `causal_residual_type hidden` 和 `logit_markov_type bias`，因此旧头不会和新头叠加计算。
`position_balance=True` 只抵消 multi-anchor 中后部绝对位置被重复监督更多次的统计偏差；`slot_decay=1.0`，
位置重要性完全交给 Prefix Survival，而不是手工位置补丁。

### 当前 Loss 和指标

三层 Action-RNN 默认总损失为：

```text
total = 0.30 * hidden_smooth_l1
      + 0.02 * hidden_cosine
      + 0.10 * action_token_ce
      + 0.90 * action_distribution_l1
      + 0.50 * prefix_survival
      + 0.10 * action_confidence_bce
      + 0.10 * cross_anchor_logit_distillation
```

`action_distribution_l1` 只比较 256 类动作分布，不再让损失被 3 万多个无关语言 token 稀释；
`prefix_survival` 与最终 acceptance length 的目标最接近；hidden 两项退为稳定表征的辅助监督。
SwanLab 和 `metrics.jsonl` 新增记录：

```text
action_token_ce_loss / component
action_distill_l1_loss / component
prefix_survival_loss / component
action_confidence_loss / component
action_accept_probability_proxy
expected_prefix_length
confidence_mae
```

在线 summary JSON 还会记录 `conditional_prefix`：`p_k` 表示已经连续接受前 k-1 个 token 后，第 k 个仍被
接受的条件概率。它比独立 `per_position` 命中率更能解释真实 Length。长期控制信号仍是 4090 上的
`SR / Length / Speedup`；四卡训练保持 `--val_split 0`，每 10 epoch 保存 checkpoint。

## 实验演进与诊断记录

这一节是给后续自己和 AI 快速接上下文用的研究日志。指标来自 3090 上各训练目录的
`metrics.jsonl/run_config.json/latest_checkpoint.txt`，读取时间更新到 2026-07-11。这里的离线 token accuracy
只说明 draft 在 teacher-forced 训练视角下是否学到模式，最终仍要用 4090 串行 LIBERO rollout 的
`SR / Length / Speedup` 判断。

### 训练阶段总览

| 阶段 | 训练目录 | 主要改动 | 训练状态 | `train/accuracy` 末值 / 最好值 | 关键诊断 |
| --- | --- | --- | --- | ---: | --- |
| Pure hidden baseline | `ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_puretrain_4gpu` | 完整 prefix/action hidden、1-layer draft、`[1,8,15,29,final]`、hidden+cos，`soft_w=0` | 200 epoch 完整训练 | 0.807 / 0.830 | p1/p6 很强，但 anchor0 的 p2-p5 明显弱，说明块并行远 slot 缺因果信息。 |
| Residual-CAD weak-path | `ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_residual_cad_weakpath_b16_4gpu` | 增加 hidden residual head、Hidden-level CAD、refined hidden supervision，b16 四卡 | 200 epoch 完整训练 | 0.799 / 0.830 | anchor0->p2 从 0.503 抬到 0.671，但 residual 后 `accuracy` 没超过 `base_accuracy`，说明残差头训练信号还不够强。 |
| Markov-ACD 诊断短跑 | `ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_tokence_soft01_b16_4gpu` | 增加 logits-level Markov bias、Logit-level CAD、residual token CE、`soft_w=0.10` | 跑到 epoch 18 手动/中途停止 | 0.897 / 0.905 | p2-p5 离线准确率明显跃升，证明“跨 anchor 蒸馏 + token/logit 信号”方向有效；但该目录仍是旧短跑诊断版，run_config 里有旧 `weak_path_loss_boost` 且 `start_index=1`。 |
| Clean Markov-ACD 完整长跑 | `ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu` | 删除所有手工位置加权；`causal_residual_start_index=0`；`slot_decay=0.90`；低权重 `soft_w=0.10`；Markov-aware refinement + Hidden/Logit CAD + residual token CE | 3090 四卡 200 epoch 已完成 | 0.999 / 1.000；`base_accuracy` 末值 0.974 / 最好 0.987 | 离线 teacher-forced 视角几乎饱和：p1-p5 到 1.000，p6 约 0.996。最终是否有效必须看 4090 online Length/Speedup，建议优先测 epoch 100/150/200。 |
| Action-RNN Prefix Survival 3-layer | `ckpt_goal_dflash_action_rnn_prefix_3layer_b8x2_4gpu` | 三层并行主干、256 类动作前缀状态头、动作分布 L1、期望前缀存活、置信度头、Logit CAD | 代码就绪，待 3090 四卡训练 | - | 针对上一版离线近饱和但在线 p2-p5 和速度未同步提升的问题；同时移除每 slot 的完整词表投影开销。 |

### 2026-07-11 最新 Markov-ACD 200 epoch 结果

最新完整训练目录：

```text
/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu
```

训练配置要点：4 卡 3090，`batch_size=16/GPU`，有效 batch 64，`lr=5e-5`，`warmup_steps=1000`，`num_epochs=200`，`slot_decay=0.90`，`soft_w=0.10`，`causal_residual_start_index=0`。

| epoch | step | loss | soft_loss | hidden_loss | CAD loss | refined hidden | residual CE | logit CAD | acc | base_acc | p1 | p2 | p3 | p4 | p5 | p6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 440 | 3.228 | 14.468 | 1.015 | 0.012 | 1.029 | 4.156 | 0.030 | 0.536 | 0.534 | 0.350 | 0.127 | 0.466 | 0.333 | 0.500 | 0.905 |
| 10 | 4460 | 2.155 | 7.199 | 0.857 | 0.319 | 0.871 | 0.438 | 2.151 | 0.894 | 0.845 | 0.897 | 0.848 | 0.841 | 0.802 | 0.802 | 0.956 |
| 50 | 22340 | 1.706 | 5.890 | 0.749 | 0.323 | 0.753 | 0.184 | 0.707 | 0.963 | 0.934 | 0.974 | 0.932 | 0.955 | 0.931 | 0.944 | 0.983 |
| 100 | 44700 | 1.556 | 5.520 | 0.706 | 0.320 | 0.702 | 0.132 | 0.250 | 0.989 | 0.962 | 0.998 | 0.984 | 0.988 | 0.984 | 0.986 | 0.986 |
| 150 | 67040 | 1.514 | 5.435 | 0.694 | 0.310 | 0.688 | 0.112 | 0.114 | 0.998 | 0.973 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 |
| 200 | 89400 | 1.517 | 5.519 | 0.690 | 0.308 | 0.685 | 0.111 | 0.102 | 0.999 | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.996 |

诊断结论：训练集上的 Markov-ACD/CAD-head 信号很强，弱路径 p2-p5 已被明显拉起；但这是 offline teacher-forced 统计，不能直接等价为接收长度。下一步优先把 epoch 100/150/200 搬到 4090，跑 CAD-head strict/relaxed 的 online rollout。

### 2026-07-12 Goal 在线结果与新方案靶点

在 4090、`SYNC_CUDA_TIMING=False`、`TIMING_SCOPE=last_task` 下，epoch 200 的完整 Goal 结果如下。
Speedup 使用同轮 AR mean step time `0.161929s` 作为分子：

| 方法 | SR | mean step time | Speedup vs AR | Length | avg accept | online per-position hit |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| OpenVLA AR | 0.774 | 0.161929s | 1.000x | - | - | - |
| SpecVLA strict | 0.768 | 0.178630s | 0.907x | 1.631 | 0.631 | - |
| SpecVLA relaxed | 0.734 | 0.141256s | 1.146x | 2.361 | 1.361 | - |
| DFlash Markov-ACD strict | 0.788 | 0.182786s | 0.886x | 2.109 | 1.008 | 0.687/0.326/0.411/0.393/0.468/0.577 |
| DFlash Markov-ACD relaxed | 0.768 | 0.158210s | 1.024x | 2.607 | 1.479 | 0.819/0.446/0.528/0.511/0.535/0.560 |

这组结果说明两件事。第一，离线准确率接近 1.0 并没有转化成同等 online hit rate，teacher-forced
前缀与推理时自生成前缀之间仍有明显 exposure gap。第二，DFlash strict 的 Length 高于 SpecVLA strict，
却更慢；Length 只描述每个验证块推进多少 token，不包含 draft 自己的成本。旧 Markov-ACD 对每个 slot
重复完整 `lm_head`/全词表修正，额外串行开销足以抵消较长前缀。这正是三层 Action-RNN 新实验同时优化
“模型能力、连续前缀目标、推理头成本”的直接依据，而不是单纯继续堆离线 accuracy。

### 位置准确率演进

`position_i_acc` 是所有 anchor 路径上预测绝对 action 位置 `pi` 的平均准确率。它能看整体训练是否健康，
但不完全等价于真实推理里从 anchor0 开始的一次块预测。

| 阶段 | p1 | p2 | p3 | p4 | p5 | p6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure hidden baseline 末值 | 0.886 | 0.690 | 0.780 | 0.704 | 0.750 | 0.962 |
| Pure hidden baseline 最好值 | 0.969 | 0.781 | 0.854 | 0.789 | 0.831 | 1.000 |
| Residual-CAD weak-path 末值 | 0.846 | 0.728 | 0.764 | 0.691 | 0.751 | 0.942 |
| Markov-ACD 诊断短跑末值 | 0.820 | 0.819 | 0.899 | 0.868 | 0.877 | 0.972 |
| Markov-ACD 诊断短跑最好值 | 0.885 | 0.844 | 0.903 | 0.892 | 0.925 | 0.990 |

### Anchor0 弱路径演进

`anchor0 -> p1..p6` 最接近 DFLASH 在线第一块 proposal 的压力测试：输入只有
`[prompt hidden, t0, MASK, MASK, ...]`，后续 slot 很难直接看到真实因果前缀。这个表最能解释为什么要做
Markov-ACD。

| 阶段 | p1 | p2 | p3 | p4 | p5 | p6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure hidden baseline 末值 | 0.886 | 0.503 | 0.716 | 0.620 | 0.692 | 0.953 |
| Pure hidden baseline 最好值 | 0.969 | 0.656 | 0.844 | 0.781 | 0.813 | 1.000 |
| Residual-CAD weak-path 末值 | 0.846 | 0.671 | 0.717 | 0.647 | 0.718 | 0.936 |
| Residual-CAD weak-path 最好值 | 0.906 | 0.766 | 0.758 | 0.766 | 0.781 | 0.984 |
| Markov-ACD 诊断短跑末值 | 0.820 | 0.860 | 0.913 | 0.881 | 0.880 | 0.963 |
| Markov-ACD 诊断短跑最好值 | 0.885 | 0.917 | 0.925 | 0.901 | 0.938 | 0.973 |

最重要的变化是 anchor0 弱路径：p2 从 `0.503 -> 0.671 -> 0.860`，p3 从
`0.716 -> 0.717 -> 0.913`，p4 从 `0.620 -> 0.647 -> 0.881`，p5 从
`0.692 -> 0.718 -> 0.880`。这说明单纯 hidden 拟合不足以解决远 slot；加入 token/logit 级 Markov 信号和
跨 anchor 蒸馏后，弱路径确实更快追上强路径。Clean Markov-ACD 的 200 epoch 长跑最终离线接近饱和，
但 4090 在线结果没有得到同等提升；因此下一轮不再继续追求 teacher-forced 饱和，而是改攻 online exposure
gap、连续前缀目标和草稿头开销。

### 3090 在线 sanity check

2026-07-05 这组 Goal 评测是在 3090 上多实验并行跑的，适合判断成功率和大致趋势，但不适合作论文最终速度。
正式速度统一在 4090 单实验串行跑，并保持 `SYNC_CUDA_TIMING=False`、`TIMING_SCOPE=last_task`。

| 方法 | SR | mean step time | Speedup vs AR | Length | avg accept | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| OpenVLA AR | 0.768 | 0.2826s | 1.00x | - | - | baseline |
| SpecVLA strict | 0.762 | 0.3211s | 0.88x | 1.396 | 0.396 | 3090 上负加速 |
| SpecVLA relaxed | 0.736 | 0.2709s | 1.04x | 2.372 | 1.372 | relaxed 后才略快 |
| DFLASH strict | 0.760 | 0.2905s | 0.97x | 2.113 | 1.043 | Length 高于 Spec strict，但速度仍不够 |
| DFLASH relaxed | 0.768 | 0.2691s | 1.05x | 2.416 | 1.339 | SR 与 AR 持平，Length 略高于 SpecVLA relaxed |

该 DFLASH relaxed 的在线 per-position hit rate 是：

| p1 | p2 | p3 | p4 | p5 | p6 | overall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.919 | 0.369 | 0.416 | 0.428 | 0.387 | 0.626 | 0.342 |

这组在线结果暴露了当时的 offline/online gap：训练里 anchor0 p2-p5 已经到 `0.65-0.72`，但在线 relaxed
hit rate 仍只有 `0.37-0.43`。后来检查发现当时主 runtime 的 `include_anchor_hidden=True` 分支没有真正启用
`sample_with_causal_residual`，只是直接对 `draft_hidden` 做 `lm_head` 后采样。因此该评测不能判定
residual/CAD 机制失败，只能说明“不在推理时使用 residual 修正，仅靠训练正则”不足以明显反超。

### 历史 Clean Markov-ACD 长跑版观察重点

该 Clean Markov-ACD 长跑已经在 3090 上完成，核心设置是：删掉所有手工位置加权，使用
`causal_residual_start_index=0` 和 `slot_decay=0.90`。训练期间优先看这些指标：

```text
train/accuracy
train/base_accuracy
train/residual_token_ce_accuracy
train/anchor_0_to_position_1_acc ... train/anchor_0_to_position_6_acc
train/causal_residual_cad_loss
train/anchor_logit_distill_loss
train/refined_hidden_loss
train/residual_token_ce_loss
```

判断标准：

1. `anchor0 -> p2-p5` 是否保持 Markov-ACD 短跑里的快速上升趋势，并在长跑中稳定到高位。
2. `anchor0 -> p1` 不能因为照顾 p2-p5 而长期掉下去；`start_index=0` 和 `slot_decay=0.90`
   就是为了把第一跳重新纳入训练重点。
3. `accuracy - base_accuracy` 如果长期为正，说明 residual/logit 修正头在线启用后有希望产生真实收益；
   如果长期为负，要重新检查残差头是否在破坏 base draft。
4. 离线准确率不是最终结论；训练后仍必须搬 checkpoint 到 4090，重点跑 CAD-head strict/relaxed 的 LIBERO rollout。

2026-07-06 已补上 `include_anchor_hidden=True` 推理分支里的 residual sampling 接线。默认旧 DFlash
launcher 仍关闭该功能；专用 residual launcher 会显式开启：

```bash
# relaxed，默认 EVAL_EPOCH=200，ACCEPT_THRESHOLD=9
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_residual_libero_goal_eval.sh

# strict，默认 EVAL_EPOCH=200，ACCEPT_THRESHOLD=0
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_residual_strict_libero_goal_eval.sh
```

summary JSON 会记录：

```text
dflash_use_causal_residual_sampling = true
generation.use_causal_residual_sampling = true
```

如果要在旧 launcher 上临时打开，也可以：

```bash
EVAL_EPOCH=200 DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=True \
  bash openvla/specdecoding/decode-scripts/run_dflash_libero_goal_eval.sh
```

相关开关：

```text
--causal_residual_type hidden
--causal_residual_rank 256
--causal_residual_start_index 0   # 所有 slot 都启用 Markov-aware residual refinement
--causal_residual_cad_w 0.10
--causal_residual_cad_type cosine
--causal_residual_cad_warmup_steps 4000
--causal_residual_cad_correct_teacher_only
--causal_residual_min_position 2
--causal_residual_max_position 5
--refined_hidden_w 0.30
--residual_token_ce_w 0.10
--logit_markov_type bias
--logit_markov_rank 256
--logit_markov_scale 1.0
--anchor_logit_distill_w 0.10
--anchor_logit_distill_temperature 2.0
--anchor_logit_distill_min_position 2
--anchor_logit_distill_max_position 5
--anchor_logit_distill_correct_teacher_only
--soft_w 0.10
--slot_decay 0.90
```

SwanLab/JSONL 会额外记录：

```text
causal_residual_cad_loss
causal_residual_cad_component
causal_residual_cad_pairs
anchor_logit_distill_loss
anchor_logit_distill_component
anchor_logit_distill_pairs
refined_hidden_loss
refined_hidden_component
base_accuracy
accuracy
```

其中 `base_accuracy` 是残差修正前的 token 命中率，`accuracy` 是残差修正后的命中率。这个差值用于判断
残差修正是否真的救到了 p2-p5。新版训练尤其要看：
`anchor_0_to_position_2_acc` 是否比旧版更快上升，以及 `accuracy - base_accuracy` 是否转正。

## 离线数据：历史 artifact 和 4090 目标目录

数据生成脚本使用 `openvla/modified_libero_rlds` 中的 `libero_goal_no_noops` split。对每个 RLDS sample，
脚本贪心运行 OpenVLA；只有当返回的 action hidden-state sequence 和 7 个 action token 在结构上兼容时，
才写出一个 `data_*.ckpt` tensor dictionary。

4090 上建议使用的数据目录：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset
```

2026-06-26 审计结果：`419G`，`28,639` 个 `.ckpt` 样本。生成日志显示枚举样本 `52,042`，
有效样本 `28,639`。数据较大是预期现象，因为每个有效样本保存完整 prefix hidden sequence、
selected/final hidden states，而不只是离散 action token。

关键字段如下：

```text
input_ids                    tokenized vision-language prompt
pixel_values                 preprocessed image tensor
loss_mask                    prompt attention mask
predicted_tokens             greedy OpenVLA action-token sequence，长度 7
hidden_state.prompt_selected 完整 prefix，拼接 selected layers
hidden_state.prompt_last     完整 prefix，final layer
hidden_state.action_selected action hidden states，selected layers
hidden_state.action_last     action hidden states，final layer
dflash_data_format           full_prefix_plus_action_hidden_v4
```

不要把旧数据格式生成的文件混进这个目录。Trainer 会检查字段和 shape，但实验前仍应手动确认数据版本和样本数。

## 方法演进摘要

更详细的数值见上面的“实验演进与诊断记录”。这里只保留设计决策主线：

1. **初始迁移。** 把 DFlash-style draft 插入 SpecVLA/OpenVLA speculative 路径。早期 draft context 不足，
   acceptance 基本不可用。
2. **Context 对齐 SpecVLA。** 数据和 runtime 改为保留完整 prefill hidden sequence 和 target-verified
   action history。当前 `include_anchor_hidden` 路径会在每次并行 tail proposal 前，先用 target 解码 anchor。
3. **Multi-anchor 训练。** 加入 multi-anchor supervision、action-dimension embedding、position balancing、
   hidden loss、cosine loss 和 per-anchor/per-position 诊断指标。
4. **Pure hidden baseline。** 证明 p1/p6 容易学，p2-p5 弱，尤其 `anchor0 -> p2` 是瓶颈。
5. **Residual-CAD weak-path。** 加入前序 token hidden 残差、Hidden-level CAD、refined hidden supervision。
   这一步把 anchor0->p2 从约 0.50 拉到约 0.67，但 residual 后 accuracy 没稳定超过 base accuracy。
6. **Markov-ACD。** 进一步加入 logits-level Markov bias、Logit-level CAD、residual token CE 和低权重 soft
   distribution。短跑诊断显示 anchor0 的 p2-p5 离线准确率大幅提升。
7. **Clean Markov-ACD 当前长跑版。** 删除所有手工位置加权，保留结构性 Markov-aware refinement 和 CAD；
   `causal_residual_start_index=0` 覆盖第一跳，`slot_decay=0.90` 轻度偏向更影响 acceptance length 的前几个 slot。epoch 120 已显示离线 p2-p5 接近饱和，下一步看 4090 online Length/Speedup。
8. **Action-RNN Prefix Survival。** 在线评测证明旧头的 teacher-forced/online gap 和逐 slot 全词表开销仍是瓶颈。
   新版改为三层并行主干、256 类低秩前缀状态、动作分布蒸馏、连续前缀存活目标和可选置信截断；跨
   anchor logit distillation 保留为本项目的因果强弱路径约束。

需要始终记住的限制：

- Block draft 的 block 内输入是非因果的，未来 slot 在单次 draft forward 中不会收到 ground-truth causal prefix。
  这是并行设计的核心建模风险，不是可以用离线 token metric 掩盖的小问题。
- 低 hidden loss 不等于有效 speculative speedup。真正决定速度的是在线 acceptance distribution 和 target-call count。
- Relaxed acceptance 可能保持实际动作效果，但 token 层面不等于 strict equality。必须做消融并诚实报告阈值。


## 后续扩展：OpenVLA-OFT Layer-ACD Early Exit

这一节现在已进入独立实现与复现阶段，但仍是 DFLASH 主实验之外的补充章节。当前完整代码、环境、训练和
评测说明见 [openvla-oft/README_SPECVLA_DFLASH_CN.md](openvla-oft/README_SPECVLA_DFLASH_CN.md)。代码固定了
[OpenVLA-OFT](https://openvla-oft.github.io/) 上游版本，并沿用 `4090 评测 / 3090 数据和训练` 的机器分工。

### 动机

OpenVLA-OFT 已经不是原始 OpenVLA 那种 7 个 action token 自回归解码，而是 parallel decoding、
action chunking 和 continuous action head。因此，当前 DFLASH speculative decoding 主线即使成功，也主要证明
“能加速自回归 OpenVLA”。审稿人可能会追问：如果 VLA action head 本来就是并行的，这个思路是否还有价值？

可以用一个小实验回答这个问题：把 Markov-ACD 的核心抽象成 **弱计算路径追强计算路径**。在 OpenVLA 中，
弱路径是短 anchor / 薄前缀；在 OpenVLA-OFT 中，弱路径可以是 **早期 LLaMA layer hidden**，强路径是
**最终层 hidden / full OpenVLA-OFT action chunk**。这样就从 speculative decoding 扩展成
parallel action-chunk VLA 的 early-exit acceleration。

### 粗略方案

当前实现名为 **Layer-ACD Early Exit**：

```text
early layer action hidden h_l
    -> Chunk-aware residual adapter
    -> frozen official OFT L1 action head
    -> predicted parallel action chunk
```

训练时固定一个 OpenVLA-OFT teacher，对同一输入保存：

```text
early hidden:      h_l, 例如 layer 16 / 20 / 24 / 28
final hidden:      h_L
teacher action:    full OpenVLA-OFT action chunk
ground-truth act:  dataset action chunk
```

然后训练一个轻量 early-exit module：

```text
refined_h_l = h_l + ChunkMixerResidual(h_l)
early_action_chunk = ActionHead(refined_h_l)
```

adapter 先把 56 个 action hidden 投到低维 bottleneck，再做两层 action-token self-attention，最后预测回
4096 维 residual。它显式保留 OFT 并行 action chunk 内的跨 token 关系，同时复用“弱路径 hidden 经轻量修正后
追强路径”的思想，而不是复制 speculative decoding。

### 训练 loss

第一版不要设计太复杂，先验证早退是否能保住控制性能：

```text
total =
    hidden_distill(refined_h_l, final_h_L)
  + action_distill(early_action_chunk, teacher_action_chunk)
  + cosine_hidden_loss(refined_h_l, final_h_L)
```

第一版 action loss 使用冻结官方 action head 输出的 teacher continuous action，而不直接混入额外标签设计；
先证明早退是否保住 OFT 自己的 action mapping，再决定是否有必要加入 ground-truth action loss。

### 最小实验矩阵

先只做 LIBERO-Goal，避免在主线没收束前铺太大：

| 方法 | 说明 |
| --- | --- |
| OFT-full | 原始 OpenVLA-OFT，完整 LLaMA 层数和原 action head。 |
| OFT-early-naive | 直接拿 early layer hidden 接 action head，不加 residual。 |
| OFT-early-residual | early layer hidden + residual refinement + action head。 |
| OFT-early-Layer-ACD | residual refinement + hidden/action distillation，是完整小实验。 |

建议测试的 exit layer：

```text
layer 16 / 20 / 24 / 28
```

核心指标：

```text
LIBERO success rate
latency / throughput
action L1 to teacher action
action L1 to ground-truth action
chunk 前 1/2/4/8 步误差
```

保守成功标准：

```text
success rate drop <= 1-2%
latency speedup >= 15-25%
```

### 风险和边界

1. OpenVLA-OFT 已经很快，early-exit 的绝对速度收益可能没有自回归 OpenVLA 明显。
2. 早期 layer hidden 可能还没有形成足够强的 action semantics，success rate 可能明显下降。
3. 这个扩展不是 speculative decoding，本论文里只能作为 generalization / future extension / small study，
   不要和主贡献混淆。
4. 如果最小实验失败，也仍然有价值：它能界定当前 Markov-ACD 思想更适合 action-token autoregressive VLA，
   而不是所有并行 action-head VLA。

当前已完成统一代码接入、3090 Goal smoke、teacher HDF5、DDP adapter trainer、hook early-exit runtime 与
4090/3090 脚本。4090 已准备 oft 环境、custom Transformers fork 和 openvla-7b-oft-finetuned-libero-goal
权重；待 DFLASH 当前主表评测释放 GPU 后，先在 4090 完成 OFT Goal 50-trial baseline，再在 3090 收集 teacher
feature 并训练 layer-16 adapter。


## 后续扩展：顶会级实验路线图

这一节回答一个更大的问题：如果当前 OpenVLA 自回归场景已经拿到非常好的速度、Length 和成功率结果，
下一步还需要补哪些实验，才能让论文故事达到机器人/VLA 顶会的完整度。这里仍是路线图，不代表已经完成。

### 论文主结论需要被哪些证据支撑

顶会论文不能只报告“某个 checkpoint 在 Goal 上更快”。更稳的主结论应该是：

> Markov-ACD 通过跨 anchor 强弱路径蒸馏，让轻量块并行 draft 在保持 target-model 校验可靠性的前提下，
> 提高 accepted length，并把 accepted length 的提升稳定转化为 wall-clock latency 下降和机器人任务成功率保持。

这句话需要四类证据同时成立：

1. **效果证据。** 在 LIBERO 多个 suite 上，DFLASH/Markov-ACD 的 `SR / Length / Speedup`
   同时优于或不弱于 OpenVLA AR 和 SpecVLA baseline。
2. **机制证据。** p2-p5 弱路径的离线准确率、在线 hit rate、accept length histogram 都能解释为什么速度提高。
3. **消融证据。** 去掉 Markov-aware residual、Hidden-level CAD、Logit-level CAD、soft distribution、
   `start_index=0`、`slot_decay=0.90` 等组件后，指标按预期下降。
4. **部署证据。** 在真实控制循环里，推理延迟降低不会破坏动作质量，最好能提升闭环响应或至少保持成功率。

### 必做实验清单

| 层级 | 实验 | 目的 | 关键指标 |
| --- | --- | --- | --- |
| Main table | OpenVLA AR / SpecVLA strict / SpecVLA relaxed / DFLASH strict / DFLASH relaxed / DFLASH CADhead strict-relaxed | 证明主结果不是单一脚本偶然现象 | `SR`、`Length`、`Speedup`、`avg_accept_length`、`overall_hit_rate` |
| Multi-suite | LIBERO Goal / Object / Spatial / Long | 证明不是只对 Goal 有效 | 每个 suite 的主表指标和 task-level breakdown |
| Threshold sweep | relaxed threshold `0/3/6/9/12` | 分离 strict acceptance 和 relaxed acceptance 的贡献 | `SR` vs `Length` vs `Speedup` 曲线 |
| Checkpoint sweep | epoch 30/60/90/120/150/180/200 | 找到最佳训练点，避免只报 latest | offline acc、online Length、SR |
| Hardware timing | 4090 单实验串行，必要时再补 A100/3090 | 说明速度结果对硬件和 timing 口径敏感 | latency mean/median/p95、GPU、`SYNC_CUDA_TIMING`、`TIMING_SCOPE` |
| Cost breakdown | target prefill、draft forward、verify、environment step | 解释速度瓶颈，不只报最终 speedup | 每部分耗时占比 |
| Online diagnostics | per-position hit rate、accept length histogram、reject position histogram | 证明 p2-p5 是否真的被救起来 | `p1..p6 hit_rate`、histogram |
| Robustness | 3 seeds 或多次 eval runs | 让成功率不是一次随机 rollout | mean/std 或 confidence interval |

### 消融实验矩阵

第一版消融要服务主故事，不要切得太碎。当前建议只保留三档 waterfall：

| Variant | 训练脚本 | 结构含义 | 主要看什么 |
| --- | --- | --- | --- |
| Pure DFlash | `run_dflash_ablation_1_pure_hidden_4gpu.sh` | 完整 prefix/action hidden + 1-layer block draft，只用 hidden/cos 几何蒸馏；无 soft、无 CAD、无残差头 | p1-p6 离线准确率、online Length 的自然上限 |
| + Anchor-Contrastive Distillation | `run_dflash_ablation_2_anchor_cad_4gpu.sh` | 加 Hidden-level CAD 和 Logit-level CAD，让短前缀弱路径追长前缀强路径；仍无 Markov 残差头 | anchor0 的 p2-p5 是否整体抬升，Length 是否开始变长 |
| + Markov-aware CAD Head | `run_dflash_ablation_3_markov_acd_4gpu.sh` | 在 CAD 上加入 Markov-aware hidden residual、logit bias、residual token CE 和低权重 soft loss | `accuracy-base_accuracy`、online hit rate、Length、Speedup 是否进一步提高 |

这三档足够回答审稿人最可能问的核心问题：

1. 单纯块并行 hidden draft 是否已经足够？
2. 跨 anchor 强弱路径蒸馏是否真的缓解 p2-p5 的薄前缀问题？
3. Markov-aware CAD head 是否把离线 token/hidden 改善转化成在线 acceptance 和速度？

主表不需要放所有细碎 loss 开关。更细的 hidden-CAD-only、logit-CAD-only、soft_w sweep、slot_decay sweep 可以放附录或后续补实验，只有当主表结果需要解释时再展开。

### 分析图和论文叙事图

如果主实验成功，至少需要准备这些图：

1. **Accuracy-to-acceptance bridge。** 横轴是训练 epoch，纵轴同时画 `anchor0->p2-p5 acc`、
   online p2-p5 hit rate、Length。目的是证明离线弱路径提升能转化为在线 acceptance。
2. **Accept length histogram。** 对比 SpecVLA relaxed 和 DFLASH relaxed，展示我们是否把更多 block 推到
   length 2/3/4，而不是只靠少数长 tail。
3. **Latency decomposition。** AR、SpecVLA、DFLASH 的 target/draft/verify/environment 耗时分解。
4. **Ablation waterfall。** 从 Base hidden 到 full Markov-ACD，每加入一个模块，p2-p5 acc、Length、
   Speedup 如何变化。
5. **Failure cases。** 收集失败 rollout：是视觉误识别、动作 token 拒绝过早、relaxed threshold 过松，
   还是 draft 快但 target correction 频繁。

### 真实机械臂实验：ALICIA-D 验证方案

你手头的玄雅科技灵动 ALICIA-D 可以作为真机验证平台。根据官方公开介绍，ALICIA-D 是桌面级六轴机械臂，
面向具身智能和数据采集场景，官方资料提到其可接入仿真、真机训练和 LeRobot 等生态；具体负载、末端、相机、
控制接口以你手头设备和厂家文档为准。

真机实验的目标不是重新证明一个大规模通用机器人策略，而是回答：

> Markov-ACD 在真实闭环控制中降低推理延迟后，是否仍保持 OpenVLA policy 的任务成功率和动作质量？

#### 实验分阶段

| 阶段 | 目标 | 推荐规模 |
| --- | --- | --- |
| Real-robot smoke test | 跑通相机、标定、动作接口、限位和急停；确认 action space 与 OpenVLA 兼容 | 每个动作 primitive 5-10 次 |
| Small imitation dataset | 采集 Alicia-D 桌面任务 demonstration，用于 fine-tune target VLA | 3-5 个任务，每任务 30-80 条 demo |
| Target policy | 先训练能稳定完成任务的 OpenVLA target，不上 draft | 每任务成功率最好先到 60-80% 以上 |
| Draft training | 用 target policy 生成/保存 action hidden 和 token 数据，再训练 DFLASH draft | 复用当前 Markov-ACD workflow |
| Real rollout comparison | 同一 target policy 下比较 AR、SpecVLA、DFLASH，改变的只有 decode mode | 每任务每方法 20-30 trials |

#### 任务设计

优先选择桌面、低风险、容易重置、能体现闭环响应的任务：

| 任务 | 描述 | 评价 |
| --- | --- | --- |
| Pick-place colored block | 把指定颜色方块放入指定容器 | 成功/失败、完成时间、碰撞/掉落 |
| Push to target | 把小物体推到桌面目标区域 | 最终距离、完成时间 |
| Insert / place into cup | 把海绵块或轻物体放入杯/盒中 | 成功/失败、重试次数 |
| Disturbed pick | 机械臂接近前随机轻微移动目标物 | 测试低延迟闭环是否更稳 |
| Long-horizon two-step | 先拿物体，再放到另一处或按颜色分类 | 测试累计 correction 是否影响策略 |

#### 真机数据格式和控制日志

每条 rollout 建议保存：

```text
timestamp
front_camera / wrist_camera image
instruction
end-effector pose
gripper state
raw target action
executed action
decode mode: AR / SpecVLA / DFLASH / DFLASH-CADhead
per-step latency
accepted length / reject position / corrected token
success label
safety stop / collision / out-of-bound flag
```

#### 真机对照原则

1. **同一 target policy。** AR、SpecVLA、DFLASH 必须加载同一个 target VLA 权重；只改变 decoding / draft。
2. **交错评测顺序。** 不要先跑完 AR 再跑 DFLASH；按随机顺序交错方法，避免光照、物体位置、机械臂温度漂移。
3. **固定随机种子和初始位姿列表。** 每个方法使用相同的 object pose set。
4. **速度和成功率一起报告。** 如果速度提升但成功率下降，必须诚实给出 trade-off。
5. **安全限制。** workspace bound、速度/加速度上限、急停、软限位、碰撞检查必须先于任何论文实验。

#### 真机结果表

最小可投稿级真机表可以这样设计：

| Task | Method | Success | Completion time | Policy latency | Control freq | Avg accepted length | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pick-place | OpenVLA AR | - | - | - | - | - | target baseline |
| Pick-place | DFLASH relaxed | - | - | - | - | - | same target |
| Disturbed pick | OpenVLA AR | - | - | - | - | - | latency-sensitive |
| Disturbed pick | DFLASH relaxed | - | - | - | - | - | latency-sensitive |

如果资源有限，真机部分可以先做 2-3 个任务、每任务每方法 20 trials，作为“real-robot validation”小节；
如果效果很好，再扩成 5 个任务并加 task-level videos。

### 当前优先级排序

1. **先完成当前 3090 训练和 4090 串行 eval。** 没有稳定主结果，不要扩散到 OFT 或真机。
2. **补齐主表和核心消融。** 这决定论文主线是否成立。
3. **补 online diagnostics 和 latency breakdown。** 这决定故事是否高级，而不是只报一个 speedup。
4. **再做 Alicia-D 真机 smoke test。** 先确认 target policy 能控制真机，再比较 decoding mode。
5. **最后做 OpenVLA-OFT Layer-ACD。** 它是泛化扩展章节，能拔高论文，但不应抢主线资源。


## 新服务器 4090 从零迁移步骤

旧 4090d 机器不再作为默认推理机。之后默认把新的 `ssh 4090` 作为主开发、数据生成和单卡推理评测机器；
3090 仍作为四卡训练机器。新 4090 的固定工作根目录约定为：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh
```

建议目录布局：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH                            代码仓库
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-goal OpenVLA Goal 权重
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/dataset/modified_libero_rlds              OpenVLA 修改版 LIBERO RLDS 数据
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset          DFLASH 离线训练数据
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs                    LIBERO 评测日志
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/specvla_checkpoint/goal      SpecVLA Goal baseline draft 权重
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/LIBERO                                    LIBERO 仿真环境源码
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf-cache                                  Hugging Face 缓存
```

### 1. Git 和基础目录

4090 应配置 deploy key，以便直接拉取和更新私有仓库：

```bash
ssh 4090
mkdir -p /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh

git clone git@github.com:guanghaichen/SpecVLA-DFLASH.git
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH
git pull --ff-only origin main
```

如果仓库已经存在，只需要：

```bash
ssh 4090
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH
git status
git pull --ff-only origin main
```

### 2. `.bashrc` 中只固定稳定根路径和镜像源

4090 后续尽量不要让脚本自动访问官方 Hugging Face。`.bashrc` 只放跨任务稳定的根目录、缓存和数据路径；
**不要在 `.bashrc` 里 export 全局 `VLA_PATH` 或 `OPENVLA_MODEL_PATH`**。评测时的 suite-specific OpenVLA
模型路径由 `openvla/specdecoding/decode-scripts/libero_eval_common.sh` 根据 `TASK_SUITE_NAME` 自动选择。
如确实要临时覆盖某个评测模型路径，用命令行前缀 `VLA_PATH_OVERRIDE=/path/to/model`。

把下面内容追加到 `~/.bashrc` 后重新登录，或执行 `source ~/.bashrc`：

```bash
# SpecVLA-DFLASH stable paths on 4090.
# Keep suite-specific eval model paths in decode-scripts/libero_eval_common.sh.
# Do not export global VLA_PATH here; it can make Object/Spatial/Long eval load the Goal checkpoint.
export SPECVLA_ROOT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh
export SPECVLA_REPO=${SPECVLA_ROOT}/SpecVLA-DFLASH
export SPECVLA_DATA=${SPECVLA_ROOT}/specvla-data
export OPENVLA_MODEL_ROOT=${SPECVLA_ROOT}/hf_files
export OPENVLA_GOAL_PATH=${OPENVLA_MODEL_ROOT}/openvla-7b-finetuned-libero-goal
export OPENVLA_OBJECT_PATH=${OPENVLA_MODEL_ROOT}/openvla-7b-finetuned-libero-object
export OPENVLA_SPATIAL_PATH=${OPENVLA_MODEL_ROOT}/openvla-7b-finetuned-libero-spatial
export OPENVLA_LONG_PATH=${OPENVLA_MODEL_ROOT}/openvla-7b-finetuned-libero-10
unset VLA_PATH
unset OPENVLA_MODEL_PATH
export LIBERO_RLDS_ROOT=${SPECVLA_ROOT}/dataset/modified_libero_rlds
export DFLASH_DATA_OUTDIR=${SPECVLA_DATA}/dflash_goal_dataset
export DFLASH_OUTPUT_DIR=${SPECVLA_DATA}/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu
export SPECVLA_CKPT_ROOT=${SPECVLA_DATA}/specvla_checkpoint
export SPECVLA_GOAL_CKPT=${SPECVLA_CKPT_ROOT}/goal
export LOG_DIR=${SPECVLA_DATA}/eval_logs
export LIBERO_PATH=${SPECVLA_ROOT}/LIBERO

# Hugging Face mirror and caches
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=${SPECVLA_ROOT}/hf-cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_DISABLE_TELEMETRY=1

# Python path used by local scripts
export PYTHONPATH=${SPECVLA_REPO}:${SPECVLA_REPO}/openvla:${LIBERO_PATH}:${PYTHONPATH}

# Optional pip mirror. If a package is missing on this mirror, temporarily unset it.
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
```

### 3. 创建 conda 环境

如果 4090 还没有 Miniconda，可以用清华镜像安装：

```bash
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh
wget -O Miniconda3-latest-Linux-x86_64.sh \
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ${HOME}/miniconda3
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda init bash
```

创建 `specvla` 环境：

```bash
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda create -n specvla python=3.10 -y
conda activate specvla
python -m pip install -U pip setuptools wheel packaging ninja
```

PyTorch 版本沿用旧环境经验：Python 3.10、PyTorch 2.2.0、CUDA 12.1。优先使用官方 CUDA wheel；
如果服务器不能直连，可先在能联网的机器下载 wheel 后传到 4090：

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

安装项目依赖：

```bash
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH/openvla
pip install -e .
```

注意：`openvla/pyproject.toml` 里包含 `dlimp @ git+https://github.com/moojink/dlimp_openvla`。
如果 4090 不能访问 GitHub，这一步会卡住。解决方式是先在可联网机器把 `dlimp_openvla`
源码或 wheel 下载好，再传到 4090 本地安装；或者临时配置能访问 GitHub 的代理后再执行 `pip install -e .`。

`flash-attn` 不是当前 DFLASH 推理评测的硬依赖。DFlash draft 自己使用 PyTorch
`scaled_dot_product_attention`；OpenVLA target 在当前加载路径里也没有强制
`attn_implementation="flash_attention_2"`。因此如果 4090 上安装 flash-attn 卡住，可以先不装，
直接跑 AR / SpecVLA / DFlash 评测。代价只是可能比 flash-attn 路径慢一些或显存更高。

只有在后续明确要复现上游 flash-attn 配置、或某个模型配置强制要求 flash-attn 时，再安装它。
安装时必须匹配 Python、PyTorch 和 CUDA，优先去
[Dao-AILab/flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases)
下载对应 wheel；没有完全匹配的 wheel 时再源码编译。

### 4. 下载 OpenVLA Goal 权重

OpenVLA Goal 目标模型来自 Hugging Face 仓库 `openvla/openvla-7b-finetuned-libero-goal`。
在国内网络下优先走 `HF_ENDPOINT=https://hf-mirror.com`：

```bash
source ~/.bashrc
conda activate specvla
mkdir -p /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files

huggingface-cli download openvla/openvla-7b-finetuned-libero-goal \
  --local-dir /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-goal \
  --local-dir-use-symlinks False \
  --resume-download
```

检查关键文件：

```bash
ls -lh /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-goal
```

后续 Object、Spatial、Long baseline 也按同一规则下载到：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-object
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-spatial
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-10
```

对应 Hugging Face repo 名通常是：

```text
openvla/openvla-7b-finetuned-libero-object
openvla/openvla-7b-finetuned-libero-spatial
openvla/openvla-7b-finetuned-libero-10
```

### 5. 下载 modified LIBERO RLDS 数据

DFLASH 数据生成脚本读取 OpenVLA 修改版 RLDS 数据。镜像下载命令：

```bash
source ~/.bashrc
conda activate specvla
mkdir -p /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/dataset

huggingface-cli download openvla/modified_libero_rlds \
  --repo-type dataset \
  --local-dir /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/dataset/modified_libero_rlds \
  --local-dir-use-symlinks False \
  --resume-download
```

下载后应能看到 `libero_goal_no_noops` 等 split。检查：

```bash
find /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/dataset/modified_libero_rlds -maxdepth 2 -type d | sort | head -30
```

### 6. 准备 SpecVLA baseline draft 权重

SpecVLA/EAGLE baseline 的 draft 权重由原始
[PineTreeWss/SpecVLA](https://github.com/PineTreeWss/SpecVLA) README 提供 Google Drive 下载入口。
Goal 权重放到：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/specvla_checkpoint/goal
```

Object、Spatial、Long 后续如果要复现四个 suite 的 baseline，建议放到：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/specvla_checkpoint/object
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/specvla_checkpoint/spatial
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/specvla_checkpoint/10
```

评测脚本会通过 `SPECVLA_CKPT_ROOT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/specvla_checkpoint`
自动拼出 suite-specific checkpoint 路径。

### 7. 安装 LIBERO 仿真环境

LIBERO 推荐作为源码目录放在 `/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/LIBERO`：

```bash
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/LIBERO
pip install -e .
```

如果 4090 无法访问 GitHub，先在其它机器下载 LIBERO 源码压缩包，再传到 `/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/LIBERO`。
4090 上没有 3090 那套 NVIDIA EGL shim 记录；如果后续评测遇到 EGL/MuJoCo 报错，先检查：

```bash
python - <<'PY2'
import os
print('MUJOCO_GL=', os.environ.get('MUJOCO_GL'))
print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES'))
PY2
```

评测 launcher 默认会设置 `MUJOCO_GL=egl`。不要手动设置错误的 `MUJOCO_EGL_DEVICE_ID`。

### 8. 数据生成 sanity check

确认模型和 RLDS 都准备好后，在 4090 上先小规模跑通数据生成：

```bash
ssh 4090
source ~/.bashrc
conda activate specvla
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH

CUDA_VISIBLE_DEVICES=0 python openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py \
  --gpu_index 0 \
  --vla_path /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-goal \
  --data_root_dir /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/dataset/modified_libero_rlds \
  --dataset_name libero_goal_no_noops \
  --outdir /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset.h5 \
  --output_format hdf5
```

正式生成结束后检查：

```bash
ls -lh /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset.h5
python - <<'PY'
import h5py
path = '/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset.h5'
with h5py.File(path, 'r') as f:
    print('complete', bool(f.attrs.get('complete')), 'samples', int(f.attrs.get('num_samples')), 'format', f.attrs.get('format'))
PY
```

历史 4090/3090 的有效样本规模约为 `28.5k`，大小约 `419G`。新 4090 重新生成后，
应在实验记录中写清楚实际样本数；不要默认和旧机器完全一致。

### 9. 从 3090 搬训练好的 checkpoint 到 4090

3090 继续负责四卡训练。训练完成后，在本地终端用 `scp -3` 从 3090 搬到 4090：

```bash
TRAIN_DIR=/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu
DEST_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu

ssh 4090 "mkdir -p ${DEST_DIR}"

for CKPT in epoch_100_step_044700 epoch_150_step_067050 epoch_200_step_089400; do
  ssh 4090 "mkdir -p ${DEST_DIR}/${CKPT}"
  scp -3 3090_wulin:${TRAIN_DIR}/${CKPT}/pytorch_model.bin 4090:${DEST_DIR}/${CKPT}/pytorch_model.bin
  scp -3 3090_wulin:${TRAIN_DIR}/${CKPT}/dflash_config.json 4090:${DEST_DIR}/${CKPT}/dflash_config.json
done

for FILE in dflash_config.json metrics.jsonl training_summary_markov_acd_start0_slotdecay090.csv training_summary_markov_acd_start0_slotdecay090.md; do
  scp -3 3090_wulin:${TRAIN_DIR}/${FILE} 4090:${DEST_DIR}/${FILE}
done
```

复制后检查：

```bash
ssh 4090 "for CKPT in epoch_100_step_044700 epoch_150_step_067050 epoch_200_step_089400; do \
  ls -lh ${DEST_DIR}/\${CKPT}/dflash_config.json ${DEST_DIR}/\${CKPT}/pytorch_model.bin; \
done"
```

## 服务器分工和训练流程

当前两台机器的分工如下：

| 机器 | GPU 情况 | 主要用途 |
| --- | --- | --- |
| 4090 | 新 RTX 4090 服务器 | 主开发、代码调试、数据生成、小规模 sanity check、最终 LIBERO 推理评测。 |
| 3090 | 8 张 RTX 3090，实验中默认只用 0-3 四张 | 完整 DFLASH 四卡训练。 |

因此，**不要在 README 中把 4090 写成四卡训练机器**。当前四卡 launcher 固定使用
`torchrun --nproc_per_node 4`，实际应该在 3090 上用 `CUDA_VISIBLE_DEVICES=0,1,2,3`
启动。4090 如果需要训练，只适合临时做单卡小规模调试，不能直接照搬四卡命令。

当前固定实验流如下：

```text
4090 维护代码和数据生成 -> GitHub main 固化代码 -> 3090 四卡训练 -> 本地 scp -3 搬 checkpoint 到 4090 -> 4090 跑五套推理评测
```

3090 的 RTX 3090 对 speculative decoding 的小 kernel、校验和调度开销更敏感，速度结果容易偏低；
因此正式比较 `SR / Length / Speedup` 时统一使用 4090。3090 只作为训练吞吐机器。

### 1. 4090：主开发、数据生成和单卡调试

4090 进入服务器和环境：

```bash
ssh 4090
source ~/.bashrc
conda activate specvla
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH
export PYTHONPATH="$PWD"
```

数据生成路径可以通过 `VLA_PATH`、`LIBERO_RLDS_ROOT`、`DFLASH_DATA_OUTDIR` 覆盖；
数据生成脚本也支持显式 `--vla_path`、`--data_root_dir`、`--outdir` 参数。这样可以避免意外触发 Hugging Face 下载。

生成 DFLASH 原始数据的入口：

```text
openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py
```

数据生成命令：

```bash
CUDA_VISIBLE_DEVICES=0 python openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py \
  --gpu_index 0 \
  --vla_path /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-goal \
  --data_root_dir /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/dataset/modified_libero_rlds \
  --dataset_name libero_goal_no_noops \
  --outdir /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset.h5 \
  --output_format hdf5
```

训练前确认数据大小和数量：

```bash
ls -lh /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset.h5
python - <<'PY'
import h5py
path = '/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset.h5'
with h5py.File(path, 'r') as f:
    print('complete', bool(f.attrs.get('complete')), 'samples', int(f.attrs.get('num_samples')), 'format', f.attrs.get('format'))
PY
```

旧 4090 历史数据目录状态如下；新 4090 重新生成或迁移后必须重新记录实际数值：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset.h5
历史样本内容来自 28,576 个有效 .ckpt；单文件 HDF5 中 `complete=true` 才能训练。
```

### 2. 3090：完整四卡训练

3090 进入服务器和环境：

```bash
ssh 3090_wulin
cd /data/wulin/c/SpecVLA-DFLASH
source /data/wulin/miniconda3/etc/profile.d/conda.sh
conda activate specvla
```

3090 当前有 8 张 RTX 3090，但默认完整训练只使用 0-3 四张卡。当前优先跑三层
Action-RNN Prefix Survival：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_action_rnn_prefix_3layer_4gpu.sh
```

当前推荐 launcher：

```text
openvla/specdecoding/train-scripts/run_dflash_action_rnn_prefix_3layer_4gpu.sh
```

当前三档核心消融训练入口如下。它们共用 batch、学习率、层选择、完整 prefix/action hidden、`slot_decay=0.90` 等训练 recipe，只改变结构信号，方便做清楚的主线对比：

```text
openvla/specdecoding/train-scripts/run_dflash_ablation_1_pure_hidden_4gpu.sh   # 纯 DFlash hidden/cos 蒸馏，无 soft、无 CAD、无残差头
openvla/specdecoding/train-scripts/run_dflash_ablation_2_anchor_cad_4gpu.sh    # 在纯 DFlash 上加入 Hidden/Logit 跨 anchor 蒸馏，无残差头
openvla/specdecoding/train-scripts/run_dflash_ablation_3_markov_acd_4gpu.sh    # 当前完整 Markov-ACD/CAD-head recipe
```

三档消融推荐按顺序跑，形成论文里的 waterfall：`Pure DFlash -> + Anchor-Contrastive Distillation -> + Markov-aware CAD Head`。
不要一开始就铺开“只关 hidden CAD / 只关 logit CAD / 只改 soft_w”的细碎消融；第一版主故事先看 7 个 action token 准确率、online Length 和 Speedup 是否逐级提高。

这些 launcher 会根据机器自动选择训练默认路径。注意这里是 Goal 训练专用默认值，不是评测时的全局 `VLA_PATH`。评测脚本会在 `libero_eval_common.sh` 里按子集选择 OpenVLA checkpoint。3090 上的训练默认路径是：

```text
OPENVLA_GOAL_PATH=/data/wulin/hf_files/openvla-7b-finetuned-libero-goal
DATAPATH=/data/wulin/c/specvla-data/dflash_goal_dataset.h5  # 若不存在则 launcher 自动回退旧 dflash_goal_dataset
OUTPUT_DIR=/data/wulin/c/specvla-data/ckpt_goal_dflash_action_rnn_prefix_3layer_b8x2_4gpu
```

Action-RNN Prefix Survival 主训练配置：

```text
torchrun --nproc_per_node 4
num_draft_layers = 3
selected_hidden_variant = replace_22_with_final
action_head_type = slot_rnn
action_head_rank = 256
action_vocab_size = 256
action_confidence_enabled = true
hidden_w / cos_w = 0.30 / 0.02
action_token_ce_w = 0.10
action_distill_l1_w = 0.90
prefix_survival_w = 0.50
action_confidence_w = 0.10
anchor_logit_distill_w = 0.10
anchor_logit_distill_temperature = 2.0
anchor_logit_distill_min/max_position = 2/6
causal_residual_type / logit_markov_type = none / none
soft_w = 0
slot_decay = 1.0
position_balance = true
hidden_noise = 0.03
batch_size = 8 per GPU，gradient_accumulation = 2，有效 batch size = 64
epochs = 200
warmup = 1000 optimizer steps
save_every = 10
val_split = 0
SwanLab = 使用环境默认配置
```

训练完成并把 checkpoint 搬到 4090 后，Goal strict + relaxed 一键评测：

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_action_rnn_goal_pair_eval.sh
```

该脚本会显式启用新动作前缀头，但默认 `DFLASH_CONFIDENCE_THRESHOLD=0.0`。先用这个设置比较纯模型能力和
头部开销；随后可用例如 `DFLASH_CONFIDENCE_THRESHOLD=0.3` 做独立阈值 sweep。summary JSON 会同时保存
`per_position`、`conditional_prefix`、`confidence_truncated_blocks`，不要只看独立位置命中率。

`run_dflash_anchor_hidden_1layer_residual_cad_4gpu.sh` 支持环境变量覆盖常用超参数，例如：
脚本内部已经按“路径、训练规模、loss 权重、Markov-ACD 结构参数”分区写了中文注释，启动时也会打印完整配置，训练前优先检查这份打印。

### 训练 IO 修复：单文件 HDF5 数据格式

2026-07-11 排查到服务器卡顿的主因不是 GPU，而是训练数据读取：旧数据是 `28,576` 个单样本 `.ckpt`，总大小约 `419G`。200 epoch 等价于重复随机读取约 `84TB` 小文件；4 卡 DDP 再叠加 DataLoader worker/prefetch，会让共享硬盘随机读非常高。SwanLab 逐指标上传和 checkpoint 写 optimizer state 是次要问题。

当前最终修复改为**单文件 HDF5**，而不是多 shard：

1. `ge_data_all_openvla_token_only_libero_goal.py` 默认 `--output_format hdf5`，直接生成一个 `.h5` 文件。
2. `pack_dflash_dataset_hdf5.py` 可把旧 `.ckpt` 无损合并成一个 `.h5` 文件，样本内容不变。
3. `train_dflash_libero_goal.py` 默认 `--dataset_format auto`，若路径是 `.h5` 或目录内存在 HDF5 数据，就按 HDF5 读取；否则再回退旧 shard / `.ckpt`。
4. HDF5 文件属性 `complete=true` 时才允许训练；合并/生成中途的半成品不会被误读。
5. 训练 launcher 和底层 `train_dflash_libero_goal.py` 默认都使用 `NUM_WORKERS=1`、`DATALOADER_PREFETCH_FACTOR=1`、关闭 pin/persistent workers，并默认不保存 optimizer state / latest 根目录副本，减少共享硬盘压力。只有明确需要断点续训时，才手动打开 `SAVE_TRAINING_STATE=True`。

3090 上把旧数据合并成单文件 HDF5：

```bash
ssh 3090_wulin
cd /data/wulin/c/SpecVLA-DFLASH
source /data/wulin/miniconda3/etc/profile.d/conda.sh
conda activate specvla

screen -S pack_dflash_hdf5
nice -n 10 ionice -c2 -n7 python openvla/specdecoding/train-scripts/pack_dflash_dataset_hdf5.py \
  --input_dir /data/wulin/c/specvla-data/dflash_goal_dataset \
  --output_file /data/wulin/c/specvla-data/dflash_goal_dataset.h5 \
  --overwrite
```

检查合并是否完成：

```bash
python - <<'PY'
import h5py
path = '/data/wulin/c/specvla-data/dflash_goal_dataset.h5'
with h5py.File(path, 'r') as f:
    print('complete', bool(f.attrs.get('complete')), 'samples', int(f.attrs.get('num_samples')), 'format', f.attrs.get('format'))
PY
```

只有看到 `complete True` 才能开始训练。launcher 会优先使用 `/data/wulin/c/specvla-data/dflash_goal_dataset.h5`；如果该文件不存在，才回退旧 `.ckpt` 目录。

2026-07-07 中途检查 Markov-ACD 训练时发现：p2-p6 提升非常明显，但每个 anchor 的第一跳/local slot0
没有吃到 Markov/Residual/token-CE 增强，导致 t1 以及各 anchor 的一步预测仍接近旧版。当前干净版本因此改为
`causal_residual_start_index=0`，让所有 slot 都进入 Markov-aware residual/logit 修正；跨-anchor CAD 本身仍保持
p2-p5，因为 p1 没有更强前缀 teacher。这个改动是结构性开关，不再额外对某个位置做手工加权。

```bash
BATCH_SIZE=16 WARMUP_STEPS=1000 LR=5e-5 RESIDUAL_CAD_W=0.10 REFINED_HIDDEN_W=0.30 \
ANCHOR_LOGIT_DISTILL_W=0.10 RESIDUAL_TOKEN_CE_W=0.10 SOFT_W=0.10 \
CAUSAL_RESIDUAL_START_INDEX=0 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_residual_cad_4gpu.sh
```

如果要启动三档消融，命令分别是：

```bash
# 1. Pure DFlash：只看块并行 hidden draft 自身上限
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_ablation_1_pure_hidden_4gpu.sh

# 2. + Anchor-Contrastive Distillation：验证跨 anchor 强弱路径蒸馏是否能抬 p2-p5
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_ablation_2_anchor_cad_4gpu.sh

# 3. + Markov-aware CAD Head：当前完整方案
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_ablation_3_markov_acd_4gpu.sh
```

如果要复现实验，请优先使用新的 `*_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu` 输出目录；
不要和旧的 puretrain、weak-path、Residual-CAD 目录混写。

2026-06-29 重新检查到的 3090 数据和上一版 puretrain 训练产物状态：

```text
数据目录: /data/wulin/c/specvla-data/dflash_goal_dataset
大小: 419G
当前样本数: 28,576 个 data_*.ckpt

训练输出目录: /data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_puretrain_4gpu
已保存 checkpoint: epoch_010_step_008930 ... epoch_200_step_178600
latest_checkpoint.txt -> epoch_200_step_178600
run_config.json 记录: world_size=4, global_effective_batch=32, train_files=28576
```

4090 和 3090 的数据文件数可能不完全相同，因此写实验记录时必须记录本机实际
`find ... | wc -l` 结果，不要默认两台机器的数据集完全一致。

训练输出中的重要文件：

```text
<output>/epoch_XXX_step_XXXXXX/pytorch_model.bin
<output>/epoch_XXX_step_XXXXXX/training_state.pt
<output>/epoch_XXX_step_XXXXXX/dflash_config.json
<output>/latest_checkpoint.txt
<output>/metrics.jsonl
<output>/swanlog/
```

`latest_checkpoint.txt` 指向默认评测 checkpoint。若中断后继续训练，应使用同一个 `--output_dir`
和 `--resume_from_checkpoint latest`，不要在对比实验中静默改变 world size、有效 batch 或 scheduler 设置。

### 3. 本地：把 3090 checkpoint 搬到 4090

训练完成后，在 **本地终端** 执行远端到远端复制。方向必须是：

```text
3090_wulin:/data/.../checkpoint -> 4090:/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/checkpoint
```

推荐加 `-3`，让数据经由本地转发，不要求 3090 能直接连到 4090：

```bash
TRAIN_DIR=/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu
CKPT=epoch_190_step_169670

scp -3 -r \
  3090_wulin:${TRAIN_DIR}/${CKPT} \
  4090:/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/${CKPT}
```

如果要复制 3090 当前 `latest_checkpoint.txt` 指向的最新 checkpoint，可以在本地终端执行：

```bash
TRAIN_DIR=/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu
CKPT=$(ssh 3090_wulin "basename \"\$(cat ${TRAIN_DIR}/latest_checkpoint.txt)\"")

scp -3 -r \
  3090_wulin:${TRAIN_DIR}/${CKPT} \
  4090:/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/${CKPT}
```

复制后检查 4090 上的 checkpoint 是否完整：

```bash
ssh 4090 "ls -lh \
  /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/${CKPT}/dflash_config.json \
  /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/${CKPT}/pytorch_model.bin"
```

保留两个旧诊断 launcher，仅用于 controlled ablation，不作为当前默认 recipe。注意它们目前仍写死了
旧 4090 风格输出路径，不能直接当作 3090 主训练命令：

```text
openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_baseline.sh
openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_consistency.sh
```

## LIBERO-Goal 评测命令

所有 LIBERO-Goal 评测 launcher 都在：

```text
openvla/specdecoding/decode-scripts/
```

这些脚本共享 `libero_eval_common.sh`。它会自动选择新 4090、3090 或历史旧 4090 路径，设置 `PYTHONPATH`，
配置 LIBERO，并在 3090 上优先使用本地 NVIDIA 570 EGL shim。当前固定流程虽然不在 3090
做正式速度评测，但保留这些路径可以方便必要时做 sanity check：

```text
/data/wulin/c/nvidia-egl-570.133.07
```

3090 训练/临时 sanity check 默认路径：

```text
OpenVLA goal model: /data/wulin/hf_files/openvla-7b-finetuned-libero-goal
SpecVLA checkpoints: /data/wulin/c/specvla-data/specvla_checkpoint/goal
DFLASH run dir: /data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu
Logs: /data/wulin/c/specvla-data/eval_logs
```

4090 正式评测默认路径：

```text
OpenVLA goal model: /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-goal
SpecVLA checkpoint: /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_libero_goal_debug_ckpt
DFLASH copied checkpoint example: /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/epoch_190_step_169670
Logs: /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs
```

如果权重被复制或重命名，可以用下面变量覆盖。DFlash 评测用 `SPEC_CKPT` 指向从 3090 搬到 4090 的
checkpoint；SpecVLA baseline 才需要 `SPECVLA_GOAL_CKPT`。OpenVLA 模型路径不要再用全局 `VLA_PATH`
覆盖 eval launcher，因为 `.bashrc` 里常固定的是 Goal 模型，容易污染 Object/Spatial/Long；如确实要覆盖，
使用 `VLA_PATH_OVERRIDE`：

```bash
VLA_PATH_OVERRIDE=/path/to/openvla-suite-model
SPEC_CKPT=/path/to/goal_ckpt
SPECVLA_GOAL_CKPT=/path/to/goal_ckpt
```

### Goal 七套核心评测

| 实验 | Launcher | Python 入口 | Draft backend | Acceptance | 日志子目录 |
| --- | --- | --- | --- | --- | --- |
| OpenVLA AR baseline | `run_openvla_ar_libero_goal_eval.sh` | `run_libero_goal_AR.py` | 无 | 自回归 | `openvla_ar` |
| SpecVLA strict baseline | `run_specvla_libero_goal_eval.sh` | `run_libero_goal_Spec.py` | `eagle` | strict，`accept_threshold=0` | `specvla_strict` |
| SpecVLA relaxed baseline | `run_specvla_relaxed_libero_goal_eval.sh` | `run_libero_goal_Spec_Relaxed.py` | `eagle` | relaxed，默认 `accept_threshold=9` | `specvla_relaxed` |
| DFLASH strict ablation | `run_dflash_strict_libero_goal_eval.sh` | `run_libero_goal_Spec.py` | `dflash` | strict，`accept_threshold=0` | `dflash_strict` |
| DFLASH relaxed 当前方法 | `run_dflash_libero_goal_eval.sh` | `run_libero_goal_Spec_Relaxed.py` | `dflash` | relaxed，默认 `accept_threshold=9` | `dflash_relaxed` |
| DFLASH CADhead strict | `run_dflash_residual_strict_libero_goal_eval.sh` | `run_libero_goal_Spec.py` | `dflash` | strict，开启 residual sampling | `dflash_strict` |
| DFLASH CADhead relaxed | `run_dflash_residual_libero_goal_eval.sh` | `run_libero_goal_Spec_Relaxed.py` | `dflash` | relaxed，开启 residual sampling | `dflash_relaxed` |

### 速度计时口径

为了复现 SpecVLA 上游代码和论文表格的速度口径，当前所有 LIBERO 评测 launcher 默认使用并显式传入：

```text
SYNC_CUDA_TIMING=False
TIMING_SCOPE=last_task
```

`SYNC_CUDA_TIMING=False` 表示计时前后不额外调用 `torch.cuda.synchronize()`，与
`PineTreeWss/SpecVLA` 的 `openvla_utils.py` 保持一致。严格同步会把 GPU kernel 的真实完成时间计入
每一步，对 speculative decoding 里 draft、verify、tree/cache 等多个小 forward 更不友好，容易放大
不同显卡之间的调度差异；因此论文复现默认不用它。

`TIMING_SCOPE=last_task` 是为了对齐上游 SpecVLA 的 timing JSON 行为：其
`total_episode_time` 在 task 循环内重新初始化，最后写出的 timing 文件只对应最后一个 task。
本仓库仍然会统计完整 suite 的成功率；`*_timing.json` 和 `*_summary.json` 中的 `timing` /
`generation.length` 默认按最后一个 task 计算，以便和上游 speed 脚本更接近。若要做更稳定的工程统计，
可以显式改成：

```bash
TIMING_SCOPE=full_suite SYNC_CUDA_TIMING=True \
  bash openvla/specdecoding/decode-scripts/run_specvla_relaxed_libero_goal_eval.sh
```

### Goal：一键复现 SpecVLA baseline

为了严格复现上游 SpecVLA 的 Goal 评测语义，使用下面的一键 launcher。它串行运行
OpenVLA AR、SpecVLA strict（`r=0`）和 SpecVLA relaxed（`r=9`），统一使用 `seed=7`、
`50 trials/task`、`SYNC_CUDA_TIMING=False`、`TIMING_SCOPE=last_task`：

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash openvla/specdecoding/decode-scripts/run_specvla_goal_upstream_compatible_eval.sh
```

脚本会使用唯一的 `RUN_TAG` 防止旧日志混入，并在
`<LOG_DIR>/reproduction/<RUN_TAG>_comparison.txt` 汇总 `SR / Length / Avg Accept / Mean Step / Speedup`。
若要先 smoke test，可写 `NUM_TRIALS_PER_TASK=1`；论文复现必须回到默认的 `50`。

本仓库在这一模式下还对齐了两个容易被忽略的源码细节：成功终止 action 不写入 timing JSON，且
Length 的 CUDA 标量会在计时结束后才 materialize，不把统计开销混入 SpecVLA 的 wall-clock 时间。
这仍不能消除 RTX 4090 与论文 A100 80G 的硬件差异，但排除了可控的评测代码差异。

### 非 Goal suite 的 AR / SpecVLA baseline

当前已补齐 Object、Spatial、Long 三个 suite 的 baseline launcher。这里的 Long 对应代码里的
`libero_10`。这些专用脚本覆盖 AR、SpecVLA strict、SpecVLA relaxed；DFLASH 使用保留
`goal` 名称的通用 launcher，但可通过 `TASK_SUITE_NAME` 评测四个已有权重的 suite。

Relaxed acceptance 的默认阈值按论文设置：Goal/Object/Spatial 使用 `r=9`，Long（`libero_10`）
使用 `r=5`。`run_specvla_relaxed_libero_10_eval.sh` 与所有支持 `TASK_SUITE_NAME=libero_10` 的
DFLASH relaxed launcher 都会自动选 `r=5`；只有显式传入 `ACCEPT_THRESHOLD` 时才覆盖该规则。
例如评测 DFLASH CAD-head 的第 200 epoch Long checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 TASK_SUITE_NAME=libero_10 EVAL_EPOCH=200 \
  bash openvla/specdecoding/decode-scripts/run_dflash_residual_libero_goal_eval.sh
```

| LIBERO suite | AR | SpecVLA strict | SpecVLA relaxed |
| --- | --- | --- | --- |
| Object | `run_openvla_ar_libero_object_eval.sh` | `run_specvla_libero_object_eval.sh` | `run_specvla_relaxed_libero_object_eval.sh` |
| Spatial | `run_openvla_ar_libero_spatial_eval.sh` | `run_specvla_libero_spatial_eval.sh` | `run_specvla_relaxed_libero_spatial_eval.sh` |
| Long (`libero_10`) | `run_openvla_ar_libero_10_eval.sh` | `run_specvla_libero_10_eval.sh` | `run_specvla_relaxed_libero_10_eval.sh` |

这些 launcher 会自动选择 4090 上的 suite-specific OpenVLA 权重和 SpecVLA checkpoint，例如：

```text
OpenVLA Object  : /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-object
SpecVLA Object  : /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/specvla_checkpoint/object
OpenVLA Spatial : /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-spatial
SpecVLA Spatial : /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/specvla_checkpoint/spatial
OpenVLA Long    : /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/hf_files/openvla-7b-finetuned-libero-10
SpecVLA Long    : /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/specvla_checkpoint/10
```

`libero_90` 暂未写一键脚本，因为当前 4090 没有准备对应的 OpenVLA fine-tuned model 和 SpecVLA
checkpoint。若后续补齐权重，可用 `init_libero_eval_env libero_90` 按同一模板扩展。

AR baseline 使用标准 OpenVLA 模型，故意不向模型传 `generate_mode`、`return_dflash_stats`
这类 SpecVLA/DFlash 专用 generation 参数。2026-06-28 曾因误传这些参数导致 AR 每个 episode
一开始就异常退出，表现为“推得很快但成功率全 0”；该问题已在 `a5817d5` 修复。

### 评测机器约定

正式速度评测默认 **不在 3090 上跑**。3090 推理时 SpecVLA/DFlash 的小 kernel、校验和调度开销
会明显吃掉投机解码收益，曾观察到 strict/relaxed speedup 明显偏低。3090 可以临时做成功率 sanity check，
但论文式 `SR / Length / Speedup` 统一在 4090 上记录。

### 评测输出和覆盖变量

`run_libero_goal_Spec.py` 和 `run_libero_goal_Spec_Relaxed.py` 现在都会为 SpecVLA/EAGLE 与
DFLASH 记录论文 Table 1 风格的 `Length`。评测结束后，每个 run 会写三类文件：

```text
*.txt             人类可读日志：成功率、Length、accept length、hit rate
*_timing.json    每个环境 step 的 action-generation 耗时
*_summary.json   汇总指标：success_rate、timing.mean、generation.length
```

`generation.length` / `generation.table1_length` 的口径是：

```text
Length = 该 run 实际推进的 action token 总数 / speculative block 数
```

对 OpenVLA action 来说，每个 policy step 目标是 7 个 action token。DFLASH 会先由 target
prefill 得到第一个 action token，再用 block draft 推进后续 token；summary 中的 `Length`
仍把这个首 token 放进总推进 token 数里，以便接近 SpecVLA 论文 Table 1 的
“每次 forward 平均预测/推进 token 数”口径。`generated_tokens` / `avg_generated_length`
只是调试字段：部分 backend 会把它记成固定 action 宽度 7，不能拿来当论文式 Length。
`avg_accept_length` 则保留更底层的含义：平均每个 block 真正接受了多少 draft token，
不等同于 Table 1 的 `Length`。DFLASH 还会额外记录 `avg_tail_progress_length`，
用于观察去掉 bootstrap 首 token 后的 tail-block 推进长度。

可用下面的脚本把 AR 和若干 speculative summary 汇总成论文式表格：

```bash
python openvla/specdecoding/test-speed/summarize_eval_summaries.py \
  --ar-summary /path/to/openvla_ar_summary.json \
  /path/to/specvla_strict_summary.json \
  /path/to/specvla_relaxed_summary.json \
  /path/to/dflash_strict_summary.json \
  /path/to/dflash_relaxed_summary.json
```

一次性覆盖变量可以写在 `bash` 前面：

```text
VLA_PATH_OVERRIDE
SPEC_CKPT
SPECVLA_CKPT
SPECVLA_GOAL_CKPT
SPECVLA_CKPT_ROOT
DFLASH_OUTPUT_DIR
EVAL_EPOCH
LOG_DIR
NUM_TRIALS_PER_TASK
RUN_ID_NOTE
USE_WANDB
SEED
```

### 4. 4090：统一推理评测

4090 是固定推理评测机器。每次从 3090 搬来 checkpoint 后，在 4090 上跑 Goal 的 AR、SpecVLA strict、
SpecVLA relaxed、DFLASH strict、DFLASH relaxed、DFLASH CADhead strict、DFLASH CADhead relaxed
七套实验，最终比较 `SR / Length / Speedup`。
其它 suite 先跑 AR / SpecVLA baseline，作为后续扩展 DFLASH 的公平对照。

进入环境：

```bash
ssh 4090
source ~/.bashrc
conda activate specvla
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH
```

设置本次要评测的 DFLASH checkpoint。这里以当前 Markov-ACD 第 200 epoch 为例：

```bash
export DFLASH_CKPT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu/epoch_200_step_089400
export NUM_TRIALS_PER_TASK=50
export CUDA_VISIBLE_DEVICES=0
```

建议每个长评测都放在 screen 里跑，例如：

```bash
screen -S eval_dflash_strict
```

七套评测命令如下。DFlash 四个命令必须显式传 `SPEC_CKPT="${DFLASH_CKPT}"`，确保评测的是刚从
3090 复制来的权重。建议串行跑，不要同时启动多个评测，否则速度数字会被 CPU、MuJoCo、图像预处理和
Python 调度共享开销污染。

1. OpenVLA AR baseline：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_openvla_ar_libero_goal_eval.sh
```

2. SpecVLA strict baseline：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_specvla_libero_goal_eval.sh
```

3. SpecVLA relaxed baseline：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_specvla_relaxed_libero_goal_eval.sh
```

4. DFLASH strict：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  SPEC_CKPT="${DFLASH_CKPT}" \
  bash openvla/specdecoding/decode-scripts/run_dflash_strict_libero_goal_eval.sh
```

5. DFLASH relaxed：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  SPEC_CKPT="${DFLASH_CKPT}" \
  bash openvla/specdecoding/decode-scripts/run_dflash_libero_goal_eval.sh
```

6. DFLASH CADhead strict：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  SPEC_CKPT="${DFLASH_CKPT}" \
  bash openvla/specdecoding/decode-scripts/run_dflash_residual_strict_libero_goal_eval.sh
```

7. DFLASH CADhead relaxed：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  SPEC_CKPT="${DFLASH_CKPT}" \
  bash openvla/specdecoding/decode-scripts/run_dflash_residual_libero_goal_eval.sh
```

如果只是想在 3090 上快速 sanity check，可以把五套评测分别绑到不同 GPU 并行跑；但这会共享 CPU、
MuJoCo、图像预处理、磁盘和 Python 调度资源，速度数值只作工程参考。2026-07-05 的 3090 临时并行评测
就是这种口径，不应直接写成论文速度。

SpecVLA strict / relaxed 的四个 suite 可以用一键脚本自动续跑。它会扫描 `eval_logs/specvla_strict` 和
`eval_logs/specvla_relaxed`，默认跳过已有 summary 或已有 txt/timing/summary artifact 的 run。因此如果 Goal strict
已经跑过或正在跑，就不会重复启动。正式启动前建议先 dry-run 看计划：

```bash
# 只打印将跳过/将运行的项，不启动 LIBERO。
DRY_RUN=True CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_specvla_main_table_eval.sh

# 确认无误后正式续跑缺失项。
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_specvla_main_table_eval.sh
```

如果某个失败 run 留下了 txt，脚本也会默认跳过它；这种情况下可用 `FORCE_RERUN=True` 强制重跑，或手动删除对应 artifact。
脚本结束后会写：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs/main_table_specvla_baselines.csv
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs/main_table_specvla_baselines.md
```

其它 suite baseline 也仍可照下面逐个跑。每条命令都会自动使用对应 suite 的 OpenVLA 和 SpecVLA checkpoint：

```bash
# Object
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_openvla_ar_libero_object_eval.sh
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_specvla_libero_object_eval.sh
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_specvla_relaxed_libero_object_eval.sh

# Spatial
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_openvla_ar_libero_spatial_eval.sh
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_specvla_libero_spatial_eval.sh
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_specvla_relaxed_libero_spatial_eval.sh

# Long / libero_10
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_openvla_ar_libero_10_eval.sh
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_specvla_libero_10_eval.sh
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_specvla_relaxed_libero_10_eval.sh
```

### DFlash CAD-head checkpoint sweep 和自动主表

当前主表优先比较五类方法：OpenVLA AR、SpecVLA strict、SpecVLA relaxed、DFlash CAD-head strict、DFlash CAD-head relaxed。
AR baseline 的四个 suite summary 已放在：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs/openvla_ar
```

DFlash CAD-head 一键评测链只跑 strict/relaxed 两个 CAD-head 版本，并自动用同 suite 的 AR `timing.mean` 计算 Speedup：

```bash
# 默认跑 libero_goal/libero_object/libero_spatial/libero_10，默认扫 120/150/180/200 四个 epoch。
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_cad_head_main_table_eval.sh

# 当前推荐先测 Goal 的 epoch 100/150/200：
TASK_SUITES="libero_goal" EVAL_EPOCHS="100 150 200" \
DFLASH_OUTPUT_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_markov_acd_start0_slotdecay090_tokence_soft01_b16_4gpu \
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_cad_head_main_table_eval.sh
```

评测链结束后会写：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs/main_table_dflash_cad_head.csv
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs/main_table_dflash_cad_head.md
```

如果已经手动跑完若干实验，只想重新汇总，不启动 LIBERO rollout：

```bash
LOG_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs
python openvla/specdecoding/test-speed/summarize_main_table_eval.py \
  --log-dir "${LOG_DIR}" \
  --ar-dir "${LOG_DIR}/openvla_ar" \
  --output-csv "${LOG_DIR}/main_table_dflash_cad_head.csv" \
  --output-md "${LOG_DIR}/main_table_dflash_cad_head.md"
```

这个汇总脚本会自动过滤普通 DFlash，只收 `dflash_use_causal_residual_sampling=true` 的 DFlash CAD-head 结果；
Speedup 一律相对同 suite 的 OpenVLA AR baseline。

评测结束后，旧的手动汇总方式仍可用。由于普通 DFLASH 和 CADhead 都写入 `dflash_strict` / `dflash_relaxed`
目录，不能只靠 `ls -t | head -1` 自动判断是哪一组；刚跑完一整套时可以先这样取最新文件，
正式记录时必须手动核对 `run_id`、`dflash_use_causal_residual_sampling`、`SPEC_CKPT` 和
`accept_threshold`：

```bash
LOG_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs

AR=$(ls -t ${LOG_DIR}/openvla_ar/*_summary.json | head -1)
SPEC=$(ls -t ${LOG_DIR}/specvla_strict/*_summary.json | head -1)
SPEC_R=$(ls -t ${LOG_DIR}/specvla_relaxed/*_summary.json | head -1)
DFLASH=$(ls -t ${LOG_DIR}/dflash_strict/*_summary.json | head -1)
DFLASH_R=$(ls -t ${LOG_DIR}/dflash_relaxed/*_summary.json | head -1)

python openvla/specdecoding/test-speed/summarize_eval_summaries.py \
  --ar-summary "$AR" "$SPEC" "$SPEC_R" "$DFLASH" "$DFLASH_R"
```

CADhead 的两个 summary 也在 `dflash_strict` / `dflash_relaxed` 目录里，识别依据是 summary JSON 中：

```text
dflash_use_causal_residual_sampling = true
generation.use_causal_residual_sampling = true
```

## 3090 LIBERO EGL 问题记录

3090 曾出现过 LIBERO/robosuite EGL 相关报错：

```text
RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer between 0 and -1
ImportError: Cannot initialize a EGL device display
```

当时观察到的根因是 NVIDIA driver 和用户态 EGL library 不匹配：

```text
nvidia-smi: driver 570.133.07
system EGL libraries: 535 系列
```

无 sudo 权限下的修复方式是解包匹配 driver 版本的 NVIDIA runfile 到用户目录，并让 GLVND 指向这个 shim。
辅助脚本：

```bash
bash openvla/specdecoding/decode-scripts/setup_3090_nvidia_egl_shim.sh
```

默认 shim 路径：

```text
/data/wulin/c/nvidia-egl-570.133.07/slim-lib
/data/wulin/c/nvidia-egl-570.133.07/egl_vendor.d/10_nvidia_570.json
```

不要手动设置 `MUJOCO_EGL_DEVICE_ID=0`，除非明确知道原因。之前 `0 to -1` 报错就是由破损 EGL 栈下的
这个设置直接触发的。选择推理 GPU 用 `CUDA_VISIBLE_DEVICES=<gpu_id>`。

## Git 和服务器工作流

当前维护逻辑：

```text
代码: 4090 主开发/提交机器 -> GitHub main -> 3090 按需同步训练代码
权重: 3090 四卡训练输出 -> 本地 scp -3 -> 4090 推理评测
```

后续默认建议：

1. 在 4090 上做代码或文档改动并验证。
2. 只提交与当前改动相关的文件，推送到
   [guanghaichen/SpecVLA-DFLASH](https://github.com/guanghaichen/SpecVLA-DFLASH)。
3. 不把未提交的 4090 改动直接复制到 3090。
4. GitHub 包含目标 commit 后，再按训练需要同步 3090。
5. 3090 训练完只复制 checkpoint 到 4090；不要把 3090 的临时改动反向覆盖 4090 代码。

### 服务器 deploy key 和 GitHub 同步

私有仓库推荐每台服务器使用单独的 GitHub deploy key。私钥只放在服务器用户的 `~/.ssh` 下，
README 只记录配置方法，不记录私钥内容。

以 3090 为例，本地已经有项目专用 deploy key：

```text
/Users/chenguanghai/.ssh/id_specvla_dflash_ro
/Users/chenguanghai/.ssh/id_specvla_dflash_ro.pub
```

把 key 放到服务器并设置权限：

```bash
scp ~/.ssh/id_specvla_dflash_ro ~/.ssh/id_specvla_dflash_ro.pub 3090_wulin:~/.ssh/
ssh 3090_wulin
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_specvla_dflash_ro
chmod 644 ~/.ssh/id_specvla_dflash_ro.pub
```

如果服务器上的仓库 remote 还是 HTTPS，会在非交互环境里报：

```text
fatal: could not read Username for 'https://github.com': No such device or address
```

改成 SSH remote，并只给当前仓库绑定这把 key：

```bash
cd /data/wulin/c/SpecVLA-DFLASH
git remote set-url origin git@github.com:guanghaichen/SpecVLA-DFLASH.git
git config core.sshCommand \
  "ssh -i ~/.ssh/id_specvla_dflash_ro -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
git ls-remote origin HEAD
```

`git ls-remote origin HEAD` 能返回 commit hash，就说明 deploy key 至少有读权限。如果要从服务器推送，
需要在 GitHub 仓库的 deploy key 设置里勾选 write access。推送前先确认远端没有新提交：

```bash
git fetch origin
git status --short --branch
git log --oneline --decorate --graph --max-count=12 --all
```

如果显示 `ahead` 且没有 `behind`，可以直接：

```bash
git push origin main
```

如果同时显示 `ahead` 和 `behind`，先 rebase 到最新远端：

```bash
git rebase origin/main
```

出现冲突时按下面顺序处理：

```bash
grep -RIn "<<<<<<<\|=======\|>>>>>>>" README.md openvla || true
git diff --check
# 手动保留两边真正需要的内容，尤其不要把 4090 路径回退成旧 4090 路径
git add <resolved-files>
GIT_EDITOR=true git rebase --continue
```

完成后必须做最小检查：

```bash
git diff --check
bash -n openvla/specdecoding/decode-scripts/*.sh
python -m py_compile \
  openvla/experiments/robot/openvla_utils.py \
  openvla/experiments/robot/robot_utils.py \
  openvla/experiments/robot/libero/eval_metrics.py
git push origin main
```

2026-07-05 的一次实际经验：3090 原本 `origin` 是 HTTPS，且没有 GitHub 凭据，导致无法推送；
配置 deploy key 后成功把 `main` 推到 GitHub。之后 3090 可以直接读写私有仓库，但原则上仍然只在需要同步
训练脚本或紧急修复时从 3090 推送；常规开发优先放在 4090。

### 新服务器迁移检查表

迁移到一台新机器时，按这个顺序检查，不要只复制代码就直接跑实验：

1. GitHub 权限：配置 deploy key，确认 `git ls-remote origin HEAD` 能读私有仓库。
2. 目录约定：建立 `SpecVLA-DFLASH`、`specvla-data`、`hf_files`、`hf-cache`、`LIBERO` 等固定目录。
3. `.bashrc`：写入 `HF_ENDPOINT=https://hf-mirror.com`、HF cache、本地模型路径、`PYTHONPATH`。
4. Conda 环境：确认 `python`、`torch`、`transformers`、`flash_attn`、`robosuite`、`mujoco`、`swanlab`。
5. 模型权重：优先从 `hf-mirror.com` 或已有服务器复制到本地路径，避免脚本隐式访问 Hugging Face 官方站。
6. 数据和 checkpoint：训练机需要 RLDS 与 DFLASH 离线数据；纯评测机器只需要 OpenVLA、SpecVLA/DFlash checkpoint 和 LIBERO。
7. MuJoCo/EGL：先用小规模 LIBERO episode sanity check，发现 EGL 报错再处理 driver/library 匹配问题。
8. 代码同步：先 `git fetch`、`git status --short --branch`，确认没有未提交改动或远端分叉。
9. 最小验证：跑 `bash -n`、`python -m py_compile`，再用 `NUM_TRIALS_PER_TASK=1` 做评测 smoke test。
10. 实验记录：保存 commit、launcher、数据目录、checkpoint、评测 seed、timing 口径。

每次实验前，至少记录下面信息：

```text
Git commit
数据目录和样本数
launcher
checkpoint path
selected-hidden variant
acceptance threshold
evaluation seed
```

这些信息是让结果可比较而不是只停留在经验描述的最低要求。

## 2026-07-14 推理 speedup 修复记录

这次问题的背景是：clean upstream 复现环境在 LIBERO goal 上重新跑通后，`test_speed_for_json.py` 得到的 speedup 为 Spec 约 `1.00x`、Spec Relaxed 约 `1.31x`，已经接近原 SpecVLA 论文复现结果；而主项目 `SpecVLA-DFLASH` 之前一度出现 SpecVLA 负加速或 speedup 异常。排查结论是：不能把原因简单归结为 MuJoCo 一个包，实际漂移点至少包括 MuJoCo、robosuite、LIBERO 指向、numpy、accelerate，以及部分基础依赖。

### 主要原因判断

最可疑、也最直接影响 LIBERO 推理复现的差异是仿真栈：

```text
SpecVLA eval script
-> LIBERO task/env wrapper
-> robosuite
-> mujoco Python package
-> MuJoCo physics/render backend
```

其中 `robosuite==1.4.1` 只声明 `mujoco>=2.3.0`，没有上限；如果不额外 pin，现在 fresh install 会解析到更新的 MuJoCo，而不是作者 W&B 快照里记录过的 `mujoco==3.3.4`。历史上主环境被手动改成 `mujoco==3.3.2`，这很可能是 speedup 和成功率开始异常的触发点之一。

但这次不是只修 MuJoCo。修复前实际看到的漂移包括：

```text
4090 old specvla:
  mujoco==3.3.2
  numpy==1.26.2
  accelerate==1.14.0
  opencv-python==5.0.0.93
  libero editable -> /media/asus/.../cgh/LIBERO

3090 old specvla:
  robosuite==1.4.0
  mujoco==3.3.2
  accelerate==1.14.0
  protobuf==6.32.0
  wrapt==2.2.2
  libero editable -> /data/wulin/c/LIBERO
```

`LIBERO` 源码 commit 在两台机器上都是 `8f1084e3132a39270c3a13ebe37270a43ece2a01`，和 clean checkout 一致；所以这里的 LIBERO 问题主要不是源码版本不同，而是路径/config/环境包混杂让排查变复杂。后续评测时应显式确认 `PYTHONPATH` 和 `~/.libero/config.yaml` 指向当前机器对应的 LIBERO 目录。

### 已对齐的旧 specvla 环境

2026-07-14 已直接修改两台机器的旧 `specvla` conda 环境，没有保留新的 aligned 环境：

```text
4090:
  /home/asus/miniconda3/envs/specvla
  repo: /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH
  LIBERO: /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/LIBERO

3090:
  /data/wulin/miniconda3/envs/specvla
  repo: /data/wulin/c/SpecVLA-DFLASH
  LIBERO: /data/wulin/c/LIBERO
```

最终关键版本对齐为：

```text
mujoco==3.3.4
robosuite==1.4.1
numpy==1.26.4
accelerate==1.9.0
opencv-python==4.12.0.88
protobuf==4.21.12
wrapt==1.14.2
pynput==1.8.2
termcolor==3.3.0
libero==0.1.0
bddl==3.6.0
transformers==4.40.1
torch==2.2.0+cu121
torchvision==0.17.0+cu121
torchaudio==2.2.0+cu121
```

这些版本的来源和优先级：

- SpecVLA README 明确写了 `torch==2.2.0`、CUDA 12.1、`libero==0.1.0`。
- SpecVLA/OpenVLA 的 `openvla/experiments/robot/libero/libero_requirements.txt` 写了 `robosuite==1.4.1`。
- 上游 SpecVLA GitHub 中提交的 W&B offline run 环境快照记录过 `mujoco==3.3.4`、`numpy==1.26.4`、`accelerate==1.9.0`、`torch==2.2.0+cu121` 等。
- `mujoco==3.3.4` 不是 README 官方 pin 的版本，但它比不指定版本更接近上游仓库残留的实验环境证据。

### 修复时使用的安装命令

4090：

```bash
/home/asus/miniconda3/envs/specvla/bin/python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --no-cache-dir --force-reinstall --no-deps \
  mujoco==3.3.4 robosuite==1.4.1 numpy==1.26.4 accelerate==1.9.0

/home/asus/miniconda3/envs/specvla/bin/python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --no-cache-dir --force-reinstall --no-deps \
  opencv-python==4.12.0.88 protobuf==4.21.12 wrapt==1.14.2
```

3090：

```bash
/data/wulin/miniconda3/envs/specvla/bin/python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --no-cache-dir --force-reinstall --no-deps \
  mujoco==3.3.4 robosuite==1.4.1 numpy==1.26.4 accelerate==1.9.0

/data/wulin/miniconda3/envs/specvla/bin/python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --no-cache-dir pynput termcolor

/data/wulin/miniconda3/envs/specvla/bin/python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --no-cache-dir --force-reinstall --no-deps \
  opencv-python==4.12.0.88 protobuf==4.21.12 wrapt==1.14.2
```

### 验证命令

4090：

```bash
ROOT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh
PY=/home/asus/miniconda3/envs/specvla/bin/python
cd "$ROOT/SpecVLA-DFLASH"

PYTHONPATH="$ROOT/LIBERO:$ROOT/SpecVLA-DFLASH/openvla:$ROOT/SpecVLA-DFLASH" "$PY" - <<'PY'
import importlib.metadata as md, torch, cv2
for p in ["mujoco", "robosuite", "numpy", "accelerate", "opencv-python", "protobuf", "wrapt", "pynput", "termcolor", "libero", "bddl", "transformers"]:
    print(f"{p}=={md.version(p)}")
import libero, robosuite, mujoco
print("torch", torch.__version__, torch.version.cuda)
print("cv2", cv2.__version__)
print("imports_ok")
PY

bash -n openvla/specdecoding/decode-scripts/libero_eval_common.sh \
  openvla/specdecoding/decode-scripts/run_openvla_ar_libero_goal_eval.sh \
  openvla/specdecoding/decode-scripts/run_specvla_libero_goal_eval.sh \
  openvla/specdecoding/decode-scripts/run_specvla_relaxed_libero_goal_eval.sh
```

3090：

```bash
ROOT=/data/wulin/c
PY=/data/wulin/miniconda3/envs/specvla/bin/python
cd "$ROOT/SpecVLA-DFLASH"

PYTHONPATH="$ROOT/LIBERO:$ROOT/SpecVLA-DFLASH/openvla:$ROOT/SpecVLA-DFLASH" "$PY" - <<'PY'
import importlib.metadata as md, torch, cv2
for p in ["mujoco", "robosuite", "numpy", "accelerate", "opencv-python", "protobuf", "wrapt", "pynput", "termcolor", "libero", "bddl", "transformers"]:
    print(f"{p}=={md.version(p)}")
import libero, robosuite, mujoco
print("torch", torch.__version__, torch.version.cuda)
print("cv2", cv2.__version__)
print("imports_ok")
PY

bash -n openvla/specdecoding/decode-scripts/libero_eval_common.sh \
  openvla/specdecoding/decode-scripts/run_openvla_ar_libero_goal_eval.sh \
  openvla/specdecoding/decode-scripts/run_specvla_libero_goal_eval.sh \
  openvla/specdecoding/decode-scripts/run_specvla_relaxed_libero_goal_eval.sh
```

### 后续注意事项

1. 不要再无版本号执行 `pip install mujoco` 或 `pip install -U robosuite mujoco`；否则 MuJoCo 会漂到当前最新版本。
2. 跑 LIBERO 推理前先确认 `MUJOCO_GL=egl`、`MUJOCO_EGL_DEVICE_ID`、`PYTHONPATH` 和 `LIBERO_PATH`。
3. 本项目的 speedup 统计脚本看的是 JSON 中每 step 的模型推理时间；仿真栈变化不一定直接改变单步计时代码，但会改变 rollout 轨迹、成功/失败路径和统计样本，因此仍会间接影响最终 speedup。
4. 如果要比较 DFLASH 与 SpecVLA，上游兼容模式应保持 `SYNC_CUDA_TIMING=False`、`TIMING_SCOPE=last_task`，这对应作者脚本的 paper-style timing。
5. 若再次出现负加速，先记录 `pip freeze | grep -E 'mujoco|robosuite|numpy|accelerate|opencv|protobuf|wrapt|libero|bddl|torch'`，再比较 JSON 的 steps/mean/median，不要只看最终 speedup。

## 环境备注

- 原 SpecVLA 环境备注：Python 3.10、PyTorch 2.2.0 + CUDA 12.1、LIBERO 0.1.0。
- 依赖安装后，在仓库中执行 `cd openvla && pip install -e .`。
- 模型和 RLDS 数据都应使用本地路径。不要让训练或数据生成意外回退到远程下载。
- 本仓库中仍可能保留 SpecVLA 历史脚本和注释。判断 DFLASH 当前行为时，以“项目地图”列出的文件为准。

## 参考文献

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
