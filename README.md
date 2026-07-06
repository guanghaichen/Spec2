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

### 当前 loss 和训练策略

当前 pure-training recipe 使用 hidden-state distillation，而不是 token-level CE：

```text
total = 1.0 * hidden_loss + 0.05 * cosine_hidden_loss
soft_w = 0
anchor_consistency_w = 0
```

Token accuracy 只作为诊断指标保留，不进入优化目标。早期 token soft-distribution 实验在这个离线 block setting
中出现较早的 validation deterioration，所以当前默认不启用；相关代码仍通过 `--soft_w`、`--soft_temperature`
保留为可控消融。当前 recipe 中 hidden-context noise 是 `0.05`。Per-anchor 和 per-position 指标会记录到
SwanLab 和本地 JSONL。

当前 recipe 的长期控制信号是 LIBERO simulator behavior，而不是离线 validation split 的 early stopping。
因此 pure-training launcher 使用 `--val_split 0`，默认不做 validation/early stopping，并且每 10 个 epoch 保存一次 checkpoint。

### 当前新分支：跨 Anchor 因果蒸馏 + 前序 Token 残差修正

2026-06-30 新增的主攻分支针对一个非常明确的现象：`anchor_a_to_position_(a+1)` 通常很高，
但 `anchor_a_to_position_(a+2..)` 明显下降。这说明 draft 擅长一步预测，却不擅长在
`[t_anchor, MASK, MASK, ...]` 的薄前缀条件下预测远 slot。

这个分支最初只做两件事：

1. **跨 Anchor 因果蒸馏。** 对同一个目标 token，用一步强路径当老师。例如预测 `t5` 时，
   `anchor=4 -> t5` 是强路径；`anchor=0/1/2/3 -> t5` 都是弱路径。训练时让弱路径靠近强路径。
2. **前序 Token 残差修正。** 弱路径不是直接硬追老师，而是先用一个共享小残差头读取“前一个 token”，
   给远 slot hidden 补一小段因果残差。slot0 默认不修正，因为一步预测本来已经很强；
   从 slot1 开始修正，默认只重点覆盖 p2-p5。

设计目标是：训练时残差头看到的前序 token 来自 target model 真实轨迹；推理时没有真实未来 token，
所以用刚生成的 draft token 作为下一 slot 的前序条件。DFlash transformer 主干仍然只跑一次，
后面只是轻量 residual head + frozen `lm_head` 的逐 slot 修正。

2026-07-06 复盘发现：早先的主 runtime 走的是 `include_anchor_hidden=True` 的
`_dflash_generate_with_anchor_hidden` 分支，当时该分支还没有真正接入 `sample_with_causal_residual`，
而是直接对 `draft_hidden` 做 `lm_head` 后采样。也就是说，旧 DFLASH 评测虽然使用了训练过的
residual/CAD checkpoint，但并没有在线启用 residual head。现在已经补上显式开关
`dflash_use_causal_residual_sampling`，旧 launcher 默认关闭以保留可复现性，`CADhead` launcher
默认开启以做对照。

2026-07-01 的 b16 训练观察到：`anchor_0_to_position_1_acc` 和 `anchor_1_to_position_2_acc`
可以较快升到 0.8 左右，但 `anchor_0_to_position_2_acc` 仍明显滞后，且 `accuracy - base_accuracy`
接近 0。这说明旧 Residual-CAD 只让弱路径追“强路径的影子”还不够，需要让弱路径更直接地追
target hidden，并给瓶颈位置更高权重。当前 launcher 已切到 weak-path recipe：

1. **弱路径 refined hidden 直追 target hidden。** 对 p2-p5 的远 slot，额外监督 residual head 输出后的
   `refined_hidden -> target_hidden`。
2. **弱路径加权。** `slot_decay=1.0`，不再衰减远 slot；p2-p5 弱路径默认 2x，
   `anchor0->p2` 默认 4x。
3. **更干净的 CAD teacher。** 只有一步强路径预测 token 正确时，才把该强路径 hidden 作为 CAD teacher。
4. **CAD 距离改为 cosine。** CAD 更关注方向/语义结构，而不是强行复刻 raw hidden 数值尺度。

2026-07-05/06，3090 上跑完 weak-path b16 训练和 Goal suite 临时并行评测，结论如下：

```text
训练输出目录:
/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_residual_cad_weakpath_b16_4gpu

latest_checkpoint:
epoch_200_step_089400

训练规模:
train_files = 28,576
world_size = 4
per_device_batch = 16
global_effective_batch = 64
epochs = 200
total_optimizer_steps = 89,400
trainable_params = 296.62M
```

训练端最后一步诊断：

| 指标 | 最后值 | 训练中最好值 |
| --- | ---: | ---: |
| `train/loss` | 0.753 | 0.748 |
| `train/hidden_loss` | 0.559 | 0.556 |
| `train/refined_hidden_loss` | 0.543 | 0.539 |
| `train/cos_loss` | 0.221 | 0.218 |
| `train/accuracy` | 0.799 | 0.830 |
| `anchor0 -> p1 acc` | 0.846 | 0.906 |
| `anchor0 -> p2 acc` | 0.671 | 0.766 |
| `anchor0 -> p3 acc` | 0.717 | 0.758 |
| `anchor0 -> p4 acc` | 0.647 | 0.766 |
| `anchor0 -> p5 acc` | 0.718 | 0.781 |
| `anchor0 -> p6 acc` | 0.936 | 0.984 |

相比上一版 puretrain，`anchor0 -> p2 acc` 从最后约 `0.503` 提升到约 `0.671`，
说明 weak-path 加权和 CAD 训练信号确实改变了离线训练行为；但这并没有充分转化为在线
speculative acceptance。

同一批 3090 临时并行评测结果：

| 方法 | SR | mean step time | Speedup vs AR | Length | avg accept | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| OpenVLA AR | 0.768 | 0.2826s | 1.00x | - | - | baseline |
| SpecVLA strict | 0.762 | 0.3211s | 0.88x | 1.396 | 0.396 | 3090 上负加速 |
| SpecVLA relaxed | 0.736 | 0.2709s | 1.04x | 2.372 | 1.372 | relaxed 后才略快 |
| DFLASH strict | 0.760 | 0.2905s | 0.97x | 2.113 | 1.043 | Length 高于 Spec strict 但速度仍不够 |
| DFLASH relaxed | 0.768 | 0.2691s | 1.05x | 2.416 | 1.339 | SR 与 AR 持平，速度略高于 Spec relaxed |

DFLASH relaxed 的在线 per-position hit rate：

```text
p1 = 0.919
p2 = 0.369
p3 = 0.416
p4 = 0.428
p5 = 0.387
p6 = 0.626
overall_hit_rate = 0.342
```

核心诊断：

1. 训练端 anchor0 的 p2-p5 已到 `0.65-0.72`，但在线 relaxed hit rate 只有 `0.37-0.43`。
   这是明显的 offline/online gap。
2. DFLASH 的 Length 已经略高于 SpecVLA relaxed，但速度收益只有 `1.05x`，说明 3090 上 draft/verify
   小 forward、Python 调度和并行评测共享资源的开销很容易吃掉投机收益。
3. 这五个 eval 在 3090 上几乎同时启动，分别占用 GPU 0-4；它们适合作 sanity check，不应作为论文最终速度。
   正式 `SR / Length / Speedup` 仍应在 4090 上单实验串行跑。
4. 由于 residual head 没有接入 anchor-hidden 推理分支，本次结果不能判定 residual-CAD 机制失败。
   更准确地说，它暴露的是“只靠训练时 CAD/weak-path 正则，不在推理时使用 residual 修正”不足以明显反超。

下一步行动建议：

1. **先跑 CADhead 推理消融，不急着重训。** 当前已经把 `sample_with_causal_residual` 接入
   `_dflash_generate_with_anchor_hidden`，专用 residual launcher 会让 slot1 之后用已生成的 draft token
   做轻量 residual 修正。
2. **复用已有 checkpoint 做消融。** 当前训练最好 `train/loss` 和 `train/accuracy` 在 epoch 178 左右；
   `anchor0 -> p2` 最好在 epoch 118 左右。优先评测 epoch 120、180、190、200，而不是立刻新训。
3. **正式速度只在 4090 串行跑。** 3090 并行评测只能用来确认成功率和大致趋势。最终报告必须记录
   `timing_scope`、`sync_cuda_timing`、GPU 型号、是否并行跑其它 eval。
4. **如果 residual sampling 接入后 Length/Speedup 仍不动，再考虑训练侧变化。** 例如提高 p2-p5 的
   online-aligned 监督、加入轻量 token/hidden rerank 头，或重新设计让训练目标更贴近在线 self-generated
   prefix 的机制。

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
--causal_residual_start_index 1
--causal_residual_cad_w 0.10
--causal_residual_cad_type cosine
--causal_residual_cad_warmup_steps 4000
--causal_residual_cad_correct_teacher_only
--causal_residual_min_position 2
--causal_residual_max_position 5
--refined_hidden_w 0.30
--weak_far_slot_boost 2.0
--anchor0_p2_boost 4.0
--slot_decay 1.0
```

SwanLab/JSONL 会额外记录：

```text
causal_residual_cad_loss
causal_residual_cad_component
causal_residual_cad_pairs
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

## 实验历程

这里记录的是设计决策，不代表实验问题已经被解决。

1. **初始迁移：** 把 DFlash-style draft 插入 SpecVLA/OpenVLA speculative 路径。早期 draft context 不足，
   acceptance 基本不可用。
2. **Context 修正：** 数据和 runtime 改为保留完整 prefill hidden sequence 和 target-verified action history。
   当前 `include_anchor_hidden` 路径会在每次并行 tail proposal 前，先用 target 解码 anchor。
3. **离线监督修正：** 加入 multi-anchor supervision、action-dimension embedding、position balancing、
   hidden loss、cosine loss 和诊断指标。早期 hard-token-CE objective 已从主 recipe 中移除。
4. **Soft-loss 和 consistency 消融：** soft token-distribution 和 cross-anchor consistency 都跑过诊断实验；
   相关 flag 仍保留，但都不是当前 pure-training recipe。
5. **当前主实验：** 使用完整 28,639 样本数据集训练 1-layer draft，五层 context 特征为
   `[1, 8, 15, 29, final]`，`soft_w=0`，`anchor_consistency_w=0`，不使用离线 validation split；
   然后用 LIBERO simulator 比较 checkpoint 的成功率、acceptance length、hit rate 和 wall-clock time。
6. **Residual-CAD 新分支：** 观察到所有 anchor 的一步预测都明显强于远 slot 后，新增
   “跨 Anchor 因果蒸馏 + 前序 Token 残差修正”。它不是直接把 DSpark 的 logits Markov head 搬过来，
   而是让远 slot 的 hidden 经前序 token 残差修正后追同一目标位置的一步强路径。

需要始终记住的限制：

- Block draft 的 block 内输入是非因果的，未来 slot 在单次 draft forward 中不会收到 ground-truth causal prefix。
  这是并行设计的核心建模风险，不是可以用离线 token metric 掩盖的小问题。
- 低 hidden loss 不等于有效 speculative speedup。真正决定速度的是在线 acceptance distribution 和 target-call count。
- Relaxed acceptance 可能保持实际动作效果，但 token 层面不等于 strict equality。必须做消融并诚实报告阈值。


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

### 2. `.bashrc` 中固定镜像源和本地路径

4090 后续尽量不要让脚本自动访问官方 Hugging Face。把下面内容追加到 `~/.bashrc` 后重新登录，
或执行 `source ~/.bashrc`：

```bash
# SpecVLA-DFLASH paths on 4090
export SPECVLA_ROOT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh
export SPECVLA_REPO=${SPECVLA_ROOT}/SpecVLA-DFLASH
export SPECVLA_DATA=${SPECVLA_ROOT}/specvla-data
export VLA_PATH=${SPECVLA_ROOT}/hf_files/openvla-7b-finetuned-libero-goal
export OPENVLA_MODEL_PATH=${VLA_PATH}
export LIBERO_RLDS_ROOT=${SPECVLA_ROOT}/dataset/modified_libero_rlds
export DFLASH_DATA_OUTDIR=${SPECVLA_DATA}/dflash_goal_dataset
export DFLASH_OUTPUT_DIR=${SPECVLA_DATA}/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_residual_cad_weakpath_b16_4gpu
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
  --outdir /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset
```

正式生成结束后检查：

```bash
du -sh /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset
find /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset -maxdepth 1 -name 'data_*.ckpt' | wc -l
```

历史 4090/3090 的有效样本规模约为 `28.5k`，大小约 `419G`。新 4090 重新生成后，
应在实验记录中写清楚实际样本数；不要默认和旧机器完全一致。

### 9. 从 3090 搬训练好的 checkpoint 到 4090

3090 继续负责四卡训练。训练完成后，在本地终端用 `scp -3` 从 3090 搬到 4090：

```bash
TRAIN_DIR=/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_residual_cad_weakpath_b16_4gpu
CKPT=epoch_190_step_169670

scp -3 -r \
  3090_wulin:${TRAIN_DIR}/${CKPT} \
  4090:/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/${CKPT}
```

复制后检查：

```bash
ssh 4090 "ls -lh \
  /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/${CKPT}/dflash_config.json \
  /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/${CKPT}/pytorch_model.bin"
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

路径可以通过 `VLA_PATH`、`LIBERO_RLDS_ROOT`、`DFLASH_DATA_OUTDIR` 覆盖；
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
  --outdir /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset
```

训练前确认数据大小和数量：

```bash
du -sh /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset
find /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset \
  -maxdepth 1 -name 'data_*.ckpt' | wc -l
```

旧 4090 历史数据目录状态如下；新 4090 重新生成或迁移后必须重新记录实际数值：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/dflash_goal_dataset
大小: 419G
样本数: 28,639 个 data_*.ckpt
```

### 2. 3090：完整四卡训练

3090 进入服务器和环境：

```bash
ssh 3090_wulin
cd /data/wulin/c/SpecVLA-DFLASH
source /data/wulin/miniconda3/etc/profile.d/conda.sh
conda activate specvla
```

3090 当前有 8 张 RTX 3090，但默认完整训练只使用 0-3 四张卡。当前优先跑
Residual-CAD weak-path 新分支：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_residual_cad_4gpu.sh
```

当前推荐 launcher：

```text
openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_residual_cad_4gpu.sh
```

上一版 pure-training baseline launcher 仍保留，做消融时使用：

```text
openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_puretrain_4gpu.sh
```

该 launcher 会根据机器自动选择默认路径。3090 上的默认路径是：

```text
VLA_PATH=/data/wulin/hf_files/openvla-7b-finetuned-libero-goal
DATAPATH=/data/wulin/c/specvla-data/dflash_goal_dataset
OUTPUT_DIR=/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_residual_cad_weakpath_b16_4gpu
```

Residual-CAD weak-path 主训练配置：

```text
torchrun --nproc_per_node 4
num_draft_layers = 1
selected_hidden_variant = replace_22_with_final
causal_residual_type = hidden
causal_residual_cad_w = 0.10
causal_residual_cad_type = cosine
causal_residual_cad_warmup_steps = 4000
causal_residual_cad_correct_teacher_only = true
causal_residual_min/max_position = 2/5
refined_hidden_w = 0.30
refined_hidden_loss_type = smooth_l1
weak_far_slot_boost = 2.0
anchor0_p2_boost = 4.0
slot_decay = 1.0
hidden_noise = 0.03
batch_size = 16 per GPU，有效 batch size = 64
epochs = 200
warmup = 1000 optimizer steps
save_every = 10
val_split = 0
SwanLab = 使用环境默认配置
```

`run_dflash_anchor_hidden_1layer_residual_cad_4gpu.sh` 支持环境变量覆盖常用超参数，例如：

```bash
BATCH_SIZE=16 WARMUP_STEPS=1000 LR=5e-5 RESIDUAL_CAD_W=0.10 REFINED_HIDDEN_W=0.30 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_anchor_hidden_1layer_residual_cad_4gpu.sh
```

如果要复现实验，请优先使用新的 `*_weakpath_b16_4gpu` 输出目录；不要和旧的 b8、旧 Residual-CAD
目录混写。

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
TRAIN_DIR=/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_residual_cad_weakpath_b16_4gpu
CKPT=epoch_190_step_169670

scp -3 -r \
  3090_wulin:${TRAIN_DIR}/${CKPT} \
  4090:/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/${CKPT}
```

如果要复制 3090 当前 `latest_checkpoint.txt` 指向的最新 checkpoint，可以在本地终端执行：

```bash
TRAIN_DIR=/data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_residual_cad_weakpath_b16_4gpu
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
DFLASH run dir: /data/wulin/c/specvla-data/ckpt_goal_dflash_anchor_hidden_1layer_finalhidden_residual_cad_weakpath_b16_4gpu
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
checkpoint；SpecVLA baseline 才需要 `SPECVLA_GOAL_CKPT`：

```bash
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

为了复现 SpecVLA 上游代码和论文表格的速度口径，当前 Goal 评测 launcher 默认使用：

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

### 非 Goal suite 的 AR / SpecVLA baseline

当前已补齐 Object、Spatial、Long 三个 suite 的 baseline launcher。这里的 Long 对应代码里的
`libero_10`。这些脚本只覆盖 AR、SpecVLA strict、SpecVLA relaxed；DFLASH 暂时只保留 Goal
脚本，等新 draft 方案稳定后再扩展。

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
VLA_PATH
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

设置本次要评测的 DFLASH checkpoint。这里以当前 weak-path b16 的第 200 epoch 为例：

```bash
export DFLASH_CKPT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/epoch_200_step_089400
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

其它 suite baseline 可以照下面跑。每条命令都会自动使用对应 suite 的 OpenVLA 和 SpecVLA checkpoint：

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

评测结束后，汇总 summary。由于普通 DFLASH 和 CADhead 都写入 `dflash_strict` / `dflash_relaxed`
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
