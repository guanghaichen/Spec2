# SpecVLA-DFLASH

本仓库研究一个具体问题：能否把 DFlash 式块并行草稿模型迁移到 OpenVLA，在保持目标模型校验可靠性的
前提下，提高 LIBERO 动作解码速度。

代码基于 [SpecVLA](https://github.com/PineTreeWss/SpecVLA)，而 SpecVLA 又基于
[OpenVLA](https://github.com/openvla/openvla)。当前仓库仍是研究代码，不是已经定稿的公开复现包。
所有结论必须同时报告成功率、接受长度和速度；训练准确率不能替代在线 LIBERO 结果。

## 1. 阅读顺序和项目地图

建议按下面的顺序理解项目：

1. 先理解 OpenVLA 为什么把一个动作表示为 7 个离散 token。
2. 再理解 SpecVLA 如何用小草稿模型逐 token 提案，并让 OpenVLA 校验。
3. 然后理解本项目为什么改用 DFlash 式块并行 hidden 生成。
4. 最后沿着实验历程看清楚：完整上下文、multi-anchor、跨 anchor 蒸馏、Markov 修正，为什么一步步演化到当前 Action-RNN Prefix Survival。

当前核心文件：

| 环节 | 文件 | 作用 |
| --- | --- | --- |
| 数据生成 | `openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py` | 用冻结 OpenVLA 生成动作 token 和多层 hidden 教师数据 |
| 数据入口 | `openvla/specdecoding/train-scripts/run_dflash_data_goal.sh` | 统一 smoke/full 数据生成命令 |
| DFlash 训练 | `openvla/specdecoding/train-scripts/train_dflash_libero_goal.py` | multi-anchor、loss、DDP、checkpoint、SwanLab |
| 训练入口 | `openvla/specdecoding/train-scripts/run_dflash_train.sh` | 当前 Action-RNN 主实验及少量结构消融 |
| Draft 模型 | `openvla/specdecoding/model/dflash.py` | 块并行主干、动作位置 embedding、Action-RNN、双路径候选 |
| 在线推测解码 | `openvla/prismatic/extern/hf/modeling_speculation.py` | draft 提案、目标模型校验、partial accept/correction、树验证 |
| LIBERO strict | `openvla/experiments/robot/libero/run_libero_goal_Spec.py` | strict token 校验与在线指标 |
| LIBERO relaxed | `openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py` | relaxed 动作接受与在线指标 |
| 推理公共配置 | `openvla/specdecoding/decode-scripts/libero_eval_common.sh` | 两台机器路径、suite 权重、计时口径和 checkpoint 解析 |

正式 shell 入口已经收敛。`train-scripts` 只保留数据生成和当前训练两个入口；`decode-scripts` 只保留
通用单项评测和少数一键工作流。旧 wrapper 已从工作树删除，需要回溯时使用 Git 历史。

## 2. 研究基础

### 2.1 OpenVLA 的动作序列

OpenVLA 把一个连续机器人动作离散成 7 个 token：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

原始 OpenVLA 按语言模型方式自回归地产生这 7 个 token。动作之间虽然具有结构关系，但在模型接口上仍是
一个长度为 7 的 token 序列。

### 2.2 SpecVLA baseline

[Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance](https://aclanthology.org/2025.emnlp-main.1367.pdf)
把投机解码迁移到 OpenVLA：小型 EAGLE 风格 draft 逐 token 产生候选，OpenVLA 目标模型并行校验候选，
只提交被接受的前缀；relaxed 模式允许动作 bin 在阈值内偏离目标 token。

下面是 SpecVLA 论文 Table 1 的数字，不是本仓库复现结果：

| LIBERO suite | AR 成功率 | SpecVLA：成功率 / Length / Speedup | Relaxed：成功率 / Length / Speedup |
| --- | ---: | --- | --- |
| Goal | 78.0% | 74.2% / 2.04 / 1.09x | 74.4% / 2.94 / 1.42x |
| Object | 89.0% | 89.0% / 1.75 / 1.15x | 85.0% / 2.38 / 1.38x |
| Spatial | 85.0% | 83.8% / 1.59 / 1.08x | 85.8% / 2.14 / 1.28x |
| Long | 52.0% | 50.8% / 1.67 / 1.13x | 55.0% / 2.10 / 1.22x |

本仓库复现论文 Speedup 时，AR 分母必须使用作者的 wrapped AR 路径：

```text
run_libero_goal_AR.py
use_spec=True
parallel_draft=False
```

它会加载 SpecVLA wrapper，但逐 token 推进。纯 OpenVLA `predict_action` 路径更快，却不是论文所用分母，
不能混入主表。

### 2.3 DFlash 灵感

[DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
用一个轻量、非因果的 block draft 一次预测整块 token，再由目标模型校验。当前可参考实现来自
[SpecForge](https://github.com/sgl-project/SpecForge/blob/main/specforge/modeling/draft/dflash.py)。

本仓库迁移的是“目标模型 hidden 条件下的轻量块并行 draft”这一核心思想，不声称复现 DFlash 原论文的
完整模型、训练配方或 LLM 加速数字。

## 3. 方法是怎样一步步形成的

### 3.1 阶段一：最初的块并行迁移

最初版本让 DFlash draft 根据很少的上下文一次预测多个 action token。它可以跑通，但草稿几乎无效。
根因不是“块并行一定不可行”，而是训练和推理没有给 draft 与 SpecVLA 同等级别的目标模型上下文。

### 3.2 阶段二：完整 prefix 与 multi-anchor

随后完成两项基础修正：

1. 数据和在线推理都保留完整 prompt/prefill 多层 hidden，而不是只保留 prompt 最后位置。
2. 每个已被目标模型验证的 action token hidden 都加入上下文，并对每个 anchor 训练一次。

以动作 token `t0...t6` 为例：

```text
anchor=0：完整 prompt + t0 的目标模型上下文，预测 t1...t6
anchor=1：完整 prompt + t0,t1 的目标模型上下文，预测 t2...t6
...
anchor=5：完整 prompt + t0...t5 的目标模型上下文，预测 t6
```

这让同一个目标位置同时拥有“远端弱路径”和“近端强路径”，也为后来的跨 anchor 蒸馏提供了依据。

### 3.3 阶段三：Pure Hidden baseline

第一版稳定 baseline 使用一层块并行 draft，以目标模型最终 hidden 的 SmoothL1 和 cosine 为主监督。
结果表明：p1 和 gripper p6 容易学习，anchor0 的 p2-p5 明显更弱。最典型的是 anchor0->p2 约 0.50，
说明后续 slot 缺少即时因果前缀。

### 3.4 阶段四：Residual-CAD

这一阶段尝试用前一个真实 token 修正 weak-path hidden，并让弱 anchor hidden 靠近同一目标位置的强
anchor hidden。历史目录名称使用 `Residual-CAD`；为了避免缩写混乱，本文统一称它为
**Hidden Cross-Anchor Distillation，Hidden 跨 Anchor 蒸馏**。

它把 anchor0->p2 从约 0.50 提高到约 0.67，证明跨 anchor 的强路径确实包含可迁移信息。但是残差修正后的
总体 accuracy 没有稳定超过未修正的 base accuracy，说明 hidden 约束不足以直接改善 token 决策。

### 3.5 阶段五：Markov-ACD

下一版加入三类更直接的信号：

1. 前一个真实 token 条件下的 hidden residual。
2. 前一个真实 token 条件下的 logits bias。
3. 同一目标位置的弱/强 anchor logits 蒸馏和 token CE。

这一阶段统一称为 **Markov-aware Anchor-Contrastive Distillation，Markov 感知跨 Anchor 对比蒸馏**，
简称 Markov-ACD。短跑中 anchor0 的 p2-p5 大幅提高；200 epoch 长跑的 teacher-forced accuracy 接近 1.0。

但在线结果暴露出两个关键问题：

- 训练时看到真实前一个 token，在线时看到 draft 自己生成的 token，存在明显 exposure gap。
- 旧修正头逐 slot 执行较重的 hidden/logits 操作；Length 即使提高，也可能被草稿开销抵消。

这一步的重要贡献是确定了方向：token 级因果信息和跨 anchor 蒸馏确实能抬高弱路径；但不能只追求离线
teacher-forced accuracy，也不能忽视推理头成本。

### 3.6 阶段六：当前 Action-RNN Prefix Survival

当前版本保留一层并行 DFlash 主干，把旧的 hidden residual 和全词表 Markov bias 合并为一个很小的
动作专用顺序残差头。

#### 块并行主干

目标模型先得到 anchor `t_a`。DFlash 在一次 transformer forward 中并行输出最多 6 个未来位置 hidden：

```text
h_(a+1), h_(a+2), ..., h_6
```

主干仍是非因果块生成；未来 slot 不会在这次 transformer forward 里得到真实未来 token。

#### Frozen-lm-head Action-RNN

Action-RNN 不替代 OpenVLA `lm_head`。每个位置先用冻结 `lm_head` 的 256 个动作 token 权重行产生 base logits，
再由低秩 RNN 输出一个 256 维残差：

```text
final_action_logits = frozen_lm_head_action_logits + action_rnn_bias
```

RNN 每一步读取：当前位置 DFlash hidden、前一个 action token、当前位置 embedding、上一步 RNN state。
输出层零初始化，因此训练起点严格等于 frozen `lm_head`。

当前稳定版把两条反传职责解耦，但前向并非互不相干：Action-RNN 仍读取 Draft hidden 和冻结 `lm_head`
产生的基础 logits。hidden/cos、低权重 teacher soft distribution 和主干跨 anchor KL 更新并行 Draft；CE、
动作分布 L1、Prefix Survival 和动作头跨 anchor KL 更新 Action-RNN。动作头读取的 Draft hidden/base logits
在动作损失分支执行 `stop_gradient`。这样重动作目标不能再通过改动共享 hidden 走捷径，同时 Draft 仍能得到
直接的 token 分布监督。

训练使用 teacher forcing。以 anchor0 为例：

```text
预测 t1：h1 + 真实 t0
预测 t2：h2 + 真实 t1 + 已累计 t0 的状态
预测 t3：h3 + 真实 t2 + 已累计 t0,t1 的状态
...
```

推理则使用自己刚生成的 token：

```text
真实 t0 -> 预测 t1 -> 预测 t2 -> ... -> 预测 t6
```

所以因果信息没有消失，而是从旧 hidden 修正迁移到了动作 logits 决策层。代价是 teacher forcing 与
self-rollout 仍有分布差异，必须看 `rollout_*` 和在线命中率。

#### 双分支跨 Anchor Logit Distillation

对同一目标位置，较远 slot 的动作分布追近端强路径分布。强路径必须预测正确，且 teacher 分支
`stop_gradient`。当前同一原则作用两次：修正前 base logits 的 KL 只训练 Draft 主干；修正后 final logits 的
KL 只训练 Action-RNN。这保留了本项目区别于单纯 DSpark 式前 token 注入的核心：不同 anchor 是同一目标
位置的多种因果视角，并且主干与动作头都必须各自吸收这种信息，不能由另一侧代偿。

#### Prefix Survival

先用 teacher/student 动作分布的 total variation 得到每个位置的近似可接受概率，再沿块累乘，直接惩罚
连续前缀中断。早期错误会同时破坏后面的累计前缀，因此无需再设置 p2、p3 等手工 boost。

#### 当前边界

DFlash transformer 仍只 forward 一次，但 Action-RNN 有最多 6 次很小的顺序状态更新。因此当前方法是
“块并行重计算 + 轻量顺序 token 修正”，不是严格意义上 6 个最终 token 完全 O(1) 同时输出。

## 4. 当前数据、模型和损失

### 4.1 正式离线数据

当前 3090 正式数据：

```text
/data/wulin/c/specvla-data/dflash_goal_dataset_envfix_20260714.h5
```

已核对元数据：

| 字段 | 当前值 |
| --- | --- |
| `complete` | `True` |
| 样本数 | 28,501 |
| 数据格式 | `full_prefix_plus_action_hidden_v4` / `dflash_hdf5_v1` |
| selected layers | `[1, 9, 16, 24, 31]` |
| 文件大小 | 约 322 GiB |
| `predicted_tokens` | `[7]` |
| `prompt_selected` | `[prompt_len, 5*4096]` |
| `action_selected` | `[6, 5*4096]` |
| `action_last` | `[6, 4096]` |

层索引从 layer 1 到 final layer 31 做五点近似等间隔抽样，不再手工选择层。训练不读取图像 tensor，因此
HDF5 不保存 `pixel_values`；final hidden 已在 selected prompt 中，也不重复保存 `prompt_last`。

历史 419 GiB、28,639 个小 `.ckpt` 的数据只用于解释旧实验。它会产生大量随机文件 IO，也与当前均匀选层
格式不一致，禁止继续作为新主实验输入。

### 4.2 Context 和位置不变量

训练与推理必须同时满足：

- 保留完整 prompt/prefill 多层 hidden。
- 保留从 t0 到当前 anchor 的已验证 action hidden。
- selected layers 与 checkpoint 中 `dflash_config.json` 一致。
- prefix position 从 0 连续增长，action context 和 block position 紧随 prefix。
- `action_dim_embed` 标识动作维度；它补充 RoPE，不替代 RoPE。

### 4.3 当前总损失

默认一层主实验：

```text
total = 1.00 * hidden_smooth_l1
      + 0.05 * hidden_cosine
      + 0.05 * backbone_teacher_soft_ce
      + 0.05 * backbone_cross_anchor_logit_kl
      + 0.10 * action_token_ce
      + 0.40 * action_distribution_l1
      + 0.20 * prefix_survival
      + 0.05 * action_head_cross_anchor_logit_kl
```

它们对参数的作用不同：

| Loss | 直接训练 Action-RNN | 直接训练 DFlash 主干 | 目的 |
| --- | --- | --- | --- |
| hidden SmoothL1 | 否 | 是 | 保持目标 hidden 表征 |
| hidden cosine | 否 | 是 | 对齐 hidden 方向 |
| backbone teacher soft CE | 否 | 是 | 让修正前 base logits 对齐 teacher 的 256 类动作分布 |
| backbone cross-anchor logit KL | 否 | 是 | 让主干远端弱路径追近端强路径 |
| action token CE | 是 | 否 | 提高真实 token top-1 命中 |
| action distribution L1 | 是 | 否 | 对齐 256 类 teacher 动作分布 |
| prefix survival | 是 | 否 | 直接鼓励连续接受前缀 |
| action-head cross-anchor logit KL | 是 | 否 | 让动作头修正后的弱路径追近端强路径 |

当前 soft CE 只在 256 个动作 token 子词表上计算，不做昂贵且无关的全词表 soft CE。当前关闭：confidence
head、旧 hidden CAD、旧 causal residual、旧 refined hidden、旧 residual CE。
`slot_decay=1.0`；只保留 `position_balance` 来抵消 multi-anchor 对后部绝对位置的重复监督偏差。
优化器使用两组学习率：Draft 主干 `2e-5`，Action-RNN `5e-5`，共同 warmup 1000 step 后线性退火。

### 4.4 SwanLab 指标怎样读

原始 loss 与 `*_component` 要分开看：

```text
component = raw_loss * 配置权重
```

真正决定总 loss 比例的是 component。核心指标：

| 指标 | 含义 |
| --- | --- |
| `base_accuracy` | Action-RNN 残差加入前的 frozen-lm-head 命中率 |
| `accuracy` | teacher-forced Action-RNN 修正后的总体命中率 |
| `action_head_accuracy_gain` | `accuracy-base_accuracy`；直接判断残差动作头是否真的带来收益 |
| `soft_loss` / `soft_component` | Draft base logits 的 teacher soft CE 原值 / 乘 `0.05` 后贡献 |
| `backbone_anchor_logit_distill_loss` | Draft base logits 的跨 anchor KL；只更新主干 |
| `anchor_logit_distill_loss` | Action-RNN final logits 的跨 anchor KL；只更新动作头 |
| `rollout_accuracy` | anchor0 使用自身预测前缀回滚时的 top-1 命中率 |
| `rollout_exposure_gap` | `accuracy-rollout_accuracy`；衡量 teacher forcing 到自回滚的分布差距 |
| `rollout_top2_accuracy` | 正确 token 是否进入 self-rollout top-2 |
| `rollout_expected_prefix_length` | anchor0 实际连续猜对的平均前缀长度，范围 0 到 6 |
| `position_k_acc` | 所有 anchor 对绝对位置 pk 的 teacher-forced 平均准确率 |
| `anchor_a_to_position_k_acc` | 指定 anchor 到指定目标位置的准确率矩阵 |
| `action_accept_probability_proxy` | 由 teacher/student 分布距离得到的单位置接受概率代理 |
| `expected_prefix_length` | 分布式可微前缀代理，不等于在线论文 Length |

`train/*` 是每 20 optimizer step 的局部窗口，可能受批次难度影响；`train_epoch/*` 是整轮均值，判断长期
收敛时应优先看后者。

当前主训练每 20 optimizer step 上传普通标量，每 200 step 上传 anchor/position 明细。`NUM_WORKERS=1`、
单文件 HDF5、不保存 optimizer state 和根目录 checkpoint 副本，用于降低共享服务器 IO 压力。

## 5. 推理和验证

### 5.1 Partial accept 与 correction

目标模型验证 proposal 后接受最长合法前缀。遇到第一个拒绝位置时，写入目标模型自己的 posterior token，
再进入下一 anchor。因此 strict 路径始终由目标模型裁决，支持部分接受和部分纠正。

目标校验只输入 `q-1` 个 proposal 是标准 next-token shift：anchor logits 校验第一个 proposal，输入前
`q-1` 个 proposal 后得到的 logits 分别校验余下 token。最后一个 proposal 不需要再次作为本块输入。

### 5.2 Relaxed action-group acceptance

当前 relaxed 评测把平移 `(x,y,z)` 和旋转 `(roll,pitch,yaw)` 分组检查总平方 bin 误差；gripper 始终要求
精确相等。它只改变 relaxed 接受规则，不改变 strict 输出语义。论文中必须同时报告阈值、成功率和 Length。

### 5.3 Calibrated single-fork tree

Action-RNN 主路径生成后，可在一个固定位置取 runner-up token，并从该状态滚出一条备选后缀。两条路径由
目标模型一次 tree-attention forward 校验，最长合法前缀胜出。

strict 评测先在真实 observation 上比较 `off/p2/p3/p4/p5` 的完整动作延迟。只有某个分叉位置统计显著地
快于 off，才进入正式评测，否则回退 off。树只作用于 anchor0 首块，避免后续短块支付无效成本。

## 6. 实验历程与目前证据

### 6.1 训练演进

| 阶段 | 关键设置 | 结果 | 得到的结论 |
| --- | --- | --- | --- |
| Pure hidden | 完整 prefix、1 层、hidden+cos | 末值 acc 0.807，最好 0.830 | p1/p6 强，anchor0 p2-p5 弱 |
| Residual-CAD | 前 token hidden residual + Hidden 跨 anchor 蒸馏 | anchor0->p2 约 0.503 -> 0.671 | 强 anchor 信息可以帮助弱路径，但 hidden 修正不够直接 |
| Markov-ACD 短跑 | 加 logits bias、token CE、logit 跨 anchor 蒸馏 | anchor0 p2-p5 约 0.86/0.91/0.88/0.88 | token/logit 因果监督有效 |
| Clean Markov-ACD 200 epoch | 删除手工 boost，覆盖 slot0，低权重 soft | teacher-forced acc 约 0.999 | 离线饱和不代表在线饱和 |
| Action-RNN Prefix Survival | 256 类动作残差头、分布 L1、前缀目标、跨 anchor KL | 当前训练进行中 | 目标是同时解决 exposure、前缀目标和推理开销 |

Pure hidden、Residual-CAD、Markov-ACD 的代表性 anchor0 结果：

| 阶段 | p1 | p2 | p3 | p4 | p5 | p6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure hidden 末值 | 0.886 | 0.503 | 0.716 | 0.620 | 0.692 | 0.953 |
| Residual-CAD 末值 | 0.846 | 0.671 | 0.717 | 0.647 | 0.718 | 0.936 |
| Markov-ACD 短跑末值 | 0.820 | 0.860 | 0.913 | 0.881 | 0.880 | 0.963 |

### 6.2 Markov-ACD 在线结果

2026-07-12 的历史 Goal rollout 使用了错误的纯 OpenVLA AR 分母，因此 Speedup 不能进入论文主表；它仍能
用于比较 Length、在线命中与开销：

| 方法 | SR | mean step | 历史 Speedup | Length | avg accept |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 AR，非 paper AR | 0.774 | 0.161929s | 1.000x | - | - |
| SpecVLA strict | 0.768 | 0.178630s | 0.907x | 1.631 | 0.631 |
| SpecVLA relaxed | 0.734 | 0.141256s | 1.146x | 2.361 | 1.361 |
| Markov-ACD strict | 0.788 | 0.182786s | 0.886x | 2.109 | 1.008 |
| Markov-ACD relaxed | 0.768 | 0.158210s | 1.024x | 2.607 | 1.479 |

离线 p1-p5 接近 1.0，但 strict 在线命中约为
`0.687/0.326/0.411/0.393/0.468/0.577`。这证明 exposure gap 是实质问题。另一方面，DFlash strict
Length 高于 SpecVLA strict 却更慢，证明 Length 不包含 draft 自身成本，不能单独作为加速结论。

### 6.3 当前 Action-RNN 训练早期快照

2026-07-15，当前 3090 主训练在 step 500、warmup 进行到一半时：

| 指标 | 数值 |
| --- | ---: |
| total loss | 2.213 |
| hidden / cosine | 1.095 / 0.833 |
| action CE / distribution L1 | 3.808 / 1.188 |
| prefix survival / cross-anchor KL | 0.829 / 0.033 |
| accuracy / base accuracy | 0.525 / 0.525 |
| rollout accuracy / top2 | 0.434 / 0.464 |
| rollout prefix length | 0.438 |
| rollout p1-p6 | 0.341 / 0.109 / 0.428 / 0.344 / 0.479 / 0.902 |

总 loss、CE、L1、hidden 都在下降，没有 NaN/OOM。需要警惕的是 RNN 修正后的 accuracy 尚未超过 base，
以及 teacher-forced 与 rollout 约有 9 个百分点差距。

到 epoch 6-12，旧耦合配方出现稳定的轮内锯齿：action CE/L1/Prefix/KL 在每轮内上升，准确率下降，下一轮
开头又恢复；学习率始终平滑，因此不是 scheduler 重启。去除 epoch 均值后，hidden 与 CE/L1/Prefix 的
相关系数约为 `-0.61/-0.53/-0.47`。epoch 12 时 teacher-forced head accuracy 约 `0.922`，base accuracy
约 `0.921`，动作头净增益仅约 `0.0005`，而 rollout 约 `0.889`。这说明高权重动作损失主要在改动共享
Draft hidden，Action-RNN 本身几乎没有学到有效残差。当前默认配方因此改为“主干/动作头梯度隔离 + 双学习率
+ hidden 主导权重”。在此基础上，当前配方再给 Draft 增加 `0.05` teacher soft CE 和 `0.05` 主干跨 anchor
KL，使其不只依靠 hidden 回归；旧输出目录不得续训到新配方。

## 7. 当前标准工作流

固定分工：

```text
3090：生成离线数据，0-3 四卡训练
本地：使用 scp -3 中转 checkpoint
4090：单卡串行 LIBERO 推理评测
GitHub：两台服务器之间唯一的代码同步基准
```

### 7.1 3090 生成数据

进入环境：

```bash
ssh 3090_wulin
source /data/wulin/miniconda3/etc/profile.d/conda.sh
conda activate specvla
cd /data/wulin/c/SpecVLA-DFLASH
```

先运行 32 条 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh smoke
```

确认 smoke 后正式生成。当前正式文件已经存在，生成脚本默认拒绝覆盖；重新生成时必须显式提供新文件名：

```bash
GPU_ID=0 OUT_FILE=/data/wulin/c/specvla-data/dflash_goal_dataset_new.h5 \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh full
```

生成结束检查：

```bash
python - <<'PY'
import h5py

path = "/data/wulin/c/specvla-data/dflash_goal_dataset_new.h5"
with h5py.File(path, "r") as h5:
    print("complete:", bool(h5.attrs["complete"]))
    print("format:", h5.attrs["dflash_data_format"])
    print("layers:", list(h5.attrs["hidden_layer_ids"]))
    print("samples:", len(h5["samples"]))
PY
```

只有 `complete=True` 才能训练。

### 7.2 3090 四卡训练

当前推荐主实验：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh main
```

脚本默认：一层 draft、global batch 64、200 epoch、Draft/Action-RNN 学习率分别为 `2e-5/5e-5`、
warmup 1000 step、每 10 epoch 保存一次。Draft 侧 soft/KL 权重均为 `0.05`；动作重损失输入默认 detach，
不要用旧 checkpoint 续训。

常用消融：

```bash
# 三层容量消融
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh three_layer

# 去掉 Prefix Survival
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh no_prefix

# 去掉跨 anchor logits 蒸馏
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh no_anchor
```

使用新 HDF5 时显式覆盖：

```bash
DATAPATH=/data/wulin/c/specvla-data/dflash_goal_dataset_new.h5 \
OUTPUT_DIR=/data/wulin/c/specvla-data/ckpt_goal_dflash_action_rnn_new \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh main
```

不要把不同数据、模型结构或 loss 配方写进同一个输出目录。

### 7.3 把 checkpoint 搬到 4090

在本地终端执行。下面以 epoch 200 为例：

```bash
CKPT_3090=$(ssh 3090_wulin \
  "find /data/wulin/c/specvla-data/ckpt_goal_dflash_action_rnn_decoupled_distill_1layer_b8x2_4gpu \
   -maxdepth 1 -type d -name 'epoch_200_step_*' | sort -V | tail -1")

ssh 4090 \
  'mkdir -p /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_action_rnn_decoupled_distill_1layer_b8x2_4gpu'

scp -3 -r "3090_wulin:${CKPT_3090}" \
  '4090:/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_action_rnn_decoupled_distill_1layer_b8x2_4gpu/'
```

### 7.4 4090 复现 SpecVLA Goal baseline

```bash
ssh 4090
source /home/asus/miniconda3/etc/profile.d/conda.sh
conda activate specvla
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH

CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_specvla_goal_upstream_compatible_eval.sh
```

该脚本串行执行 paper wrapped AR、strict `r=0`、relaxed `r=9`，并生成对比摘要。

单独执行：

```bash
bash openvla/specdecoding/decode-scripts/run_specvla_paper_ar_eval.sh goal
bash openvla/specdecoding/decode-scripts/run_specvla_eval.sh goal strict
bash openvla/specdecoding/decode-scripts/run_specvla_eval.sh goal relaxed
```

Object、Spatial、Long 只需替换第一个位置参数：

```bash
bash openvla/specdecoding/decode-scripts/run_specvla_paper_ar_eval.sh object
bash openvla/specdecoding/decode-scripts/run_specvla_eval.sh spatial relaxed
bash openvla/specdecoding/decode-scripts/run_specvla_eval.sh 10 relaxed
```

Long relaxed 默认 `r=5`；其它 suite 默认 `r=9`。

四个 suite strict/relaxed 自动续跑并汇总：

```bash
DRY_RUN=True bash openvla/specdecoding/decode-scripts/run_specvla_main_table_eval.sh
bash openvla/specdecoding/decode-scripts/run_specvla_main_table_eval.sh
```

### 7.5 4090 评测当前 DFlash Goal

当前 Goal 权重不能用于 Object、Spatial 或 Long。成对评测：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 EVAL_EPOCH=200 \
DFLASH_OUTPUT_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_action_rnn_decoupled_distill_1layer_b8x2_4gpu \
  bash openvla/specdecoding/decode-scripts/run_dflash_action_rnn_goal_pair_eval.sh
```

单独执行：

```bash
EVAL_EPOCH=200 bash openvla/specdecoding/decode-scripts/run_dflash_goal_eval.sh strict
EVAL_EPOCH=200 bash openvla/specdecoding/decode-scripts/run_dflash_goal_eval.sh relaxed
```

五个 seed 重复性评测：

```bash
EVAL_EPOCH=200 REPEAT_SEEDS="7 8 9 10 11" \
  bash openvla/specdecoding/decode-scripts/run_dflash_goal_repeat_eval.sh
```

### 7.6 输出和论文指标

每次评测产生：

```text
*.txt             人类可读日志
*_timing.json     每个环境 step 的动作生成耗时
*_summary.json    SR、timing、Length、接受统计
```

重点指标：

| 指标 | 含义 |
| --- | --- |
| `success_rate` | LIBERO 闭环任务成功率 |
| `timing.mean` | 当前计时范围内的平均动作生成时间 |
| `speedup` | 同机器、同 suite 的 paper wrapped AR mean / 当前 mean |
| `generation.length` | 论文 Table 1 风格的每个 speculative block 平均推进 token 数 |
| `avg_accept_length` | 每个 block 平均接受的 draft token 数，不等于 Length |
| `per_position` | 各 proposal 位置在线命中率 |
| `conditional_prefix` | 已接受前 k-1 个 token 后，第 k 个继续被接受的条件概率 |

固定计时协议：

```text
SEED=7
SYNC_CUDA_TIMING=False
TIMING_SCOPE=last_task
```

AR、strict、relaxed 必须同机、同 GPU、串行执行。4090 是正式速度机器；3090 只做训练或临时成功率 sanity。

## 8. 脚本组织

### 8.1 训练入口

| 脚本 | 用法 |
| --- | --- |
| `run_dflash_data_goal.sh` | `smoke` 或 `full` 生成单文件 HDF5 |
| `run_dflash_train.sh` | `main`、`three_layer`、`no_prefix`、`no_anchor` |

`train_dflash_libero_goal.py` 是底层训练实现，不建议日常手写几十个 CLI 参数。

### 8.2 推理入口

| 脚本 | 用法 |
| --- | --- |
| `run_specvla_paper_ar_eval.sh` | 一个 suite 的论文 AR 分母 |
| `run_specvla_eval.sh` | 一个 suite 的 strict/relaxed |
| `run_specvla_goal_upstream_compatible_eval.sh` | Goal AR+strict+relaxed 一键复现 |
| `run_specvla_main_table_eval.sh` | 四 suite strict/relaxed 自动续跑与汇总 |
| `run_dflash_goal_eval.sh` | 当前 Goal DFlash 单项 strict/relaxed |
| `run_dflash_action_rnn_goal_pair_eval.sh` | 当前 Goal DFlash strict+relaxed |
| `run_dflash_goal_repeat_eval.sh` | 多 seed 稳定性评测 |
| `setup_3090_nvidia_egl_shim.sh` | 仅处理 3090 特定 EGL 驱动问题 |

所有 suite 路径、权重路径和公共计时参数集中在 `libero_eval_common.sh`，不要在 `.bashrc` 中全局导出
suite-specific `VLA_PATH`。

## 9. 环境与服务器维护

当前关键环境版本：

```text
Python 3.10
torch 2.2.0+cu121
transformers 4.40.1
mujoco 3.3.4
robosuite 1.4.1
numpy 1.26.4
accelerate 1.9.0
swanlab 0.7.20
protobuf 4.21.12
LIBERO commit 8f1084e3132a39270c3a13ebe37270a43ece2a01
```

`swanlab==0.7.20` 与 `protobuf==4.21.12` 是当前 TensorFlow/RLDS 环境的兼容组合。不要无版本号升级
MuJoCo、robosuite、SwanLab 或 protobuf。

新服务器迁移顺序：Git/deploy key、目录、conda、模型权重、LIBERO、数据或 checkpoint、EGL smoke、评测
smoke。训练机需要 RLDS 和 HDF5；纯评测机不需要 RLDS。

Hugging Face 优先使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

`.bashrc` 只放稳定根路径、HF cache 和镜像；suite 模型路径由评测脚本选择。

## 10. Git 与同步规则

```text
代码：4090 开发 -> GitHub main -> 3090 拉取
权重：3090 训练 -> 本地 scp -3 -> 4090 评测
```

同步前：

```bash
git fetch origin
git status --short --branch
git log --oneline --decorate --max-count=8 --all
```

4090 推送后，3090 在仓库干净时：

```bash
git pull --ff-only origin main
```

如果 3090 有正在运行的训练，拉取代码不会改变已经启动的 Python 进程；但不要移动、删除当前训练的数据
和输出目录。每个实验至少记录 Git commit、数据文件、launcher、checkpoint、seed、接受阈值和计时口径。

## 11. 后续研究计划

### 11.1 当前 OpenVLA 主线

按优先级：

1. 完成当前 Action-RNN 训练，检查 warmup 后 `accuracy-base_accuracy`、rollout p2-p5 和 prefix length。
2. 在 4090 评测 epoch 100/150/200，联合报告 SR、Length、Speedup 和在线位置命中率。
3. 做 `main / no_prefix / no_anchor / three_layer` 消融，判断收益来自哪里。
4. 分别消融 action-group acceptance 和 calibrated tree，报告净延迟而非只报 Length。
5. 在多个 seed 和真实机械臂上验证稳定性。

### 11.2 论文需要的完整证据

- 主表：paper AR、SpecVLA strict/relaxed、DFlash strict/relaxed。
- 训练到在线的诊断图：teacher-forced、self-rollout、online hit rate。
- 前缀图：条件接受概率与 expected prefix length。
- 速度分解：DFlash transformer、Action-RNN、target verify、环境外开销。
- 消融：跨 anchor、Prefix Survival、主干层数、树、动作组 relaxed。
- 鲁棒性：checkpoint、seed、硬件、任务长度。
- 真机：ALICIA-D 上比较成功率、动作延迟、控制频率和失败类型。

### 11.3 OpenVLA-OFT 后续扩展

OpenVLA-OFT 已经并行输出动作，不适合直接套用 action-token speculative decoding。后续可把“弱计算路径追强
计算路径”的思想迁移为 layer early exit：冻结 OFT，用中间层 hidden 经过小型 refinement head 逼近最终层
动作输出，并由置信或一致性决定是否早退。

这个方向必须单独回答三个问题：早退是否真的跳过足够多层、动作误差是否保持任务成功率、门控和 refinement
开销是否小于省下的 LLM 层。当前只作为扩展计划，不与 OpenVLA 主线混在同一套结果中。

## 12. 重要限制

- 当前只训练了 LIBERO-Goal draft，不能拿同一权重评测其它 suite。
- teacher-forced accuracy 可能严重高估在线能力。
- Length 高不保证 Speedup 高；必须计算草稿头和校验树开销。
- relaxed acceptance 不是 strict lossless，必须报告阈值与成功率。
- Action-RNN 引入轻量顺序步骤，论文表述不能宣称最终 token 完全并行。
- 当前没有证据证明已经稳定超过 SpecVLA，结论必须等正式在线评测。

## 13. 参考文献

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
