# SpecVLA-DFLASH

本仓库研究一个具体问题：能否把 DFlash 式块并行草稿模型迁移到 OpenVLA，在保持目标模型校验可靠性的
前提下，提高 LIBERO 动作解码速度。

当前完整方案由三层组成：块并行 Draft 负责低成本提案；高维主导的 Base/Final 线性交接、跨 Anchor 蒸馏和
DSpark/Domino 启发的 Action-RNN 负责提高弱路径命中；动作组级宽松校验与校准式稀疏多候选树负责在推理
阶段扩大可接受候选覆盖。它们最终都必须接受成功率、Length 和端到端 Speedup 的共同检验。

代码基于 [SpecVLA](https://github.com/PineTreeWss/SpecVLA)，而 SpecVLA 又基于
[OpenVLA](https://github.com/openvla/openvla)。当前仓库仍是研究代码，不是已经定稿的公开复现包。
所有结论必须同时报告成功率、接受长度和速度；训练准确率不能替代在线 LIBERO 结果。

## 1. 阅读顺序和项目地图

建议按下面的顺序理解项目：

1. 先理解 OpenVLA 为什么把一个动作表示为 7 个离散 token。
2. 再理解 SpecVLA 如何用小草稿模型逐 token 提案，并让 OpenVLA 校验。
3. 然后理解本项目为什么改用 DFlash 式块并行 hidden 生成。
4. 最后沿着实验历程看清楚：完整上下文、multi-anchor、跨 anchor 蒸馏、Action-RNN，以及动作组级宽松校验和稀疏多候选树为什么逐步形成。

当前核心文件：

| 环节 | 文件 | 作用 |
| --- | --- | --- |
| 数据生成 | `openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py` | 用冻结 OpenVLA 生成动作 token 和多层 hidden 教师数据 |
| 数据入口 | `openvla/specdecoding/train-scripts/run_dflash_data_goal.sh` | 统一 smoke/full 数据生成命令 |
| 数据无损打包 | `openvla/specdecoding/train-scripts/pack_dflash_hdf5.py` | 把每样本 group 的 HDF5 v1 重排为连续 packed v2 |
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

### 2.4 当前完整方法与创新边界

当前方法不是单一训练技巧，而是一条从“产生更好的草稿”到“用更低代价校验草稿”的完整链：

```text
完整 prompt + 已验证 action hidden
              ↓
     一次 DFlash 块并行 forward
              ↓
   frozen lm_head 得到 Base logits
              ↓
 Action-RNN 注入块内短程因果关系
              ↓
 主路径 + 可选单分叉 runner-up 路径
              ↓
 目标模型一次线性/树形并行校验
              ↓
 strict 精确接受 或 action-group 宽松接受
              ↓
 提交最长合法前缀；拒绝点由目标模型纠正
```

各部分的来源和本项目的增量必须区分清楚：

| 组成 | 思想来源 | 本项目中的落地 | 定位 |
| --- | --- | --- | --- |
| hidden 条件下的块并行主干 | DFlash | 适配 OpenVLA 七维动作、完整多层 prompt hidden 和已验证 action hidden | 基础骨架 |
| 轻量顺序因果修正 | [DSpark](https://arxiv.org/abs/2607.05147)/同类半自回归 draft 的启发 | frozen `lm_head` 后增加低秩 Action-RNN，训练读真实前 token，推理读自身前 token | 吸收短程因果建模精髓，不声称复现 DSpark 全系统 |
| Base/Final 训练次序 | [Domino](https://arxiv.org/abs/2605.29707) 的 base-anchored curriculum | 单次训练内将 Base 比例从 1 线性降到 0，Final 同步从 0 升到 1；Soft 与 hard CE 使用同一交接 | 吸收“先立主干、再学因果修正”的核心经验 |
| 跨 Anchor 长程因果迁移 | 本项目 | 同一目标位置上，让完整前缀强路径的 base 分布蒸馏短前缀弱路径 | 训练端主要自有设计 |
| 高维主导的自动损失标定 | 本项目结合历史训练诊断形成 | 正式更新前只读少量 batch，固定缩放低维损失，使初始高维:低维约为 9:1 | 防止初始值 20-40 的 Soft KL 掩盖 Hidden 学习，标定后不动态追权 |
| 动作组级宽松校验 | 本项目在 SpecVLA relaxed 上的结构化扩展 | 平移组、旋转组使用组内平方误差预算，夹爪保持精确 | 推理端自有设计之一 |
| 校准式稀疏多候选树 | 本项目 | 一个 runner-up 单分叉形成双路径稀疏树，一次目标 forward 校验，实测无收益则关闭 | 推理端自有设计之二 |

因此，论文主线应表述为：**块并行主干负责低成本长程预测，Action-RNN 补充显式短程因果，跨 Anchor 蒸馏
补充训练时可获得的长前缀知识；动作组校验与稀疏候选树再把更高的候选覆盖率转化为在线接受长度和速度。**
其中 Domino 与 DSpark 是清晰标注的机制启发，跨 Anchor、自动损失标定及两项推理机制是当前拟通过
消融证明的自有增量。最终的新颖性表述仍应以完整文献检索和实验结果为准。

与 Domino 的关系需要准确表述：当前训练骨架与其核心思想高度一致，都是在同一次训练中令 Base 任务权重
线性下降、Final 任务权重线性上升，并让 Final 梯度继续穿过修正头更新 Draft；但本项目不是原样复现。
Domino 的公开实现主要交接 Base/Final CE，本项目交接的是 `Soft KL + 小权重 hard CE`，并额外保留完整
Hidden/Cos、Draft-only 跨 Anchor、Action-RNN L1/Prefix 及启动前固定损失标定。这些差异来自 VLA 离线
多层 hidden、块内动作因果关系和历史训练退化，而不是为了形式上制造不同。

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

### 3.6 阶段六：Action-RNN 与 Base/Final 线性交接

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

`Base` 是并行 Draft hidden 经过冻结动作 `lm_head` 后的原始分布；`Final` 是 Base 加上 Action-RNN 残差后的
最终分布。曾尝试先 Hidden/Cos、再 Final 微调的两次独立训练，但阶段二大部分更新被用于修复 Draft 退化，
说明两个目标不能靠突然切换来衔接。当前恢复为一次训练，并使用 Domino 式全程线性交接：

```text
lambda: 1 -> 0
L_token = lambda * L_token_base + (1-lambda) * L_token_final
```

Soft KL 和 hard CE 都使用同一个 `lambda`，避免“Soft 只管 Base、CE 才交接”的人为职责划分。Hidden/Cos
始终直接监督 Draft，Draft-only 跨 Anchor KL 始终补强弱路径；Final Soft/CE 不 `detach`，因此后程同时改善
Draft 与 RNN。动作分布 L1 和 Prefix Survival 使用 detached Draft 输入，只训练 Action-RNN，并随 Final 比例
渐入。训练前自动标定一个固定低维总缩放，使初始高维:低维约为 9:1；训练中不再动态改变这个总缩放。

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

#### Draft-only 跨 Anchor Logit Distillation

对同一目标位置，较远 slot 的动作分布追近端强路径分布。强路径必须预测正确，且 teacher 分支
`stop_gradient`。当前主实验只在修正前 base logits 上计算 KL，并只训练 Draft 主干；Action-RNN 不再进行
同模型 final logits 自蒸馏，而是只接受真实 token、teacher 分布和连续前缀的直接监督。这使跨 anchor 模块
职责更明确：不同 anchor 提供同一目标位置的多种因果视角，专门用于补强并行 Draft 的远端弱路径。

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
/data/wulin/c/specvla-data/dflash_goal_dataset_envfix_20260714.h5            # 原始 v1，保留作校验/回退
/data/wulin/c/specvla-data/dflash_goal_dataset_envfix_20260714_packed_v2.h5  # 正式训练读取
```

已核对元数据：

| 字段 | 当前值 |
| --- | --- |
| `complete` | `True` |
| 样本数 | 28,501 |
| 训练语义格式 | `full_prefix_plus_action_hidden_v4` |
| 物理存储格式 | `dflash_hdf5_packed_v2` |
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

单文件并不自动等于顺序读。v1 内部仍有约二十万个 sample group/dataset；旧块采样又会让四个 rank 同时跳到
四个无关物理区域。packed v2 对训练实际读取的 BF16 hidden/token 做**逐 bit 无损重排**：完整 prompt 使用一块
连续数组加 offset，action hidden 与 token 使用固定形状连续数组；一个 local batch 只需一次 prompt hyperslab
读取。它不量化、不压缩、不改变样本、selected layer、position id 或监督标签。

当前 `DistributedBlockSampler` 以四卡相邻超级块为随机单位：默认每卡连续读取 64 个样本，四卡同一时刻处理
同一个 256 样本物理区域的相邻切片；每轮打乱超级块并轮换 rank 切片，完整覆盖全体样本，仅尾部按 DDP 规则
少量补齐。启动脚本固定 `NUM_WORKERS=1`、`prefetch_factor=1` 并开启 persistent worker，使每个 rank 全程复用
一个只读 HDF5 句柄。低优先级 `ionice/nice` 继续保护共享服务器。checkpoint 仍每 10 epoch 保存一次，但不保存
约 1.2 GiB optimizer state，也不额外复制根目录 latest 权重。

### 4.2 Context 和位置不变量

训练与推理必须同时满足：

- 保留完整 prompt/prefill 多层 hidden。
- 保留从 t0 到当前 anchor 的已验证 action hidden。
- selected layers 与 checkpoint 中 `dflash_config.json` 一致。
- prefix position 从 0 连续增长，action context 和 block position 紧随 prefix。
- `action_dim_embed` 标识动作维度；它补充 RoPE，不替代 RoPE。

### 4.3 当前单阶段高维主导训练

当前主实验一次运行 200 epoch。令 `B` 为 Base logits、`F` 为 Final logits、`lambda` 从训练开始的 1 线性
下降到结束时的 0：

```text
L_base_token  = 0.05 * soft_KL(B, teacher) + 0.01 * hard_CE(B, token)
L_final_token = 0.05 * soft_KL(F, teacher) + 0.01 * hard_CE(F, token)
L_token       = lambda * L_base_token + (1-lambda) * L_final_token

L_high = 1.00 * hidden_smooth_l1 + 0.05 * hidden_cosine
L_low  = L_token
       + 0.05 * backbone_cross_anchor_logit_KL
       + (1-lambda) * (0.05 * action_distribution_L1 + 0.05 * prefix_survival)

L_total = L_high + alpha * L_low
```

`alpha` 不是按 epoch 调出来的手工曲线。正式 optimizer step 之前，程序只前向读取默认 8 个 batch，同时计算
Base 和 Final 两端中较大的低维损失，再一次性求出固定 `alpha`，使初始低维占比不超过 10%。标定过程不反传、
不更新 optimizer/scheduler，也不写成训练 step。此后 `alpha` 固定，避免 Soft KL 初值 20-40 时压过 Hidden，
也避免后程动态追权导致总损失口径漂移。

| Loss | 原始权重 | Base/Final安排 | 更新 Draft | 更新 RNN | 目的 |
| --- | ---: | --- | ---: | ---: | --- |
| hidden SmoothL1 | 1.00 | 全程 | 是 | 否 | 保持高维 teacher 表征 |
| hidden cosine | 0.05 | 全程 | 是 | 否 | 对齐 hidden 方向 |
| soft KL | 0.05 | `lambda*Base+(1-lambda)*Final` | 是 | Final段是 | 对齐完整 teacher 分布 |
| hard CE | 0.01 | `lambda*Base+(1-lambda)*Final` | 是 | Final段是 | 轻量提高 strict top-1 |
| backbone cross-anchor KL | 0.05 | 全程只看 Base | 是 | 否 | 把强路径分布迁移给弱路径 |
| action distribution L1 | 0.05 | 随 Final 渐入 | 否 | 是 | 给修正头局部分布监督 |
| prefix survival | 0.05 | 随 Final 渐入 | 否 | 是 | 鼓励连续接受前缀 |

Draft 学习率为 `2e-5`，warmup 1000 step；Action-RNN 为 `5e-5`，warmup 500 step，之后都线性退火。
global batch 为 64，`slot_decay=0.90`、`position_balance=True`、gradient clip `0.5`。confidence head、旧 hidden
CAD、旧 causal residual、旧 refined hidden、旧 residual CE 和 Action-RNN 自身的跨 Anchor KL 均关闭。

### 4.4 SwanLab 指标怎样读

`metrics.jsonl` 保留全量原始指标。原始 loss 与 `*_component` 的关系为：

```text
component = raw_loss * 配置权重
```

真正决定总 loss 比例的是 component。核心指标：

| 指标 | 含义 |
| --- | --- |
| `base_accuracy` | Action-RNN 残差加入前的 frozen-lm-head 命中率 |
| `accuracy` | teacher-forced Action-RNN 修正后的总体命中率 |
| `action_head_accuracy_gain` | `accuracy-base_accuracy`；直接判断残差动作头是否真的带来收益 |
| `base_action_token_ce_loss` | 修正前 Base logits 的诊断 CE；阶段二不计入总 loss |
| `action_token_ce_loss` | 修正后 Final logits 的 hard CE；阶段二以 0.05 计入总 loss |
| `token_curriculum_component` | 兼容字段名；当前阶段二固定等于 `0.05*FinalHardCE`，没有 curriculum |
| `base_soft_loss` | Base 分布诊断 KL；阶段二不计入总 loss |
| `final_soft_loss` / `soft_loss` | Final teacher KL；阶段二两者相同，以 0.05 计入总 loss |
| `backbone_anchor_logit_distill_loss` | Draft base logits 的跨 anchor KL；只更新主干 |
| `anchor_logit_distill_loss` | 旧 Action-RNN 跨 anchor KL；当前主实验关闭，仅保留兼容代码 |
| `rollout_accuracy` | anchor0 使用自身预测前缀回滚时的 top-1 命中率 |
| `rollout_exposure_gap` | `accuracy-rollout_accuracy`；衡量 teacher forcing 到自回滚的分布差距 |
| `rollout_top2_accuracy` | 正确 token 是否进入 self-rollout top-2 |
| `rollout_runner_up_rescue_rate` | `rollout_top2_accuracy-rollout_accuracy`；正确 token 恰好是第二候选的位置占全部位置的比例，是单分叉的潜力而非真实接受率 |
| `rollout_conditional_runner_up_rescue_rate` | 仅在 top-1 错误位置中，正确 token 位于第二候选的比例 |
| `base_rollout_expected_prefix_length` | anchor0 的并行 Draft base logits 连续猜对长度 |
| `rollout_expected_prefix_length` | anchor0 实际连续猜对的平均前缀长度，范围 0 到 6 |
| `rollout_prefix_length_gain` | Action-RNN 自回滚连续长度减去 Draft base 连续长度；应长期为正 |
| `base/rollout_distribution_overlap` | Draft base / Action-RNN 自回滚分布与 teacher 分布的重合率 `sum(min(p,q))` |
| `base/rollout_expected_accept_length_proxy` | 将逐位置分布重合率累乘后求和得到的理论 accepted-prefix 代理，越高越好 |
| `action_head_accept_length_proxy_gain` | Action-RNN 相对 Draft base 的理论接受长度增益；应长期为正 |
| `hidden_cosine_similarity` | `1-cos_loss`，监控高维 hidden 表征是否真正趋近 teacher |
| `representation_token_gap` | `base_accuracy-hidden_cosine_similarity`；过早大幅为正提示 token 捷径/记忆风险 |
| `rollout_prefix_k_success_rate` | anchor0 从 p1 开始至少连续猜对 k 步的概率；随 k 单调不增 |
| `position_k_acc` | 所有 anchor 对绝对位置 pk 的 teacher-forced 平均准确率 |
| `anchor_a_to_position_k_acc` | 指定 anchor 到指定目标位置的准确率矩阵 |
| `action_accept_probability_proxy` | 由 teacher/student 分布距离得到的单位置接受概率代理 |
| `expected_prefix_length` | 分布式可微前缀代理，不等于在线论文 Length |

并且有严格关系：

```text
rollout_expected_prefix_length = sum(k=1..6, rollout_prefix_k_success_rate)
```

`rollout_accuracy` 是六个位置的平均自回滚命中率；后部位置在前部已错时重新命中也会计入。连续前缀成功率
要求 p1 到 pk 全部正确，因此更贴近 strict 投机验证。训练过程中优先看
`rollout_expected_prefix_length`、`rollout_expected_accept_length_proxy` 及两项 Action-RNN 增益；普通 accuracy
只能作为辅助。所有这些仍是离线训练集代理，不能代替 LIBERO 在线 `Length/Speedup/Success Rate`。

`train/*` 是每 20 optimizer step 的局部窗口，可能受批次难度影响；`train_epoch/*` 是整轮均值并完整保存在
`metrics.jsonl`，判断长期收敛时应优先看后者。

当前单阶段训练只创建一个输出目录和一个 SwanLab run，同时记录 Base、Final、跨 Anchor、RNN 与 Hidden/Cos。
在线表名使用中文并按以下类别组织：

```text
训练损失 / 训练准确率 / 训练自回滚
训练连续前缀成功率 / 训练逐位置自回滚 / 训练阶段 / 训练优化器
```

当前主训练每 20 optimizer step 上传核心标量和连续前缀成功率，每 200 step 上传 rollout 逐位置明细。
完整英文科研记录仍写入 `metrics.jsonl`。损失构成额外记录 `high_dim_component`、
`low_dim_unscaled_component`、`low_dim_component` 和 `low_dim_fraction`，用于确认 9:1 预算没有失控。

SwanLab 的 loss 曲线是最近 20 optimizer step 的窗口均值，可能随 batch 难度在单个 epoch 内上扬；跨 epoch
收敛应读取当前输出目录 `metrics.jsonl` 中的 `train_epoch/*`。Base/Final 是同一条连续训练时间轴，可以直接看
整程 total loss，同时必须结合高维/低维 component 和 `lambda` 判断变化来源。

## 5. 推理和验证

### 5.1 Partial accept 与 correction

目标模型验证 proposal 后接受最长合法前缀。遇到第一个拒绝位置时，写入目标模型自己的 posterior token，
再进入下一 anchor。因此 strict 路径始终由目标模型裁决，支持部分接受和部分纠正。

目标校验只输入 `q-1` 个 proposal 是标准 next-token shift：anchor logits 校验第一个 proposal，输入前
`q-1` 个 proposal 后得到的 logits 分别校验余下 token。最后一个 proposal 不需要再次作为本块输入。

### 5.2 动作组级宽松校验

SpecVLA 原始 relaxed 规则逐 token 判断 `|draft_i-target_i| <= r`，等价于在每个动作维度上分别设置 bin
误差上限。本项目进一步利用七维动作的物理结构，把连续运动维度分成：

```text
平移组 G_t = (x, y, z)
旋转组 G_r = (roll, pitch, yaw)
离散组       = gripper
```

对于当前块中可见的某个运动组 `G`，令 `delta_i` 为 draft 与 target 的 bin 距离，动作组通过条件为：

```text
sum(delta_i^2, i in G) <= |G_visible| * r^2
```

通过时，该组当前可见的维度整体视为可接受。因此，一个维度即使略微超过逐 token 阈值，只要其它维度误差
较小、组内总误差仍在同等最坏角点预算内，也可以被挽救。组条件未通过时保留原逐 token 判断，不额外惩罚
本来已通过的维度。`gripper` 是类别决策，相邻 token id 不代表“接近的开合程度”，所以始终要求与目标模型
完全相等。最终仍取从当前位置开始的最长连续可接受前缀，首个拒绝位置由目标模型纠正。

这项机制与 Draft 结构正交，不增加模型 forward；它试图把 token 空间的离散误差转换为更符合动作几何的
接受规则。代价是它不再属于 strict lossless 校验，必须同时报告阈值 `r`、任务成功率、Length 和 Speedup。
代码额外记录：

| 指标 | 含义 |
| --- | --- |
| `action_group_rescued_blocks` | 组级规则比逐 token 规则接受更长前缀的块数 |
| `action_group_extra_accepted` | 组级规则额外接受的 token 总数 |

### 5.3 校准式稀疏多候选树校验

当前实现是稀疏多候选树的最小固定预算版本：**单分叉、双路径**，并不是任意宽度的 beam search。流程如下：

1. DFlash transformer 只执行一次，Action-RNN 从 Base logits 生成贪心主路径。
2. 在一个固定分叉位置取主路径 logits 的第二候选 token。
3. 复用该位置之前的前缀和 RNN state，只用轻量 Action-RNN 滚出第二候选的后缀，不再执行 DFlash forward。
4. 主路径和备选路径共享分叉前节点；tree mask 保证两条后缀只能看见各自的因果祖先。
5. 目标模型用一次 tree-attention forward 同时校验两条路径，选择可接受前缀更长的一条。
6. 只把胜出路径中真正提交的 hidden 和 KV 节点写回缓存；拒绝点仍由目标模型 posterior 纠正。

若块长为 `q`，线性校验输入 `q-1` 个节点；在索引 `b` 单分叉后，稀疏树只额外加入
`q-1-b` 个备选后缀节点。它避免完整 top-k 树的指数膨胀，目标是用少量额外验证宽度覆盖主路径中高价值的
runner-up，而不是盲目增加候选数。默认只在 anchor0 的首个最长块启用，后续短块不再支付树开销。

树是否提速不能由 top-2 命中率直接决定，因此 strict 评测先在真实 observation 上对
`off/p2/p3/p4/p5` 做配对的完整动作延迟校准。候选位置必须同时满足：确实触发过分叉、相对 off 的中位延迟
更低、经过多重比较修正的单侧符号检验显著；否则自动选择 `off`。这使树成为“有实测净收益才打开”的系统
机制，而不是固定增加开销的启发式补丁。代码记录：

| 指标 | 含义 |
| --- | --- |
| `tree_triggered_blocks` | 实际构造双路径树的块数 |
| `tree_selected_alternate_blocks` | 最终选择备选路径的块数 |
| `tree_extra_verified_nodes` | 相比线性校验额外送入目标模型的树节点数 |
| `tree_extra_accepted` | 备选路径比主路径额外接受的 token 数 |
| `tree_mean_branch_score` | runner-up 相对主候选的平均概率比代理 |

### 5.4 训练创新与推理创新怎样闭环

高维主导线性交接、跨 Anchor 和 Action-RNN 解决“候选能否覆盖目标 token”；动作组校验和稀疏候选树解决
“已有候选怎样以有限验证成本转化为更长接受前缀”。两类推理机制彼此正交：树增加离散候选覆盖，动作组规则
放宽连续运动误差；二者可以组合，但必须分别做 `off/on` 消融并计入完整动作延迟。strict + tree 仍保持目标
token 精确校验；action-group relaxed 则以成功率约束换取更大的接受空间。

## 6. 实验历程与目前证据

### 6.1 训练演进

| 阶段 | 关键设置 | 结果 | 得到的结论 |
| --- | --- | --- | --- |
| Pure hidden | 完整 prefix、1 层、hidden+cos | 末值 acc 0.807，最好 0.830 | p1/p6 强，anchor0 p2-p5 弱 |
| Residual-CAD | 前 token hidden residual + Hidden 跨 anchor 蒸馏 | anchor0->p2 约 0.503 -> 0.671 | 强 anchor 信息可以帮助弱路径，但 hidden 修正不够直接 |
| Markov-ACD 短跑 | 加 logits bias、token CE、logit 跨 anchor 蒸馏 | anchor0 p2-p5 约 0.86/0.91/0.88/0.88 | token/logit 因果监督有效 |
| Clean Markov-ACD 200 epoch | 删除手工 boost，覆盖 slot0，低权重 soft | teacher-forced acc 约 0.999 | 离线饱和不代表在线饱和 |
| 三阶段 Action-RNN Prefix Survival | epoch 21/101 硬切模块，动作目标与主干完全 detach | 未完成，改为课程版 | 硬边界与完全隔离不利于 Draft/RNN 共同优化 |
| 旧统一余弦课程 Action-RNN | Soft/CE 从第 1 轮生效，Base/Final 连续交接 | 约 43 epoch 时 token acc 已接近 99%，hidden 尚未稳定 | token 监督入场过早，出现低维动作子空间捷径 |
| Hidden-first 延迟余弦课程 | hidden/cos 全程；`g=s^4` 延迟 Soft/CE/跨 Anchor/RNN | 初期 raw token loss 抖动不影响梯度，但后程加权总 loss 的解释仍复杂 | 单次课程无法提供完全独立、干净的两种优化过程 |
| 独立两阶段 | 100 epoch Hidden/Cos；再重置优化器做 Final/KL/CE/跨 Anchor/RNN | 阶段二初期 Draft hidden/连续前缀退化，RNN 大量增益用于补偿退化 | 突然切换优化目标会产生表示冲突，废弃为消融 |
| 当前线性交接主实验 | 200 epoch；Soft/CE 同步 Base->Final；低维组自动固定为初始约 10% | 已完成；rollout acc 0.853，Final 前缀代理 3.576 | 高维主线没有退化，Action-RNN 得到稳定但有限的正增益，需由 4090 在线评测裁决 |

Pure hidden、Residual-CAD、Markov-ACD 的代表性 anchor0 结果：

| 阶段 | p1 | p2 | p3 | p4 | p5 | p6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure hidden 末值 | 0.886 | 0.503 | 0.716 | 0.620 | 0.692 | 0.953 |
| Residual-CAD 末值 | 0.846 | 0.671 | 0.717 | 0.647 | 0.718 | 0.936 |
| Markov-ACD 短跑末值 | 0.820 | 0.860 | 0.913 | 0.881 | 0.880 | 0.963 |

### 6.2 已废弃的独立两阶段实验

两阶段实验先运行 100 epoch Hidden/Cos，再从阶段一权重初始化一个全新的 100 epoch Final/RNN 任务。阶段二
重新创建 optimizer、scheduler 和 SwanLab，本意是避免低维动作监督在 hidden 成熟前诱导记忆捷径；实际却
产生了新的表示切换冲突。阶段一终点与阶段二约 epoch 19 的可比指标如下：

| 指标 | 阶段一 epoch 100 | 阶段二约 epoch 19 | 变化 |
| --- | ---: | ---: | ---: |
| Hidden loss | 0.74734 | 0.75439 | 退化 |
| Hidden cosine similarity | 0.58466 | 0.56247 | 退化 |
| Base accuracy | 0.62864 | 0.62950 | 基本不变 |
| Final teacher-forced accuracy | - | 0.63685 | 仅小幅超过 Base |
| 自回滚总体准确率 | 0.51748 | 0.52563 | 小幅提高 |
| Base连续前缀长度代理 | 1.0204 | 0.9286 | 明显退化 |
| Final连续前缀长度代理 | - | 1.0652 | 相对阶段一净增益很小 |
| 理论接受长度代理 | 1.0368 | 1.0726 | 仅小幅提高 |

阶段二内部看，Action-RNN 相对“已经退化的当前 Base”能增加约 `0.136` 的连续前缀长度；但相对阶段一终点，
最终净增益只有约 `0.045`。同时低维加权项约占总损失 38%，Draft 裁剪前梯度范数由阶段一约 `0.35` 上升到
约 `5.7`。因此 RNN 的很大一部分能力被用于补偿阶段切换造成的 Draft 退化，而不是在稳定底稿上增加因果
修正。这是废弃两阶段、恢复单次平滑交接的直接证据，并非仅凭总 loss 曲线作出的判断。

### 6.3 当前 Base/Final 线性交接实验

当前实验从随机初始化一次训练 200 epoch，不加载上述阶段一 checkpoint。实验目录和入口分别为：

```text
/data/wulin/c/specvla-data/ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2
bash openvla/specdecoding/train-scripts/run_dflash_train.sh joint
```

本版只让一件事随训练进度改变：`lambda_base` 从 1 线性下降到 0，`lambda_final` 同步从 0 上升到 1。Soft KL
与 hard CE 一起交接；Hidden/Cos、Draft-only 跨 Anchor 和自动求得的低维总缩放保持固定口径；RNN 专属
L1/Prefix 随 Final 比例进入。正式 step 0 前 8 个只读 batch 负责求固定 `alpha`，目标是初始低维损失占比不
超过10%。本实验首先需要验证：

1. `low_dim_fraction` 初期接近且不显著超过 0.10，整程不出现后期权重口径突变。
2. Hidden loss与相似度不再像两阶段切换后那样退化。
3. Base连续前缀代理保持成长，Final相对Base的增益不是建立在Base下降之上。
4. p2-p5、连续前缀成功率和在线接受长度同步提高，而不只是teacher-forced accuracy饱和。
5. 最终仍以4090上的Success Rate、Length和端到端Speedup判断，不以训练代理宣布成功。

本次训练于 2026-07-24 正常完成 200 epoch。后程没有发生两阶段实验中的表示退化；约 epoch 140 后进入
平台，RNN 增益仍缓慢上升。以下是 epoch 200 的整轮均值，位置准确率为该轮所有记录 step 的均值：

| 指标 | epoch 200 |
| --- | ---: |
| Total / Hidden / Cos loss | 0.80796 / 0.76437 / 0.41954 |
| Base / Final soft loss | 1.48529 / 1.45862 |
| Draft 跨 Anchor KL | 1.50708 |
| Base / Final action CE | 0.52285 / 0.48891 |
| Action L1 / Prefix Survival | 0.32810 / 0.29844 |
| Base / Final teacher-forced accuracy | 0.86881 / 0.88343 |
| Self-rollout accuracy | 0.85302 |
| Self-rollout p1-p6 | 0.962 / 0.748 / 0.843 / 0.785 / 0.812 / 0.969 |
| Base / Final 连续前缀长度代理 | 3.352 / 3.576 |
| Base / Final 接受长度代理 | 2.725 / 2.828 |
| Action-RNN 接受长度代理净增益 | +0.102 |
| Hidden cosine similarity | 0.580 |

与早期版本相比，p2-p5 和连续前缀显著提高，且 Final 的提升没有建立在 Base 退化之上；但这些仍是离线
teacher/self-rollout 代理，不是 LIBERO 在线命中率。epoch 180 的连续前缀均值略高（3.577），epoch 200 的
RNN 接受长度增益最高（约 +0.102），因此正式评测优先搬运 `epoch 180` 与 `epoch 200`，不根据训练集指标
继续挑更多 checkpoint。

### 6.4 Markov-ACD 在线结果

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

### 6.5 已废弃的旧余弦课程早期快照

2026-07-15，旧 3090 主训练在 step 500、warmup 进行到一半时：

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
以及 teacher-forced 与 rollout 约有 9 个百分点差距。该快照属于三阶段方案之前的联合训练，仅作为问题诊断，
不能当作新版收敛结果。

到 epoch 6-12，旧耦合配方出现稳定的轮内锯齿：action CE/L1/Prefix/KL 在每轮内上升，准确率下降，下一轮
开头又恢复；学习率始终平滑，因此不是 scheduler 重启。去除 epoch 均值后，hidden 与 CE/L1/Prefix 的
相关系数约为 `-0.61/-0.53/-0.47`。epoch 12 时 teacher-forced head accuracy 约 `0.922`，base accuracy
约 `0.921`，动作头净增益仅约 `0.0005`，而 rollout 约 `0.889`。这说明高权重动作损失主要在改动共享
Draft hidden，Action-RNN 本身几乎没有学到有效残差。该版随后在 hidden 尚未收敛时就出现接近 99% 的
训练 token acc，已判定为 token 级监督过早。后续高阶延迟余弦课程把太多损失同时改变权重；独立两阶段又在
切换时产生表示冲突。当前版本改为只有 Base/Final 路径比例线性变化，Soft 与 CE 同步交接，低维组只在启动前
标定一次固定总尺度。旧余弦和两阶段输出目录均不得续训到新配方。

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

`run_dflash_data_goal.sh` 会先生成语义完整的 raw v1，再自动无损打包为训练用 packed v2。确认 smoke 后正式
生成；当前正式文件已经存在，重新生成时必须同时提供新的中间文件和最终文件名：

```bash
GPU_ID=0 \
RAW_OUT_FILE=/data/wulin/c/specvla-data/dflash_goal_dataset_new_raw_v1.h5 \
OUT_FILE=/data/wulin/c/specvla-data/dflash_goal_dataset_new_packed_v2.h5 \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh full
```

默认 `KEEP_RAW=True`，便于逐 bit 回查；空间紧张时可显式设 `KEEP_RAW=False`，脚本也只会在 packed 文件完整
关闭并标记成功后删除中间 v1。不要手工删除 `.partial` 以外的文件。

生成结束检查：

```bash
python - <<'PY'
import h5py

path = "/data/wulin/c/specvla-data/dflash_goal_dataset_new_packed_v2.h5"
with h5py.File(path, "r") as h5:
    print("complete:", bool(h5.attrs["complete"]))
    print("storage:", h5.attrs["format"])
    print("semantics:", h5.attrs["dflash_data_format"])
    print("layers:", list(h5.attrs["hidden_layer_ids"]))
    print("samples:", int(h5.attrs["num_samples"]))
PY
```

只有 `complete=True`、`storage=dflash_hdf5_packed_v2` 才能作为当前正式训练输入。

### 7.2 3090 四卡训练

当前主实验只启动一次：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh joint
```

默认输出：

```text
/data/wulin/c/specvla-data/ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2/
├── metrics.jsonl
├── run_config.json
├── swanlog/
├── latest_checkpoint.txt
└── epoch_200_step_*/
```

脚本启动时先执行默认 8 个只读 batch 的损失标定，随后才从 epoch 1 / step 0 正式训练。要换 GPU 或新输出
目录，只需覆盖环境变量：

```bash
DATAPATH=/data/wulin/c/specvla-data/dflash_goal_dataset_new_packed_v2.h5 \
OUTPUT_DIR=/data/wulin/c/specvla-data/ckpt_goal_dflash_joint_new \
CUDA_VISIBLE_DEVICES=2,3,4,6 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh joint
```

脚本发现输出目录已有 `run_config.json`、`metrics.jsonl` 或 `swanlog/` 时会拒绝启动，防止日志串线。共享盘
默认使用一张卡一个 DataLoader worker；不要为了短期吞吐直接把 worker 开到 4。要试探吞吐上限，可先改成
`NUM_WORKERS=2`，同时观察系统 `iowait` 和其他用户交互延迟。

### 7.3 把 checkpoint 搬到 4090

在本地终端执行。当前正式候选固定为 epoch 180 和 200：

```bash
ssh 4090 \
  'mkdir -p /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2'

for epoch in 180 200; do
  CKPT_3090=$(ssh 3090_wulin \
    "find /data/wulin/c/specvla-data/ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2 \
     -maxdepth 1 -type d -name \"epoch_$(printf '%03d' ${epoch})_step_*\" | sort -V | tail -1")
  scp -3 -r "3090_wulin:${CKPT_3090}" \
    '4090:/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2/'
done
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

当前 Goal 权重不能用于 Object、Spatial 或 Long。四路机制消融一键评测：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 EVAL_EPOCH=200 \
DFLASH_OUTPUT_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2 \
  bash openvla/specdecoding/decode-scripts/run_dflash_action_rnn_goal_4way_eval.sh
```

四路依次为 `RNN+strict`、`RNN+树+strict`、`RNN+动作组`、`RNN+树+动作组`。树版 strict 在真实
observation 上校准 `off/p2/p3/p4/p5`，最后一组复用该位置；若没有候选位置在完整动作延迟上显著优于
`off`，树分叉位置为 0，树版自动退化为线性验证。评测 epoch 180 时只需把上面的 `EVAL_EPOCH` 改为 180。

单独执行：

```bash
EVAL_EPOCH=100 bash openvla/specdecoding/decode-scripts/run_dflash_goal_eval.sh strict
EVAL_EPOCH=100 bash openvla/specdecoding/decode-scripts/run_dflash_goal_eval.sh relaxed
```

推理机制消融必须固定同一 checkpoint、seed 和计时口径：

```bash
# 线性 strict：关闭候选树
DFLASH_TREE_MODE=off DFLASH_TREE_AUTO_CALIBRATE=False \
  EVAL_EPOCH=100 bash openvla/specdecoding/decode-scripts/run_dflash_goal_eval.sh strict

# 原逐 token relaxed：关闭动作组规则与候选树
DFLASH_ACCEPTANCE_MODE=token DFLASH_TREE_MODE=off \
  EVAL_EPOCH=100 bash openvla/specdecoding/decode-scripts/run_dflash_goal_eval.sh relaxed

# 动作组 relaxed，但关闭候选树
DFLASH_ACCEPTANCE_MODE=action_group DFLASH_TREE_MODE=off \
  EVAL_EPOCH=100 bash openvla/specdecoding/decode-scripts/run_dflash_goal_eval.sh relaxed
```

五个 seed 重复性评测：

```bash
EVAL_EPOCH=100 REPEAT_SEEDS="7 8 9 10 11" \
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
| `run_dflash_data_goal.sh` | `smoke` 或 `full` 生成 raw v1 并自动无损打包 packed v2 |
| `pack_dflash_hdf5.py` | 将既有 v1 数据一次性迁移为 packed v2；日常无需单独调用 |
| `benchmark_dflash_hdf5.py` | 四进程只读 A/B 测试 legacy v1 与 packed v2 的真实数据吞吐 |
| `run_dflash_train.sh joint` | 当前主实验：200 epoch 高维主导 Base/Final 线性交接 |
| `run_dflash_train.sh stage1/stage2` | 仅用于复现已废弃的两阶段消融 |

`train_dflash_libero_goal.py` 是底层训练实现，不建议日常手写几十个 CLI 参数。历史单次课程仍保留在 Python
兼容参数中用于解释旧 checkpoint，但当前主实验只启用 `domino_linear_curriculum`。

### 8.2 推理入口

| 脚本 | 用法 |
| --- | --- |
| `run_specvla_paper_ar_eval.sh` | 一个 suite 的论文 AR 分母 |
| `run_specvla_eval.sh` | 一个 suite 的 strict/relaxed |
| `run_specvla_goal_upstream_compatible_eval.sh` | Goal AR+strict+relaxed 一键复现 |
| `run_specvla_main_table_eval.sh` | 四 suite strict/relaxed 自动续跑与汇总 |
| `run_dflash_goal_eval.sh` | 当前 Goal DFlash 单项 strict/relaxed |
| `run_dflash_action_rnn_goal_4way_eval.sh` | 同一 Goal checkpoint 的 RNN/树 × strict/动作组四路消融 |
| `run_dflash_action_rnn_goal_pair_eval.sh` | 兼容旧流程的树 strict+树动作组成对评测 |
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

1. 当前 Action-RNN 训练已经完成；离线 p2-p5、连续前缀和 RNN 增益均有提高。
2. 在 4090 优先评测 epoch 180/200 的四路机制组合，联合报告 SR、Length、Speedup 和在线位置命中率。
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

@article{huang2026domino,
  title={Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding},
  author={Huang, Jianuo and others},
  journal={arXiv preprint arXiv:2605.29707},
  year={2026}
}

@article{cheng2026dspark,
  title={DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation},
  author={Cheng, Xin and others},
  journal={arXiv preprint arXiv:2607.05147},
  year={2026}
}
```
