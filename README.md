# SpecVLA-DFLASH

本仓库研究一个具体问题：能否把 DFlash 式块并行草稿模型迁移到 OpenVLA，在保持目标模型校验可靠性的
前提下，提高 LIBERO 动作解码速度。

当前主线由三层组成：块并行 Draft 与 Action-RNN 产生动作 proposal；动作词表投影、首 token 早拒绝与
历史块融合校验降低每块固定成本；首 token 哨兵时序级联在稳定段把上一条已验证动作直接并入当前多模态
prefill 严格校验，其余时刻再由 target 哨兵选择历史 proposal 或 DFlash，target 一旦拒绝便用真实纠正前缀
切回 DFlash。动作组宽松校验与 DDTree 已完成实现和消融，但当前完整评测没有证明它们优于
SpecVLA relaxed，因此不再作为默认最佳方案。所有模块仍必须同时接受成功率、Length 和端到端 Speedup 检验。

代码基于 [SpecVLA](https://github.com/PineTreeWss/SpecVLA)，而 SpecVLA 又基于
[OpenVLA](https://github.com/openvla/openvla)。当前仓库仍是研究代码，不是已经定稿的公开复现包。
所有结论必须同时报告成功率、接受长度和速度；训练准确率不能替代在线 LIBERO 结果。

## 1. 阅读顺序和项目地图

建议按下面的顺序理解项目：

1. 先理解 OpenVLA 为什么把一个动作表示为 7 个离散 token。
2. 再理解 SpecVLA 如何用小草稿模型逐 token 提案，并让 OpenVLA 校验。
3. 然后理解本项目为什么改用 DFlash 式块并行 hidden 生成。
4. 最后沿着实验历程看清楚：完整上下文、multi-anchor、跨 anchor 蒸馏、Action-RNN、失败的树/动作组尝试，
   以及为什么研究重点转向固定成本和选择性免校验。

当前核心文件：

| 环节 | 文件 | 作用 |
| --- | --- | --- |
| 数据生成 | `openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py` | 用冻结 OpenVLA 生成动作 token 和多层 hidden 教师数据 |
| 数据入口 | `openvla/specdecoding/train-scripts/run_dflash_data_goal.sh` | 统一 smoke/full 数据生成命令 |
| 数据无损打包 | `openvla/specdecoding/train-scripts/pack_dflash_hdf5.py` | 把每样本 group 的 HDF5 v1 重排为连续 packed v2 |
| DFlash 训练 | `openvla/specdecoding/train-scripts/train_dflash_libero_goal.py` | multi-anchor、loss、DDP、checkpoint、SwanLab |
| 训练入口 | `openvla/specdecoding/train-scripts/run_dflash_train.sh` | 当前 Action-RNN 主实验及少量结构消融 |
| Draft 模型 | `openvla/specdecoding/model/dflash.py` | 块并行主干、动作位置 embedding、Action-RNN |
| 在线推测解码 | `openvla/prismatic/extern/hf/modeling_speculation.py` | draft 提案、目标模型校验、partial accept/correction、树验证 |
| LIBERO strict | `openvla/experiments/robot/libero/run_libero_goal_Spec.py` | strict token 校验与在线指标 |
| LIBERO relaxed | `openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py` | relaxed 动作接受与在线指标 |
| 推理公共配置 | `openvla/specdecoding/decode-scripts/libero_eval_common.sh` | 两台机器路径、suite 权重、计时口径和 checkpoint 解析 |
| 时序级联入口 | `openvla/specdecoding/decode-scripts/run_dflash_temporal_cascade_goal_eval.sh` | shadow、严格路由、prefill 融合和稳定段免校验 |
| 时序门校准 | `openvla/specdecoding/test-speed/analyze_dflash_temporal_shadow.py` | 从 shadow summary 统计覆盖、错误和 95% 风险上界 |

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
当前多模态 prompt + 上一条 target 已验证动作
                       ↓
        稳定至少 3 次？──是──→ 历史动作附入 prompt
                       │                ↓
                       │       必做 prefill 同时严格校验
                       │                ↓
                       │       接受前缀 / target 纠正 / 裁 KV
                       ↓ 否             ↓ 未完成
              target 首 token 哨兵与时序路由
                       ↓
          历史 proposal 可用？──是──→ 融合 target 校验
                       │                         ↓ 拒绝
                       ↓ 否                 target 纠正前缀
              一次 DFlash 块并行 forward ←──────┘
                       ↓
        frozen lm_head → Action-RNN 因果修正
                       ↓
                  target 块校验
                       ↓
 strict 精确接受 / action-group 宽松接受 / 独立免校验消融
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
| DDTree 动态多候选树 | [DDTree](https://arxiv.org/abs/2604.12989) | 用累计对数概率最佳优先扩展固定预算节点；共享前缀、祖先注意力、一次目标校验、目标 token 驱动遍历和缓存压紧 | 推理端系统设计之二 |
| 首 token 哨兵时序级联 | 本项目，受 HeiSD 时序相似性分析启发但不使用离线轨迹库 | 普通路径由当前 `t0` 和 prompt hidden 判断是否复用上一动作；target 首次拒绝后立刻切回带真实纠正前缀的 DFlash | 当前推理端主线 |
| 验证式时序 Prefill 融合（VTPF） | 本项目 | 连续 3 次 target 确认相同动作后，把历史 `c0..c5` 附在当前 prompt 后；一次必做的多模态 prefill 同时严格校验 `c0..c6`，拒绝后裁掉错误 KV 并回退 DFlash | 降低 target 固定调用数的结构性增量 |
| 固定成本无损优化 | 本项目的实现诊断 | 目标 `lm_head` 仅投影 256 个动作 token；首 proposal 已被 anchor 判错时不再验证无效后缀；已知历史 proposal 时把 anchor 与整块校验融合为一次 target forward | 当前默认开启 |

因此，当前论文主线应表述为：**块并行主干、Action-RNN 和跨 Anchor 蒸馏学习动作候选；目标首 token
既是严格校验的第一个 posterior，也是低成本时序哨兵；VTPF 进一步把高置信历史 proposal 的校验并入当前
必做 prefill，而普通时序路由负责其余动作。** Domino 与 DSpark 是清晰标注的训练启发，VTPF、target
反馈切源、融合校验、首 token 早拒绝和动作词表投影是当前推理端增量。最终新颖性仍须结合 HeiSD、缓存类工作与完整
消融审慎表述。

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
| `rollout_runner_up_rescue_rate` | `rollout_top2_accuracy-rollout_accuracy`；正确 token 恰好是第二候选的位置占全部位置的比例，是动态树增加宽度的潜力而非真实接受率 |
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

### 5.3 DDTree 动态多候选树校验

2026-07-25 的第二次修订已删除“固定位置单分叉、双路径、在线延迟校准”整套旧实现，改为与 DDTree 核心
算法一致的动态树。树不会在评测中偷偷切回 `off`，也不会为不同分叉反复调用目标模型。流程如下：

1. DFlash transformer 只执行一次，Action-RNN 得到各 slot 的动作分布和贪心主路径。
2. 将每个候选节点的分数定义为从根到该节点的累计对数概率，用优先队列每次弹出联合概率最高的节点；
   同时把它的同层次候选和下一深度候选放回队列，直到用完固定节点预算。高置信路径自然长得更深，
   不确定位置自然保留多个分支，不再手工指定 `p2/p3/p4/p5`。
3. 候选被编译为父子节点图和祖先可见 mask。由于本项目先用目标模型解码 anchor hidden，anchor 已在 KV
   cache 中并作为虚拟根；候选节点在同一次 target tree-attention forward 中并行校验。
4. strict 从虚拟根开始，只沿目标模型 posterior 指定的 child 前进；树中没有该 token 时立刻停止，并把
   目标 token 作为纠正。它不会比较整条候选的未来接受长度，也不会改变目标模型答案。
5. 只把真实走过的节点按索引压紧并提交 hidden/KV。action-group 模式则在每条叶路径上使用同一组级预算，
   选择最长合法近似前缀，因此仍属于 relaxed 结果。

默认 `DFLASH_TREE_BUDGET=0` 表示当前块自动使用 `q-1` 个候选节点。连同已经单独解码的 anchor，这与线性
验证送入目标模型的节点总数相同；DDTree 做的是重新分配这些节点的深度和宽度，而不是额外增加目标模型
forward。显式设为正整数可以研究更宽的树，但会增加 target 验证宽度，必须单独做开销消融。构树需要一次
很小的 GPU 到 CPU 的 top-k 分布复制和优先队列操作，这是 DDTree 本身的开销，不能从计时中排除。代码记录：

| 指标 | 含义 |
| --- | --- |
| `tree_triggered_blocks` | 实际构造动态树的块数 |
| `tree_selected_alternate_blocks` | 动态树相对贪心路径额外挽救连续 token 的块数 |
| `tree_extra_verified_nodes` | 相比线性校验额外送入目标模型的树节点数 |
| `tree_extra_accepted` | 动态树比贪心路径额外接受的 token 数 |
| `tree_average_verified_nodes` | 每个触发块的平均候选节点数 |
| `tree_average_max_depth` | 每棵动态树的平均最大深度 |

### 5.4 首 token 哨兵时序级联

一次动作的目标模型 prefill 无论如何都要执行，因为它负责融合当前图像和指令，并给出绝对可信的第一个动作
token `t0`。本项目不再把这一步只当作启动 token，而把它作为时序候选的低成本哨兵：

1. **验证式时序 Prefill 融合（Verified Temporal Prefill Fusion, VTPF）。** 当同一动作已被 target 连续确认
   至少 3 次时，上一动作的 7-token proposal 在当前图像进入 LLM 前就已知。实现把历史 `c0..c5` 附在当前
   prompt 后，使本来就必须执行的多模态 prefill 在最后 7 个 logits 上同时校验 `c0..c6`。只提交 target
   连续接受的候选；首个拒绝位置写入 target correction，未接受候选的 KV 立即裁掉，余下位置切回 DFlash。
   它不跳过 target，也不增加 target forward，而是让 prefill 同时承担第一块严格验证。
2. **严格时序路由。** 未进入 VTPF 时，若当前 `t0` 与上一条已验证动作的 `t0` 相同，且最后层 prompt hidden cosine 不低于
   路由阈值，则优先用上一动作的 `t1..t6` 作为 proposal。目标模型仍按原 strict 规则验证，因此候选错误时
   仍会 partial accept/correction；这层只改变 proposal 来源，不主动放宽目标答案。
3. **融合哨兵验证。** 历史 proposal 在 anchor 解码前已经存在，因此把
   `[当前 anchor, 历史 proposal[:-1]]` 一次送入 target；这一 causal forward 的各位置 logits 恰好校验整块
   proposal，同时产出后续回退需要的 target hidden/KV。它把旧实现每个历史块的“anchor forward + verify
   forward”合并为一次，不改变 target 驱动的 prefix accept/correction。
4. **Target 反馈切源。** 历史 proposal 一旦在任意位置被 target 拒绝，本动作剩余位置立即切回 DFlash。
   DFlash 此时读到的不是猜测前缀，而是刚被 target 接受或纠正的真实前缀；这避免对已经失效的历史尾部重复
   做 target 校验，也把训练时 multi-anchor 覆盖真正用于在线修补。
5. **稳定段免校验。** 只有上一动作已连续至少 4 次被目标模型确认完全相同、当前 `t0` 仍相同、hidden
   cosine 不低于 0.998 时，才复用动作尾部并跳过本次 anchor、Draft、Action-RNN 和 target verify。
6. **强制刷新。** `max_consecutive=1`，任何一次免校验后下一步必须重新经过目标模型严格校验，不能让缓存
   错误沿控制轨迹连续传播。
7. **先 shadow 后 active。** `shadow` 完整执行目标校验，仅记录“假如跳过是否正确”。门槛必须在当前
   checkpoint、模型环境和 suite 上重新校准；训练集准确率不能替代这一步。

这形成一个由便宜到昂贵的级联：长期稳定段先由 VTPF 在必做 prefill 内完成严格历史校验；中等相似段在
prefill 后严格验证历史 proposal；其余位置回退到 DFlash/Action-RNN；approximate 分支才允许单步免校验。
它与 HeiSD 的共同点是利用时序冗余；区别是本项目不检索离线轨迹数据库，而用在线 target 已确认动作、
prefill 内嵌因果校验和拒绝后 KV 裁剪形成局部闭环。当前仍应把这种
差异写成实现与机制差异，而不能在缺少完整文献/消融前夸大原创性。

此外有两项严格不改变接受规则的固定成本优化：

- `action_only` 只计算目标 `lm_head` 的 256 个动作词表行；4,434 次 shadow argmax 检查中与全词表为零差异。
- anchor 已判错首 proposal 时，strict 前缀长度必为 0，因此直接提交目标 correction，不再运行无意义的
  proposal 后缀 target forward。DDTree 模式不使用此捷径，因为其它根分支仍可能命中。

阶段 profiler 给出的直接依据是：4090 上 target anchor 约 `17.9 ms`、target verify 约 `18.8 ms`，而
1 层 Draft+Action-RNN 约 `1.9 ms`。因此融合历史块的 target forward 比继续压缩 Draft 更能降低当前固定成本。

### 5.5 训练创新与推理创新怎样闭环

高维主导线性交接、跨 Anchor 和 Action-RNN 解决“模型能否产生有用候选”；时序级联解决“当前最值得验证的
候选究竟来自 Draft 还是最近一条目标模型已验证动作”；首 token 早拒绝和稳定段免校验则减少无效目标计算。
动作组与 DDTree 仍作为独立消融保留，但 2026-07-26 的完整结果表明它们没有成为当前最佳路径。任何 relaxed
或 verify-skip 结果都必须与 strict 路由分栏报告，不能用更长 Length 掩盖成功率或固定开销。

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

### 6.5 2026-07-25 双路径验证器事故与完整 DDTree 替换

旧双路径实现虽然构造了祖先 mask，却在校验后分别计算两条完整路径的接受长度，再选择“接受最长”的路径；
hidden/KV 也依赖这个胜出路径索引提交。这不是标准 DDTree 的目标驱动遍历。实际 epoch 200 校准中，多个固定
分叉位置相对 tree-off 改变了 7 维动作中的 2 维，严格安全检查将其全部淘汰，四路脚本因此退化为重复执行
tree-off。该现象不能解释为“目标模型答案不唯一”，根因是验证与提交语义没有以目标 token 的逐节点选择为
唯一真值。

第一轮修复只把验证改成目标驱动遍历，候选生成仍是固定位置单分叉，并用 4 个真实 observation 在线比较
`off/p2/p3/p4/p5` 的延迟。它虽然恢复了 strict 正确性，却把一个很小、很可能无收益的树外加校准开销带进
评测；校准选择 `off` 后，所谓树实验实际仍是线性验证。因此该版本不作为最终实现。

第二轮修复完整替换候选生成、验证和脚本：累计概率最佳优先扩树、固定节点预算、一次树校验、目标驱动遍历、
按接受节点压紧 KV；同时删除固定分叉参数、首块限定、在线校准和自动关树。默认预算与线性校验等成本。
它没有改 Draft、Action-RNN、checkpoint 或训练损失，epoch 180/200 权重可直接评测，无需重训。

完整替换后的验证证据：

| 检查 | 结果 |
| --- | --- |
| 最佳优先预算、动态祖先 mask、目标遍历/纠正、树与逐路径 target logits 一致性 | `8 passed` |
| DDTree 指标聚合 | `1 passed` |
| Action-RNN 与训练回归测试 | `15 passed` |
| epoch 200 真实 Goal smoke（每任务 1 次，仅查功能） | 最后 task：516 个树块、57 个备选分支挽救、额外接受 59 token、额外 target 节点为 0；SR/速度不作正式结论 |

正式结论仍必须来自完整 50-trial 四路评测。strict 使用 DDTree 目标遍历；action-group relaxed 因定义上允许
近似动作，继续使用整条路径的组误差预算，不能作为 lossless 结果报告。

### 6.6 Goal 完整基线、四路结果与研究转向

epoch 200 在 Goal 上的 50 trials/task 完整结果如下。Speedup 均使用同一套 paper wrapped AR
`0.182718 s/action` 作为分母：

| 方法 | SR | mean step | Length | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Paper wrapped AR | 0.742 | 0.182718s | - | 1.000x |
| SpecVLA strict | 0.768 | 0.178764s | 1.631 | 1.022x |
| SpecVLA relaxed | 0.734 | 0.141228s | 2.361 | 1.294x |
| DFlash 线性 strict | 0.776 | 0.175287s | 2.159 | 1.042x |
| DFlash + DDTree strict | 0.776 | 0.172794s | 2.221 | 1.057x |
| DFlash + action-group | 0.734 | 0.158771s | 2.530 | 1.151x |
| DFlash + DDTree + action-group | 0.720 | 0.156579s | 2.632 | 1.167x |
| **DFlash + VTPF strict（2026-07-27）** | **0.790** | **0.142036s** | **2.422** | **1.286x** |

前七行给出三个明确结论：当时的 Draft 在 strict 下比 SpecVLA 略快且 SR 不低，但优势很小；DDTree 在固定节点
预算下只带来约 1.4% 相对加速；action-group 与树能增加 Length，却没有超过 SpecVLA relaxed，并伴随 SR
下降。因此继续大规模重训 Draft 的边际收益不划算，推理研究转向“降低每块固定 target 成本”和“经 shadow
校准的选择性免校验”。最后一行是该研究转向形成 VTPF 后的正式结果：在不使用 relaxed 或免校验的前提下，
速度基本追平 SpecVLA relaxed，同时取得本表最高 SR。完整诊断见 6.8。

### 6.7 首 token 哨兵时序级联小实验

以下均为 2026-07-26 的小规模机理实验，不是论文主表。10-task 实验每个 task 仅 1 个 episode；计时仍只取
最后一个 task，以保持与正式脚本一致：

| 版本 | SR（10 episodes） | 最后 task mean | Length | blocks | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 原始 DFlash shadow 路径 | 0.60 | 0.165500s | 2.339 | 898 | 对照 |
| cosine 0.990 历史 proposal，严格校验 | 0.60 | 0.135403s | 2.861 | 734 | 历史 proposal 是有效候选源 |
| 拒绝后切回 DFlash | 0.60 | 0.135415s | 2.905 | 723 | 块数改善，单独计时收益很小 |
| 再加融合哨兵验证 | 0.60 | 0.127002s | 2.941 | 714 | 相对原始路径快 23.3% |
| 每步都做 VTPF（stable=1） | 0.60 | 0.133787s | 1.981 | 1060 | 弱历史候选挤掉 Draft，否决 |
| 哨兵路由 + VTPF（stable=2） | 0.80 | 0.155863s | 2.147 | 551 | 仅 29 次融合中 9 次全中，平均接受 3.138；触发过早，否决 |
| 哨兵路由 + VTPF（stable=3） | 0.60 | **0.113715s** | **3.079** | **682** | 相对原始路径快 31.3%，当前最佳 strict 小实验 |

同一首 task 的 profiler 显示，融合前 target anchor/verify 分别约 `17.9/18.8 ms`，融合历史块一次约
`18.8 ms`；正式无 profiler 均时从原始 DFlash `0.161748s` 降到 `0.123104s`。只要求 `t0` 相同、取消
cosine 筛选时，均时退化到 `0.132906s` 且 blocks 增至 860，故当前 0.990 阈值有实际消融依据。

VTPF 的 1-task profiler 进一步显示：普通 prefill 约 `55.3 ms`，在 prompt 后附 6 个历史 action token 的
融合 prefill 约 `54.9 ms`，没有可测的额外 target 成本。现有 1,805 条跨任务 shadow 中，历史动作在只确认
1 次时整条复现率为 28.2%，连续确认 2/3/4 次后升至 69.2%/88.0%/93.4%；`stable=2` 邻域消融在最后
计时 task 也只有 `9/29` 次完整融合命中，而 `stable=3` 达到 `111/129`，故默认三次确认来自 shadow 精度
跃迁和在线消融的共同证据，不是逐位置手调。VTPF 把 prefill 计作一个实际推进 action token 的验证块，所以其 Length 保持
不超过 7；最终论文仍应以同环境端到端 mean/speedup 和 SR 为主，不能只凭该 Length 宣称加速。

严格性边界也已显式检查：两次普通 DFlash 运行的 300 条 token trace 完全一致；普通 DFlash、未融合时序路由、
融合路由相对同调用内逐 token AR 参考分别有约 `18/20/30` 条动作差异。这些差异来自 BF16 在单 token 与块
forward 形状下的近临界 argmax 数值变化；每条实际输出仍来自当前 target 块 posterior 的接受或纠正。论文中
应称其为 target-verified strict，而不能宣称 bitwise 等同逐 token AR。另一方面，`action_only` 在 4,434 个
shadow logits 位置与全词表 argmax 为 `0` 差异，可以作为无损工程优化报告。

稳定段免校验仍是 approximate 分支。已有 1,805 条跨 10-task shadow 标签中，
`stable>=4 + t0相同 + cosine>=0.998` 选中 105 条、观测错误为 0，但样本集中在两个 task，零错误的 95%
Wilson 风险上界仍约 3.5%。因此它默认最多连续跳过一次，并必须与 strict 融合路由分栏报告；在更大 shadow
校准前不能用这 105 条样本宣称可靠性已经得到统计证明。

视觉 prefill-skip 也做过否定性 shadow：`stable>=2 + pixel relative-L2<=0.0006` 虽有 117/117 条观测正确，
却全部集中在单个 episode；放宽到可跨 3 个 episode 的门槛后出现 1-2% 错误。因此当前只保留诊断记录，
不允许视觉相似度直接跳过 target prefill。另两项工程 A/B 也被否决：只返回 5 层 hidden 与原生
`DynamicCache` 分别仅有噪声级和约 0.14% 收益，均未进入主线。

正式 `route/prefill` 还移除了只供 shadow 使用的多层 cosine、pooled hidden 和动作分布逐项 CPU 同步；同一
首 task 的轨迹与 `684` 个块完全不变，均时从 `0.112812s` 到 `0.112446s`。该差异只有约 0.3%，按工程
清理保留，但不作为论文加速贡献单独宣称。

### 6.8 2026-07-27 VTPF 正式 50-trial 结果

原始 `summary.json`、逐动作 timing 和 500-episode 文本日志已固化在
[`artifacts/eval/libero_goal/vtpf_strict_e200_20260726`](artifacts/eval/libero_goal/vtpf_strict_e200_20260726)，
并附运行 commit、命令和 SHA-256；后续不得覆盖该目录。

正式命令使用 `prefill` strict 模式、epoch 200、Goal 每任务 50 次，共 500 episodes。它采用精确 token
接受、`tree=off`、`verify_skip_mode=route`，日志确认 `verify_skipped_blocks=0`，不存在 action-group、树或
免校验带来的精度放宽。

| 指标 | 正式结果 |
| --- | ---: |
| 成功数 / 总数 | 395 / 500 |
| SR | **0.790** |
| mean / median step time | **0.142036s / 0.171017s** |
| AR-relative Speedup | **1.286x** |
| Length / avg accept length | **2.422 / 1.466** |
| blocks / action | **2.890** |

按上游口径，速度与生成统计取最后一个 task。其 10,322 个动作中，VTPF 触发 2,255 次，完整 token 匹配
1,927 次，触发后的平均接受长度为 6.273；另有历史 proposal 融合校验 393 次。实际 timing 中 2,195 个动作
低于 `80 ms`，占 21.3%，平均仅 `58.7 ms`；其余大部分动作仍处于 `149–172 ms`。因此 mean 的下降来自一批
真实消除了 Draft/后续 target verify 的快速动作，而不是计时噪声。相同 50 个 episode 的重采样给出 Speedup
95% 区间约 `[1.212, 1.373]`。

| 对照 | SR | mean step | Speedup | VTPF 相对结论 |
| --- | ---: | ---: | ---: | --- |
| Paper wrapped AR | 0.742 | 0.182718s | 1.000x | 延迟下降 22.3% |
| SpecVLA strict | 0.768 | 0.178764s | 1.022x | VTPF 相对快 1.259x |
| SpecVLA relaxed | 0.734 | 0.141228s | 1.294x | VTPF 仅慢 0.57%，但保持 strict |
| 旧 DFlash strict | 0.776 | 0.175287s | 1.042x | VTPF 相对快 1.234x |

结果也存在必须如实记录的边界。500 episodes 中，成功轨迹平均每条触发 VTPF 6.47 次、完整命中率 66.4%；
失败轨迹平均触发 92.2 次、完整命中率 89.6%，说明停滞重复动作显著放大总体收益。但最后 task 只看成功轨迹时，
VTPF 仍为 `0.155723s`，对应成功轨迹 AR 的 `0.183081s`，仍有 **1.176x** 加速；失败轨迹加速为 1.407x。
所以收益并非只存在于失败轨迹，只是整体 1.286x 被失败停滞段进一步放大。

SR 的 95% Wilson 区间约为 `[0.752, 0.823]`。相同初始状态的配对结果相对 SpecVLA strict 的 McNemar
`p=0.320`，不能宣称 VTPF 显著提高成功率；当前严谨表述应是“未观察到成功率退化，并取得最佳点估计”。
该结果验证了 VTPF 的结构性加速价值，但后续仍需多 seed 或不同 suite/权重验证其泛化。

### 6.9 已废弃的旧余弦课程早期快照

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

当前 Goal 权重不能用于 Object、Spatial 或 Long。当前推荐先做时序级联 shadow，再运行严格 VTPF 主线：

```bash
# 10-task 小规模 shadow：不跳过 target，只收集门控标签
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=1 MAX_EVAL_TASKS=10 \
TIMING_SCOPE=full_suite \
  bash openvla/specdecoding/decode-scripts/run_dflash_temporal_cascade_goal_eval.sh shadow

# 把上一条命令生成的 *_summary.json 传给校准器
python openvla/specdecoding/test-speed/analyze_dflash_temporal_shadow.py \
  /绝对路径/EVAL-...-dflash_strict_summary.json

# 50 trials/task 严格主实验：stable>=3 时并入 prefill，其余动作走哨兵路由；全程 target 校验
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_temporal_cascade_goal_eval.sh prefill

# 严格消融：关闭 VTPF，只保留 prefill 后的历史 proposal 融合校验
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_temporal_cascade_goal_eval.sh route

# approximate 消融：在严格路由上增加已校准的单步稳定段免校验
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_temporal_cascade_goal_eval.sh cascade
```

`prefill` 默认使用 `stable_actions>=3`，未进入 VTPF 的动作继续使用 `route`；`route` 默认使用
`cosine=0.990`、`stop_on_reject=True`、`fuse_verify=True`。两者的所有历史 proposal 均由 target 逐位置
判定，并在配置层使用独立的 `verify_skip_mode=route`，因此不会因门槛取值或浮点饱和误触发免校验。
`cascade` 才使用 `verify_skip_mode=active`，额外加入 `cosine=0.998`、`stable_actions>=4`、`max_consecutive=1` 的 approximate
免校验门，不能与 strict route 混成一个结果。

旧四路树/动作组机制消融一键评测：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 EVAL_EPOCH=200 \
DFLASH_OUTPUT_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2 \
  bash openvla/specdecoding/decode-scripts/run_dflash_action_rnn_goal_4way_eval.sh
```

四路依次为 `RNN+strict`、`RNN+DDTree+strict`、`RNN+动作组`、`RNN+DDTree+动作组`。默认
`DFLASH_TREE_BUDGET=0`，即每个块使用与线性验证相同的 `q-1` 个候选节点；树始终启用，不做在线校准，
也不会自动退化成 `off`。评测 epoch 180 时只需把上面的 `EVAL_EPOCH` 改为 180。

如果前面的组已经完成，可用 `START_GROUP` 从指定组继续。例如第1组完成后，从第2组续跑：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 EVAL_EPOCH=200 START_GROUP=2 \
  bash openvla/specdecoding/decode-scripts/run_dflash_action_rnn_goal_4way_eval.sh
```

脚本会复用前面已完成的结果，并在第四组结束后统一汇总四组结果。

单独执行：

```bash
EVAL_EPOCH=100 bash openvla/specdecoding/decode-scripts/run_dflash_goal_eval.sh strict
EVAL_EPOCH=100 bash openvla/specdecoding/decode-scripts/run_dflash_goal_eval.sh relaxed
```

推理机制消融必须固定同一 checkpoint、seed 和计时口径：

```bash
# 线性 strict：关闭候选树
DFLASH_TREE_MODE=off \
  EVAL_EPOCH=100 bash openvla/specdecoding/decode-scripts/run_dflash_goal_eval.sh strict

# 固定预算动态 DDTree strict；0 表示每块 q-1 节点，不额外增加 target 节点数
DFLASH_TREE_MODE=ddtree DFLASH_TREE_BUDGET=0 \
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
| `run_dflash_temporal_cascade_goal_eval.sh` | 当前主线：`shadow`、严格 `route`、严格 `prefill`、approximate `cascade` |
| `analyze_dflash_temporal_shadow.py` | 汇总时序门覆盖率、错误率和 Wilson 风险上界 |
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
