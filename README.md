# SpecVLA-DFLASH

本仓库研究一个具体问题：能否把 DFlash 式块并行草稿模型迁移到 OpenVLA，在保持目标模型校验可靠性的
前提下，提高 LIBERO 动作解码速度。

当前训练主线是独立的一层 Minimal Draft：保留完整目标上下文、multi-anchor、Hidden/Cos 和小权重 Soft KL，
不再默认依赖 Action-RNN。当前推理主线分为三档：strict 使用验证式时序 Prefill 融合（VTPF）；balanced
relaxed 使用单步目标锚定时序降采样（VTPF-TD-Fast）；high-speed relaxed 使用 VTPF-PacedHarmonic。
PacedHarmonic 以 `T-H-H,T-H` 节拍约束长期 target 调用密度，并把第二次 hold 的连续动作增量缩放为一半，
从而把“多久查询 target”和“陈旧动作执行多大”解耦。动作组宽松校验、DDTree、时序多候选 Prefill 树、
短前缀认证和单独 VisualBudget 均保留为消融。所有模块仍必须同时接受成功率、Length 和端到端 Speedup 检验。

代码基于 [SpecVLA](https://github.com/PineTreeWss/SpecVLA)，而 SpecVLA 又基于
[OpenVLA](https://github.com/openvla/openvla)。当前仓库仍是研究代码，不是已经定稿的公开复现包。
所有结论必须同时报告成功率、接受长度和速度；训练准确率不能替代在线 LIBERO 结果。

论文立论、理论命题、反例边界与后续证据实验的完整内部蓝图见
[`docs/paper_story_theory_blueprint_zh.txt`](docs/paper_story_theory_blueprint_zh.txt)。该文档用于约束后续写作，
其中“已有证据”和“待验证假设”必须始终分开，不能直接当作论文结论引用。

截至 2026-07-31，最重要的正式结果均来自同一台 RTX 4090、同一 seed 7、Goal 500 episodes 和同一
paper-wrapped AR 分母。Golden 行使用 e200，Minimal 行使用独立的一层 e100 checkpoint：

| 方法 | 校验性质 | SR | mean step | Speedup | Length |
| --- | --- | ---: | ---: | ---: | ---: |
| Paper-wrapped AR | 精确基线 | 0.742 | 0.182718s | 1.000x | - |
| SpecVLA strict | target-verified | 0.768 | 0.178764s | 1.022x | 1.631 |
| SpecVLA relaxed (`r=9`) | 近似接受 | 0.734 | 0.141228s | 1.294x | 2.361 |
| **Golden + VTPF strict** | **target-verified** | **0.790** | **0.142036s** | **1.286x** | **2.422** |
| **Golden + VTPF-TD-Fast** | **单步时序近似** | **0.754** | **0.070050s** | **2.608x** | **3.653** |
| **Minimal e100 + VTPF-PacedHarmonic** | **节拍谐波时序近似** | **0.746** | **0.052448s** | **3.484x** | **3.614** |

Minimal e100 `VTPF-PacedHarmonic` 已完成 500-episode 正式评测：SR `373/500=0.746`、mean
`0.052448s`、**3.484x**、Length `3.614`。相对相同 500 个初始状态上的 AR，成功率点估计为 `+0.4`
个百分点，精确 McNemar `p=0.934`；严谨结论是“本次未观察到成功率下降”，不是等价性证明。相对
VisualBudget `p=0.15`，成功率提高 `7.4` 个百分点且配对检验显著（`p=0.00215`），同时保留 3x 以上速度。
完整方法、留出筛选和原始证据见 6.15。

独立 Minimal Draft 的 epoch 100 已完成相同的 500-episode 在线评测。它用远少于 Golden 的训练组件，仍得到
`1.295x` 的 VTPF strict 和 `2.534x` 的 VTPF-TD-Fast；这证明当前主性能并不依赖 Action-RNN、跨 Anchor
蒸馏或 Domino 交接。完整数字和训练时长结论见 6.13。

PacedHarmonic 的原始日志、逐动作 timing、summary、配对统计、筛选/留出诊断、launcher 快照、checkpoint
配置、环境身份和 SHA-256 固化在
[`artifacts/eval/libero_goal/vtpf_paced_harmonic_e100_20260731`](artifacts/eval/libero_goal/vtpf_paced_harmonic_e100_20260731)。
此前 Golden、SpecVLA 和 AR 证据仍保存在
[`artifacts/eval/curated_20260720_20260728`](artifacts/eval/curated_20260720_20260728)；README 表格只是可读摘要。

当前有六条必须隔离理解的实验线：

- **Golden reference**：复杂版一层 Draft + Action-RNN + 跨 Anchor + Domino 交接，其 epoch 200 配合
  VTPF strict 得到目前已固化的最好正式结果。代码状态以 tag `golden-vtpf-e200-20260726` 为准，权重和旧
  launcher 不得覆盖。相同权重的 RNN-off 在线试验成功率近乎不变且更快，因此后续推理默认关闭 Action-RNN；
  这不抹去 golden，只把它作为可回退训练基线。
- **Minimal Draft 对照**：新建的独立训练配方，只保留一层块并行 Draft、完整目标上下文、multi-anchor、
  Hidden/Cos 和小权重 Soft KL。epoch 100 在线结果已基本复现 Golden 主性能，现作为后续各 LIBERO 子集的
  默认干净训练基线；Golden 继续作为不可覆盖的回退证据。
- **VTPF-TD relaxed**：只改推理，直接复用任意可用 DFlash checkpoint，不需要重训。保护档使用当前图像
  变化门控；速度档固定执行 `target -> hold -> target`。任何 hold 都不增加“已验证历史”，且下一步强制
  回到 target，避免误差连续传播。Goal 正式 500-episode 结果为 SR `0.754`、最后 task 口径 `2.608x`；
  相对 VTPF strict 的 SR 下降 3.6 个百分点、速度提高约 2.03 倍。
- **VTPF-TD-Adaptive 消融**：与以上正式结果完全隔离。第一跳保持 TD-Fast，第二跳必须同时满足“至少两个
  target 关键帧给出完全相同的 7-token 动作”和“当前图像相对最近 target 锚点的累计相对 L2 不超过阈值”；
  两次 hold 后强制 target。正式结果为 `2.599x / SR 0.746`：相对同一 Minimal TD-Fast 快约 2.6%，但少
  4/500 个成功；相对 Golden TD-Fast 没有净速度优势，因此只保留为风险自适应消融，不替换主方案。
- **VTPF-TD-VisualBudget 速度消融**：去掉导致覆盖率过低的“两个 target 动作必须逐 token 完全相同”前置条件，
  第一跳仍沿用 Fast；第二跳只由相对最近 target 锚点的累计低频视觉漂移预算控制，最多连续保持两次。
  `0.15` 正式结果为 `3.679x / SR 0.672`：跨过 3x，但成功率代价不可忽略。它与旧 Adaptive 分入口、
  分日志保存，当前保留为 aggressive speed 点而不是默认替代方案。
- **VTPF-PacedHarmonic high-speed 主线**：在 VisualBudget 的两次 hold 上增加长期节拍预算和第二次 hold 的
  谐波动作缩放，并把严格 VTPF prefill 候选稳定门槛降为 1。正式结果为 `3.484x / SR 0.746`；它没有
  复用视觉特征、没有动作组 relaxed 接受，也没有候选树，是当前同时越过 3x 与 AR 成功率点估计的主结果。

## 1. 阅读顺序和项目地图

建议按下面的顺序理解项目：

1. 先理解 OpenVLA 为什么把一个动作表示为 7 个离散 token。
2. 再理解 SpecVLA 如何用小草稿模型逐 token 提案，并让 OpenVLA 校验。
3. 然后理解本项目为什么改用 DFlash 式块并行 hidden 生成。
4. 最后沿着实验历程看清楚：完整上下文、multi-anchor、跨 anchor 蒸馏、Action-RNN、失败的树/动作组尝试，
   以及为什么研究重点最终转向整次 target prefill 的受控省略。

当前核心文件：

| 环节 | 文件 | 作用 |
| --- | --- | --- |
| 数据生成 | `openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py` | 用冻结 OpenVLA 生成动作 token 和多层 hidden 教师数据 |
| 数据入口 | `openvla/specdecoding/train-scripts/run_dflash_data_goal.sh` | 统一 smoke/full 数据生成命令 |
| 数据无损打包 | `openvla/specdecoding/train-scripts/pack_dflash_hdf5.py` | 把每样本 group 的 HDF5 v1 重排为连续 packed v2 |
| DFlash 训练 | `openvla/specdecoding/train-scripts/train_dflash_libero_goal.py` | multi-anchor、loss、DDP、checkpoint、SwanLab |
| 训练入口 | `openvla/specdecoding/train-scripts/run_dflash_train.sh` | 当前 Minimal Draft 与 Golden/历史结构消融 |
| Draft 模型 | `openvla/specdecoding/model/dflash.py` | 块并行主干、动作位置 embedding、Action-RNN |
| 在线推测解码 | `openvla/prismatic/extern/hf/modeling_speculation.py` | draft 提案、目标模型校验、partial accept/correction、树验证 |
| LIBERO strict | `openvla/experiments/robot/libero/run_libero_goal_Spec.py` | strict token 校验与在线指标 |
| LIBERO relaxed | `openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py` | relaxed 动作接受与在线指标 |
| 推理公共配置 | `openvla/specdecoding/decode-scripts/libero_eval_common.sh` | 两台机器路径、suite 权重、计时口径和 checkpoint 解析 |
| 时序级联入口 | `openvla/specdecoding/decode-scripts/run_dflash_temporal_cascade_goal_eval.sh` | shadow、严格路由、prefill 融合和稳定段免校验 |
| 时序 Prefill 树 | `openvla/specdecoding/decode-scripts/run_dflash_temporal_prefill_tree_goal_eval.sh` | golden e200 的 strict/动作组 relaxed 多候选 prefill 树 |
| VTPF-TD 速度档 | `openvla/specdecoding/decode-scripts/run_dflash_vtpf_temporal_decimation_goal_eval.sh` | target 与单步 hold 交替，跳过整次 prefill |
| VTPF-TD 保护档 | `openvla/specdecoding/decode-scripts/run_dflash_vtpf_guarded_bypass_goal_eval.sh` | 在单步 hold 前增加当前图像变化门 |
| VTPF-TD 自适应档 | `openvla/specdecoding/decode-scripts/run_dflash_vtpf_adaptive_decimation_goal_eval.sh` | 仅在双重证据成立时把一次 hold 扩为两次；独立实验入口 |
| VTPF-TD 视觉预算档 | `openvla/specdecoding/decode-scripts/run_dflash_vtpf_visual_budget_goal_eval.sh` | 累计视觉漂移预算决定第二次 hold；已完成的 aggressive 速度上界 |
| VTPF-TD 谐波保持消融 | `openvla/specdecoding/decode-scripts/run_dflash_vtpf_age_decayed_goal_eval.sh` | VisualBudget 第二次 hold 的连续动作缩放为 `1/2`；只用于拆分消融 |
| VTPF-TD 节拍预算消融 | `openvla/specdecoding/decode-scripts/run_dflash_vtpf_paced_budget_goal_eval.sh` | 第二次 hold 后偿还 target 债务，形成 `T-H-H,T-H` 节拍 |
| **VTPF-PacedHarmonic 主入口** | `openvla/specdecoding/decode-scripts/run_dflash_vtpf_paced_harmonic_goal_eval.sh` | `stable=1` 严格 prefill 候选 + 节拍预算 + 第二 hold 谐波缩放 |
| 自适应 hold 决策 | `openvla/specdecoding/model/temporal_hold.py` | 无 CUDA 依赖的固定/风险受限策略与硬上限，可独立单测 |
| 短前缀认证消融 | `openvla/specdecoding/decode-scripts/run_dflash_vtpf_prefix_cert_goal_eval.sh` | target 精确认证短前缀后信任尾部；当前净收益很小 |
| 时序门校准 | `openvla/specdecoding/test-speed/analyze_dflash_temporal_shadow.py` | 从 shadow summary 统计覆盖、错误和 95% 风险上界 |
| P0 成本/持久性/验证审计 | `openvla/specdecoding/decode-scripts/run_dflash_p0_evidence.sh` | 成对生成逐阶段耗时、时序冗余、fused-vs-serial 审计和 ICLR 规格图表 |
| P0 同状态恢复实验 | `openvla/specdecoding/decode-scripts/run_dflash_p0_counterfactual.sh` | 从同一 MuJoCo 状态分叉历史动作与控制对照，恢复冻结 target 后测量单侧伤害 |
| 论文证据构建器 | `openvla/specdecoding/evidence/` | 保存 SHA-256 原始证据、CSV、5.5 英寸矢量 PDF 和 300 dpi PNG |

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

### 2.4 方法组件与创新边界

仓库包含从“产生更好的草稿”到“减少目标调用”的完整研究链，但并非每个历史组件都在当前默认路径开启：

```text
当前图像 + 最近 target 已验证动作
                       ↓
           VTPF-TD relaxed 门通过？──是──→ 单步 hold，跳过整次 target
                       │                            ↓
                       ↓ 否                  下一步强制 target 回锚
              当前多模态 prompt
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
        frozen lm_head → 可选 Action-RNN 因果修正
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
| 目标锚定时序降采样（VTPF-TD） | 本项目 | target 关键帧与最多一次历史动作保持交替；保护档再加图像变化门，保持帧不积累已验证历史 | 当前 relaxed 主线，直接减少 target prefill 次数 |
| 风险受限自适应降采样（VTPF-TD-Adaptive） | 本项目 | 保留第一跳；只有 target 动作重复证据和相对最近 target 的累计视觉变化同时通过时才增加第二跳；随后强制 target | 与正式 TD-Fast 隔离的候选推理模块，不需要重训 |
| 节拍谐波时序策略（VTPF-PacedHarmonic） | 本项目 | 用 `T-H-H,T-H` 的长期 target 预算约束模型调用密度；第二 hold 只执行半幅连续增量；上一动作在每个 target prefill 中仍由 target 严格逐 token 校验 | 将“查询 target 的频率”和“陈旧控制量的幅度”解耦，不依赖任务标签或成功反馈 |
| 固定成本无损优化 | 本项目的实现诊断 | 目标 `lm_head` 仅投影 256 个动作 token；首 proposal 已被 anchor 判错时不再验证无效后缀；已知历史 proposal 时把 anchor 与整块校验融合为一次 target forward | 当前默认开启 |

因此，当前论文主线应表述为：**一层块并行 Draft 产生候选；strict VTPF 把可信历史 proposal 的校验并入
必做 prefill；relaxed VTPF-TD 用有界单步保持减少目标关键帧密度。** Action-RNN、跨 Anchor、Domino 交接和
DDTree 是已经实现并必须如实报告的历史/消融组件，不再因为结构复杂而自动算作主线贡献。最终新颖性仍须
结合 HeiSD、缓存类工作与完整消融审慎表述。

与 Domino 的关系需要准确表述：Golden 复杂版训练骨架与其核心思想高度一致，都是在同一次训练中令 Base 任务权重
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

### 3.6 阶段六：Golden Action-RNN 与 Base/Final 线性交接

Golden 复杂版保留一层并行 DFlash 主干，把旧的 hidden residual 和全词表 Markov bias 合并为一个很小的
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
说明两个目标不能靠突然切换来衔接。Golden 配方随后恢复为一次训练，并使用 Domino 式全程线性交接：

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
`stop_gradient`。Golden joint 只在修正前 base logits 上计算 KL，并只训练 Draft 主干；Action-RNN 不再进行
同模型 final logits 自蒸馏，而是只接受真实 token、teacher 分布和连续前缀的直接监督。这使跨 anchor 模块
职责更明确：不同 anchor 提供同一目标位置的多种因果视角，专门用于补强并行 Draft 的远端弱路径。

#### Prefix Survival

先用 teacher/student 动作分布的 total variation 得到每个位置的近似可接受概率，再沿块累乘，直接惩罚
连续前缀中断。早期错误会同时破坏后面的累计前缀，因此无需再设置 p2、p3 等手工 boost。

#### Golden 结构边界

DFlash transformer 仍只 forward 一次，但 Action-RNN 有最多 6 次很小的顺序状态更新。因此 Golden 复杂版是
“块并行重计算 + 轻量顺序 token 修正”，不是严格意义上 6 个最终 token 完全 O(1) 同时输出。

## 4. 当前数据、模型和损失

### 4.1 正式离线数据

当前 3090 Goal 正式数据：

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

简化版与 golden 使用**完全相同**的 packed v2，不需要重新生成数据。当前文件也没有 RNN、CAD 或跨 Anchor
专属标签；这些监督均由训练代码在线构造。各物理字段仍有不可替代的用途：

| packed v2 字段 | 简化版用途 | 能否删除 |
| --- | --- | --- |
| `prompt_selected` | 完整 prompt 的五层 teacher hidden，上下文主体 | 否；它约占文件绝大部分，但删除会退回已证伪的压缩 prompt 方案 |
| `prompt_position_ids` / `prompt_offsets` | 恢复变长 prompt 和 RoPE 位置 | 否 |
| `action_selected` | 为 anchor>0 提供已验证 action history 的五层 hidden | 否；multi-anchor 依赖它 |
| `action_last` | Hidden/Cos 标签，并通过冻结动作 `lm_head` 产生 Soft-KL teacher 分布 | 否 |
| `predicted_tokens` | 构造每个 anchor 的真实 token embedding 与目标序列 | 否；即使 hard CE=0 仍要使用 |
| `layer_ids` | 防止数据、训练和推理抽层错位 | 否 |

因此“简化模型”不等于“删 teacher 上下文”。在不牺牲 Draft 能力的前提下，可删的内容早已在 packed v2
阶段去除：`input_ids`、`loss_mask`、图像 tensor 和重复 `prompt_last` 均不进入训练文件。

历史 419 GiB、28,639 个小 `.ckpt` 的数据只用于解释旧实验。它会产生大量随机文件 IO，也与当前均匀选层
格式不一致，禁止继续作为新主实验输入。

同一个生成入口也支持 `libero_object`、`libero_spatial` 和 `libero_10`。`TASK_SUITE_NAME` 会同时决定对应的
OpenVLA 微调权重、`*_no_noops` RLDS、动作反归一化 `norm_stats` 键和输出文件名；生成器会对这四者做强校验，
不再允许用 Goal 的统计量静默生成其它子集。三个子集默认输出分别是：

```text
/data/wulin/c/specvla-data/dflash_object_dataset_packed_v2.h5
/data/wulin/c/specvla-data/dflash_spatial_dataset_packed_v2.h5
/data/wulin/c/specvla-data/dflash_10_dataset_packed_v2.h5
```

单文件并不自动等于顺序读。v1 内部仍有约二十万个 sample group/dataset；旧块采样又会让四个 rank 同时跳到
四个无关物理区域。packed v2 对训练实际读取的 BF16 hidden/token 做**逐 bit 无损重排**：完整 prompt 使用一块
连续数组加 offset，action hidden 与 token 使用固定形状连续数组；一个 local batch 只需一次 prompt hyperslab
读取。它不量化、不压缩、不改变样本、selected layer、position id 或监督标签。

当前 `DistributedBlockSampler` 以两卡相邻超级块为随机单位：默认每卡连续读取 64 个样本，两卡同一时刻处理
同一个 128 样本物理区域的相邻切片；每轮打乱超级块并轮换 rank 切片，完整覆盖全体样本，仅尾部按 DDP 规则
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

### 4.3 Golden reference 的单阶段高维主导训练

Golden reference 一次运行 200 epoch。令 `B` 为 Base logits、`F` 为 Final logits、`lambda` 从训练开始的 1 线性
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

### 4.4 当前独立 Minimal Draft 主线

Minimal 配方不是修改旧 joint 默认值，而是显式的 `minimal` 模式和独立输出目录。它是当前后续四个 LIBERO
子集统一使用的训练主线，并保留：

```text
一层并行 DFlash + 完整 prompt hidden + 已验证 action-history hidden
+ multi-anchor 全覆盖 + action-dimension embedding
+ 1.00 * Hidden SmoothL1 + 0.05 * Hidden cosine + 0.05 * teacher Soft KL
```

它移除：Action-RNN、hard CE、跨 Anchor KL、Domino Base/Final 交接、分布 L1、Prefix Survival、置信度头、
Hidden CAD、Markov residual 和手工位置 boost。`action_only` 不是新预测头：它只从冻结 OpenVLA `lm_head`
中切出 256 个动作词表权重行，既保留目标投影语义，又避免完整词表计算。

multi-anchor 不属于被移除的跨 Anchor 蒸馏。DFlash 推理一旦被 target 纠正，就会从新的 anchor 继续生成，
所以训练必须覆盖 `anchor=0...5`；当前长度只有 7，全量覆盖比随机抽一个 anchor 更稳定。跨 Anchor KL 则是
把强路径分布额外蒸馏给弱路径的损失，本次归零。这个区分也是简化消融成立的前提。

Minimal 不做低维自动标定或课程切换，总损失口径固定：

```text
L_minimal = 1.00 * SmoothL1(hidden_draft, hidden_teacher)
          + 0.05 * (1 - cosine(hidden_draft, hidden_teacher))
          + 0.05 * KL(teacher_action_distribution || draft_action_distribution)
```

正式协议固定为 100 epoch、两卡、每卡 batch 32、梯度累积 1、每 rank 一个 DataLoader worker；global batch
为 64，与 Goal 正式实验保持一致。学习率 `2e-5`、warmup 1000 step、slot decay、位置平衡、hidden
noise、packed sampler 和每 10 epoch checkpoint 间隔保持不变。新输出目录按子集隔离，例如 Goal 为
`ckpt_goal_dflash_minimal_1layer_hidden_soft_b32x1_2gpu_packedv2`，绝不复用 Golden 或旧四卡目录。

### 4.5 SwanLab 指标怎样读

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
| `anchor_logit_distill_loss` | 旧 Action-RNN 跨 anchor KL；Golden/Minimal 当前入口均关闭，仅保留兼容代码 |
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

#### 5.1.1 从普通 AR 到 VTPF：计算到底省在哪里

这一节只回答一个问题：一条 7-token 动作从 `t0` 到 `t6` 究竟经过哪些 forward，以及 DFlash、VTPF 和
VTPF-TD 分别省掉了什么。记当前图像与指令形成的多模态 prompt 为 `P`，Draft proposal 记为
`d1...d6`，目标模型在对应因果前缀下的 greedy token 记为 `y1...y6`。

普通 OpenVLA 自回归会重复调用目标模型：

```text
P                         -> target 得到 t0
P + t0                    -> target 得到 t1
P + t0 + t1               -> target 得到 t2
...
P + t0 + ... + t5         -> target 得到 t6
```

传统投机解码已经会让目标模型一次校验多个 token，但小 Draft 仍需自回归运行多次才能产生整块 proposal。
DFlash 改进的是 proposal 产生端：Draft 在一次 forward 中并行给出 `d1...d6`，目标模型仍进行标准的块
校验。当前普通 DFlash 的首块完整流程为：

```text
1. 多模态 prefill
   输入 P -> target 得到可信 t0，并建立 KV(P)

2. target anchor
   在 KV(P) 后只输入 t0 -> 得到 H(t0)、目标答案 y1，并建立 KV(P,t0)

3. DFlash 块并行生成
   输入完整 prompt hidden、H(t0) 与动作槽 -> 一次得到 d1...d6

4. target verify
   在 KV(P,t0) 后只输入 d1...d5 -> 一次得到 y2...y6
```

步骤 4 在数学上等价于让目标模型看到 `P+t0+d1+...+d5`，但工程上 `P+t0` 已在 KV Cache 中，不会重复
计算。因果语言模型在每个输入位置预测下一个 token，因此映射关系是：

| 已处理到的位置 | 该位置 logits 校验的 proposal |
| --- | --- |
| `t0` | `d1` 是否等于 `y1` |
| `d1` | `d2` 是否等于 `y2` |
| `d2` | `d3` 是否等于 `y3` |
| `d3` | `d4` 是否等于 `y4` |
| `d4` | `d5` 是否等于 `y5` |
| `d5` | `d6` 是否等于 `y6` |

例如 `d1,d2,d3` 正确而 `d4` 错误，系统只接受 `d1...d3`，把目标模型在 `d3` 后给出的 `y4` 写作
correction，并丢弃同一次 forward 中 `d4` 之后的 posterior。原因是后面的 posterior 已经以错误的 `d4`
为条件，不能代表目标模型在正确前缀上的自回归答案。下一小块以纠正后的 `y4` 为新 anchor，只补 `t5,t6`。
因此“正确”始终表示与当前目标模型的因果 greedy 路径一致，而不是另有一份离线环境标签。

Prefill 与 verify 虽然都运行 OpenVLA，却不是同等成本。Prefill 要处理当前图像、全部图像 token 和文本
prompt，并从零建立 KV；anchor/verify 复用已有 KV，只处理 1 个或数个动作 token。4090 阶段 profiler 的
数量级为：

| 阶段 | 输入与缓存状态 | 实测均值 |
| --- | --- | ---: |
| 普通多模态 prefill | 完整图像与 prompt，无 KV | 约 `55.3 ms` |
| VTPF 融合 prefill | 完整图像与 prompt，再附 6 个历史 token | 约 `54.9 ms` |
| target anchor | 1 个新动作 token，有 KV | 约 `17.9 ms` |
| 一层 Draft + Action-RNN | 一次块并行生成 | 约 `1.9 ms` |
| target verify | 约 5 个新动作 token，有 KV | 约 `18.8 ms` |

`17.9/18.8 ms` 是缓存后的子阶段 profiler，`55.3/54.9 ms` 是完整多模态 prefill，不能当成同一计时盒子
随意比较；但它们清楚说明两个事实：附加 6 个 token 相对长 prompt 的边际成本不可测，而 batch-1 短序列
target forward 的固定成本很高，1-token anchor 与 5-token verify 几乎同价。

VTPF 利用的观察是：同一条由 target 确认的 7-token 动作连续重复越久，下一控制步仍完全相同的概率越高。
若过去至少 3 个 target-verified 动作均为 `C=[c0...c6]`，三次重复只作为候选可靠性门，不会把三份动作
都送入模型。当前输入实际是：

```text
当前 P + c0 + c1 + c2 + c3 + c4 + c5
```

这一次必做的融合 prefill 同时给出 `y0...y6`：prompt 最后位置预测 `y0`，`c0` 位置预测 `y1`，依次到
`c5` 位置预测 `y6`。若 `C==[y0...y6]`，当前 target 已在当前图像上严格确认整条历史候选，系统直接执行
`C`；不再运行 target anchor、Draft 或后续 target verify。VTPF 因而不是“把 Draft 放进 prefill”，而是
用可信历史动作替代 Draft proposal，并把 target 校验融合进 prefill。普通 DFlash 的
`prefill + anchor + Draft + verify` 在完整命中时被压成一个 `fused prefill`。

若融合 prefill 在 `c4` 首次失败，系统保留已验证的 `c0...c3`，写入 target correction `y4`，裁掉错误
候选 KV，只让 DFlash 补 `t5,t6`；若 `c0` 就失败，则直接使用 prefill 给出的当前可信 `y0`，再从 `t0`
进入普通 DFlash。它不会丢掉本次 prefill 后从头重算。

历史尚未稳定到 3 次时，系统仍会在普通 prefill 得到绝对可信 `t0` 后检查：当前 `t0` 是否等于历史
`t0`，以及当前帧与上一帧“最后 prompt 位置的所选多层 target hidden”余弦相似度是否至少为 `0.99`。
余弦相似度

```text
cos(h_now, h_prev) = dot(h_now, h_prev) / (norm(h_now) * norm(h_prev))
```

衡量的是 OpenVLA 高层上下文表示的方向相似程度，不是 99% 正确概率。相同 `t0` 只能说明第一个动作维度
一致；高 hidden cosine 进一步筛掉场景或内部状态明显变化的情况。门通过后，历史 `c1...c6` 代替 Draft
proposal，但仍由当前 target 严格校验。实现把原本分开的 anchor 与 verify 合成一次缓存后 target forward，
因此这条路由从普通 DFlash 的 3 次 target 调用降为 2 次，并跳过 Draft；该 cosine 只决定 proposal 来源，
不决定是否接受。

VTPF-TD 再进一步减少跨控制步的 target 调用。Target 重算步正常读取当前观测并保存最终 target-verified
动作 `A_k`；紧随其后的单步 hold 不运行 prefill、Draft 或 verify，直接再次执行 `A_k`；再下一步强制
target 回锚。正式速度档形成 `target -> hold -> target -> hold`，所以接近跳过一半目标模型调用。由于 hold
没有询问当前图像对应的 target 答案，它属于 bounded one-step relaxed 推理，而不是 strict VTPF。

| 路径 | 一条动作中的 target forward | Draft forward | 当前 target 是否裁决 |
| --- | ---: | ---: | --- |
| 普通 DFlash 首块完整命中 | 通常至少 3 次：prefill、anchor、verify | 1 次 | 是 |
| 未满 3 次的严格历史路由 | 2 次：prefill、融合 anchor/verify | 0 次 | 是 |
| VTPF stable=3 完整命中 | 1 次：融合 prefill | 0 次 | 是 |
| VTPF-TD hold | 0 次 | 0 次 | 否，下一步强制回锚 |

所以 VTPF strict 的核心收益不是笼统的“少做一次校验”，而是：**把历史候选整条动作的严格校验嵌入本来
必做的多模态 prefill，在完整命中时同时消除 target anchor、Draft 和后续 target verify。**

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
2. **严格时序路由。** 未进入 VTPF 时，若当前 `t0` 与上一条已验证动作的 `t0` 相同，且最后 prompt 位置的
   所选多层 target hidden cosine 不低于路由阈值，则优先用上一动作的 `t1..t6` 作为 proposal。目标模型仍按原 strict 规则验证，因此候选错误时
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

### 5.5 VTPF 时序多候选 Prefill 树

单路径 VTPF 只尝试“保持上一动作”。新实验在不增加 target forward 次数的前提下，给同一次必做 prefill
加入最多三条完整动作候选：

1. `hold`：重复上一条实际执行动作。
2. `constant_velocity`：在 OpenVLA 归一化连续动作空间用最近两条动作做恒速度外推，再量化回动作 token；
   gripper 仍按离散类别保持，不做数值外推。
3. `recent`：回看上上条实际执行动作。

候选的 `t0...t5` 按公共 token 前缀编译为 trie。每个节点只看 prompt 和本分支祖先，同深度节点使用同一
RoPE 位置；目标模型在一次 multimodal prefill 中输出根节点和所有树节点的 posterior，从而同时校验每条
候选的 `t0...t6`。选择规则依次最大化接受前缀、精确前缀，再最小化已接受动作距离；若首个不匹配出现，
写入 target correction，只提交胜出分支的 KV，余下位置回退原 VTPF/DFlash 链。

strict 模式只接受 target 精确 token，因此不改变目标裁决；`relaxed` 模式沿用动作组阈值 `r=9`，属于近似
推理，必须单独报告 SR。树增加的是 prefill 序列中的少量节点，而不是额外 target 调用；是否净加速仍必须
看完整 50-trial timing，不能只看 Length。

独立 smoke（1 task、1 episode）已验证两种真实路径可用。relaxed 中 298 个动作进入树，平均 2.67 条候选、
11.82 个树节点，34 次选择非 hold 分支，较 hold 单路径额外接受 102 个 token，`length=2.945`；strict 中
124 个动作进入树，8 次选择备选分支并额外接受 13 个 token，`length=1.901`。后续完整评测没有得到足以抵消
树构造与更宽 prefill 开销的净加速，relaxed 的成功率代价也偏大，因此该分支只保留为消融。

### 5.6 目标锚定时序降采样（VTPF-TD）

短前缀认证实验暴露了固定成本：普通 target prefill 与附带少量 action token 的认证 prefill 都约为 `55 ms`；
只少校验几个尾部 token 并不能显著降低动作延迟。VTPF-TD 因此不再压缩同一次 prefill，而是受控地省掉整次
prefill。它建立在 VTPF 已有的 target-verified 历史和逐 episode 状态管理上，流程为：

1. **Target 关键帧。** 正常执行当前图像的 OpenVLA prefill、VTPF/DFlash proposal 和 target 校验，保存最终
   执行的 7-token 动作作为最近一条可信动作。
2. **单步保持帧。** 条件通过时，在进入目标模型前直接复用最近可信动作，省掉 target prefill、Draft 和
   后续 verify；统计项为 `temporal_prefill_bypassed_actions`。
3. **不伪造证据。** 保持帧不增加 target-verified run length，也不会把近似动作当作新的教师历史。
4. **强制回锚。** `max_consecutive=1`，任何一次保持后下一控制步必须重新经过 target，形成
   `target -> hold -> target`，把最坏陈旧度限制为一个环境步。
5. **episode 隔离。** reset 会清空动作、图像签名、verified run 和连续保持计数，不允许跨轨迹复用。

代码提供两个工作点：

- **保护档（VTPF-TD-Guard）**：对 processor 输出的当前/上一图像做 `16x16` 池化，以相对 L2 不高于 `0.03`
  作为额外门；仍最多保持一帧。
- **速度档（VTPF-TD-Fast）**：不读取图像门，稳定历史存在后固定交替 target 与单步保持，获得接近 50% 的
  target prefill 跳过率。它的假设是 LIBERO 20 Hz 控制下相邻观测的最优动作具有短时平滑性。
- **自适应档（VTPF-TD-Adaptive，待正式评测）**：第一帧 hold 与 Fast 完全一致；准备连续保持第二帧时，
  必须满足 `verified_action_run_length>=2`，即最近两个真实 target 关键帧输出完全相同的 7-token 动作，并且
  当前 `16x16` 图像签名相对**最近 target 关键帧**的累计相对 L2 不超过 `0.03`。通过后只多保持这一帧，
  下一帧无条件 target；任一证据缺失或超阈值都立即 target。它不是逐帧小变化的累加放行，因此缓慢漂移不能
  通过反复更新参考帧绕过风险门。

自适应档的计算约束同样是设计的一部分：每个动作只做一次很小的池化；只有申请第二跳时才计算一次锚点 L2
并取标量，不计算 target hidden、logits 或额外 Draft forward。旧 Fast 的最后 task timing 中，hold/target
均值约为 `0.000238s/0.139699s`，所以一次成功扩展省下的是完整 target 路径，而不是用一个接近 target 成本的
门控换取名义跳过。该策略的最坏连续陈旧度从一帧增至两帧，且被硬上限截断；它仍属于 relaxed 推理。

保持帧在推进意义上记录 `Length=7`，但 `verified_accepted_tokens=0`、`compared_tokens=0`，并单独记录
`prefill_bypass`；因此论文不能把这部分称为 target 接受长度，只能称为“执行推进长度”或“目标调用跳过率”。
该方法看到的是上一关键帧的 target 答案而非当前图像的 target 答案，所以明确属于 relaxed 推理，必须同时
报告 SR 与速度。它不修改 checkpoint，也不需要重训 Draft。

作为对照，VTPF-PrefixCert 用当前 target 精确认证历史动作前 `m` 个 token，再信任尾部。`m=4, history=3`
在同种子 10-episode 小样本中维持 SR，但只比 clean strict 快约 `0.65%`；原因正是认证 prefill 本身没有被
省掉。该机制保留作风险/固定成本消融，不作为推荐入口。

#### VTPF 与 VTPF-TD 的关系

两者共享“历史只能来自 target 已确认动作”和“episode reset 清空历史”两个安全不变量，但解决的是不同固定
成本：VTPF 把本来分开的 prefill 与第一块验证合成一次；VTPF-TD 则在允许近似控制时省掉整次 target 调用。

| 项目 | VTPF strict | VTPF-TD-Fast relaxed |
| --- | --- | --- |
| 当前图像是否进入 target | 每个动作都进入 | 关键帧进入；hold 帧不进入 |
| 历史动作来源 | 最近 target-verified 动作 | 最近 target-verified 关键帧动作 |
| 触发条件 | 同一动作已连续被 target 确认至少 3 次 | 已有可信历史，且上一步不是 hold |
| 一次触发做什么 | 把历史候选编入必做 prefill 并严格校验 | 直接执行一次历史动作，跳过 prefill/Draft/verify |
| 候选错误怎么办 | target partial accept/correction，裁掉错误 KV，余下回退 DFlash | 下一动作强制 target 回锚；不会连续 hold |
| 是否改变当前 target 答案 | 否，当前动作仍由 target posterior 裁决 | 是，hold 帧没有查询当前 target |
| 应报告的性质 | target-verified strict | bounded one-step approximate temporal decimation |
| 核心收益来源 | 减少一次动作内的 target forward 数 | 约一半动作省掉整次 target prefill |

VTPF 的单动作伪流程：

```text
读取当前图像和 prompt
  -> 若历史未稳定：普通 target anchor -> DFlash -> target verify
  -> 若历史稳定：prompt + 历史 c0..c5 一次 target prefill
       -> 连续精确命中：直接提交接受前缀
       -> 首次不命中：提交 target correction、裁 KV、从新 anchor 回退 DFlash
```

VTPF-TD-Fast 的跨动作伪流程：

```text
动作 k：完整 target/VTPF/DFlash 校验 -> 保存实际执行动作 A_k
动作 k+1：直接执行 A_k -> 不增加 verified run
动作 k+2：强制完整 target 回锚 -> 保存新的可信动作 A_(k+2)
```

因此 `Length` 在两种方法中不能机械地作同一语义解释。VTPF 的推进来自 target 校验；VTPF-TD 的 hold 会按
控制推进记为 7，但 `verified_accepted_tokens=0`。公平比较必须以端到端 latency、target bypass rate 和 SR
为主，Length 与 `avg_accept_length` 作为机制诊断分栏报告。

### 5.7 训练创新与推理创新怎样闭环

训练端负责让一层块并行 Draft 产生有用候选，strict VTPF 负责把可信历史候选并入必做 prefill；VTPF-TD
则在允许近似控制时直接减少 target 调用频率。Action-RNN、跨 Anchor、动作组与 DDTree 均保留为结构消融，
但当前在线证据没有证明它们能带来稳定净收益。任何 relaxed 或 verify-skip 结果都必须与 strict 路由分栏
报告，不能用更长 Length 掩盖成功率或固定开销。

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
| Golden 线性交接实验 | 200 epoch；Soft/CE 同步 Base->Final；低维组自动固定为初始约 10% | 已完成；rollout acc 0.853，Final 前缀代理 3.576 | 高维主线没有退化，但在线消融显示 Action-RNN 净收益很小；保留为可回退 Golden |

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

### 6.3 Golden Base/Final 线性交接实验

该 Golden 实验从随机初始化一次训练 200 epoch，不加载上述阶段一 checkpoint。实验目录和入口分别为：

```text
/data/wulin/c/specvla-data/ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2
bash openvla/specdecoding/train-scripts/run_dflash_train.sh joint
```

该配方只让一件事随训练进度改变：`lambda_base` 从 1 线性下降到 0，`lambda_final` 同步从 0 上升到 1。Soft KL
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

### 6.6 4090 完整基线、Goal 四路结果与研究转向

四个 suite 的 paper-wrapped AR、SpecVLA strict 与 SpecVLA relaxed 均在同一 4090、seed 7、50 trials/task、
`SYNC_CUDA_TIMING=False`、`TIMING_SCOPE=last_task` 下完成。每个单元格为 `SR / Length / Speedup`；AR 没有
speculative block，故不定义 Length。

| Suite | Paper-wrapped AR | SpecVLA strict | SpecVLA relaxed |
| --- | --- | --- | --- |
| Goal | 0.742 / - / 1.000x | 0.768 / 1.631 / 1.022x | 0.734 / 2.361 / 1.294x (`r=9`) |
| Object | 0.884 / - / 1.000x | 0.876 / 1.806 / 1.096x | 0.850 / 2.401 / 1.308x (`r=9`) |
| Spatial | 0.870 / - / 1.000x | 0.850 / 1.579 / 1.001x | 0.862 / 1.936 / 1.145x (`r=9`) |
| Long (`libero_10`) | 0.514 / - / 1.000x | 0.544 / 1.564 / 0.994x | 0.498 / 1.837 / 1.107x (`r=5`) |

这些是本仓库实际复现值，不是 2.2 节的论文原表。原始 36 份日志与统一 CSV 在
[`artifacts/eval/curated_20260720_20260728/baseline`](artifacts/eval/curated_20260720_20260728/baseline)。
当前 DFlash 只训练 Goal，因此其它三个 suite 只提供基线，不伪用 Goal Draft。

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
并附运行 commit、命令和 SHA-256；用户整理后的同一原始运行也保存在
[`artifacts/eval/curated_20260720_20260728/dflash_strict/复杂版Draft+VTPF`](artifacts/eval/curated_20260720_20260728/dflash_strict/复杂版Draft+VTPF)。
两个目录都不得覆盖。

正式复现身份如下：

| 项目 | 固定值 |
| --- | --- |
| Draft 训练数据 | `dflash_goal_dataset_envfix_20260714_packed_v2.h5`；28,501 samples；`layers=[1,9,16,24,31]` |
| Draft 训练配方 | `run_dflash_train.sh joint`；一层；block 7；global batch 64；200 epochs；完整参数见归档 `dflash_config.json` |
| checkpoint | `ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2/epoch_200_step_089600` |
| Draft SHA-256 | `e10127daa030ab5d7fbe639090078d3380c91a6d98b9302b31cf4d2f9dc5dac8` |
| 机制代码 | VTPF 引入 commit `d60c555`；冻结复现 tag `golden-vtpf-e200-20260726` (`ea7bcbb`) |
| run id | `EVAL-libero_goal-openvla-2026_07_26-19_45_10--dflash-temporal-prefill-fusion-goal-e200` |
| 评测协议 | RTX 4090；seed 7；50 trials/task；`sync=False`；`timing_scope=last_task` |

原始启动命令为：

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_temporal_cascade_goal_eval.sh prefill
```

正式命令使用 `prefill` strict 模式、epoch 200、Goal 每任务 50 次，共 500 episodes。它采用精确 token
接受、Action-RNN 开、`target_logits=action_only`、`tree=off`、`verify_skip_mode=route`，日志确认
`verify_skipped_blocks=0`，不存在 action-group、树或免校验带来的精度放宽。

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

### 6.9 2026-07-27 简化 Draft 与时序 Prefill 树分支

为避免后续消融暴露“复杂训练组件没有净在线贡献”却无法回退，本次先把上一版完整状态固定为 tag
`golden-vtpf-e200-20260726`。随后只新增两个相互独立的实验：

- 3090 的 `minimal` 训练：复用 28,501 条 packed v2，只保留 multi-anchor Hidden/Cos/Soft KL；输出目录、
  SwanLab run 和 checkpoint 与 golden 完全隔离。
- 4090 的 temporal-prefill-tree 推理：不改权重，在必做 prefill 中并行校验三类时序候选；旧
  `run_dflash_temporal_cascade_goal_eval.sh` 原样保留。

当前验证状态：两台机器上的 Action projection、trie、分支选择、KV 压紧、HDF5 和指标测试均为
`34 passed`；真实 strict/relaxed smoke 都已触发并采用备选树分支。这里只证明 workflow 连通，不能提前
声称简化 Draft 等价于 golden，也不能用 1 episode 宣称 Prefill 树提高成功率或速度。

同一 golden e200、同一 seed 的 100-episode RNN 消融中，RNN-on/off 的 SR 分别为 `0.78/0.77`，而 RNN-off
动作均值由约 `0.14379s` 降到 `0.13399s`，快约 `7.3%`。成功率差只有 1 个 episode，尚不足以证明统计差异；
但结合训练日志中 Action-RNN 净增益长期很小，当前最稳妥结论是：**Action-RNN 没有显示出可复现的在线
收益，默认推理关闭，3090 的 Minimal Draft 训练继续保持简化配方。**

### 6.10 2026-07-28 VTPF-TD relaxed 快速研发

本轮全部复用 golden e200，关闭 Action-RNN、DDTree 和动作组规则，不做任何重训。先用 1 trial/task 快速
筛选机制，再对候选点做 3 或 5 trials/task 确认，最后运行 50 trials/task。下表的 `mean` 是全 suite 动作
均值；`last-task mean` 才与本仓库论文口径一致。前六行小样本只用于选型，最后一行为正式 500-episode 结果。

| 配置 | 轨迹 | SR | full-suite mean | last-task mean | Length | bypass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean VTPF strict，RNN-off | 10 | 0.60 | 0.143931s | 0.112436s | 2.400 | 0 |
| PrefixCert `m=4, history=3` | 10 | 0.60 | 0.143008s | 0.115539s | 2.417 | 0 |
| Guard `pixel<=0.003, history=2` | 10 | 0.60 | 0.139577s | 0.103210s | 2.409 | 133 |
| TD-Fast 初始 pilot | 10 | 0.80 | 0.077940s | 0.082580s | 3.400 | 761 |
| **TD-Fast 确认** | **50** | **0.72** | **0.076206s** | **0.079781s** | **3.473** | **4,225** |
| **TD-Guard `pixel<=0.03, history=1`** | **30** | **0.80** | **0.112370s** | **0.117607s** | **2.786** | **1,311** |
| TD-Fast 去视觉门优化复测 | 10 | 0.80 | 0.077829s | 0.082550s | 3.400 | 761 |
| **TD-Fast 正式评测** | **500** | **0.754** | 未记录 | **0.070050s** | **3.653** | **5,575** |

关键结论：

- PrefixCert 的认证 prefill 与普通 prefill 成本接近，论文计时口径反而略慢，否决为主方案。
- TD-Guard 在相同 seed 的前 3 个初始状态上得到 `24/30`，golden strict 对应为 `23/30`，暂未观察到成功率
  下降；相对 paper AR 的最后 task mean `0.182718s`，诊断加速约 `1.55x`。
- TD-Fast 的 50 条轨迹结果为 `36/50=0.72`。paper AR 正式 SR 为 `0.742`，点估计下降 2.2 个百分点；按最后
  task mean 计算 Speedup 为 `0.182718/0.079781 = 2.29x`。与同一初始状态的 golden VTPF strict 前 5 次
  结果 `41/50=0.82` 相比则下降 10 个百分点，两种参照必须同时披露。
- 去掉视觉门的 GPU 池化和标量同步后，10 条轨迹的动作、SR、Length、bypass 数完全一致，速度差很小；说明
  TD-Fast 的收益来自真正跳过约一半 target prefill，而不是门控计算优化。
- 正式 500-episode 结果为 `377/500=0.754`，VTPF strict 为 `395/500=0.790`，paper AR 为
  `371/500=0.742`。VTPF-TD 相对 VTPF 少 18 个成功，绝对下降 3.6 个百分点；同初始状态配对中两者共同
  成功 319 条、VTPF 独赢 76 条、TD 独赢 58 条、共同失败 47 条，McNemar 精确检验 `p=0.142`。
- 正式最后 task mean 为 `0.070050s`，相对 paper AR 为 **2.608x**，相对 VTPF 的 `0.142036s` 快
  **2.028x**。即使只看成功轨迹，TD/VTPF 仍为 `0.079353s/0.155723s`，加速约 1.96 倍，因此收益不是
  失败轨迹停滞造成的假象。生成统计含 11,165 个动作状态，其中 5,575 次绕过 target prefill，跳过率
  49.93%；去除计时 warmup 后的 timing 样本数为 11,137，两个分母不能混用。

正式运行仍使用 6.8 的同一个 Golden e200 权重与目标模型；变化只发生在推理侧。实现 commit 为 `5626cb6`，
正式证据 commit 为 `208482a`，run id 为
`EVAL-libero_goal-openvla-2026_07_28-16_00_48--dflash-vtpf-temporal-decimation-goal-e200-h1`。完整命令：

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_temporal_decimation_goal_eval.sh
```

该 launcher 强制 Action-RNN 关、DDTree 关、动作组关、视觉门关、`max_consecutive=1`；非 hold 动作仍使用
`token/threshold=0` 的 target 校验。因此它与 VTPF 的主要实验变量是“是否在两个 target 关键帧之间插入
一个未经当前 target 查询的保持帧”，不是同时改变 Draft、树或接受阈值。

原始 pilot、Guard 和正式 500-episode summary、timing、文本日志固化在
[`artifacts/eval/libero_goal/vtpf_temporal_decimation_e200_20260728`](artifacts/eval/libero_goal/vtpf_temporal_decimation_e200_20260728)。
用户整理后的正式运行同时保存在
[`artifacts/eval/curated_20260720_20260728/dflash_relaxed/复杂版去掉RNN的Draft+VTPF-TD`](artifacts/eval/curated_20260720_20260728/dflash_relaxed/复杂版去掉RNN的Draft+VTPF-TD)。
当前结果已经达到“少量成功率代价、显著加速”的研发目标。后续需要多 seed、其它 suite 的独立 draft 和
真机实验验证泛化，不能把单个 Goal seed 直接外推为全场景结论。

### 6.11 2026-07-30 风险受限自适应降采样消融

固定 `T-H-T` 已把 target 比例压到约 50%，但所有稳定段和变化段都使用同一个保持预算。新增实验分支
`VTPF-TD-Adaptive` 只尝试解决这个调度问题，不改 Draft、checkpoint、VTPF 校验或动作接受规则：

```text
普通段：T -> H -> T
高证据稳定段：T(A) -> H(A) -> T(A) -> H(A) -> H(A) -> T
                                               ^ 第二次 H 才需要双重门
```

双重门由在线已有证据组成：`T(A)` 的两个最近 target 关键帧必须给出完全相同的 7-token 动作，且第二跳时的
当前图像相对最近 target 锚点的累计相对 L2 必须 `<=0.03`。hold 不会增加 verified run，参考图像也只在
真实 target 路径更新；两次连续 hold 后由硬上限强制 target。因此不存在用自身重复输出伪造稳定证据，或用
逐帧更新参考图像掩盖慢漂移的问题。

阈值不是按当前 50-trial 结果调出的：历史带图像诊断的 exact-action 样本中，单步 pixel relative L2 的
中位数约 `0.0048`、p90 约 `0.0331`，所以预注册 `0.03` 作为保守工作点。正式 TD-Fast 最后 task 中，hold
和 target 路径均值约为 `0.000238s/0.139699s`；自适应门只在第二跳候选上增加一次小池化后的 L2 标量读取，
成功扩展时省掉完整 target 路径。另一方面，历史相邻 target 完整动作 exact 比例只有约 `14.1%`，因此本分支
预期是有限但真实的提速，而不是对 `3x` 作无依据承诺。

输出 summary 新增 `generation.temporal_hold`：必须同时查看 `hold_rate`、
`adaptive_extended_holds/adaptive_extension_candidates`、`target_prefill_actions`、拒绝原因直方图、SR 和 timing。
Minimal e100 的完整 500-episode 结果为 SR `0.746`、最后 task mean `0.070320s`、Speedup `2.599x`、Length
`3.544`。共做出 10,255 次时序决策，其中 5,505 次 hold；4,704 次第二跳候选中只有 779 次通过双重门，
扩展率 `16.6%`。它相对同一 Minimal e100 TD-Fast 的 `0.072119s` 快约 `2.56%`，但少 4/500 个成功；同时
仍略慢于 Golden TD-Fast 的 `0.070050s`。因此该机制证明风险门确实能选择性扩展 hold，却没有形成足以替代
固定 TD-Fast 的净收益，论文中应作为调度消融而非主方法。

它没有达到 `3x` 的原因不是 4090 算力不足，而是 target 路径仍然太密：Adaptive 的 target-prefill 占比为
`4750/10255=46.3%`，只比 TD-Fast 约 50% 的占比降低 3.7 个百分点。以当前 AR 分母计，`3x` 要求最后 task
mean 不高于 `0.060906s`；在 hold 近乎免费但门控/环境开销仍存在的实测条件下，target 占比需要进一步降到
约 38%-40%。当前第二跳仅 `779/4704=16.6%` 通过，主要拒绝原因是 3,842 次
`insufficient_verified_run`，所以它理论上只能提供小幅增益。后续任何 `>3x` 候选都必须通过 paired 小实验
同时证明 target 占比、实际 timing 和 SR，而不能只继续放宽阈值后按 Length 推测。

### 6.12 已废弃的旧余弦课程早期快照

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
切换时产生表示冲突。后续 Golden 版本改为只有 Base/Final 路径比例线性变化，Soft 与 CE 同步交接，低维组只在启动前
标定一次固定总尺度。旧余弦和两阶段输出目录均不得续训到新配方。

### 6.13 2026-07-30 Minimal e100 在线结果与训练时长结论

Minimal Draft 的 epoch 100 来自 200-epoch 线性学习率计划的中点，同一 checkpoint、seed 7、RTX 4090 和
paper-wrapped AR 分母完成了四组 Goal 500-episode 评测：

| Minimal e100 推理配置 | 校验性质 | SR | mean step | Speedup | Length | avg accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 线性 DFlash strict | target-verified | **0.792** | 0.160411s | 1.139x | 2.172 | 1.111 |
| VTPF strict | target-verified | **0.776** | 0.141044s | **1.295x** | 2.432 | 1.457 |
| VTPF-TD-Fast | 单步时序近似 | **0.754** | 0.072119s | **2.534x** | 3.573 | 1.051 |
| VTPF-TD-Adaptive | 最多双 hold 近似 | **0.746** | 0.070320s | **2.599x** | 3.544 | 0.906 |

原始日志与配置快照固化在
[`artifacts/eval/libero_goal/minimal_e100_20260729_30`](artifacts/eval/libero_goal/minimal_e100_20260729_30)。

训练日志也显示明显的边际收益递减：epoch 100 到 200，Hidden loss 仅由 `0.79599` 降到 `0.79462`，Cosine
由 `0.45355` 降到 `0.45187`，self-rollout accuracy 由 `0.9405` 升到 `0.9506`。在线结果进一步证明 e100
已经保留主性能：VTPF strict 与 Golden e200 的 `1.286x` 相当，TD-Fast 比 Golden 的 `2.608x` 只低约
2.8%，SR 则同为 `0.754`。

因此后续四个 LIBERO 子集统一采用 **固定 100 epoch**，保存 e60/e80/e100 做在线筛选；不按离线 loss 或
teacher-forced accuracy 自动早停，因为它们不能可靠预测在线 p2-p6、SR 或端到端速度。Goal 的现有 e100
来自原 200-epoch scheduler 的中点，本项目将它作为本轮 Goal 的正式 100-epoch checkpoint，**不为统一目录名
或 scheduler 形式重复消耗一次训练**；论文和 artifact 中继续如实保留它的历史来源。后续 object、spatial、10
直接使用新的 100-epoch 完整退火协议。两者并非逐 step 完全相同，但当前决策优先避免没有在线收益证据的重复训练。

### 6.14 2026-07-30 VTPF-TD-VisualBudget：3x 成立，但 `0.15` 过于激进

6.11 的旧 Adaptive 被“最近两个 target 必须给出逐 token 完全相同动作”卡住，第二跳覆盖率只有 `16.6%`。
VisualBudget 保留更重要的安全结构，但把这个低覆盖条件移除：第一跳与 TD-Fast 相同；准备第二跳时，计算当前
processor 图像相对最近一次真实 target 图像的 `16x16` 低频特征相对 L2。漂移不超过预算才复用上一条
target-verified 动作；hold 从不增加 verified run，target 图像锚点也不会被 hold 更新；连续两次 hold 后
无条件强制 target。它不改变 Draft、权重、token 校验或动作接受规则，也不需要重训。

为避免再次用失败长轨迹制造“假 3x”，筛选时同时检查总 SR、最后 task 汇总 timing 和每条轨迹 timing：

| 视觉预算 | 初始状态 | SR | last-task mean | Speedup | target 比例 | 第二跳覆盖 | 结论 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.06 | 每任务 0-2 | 24/30 | 0.054955s | 3.325x | 40.0% | 50.2% | 汇总过 3x，但两条成功轨迹仅 2.57x/2.38x，否决 |
| 0.10 | 每任务 0-2 | 25/30 | 0.063563s | 2.875x | 38.1% | 62.7% | SR 最好，但未稳定过 3x |
| **0.15** | **每任务 0-2** | **23/30** | **0.056270s** | **3.247x** | **34.7%** | **89.3%** | 三条逐轨迹均超过 3x |
| **0.15** | **每任务 3-5** | **22/30** | **0.053744s** | **3.400x** | **34.2%** | **93.8%** | 错开初始状态复核；三条逐轨迹均超过 3x |
| **0.15** | **正式 0-49** | **336/500** | **0.049670s** | **3.679x** | **34.3%** | **92.3%** | 速度成立，但 SR 显著退化 |

两个 `0.15` pilot 合并为 `45/60=0.750`、`3.331x`，但小样本没有暴露正式结果的 SR 退化。正式 500 条中，
VisualBudget 为 `336/500`，Minimal TD-Fast 为 `377/500`；配对结果是共同成功 280、VisualBudget 独赢 56、
TD-Fast 独赢 97、共同失败 67，McNemar 精确检验 `p=0.00115`。7/10 个任务下降，所以不能解释为随机波动。

| Goal task id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VisualBudget 成功数 | 21 | 39 | 41 | 11 | 47 | 36 | 29 | 46 | 37 | 29 |
| TD-Fast 成功数 | 21 | 47 | 39 | 20 | 49 | 40 | 36 | 48 | 42 | 35 |
| 差值 | 0 | -8 | +2 | -9 | -2 | -4 | -7 | -2 | -5 | -6 |

正式最后 task 汇总为 `0.049670s / 3.679x`；29 条成功轨迹单独为 `0.057676s / 3.168x`，21 条失败轨迹
为 `0.044138s / 4.140x`。失败长轨迹确实放大了总 speedup，但成功轨迹自身也稳定超过 3x。正确结论是：
**把 target 比例压到约 34% 足以跨过 3x，但纯视觉预算不能可靠约束陈旧动作风险。** `0.15` 应作为速度上界
和 aggressive Pareto 点，不能写成无精度退化的默认方案。

正式结果后又检查了更保守的全局阈值。`0.10` 的两批 pilot 合并为 `46/60`、表面 `3.251x`，但第一批最后
task 三条全成功却只有 `2.875x`，第二批三条全失败才达到 `3.475x`，仍受失败轨迹构成偏置影响。`0.12`
在同一组 0-5 初始状态上跑到前 30 条时只有 `22/30`，低于 `0.15` 对应的 `25/30`，因此提前终止。闭环控制
会因为一次 hold 决策改变后续整条状态轨迹，SR 不随视觉阈值单调变化；继续微调一个全局 L2 数字既不严谨，
也不能根治风险。当前应保留 TD-Fast 作为 balanced 主档，把 `0.15` 作为 aggressive speed 档；下一步若要
同时保持 3x 与 SR，必须引入比全局图像漂移更具任务语义的风险证据，而不是继续扫阈值。

LIBERO 的 `SEED` 不改变 `initial_states[episode_idx]`。为做不重复的小试验，strict/relaxed 评测入口新增
`trial_start_index`，默认 0；`TRIAL_START_INDEX=3, NUM_TRIALS_PER_TASK=3` 才真正选择每任务第 4-6 个初始
状态。该参数只改变评测样本范围，不改变模型或动作逻辑。两批原始文本、逐动作 timing 和完整 summary 固化在
[`artifacts/eval/libero_goal/vtpf_visual_budget_e100_20260730`](artifacts/eval/libero_goal/vtpf_visual_budget_e100_20260730)。

### 6.15 2026-07-31 VTPF-PacedHarmonic：把速度预算与控制误差解耦

VisualBudget 的正式结果说明，约 34% 的 target 比例足以超过 3x，但连续重复完整的旧动作会放大闭环过冲。
本轮没有继续扫描任务相关阈值，而是把“多久询问一次 target”和“陈旧动作执行多大”拆成两个独立约束：

1. **节拍预算（paced target budget）**：第二次 hold 使用后形成一笔 temporal debt；下一 target 周期只允许
   一次 hold，得到确定的 `T-H-H, T-H` 节拍。它把长期 target 比例约束在 40% 左右，不依赖任务 id、阶段
   标签或成功轨迹反馈。
2. **谐波动作保持（harmonic action hold）**：第一次 hold 原样执行；第二次 hold 只把 6 个连续控制维度乘以
   `1 / hold_depth = 0.5`，夹爪维度保持离散值不变，随后强制回到 target。它不改变 token、Draft proposal
   或验证结论，只限制陈旧位姿增量的累计幅度。
3. **常开严格 prefill 候选（`stable_actions=1`）**：每个 target 帧都允许上一条动作进入 VTPF prefill；候选
   token 仍由 target 逐位置判断，错误前缀在首个不匹配位置得到 correction。相比旧 `stable_actions=3`，它
   不增加免校验 hold，却显著增加单次 target prefill 直接推进动作的机会。

这不是按任务阶段编写的状态机。若两次 target 之间最多允许 1 或 2 次 hold，而长期平均预算为 1.5 次，交替
安排 2、1 次 hold 是最大间隔最小的均匀节拍；它对应约 40% 的 target 调用率。OpenVLA 前六维是连续增量控制，
同一陈旧增量重复第 `d` 次时按 `1/d` 执行，可降低零阶保持造成的线性累计过冲；夹爪是离散状态，因而不缩放。
`0.15` 仍只是此前 VisualBudget 已注册的第二 hold 视觉扩展上限，不能解释成形式化安全证书；本轮没有针对
task、物体或成功轨迹新增阈值。

筛选严格区分反复观察过的初始状态 0-4 与留出状态 5-9；只有后者用于决定是否启动正式评测：

| 候选 | 状态范围 | SR | last-task mean | Speedup | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 仅节拍预算，`stable=3`、无谐波保持 | 正式 0-49 | 353/500 | 0.057205s | 3.194x | 达到速度目标，但 SR 只有 0.706 |
| VisualBudget + 谐波保持 | 0-4 | 41/50 | 0.057596s | 3.172x | 设计集很好，但不能据此定稿 |
| VisualBudget + 谐波保持 | 5-9 | 33/50 | 0.054226s | 3.370x | 留出集未改善，单独使用被否决 |
| 节拍预算 + 谐波保持，`stable=3` | 5-9 | 38/50 | 0.062375s | 2.929x | SR 达标但没有稳定超过 3x |
| **VTPF-PacedHarmonic，`stable=1`** | **5-9** | **39/50** | **0.052489s** | **3.481x** | **唯一同时通过留出 SR 与 3x 门槛的候选** |

同轮还排除了两条更复杂的路线：复用视觉表示的 50 条 pilot 为 `SR 0.600 / 3.198x`，目标动作变化证书为
`SR 0.700 / 2.633x`。前者直接损伤视觉条件，后者没有速度收益；两者实现均已从正式工作树删除，只在实验
日志中保留否定证据，避免失败原型继续污染模型接口。

正确性影子诊断额外串行生成纯 target AR 参考链，不计入正式速度。在相同初始状态的单 episode 诊断中，原 `stable=3` DFlash
target 路径有 `20/124=16.1%` 的整动作 token 链与 AR 不同；`stable=1` 为 `17/125=13.6%`。该单样本只
排除了明显回归，不能充当等价性证明。这也限定了论文表述：VTPF 候选接受本身是逐 token target 校验，但当前 DFlash 整体实现不能被
宣称为与纯 AR 逐动作位完全等价。正式评测必须关闭 `DFLASH_DEBUG_COMPARE_TARGET_AR` 和
`DFLASH_PROFILE_STAGES`，否则额外 AR 链与 CUDA 同步会污染计时。

实现身份为 commit `82cdffe`，独立复现入口为 `run_dflash_vtpf_paced_harmonic_goal_eval.sh`。e100、seed 7、
Goal `50 trials/task` 的正式 500-episode 结果如下：

| 指标 | 正式结果 |
| --- | ---: |
| 成功率 | **373/500 = 0.746** |
| 各 task 成功数 | `[32, 44, 43, 23, 47, 38, 29, 48, 41, 28]` |
| last-task mean step | **0.0524479s** |
| 相对 paper-wrapped AR 的 Speedup | **3.4838x** |
| Length / Table-1 Length | 3.6144 |
| avg_accept_length | 0.9523 |
| target prefill / hold 比例 | 0.4046 / 0.5954 |
| 第一 / 第二 hold 次数 | 4707 / 2243 |
| VTPF prefill fused / full-match | 4672 / 1036 |

相同 500 个初始状态上，PacedHarmonic 相对 AR 的成功率差为 `+0.4` 个百分点；前者独赢 73 条、后者独赢
71 条，精确 McNemar `p=0.9336`，配对 bootstrap 95% CI `[-4.4,+5.2]` 个百分点。这个结果支持“未观察
到成功率下降”，但区间仍包含中等幅度变化，不能声称已经证明等价。相对 VisualBudget `p=0.15`，成功率
提高 `7.4` 个百分点，配对 95% CI `[+2.8,+12.0]`，McNemar `p=0.00215`；说明谐波缩放并非只换来
偶然的总体均值，而是显著修复了此前的闭环精度损失。

完整原始证据、统计脚本输出、环境和 launcher 快照固化在
[`artifacts/eval/libero_goal/vtpf_paced_harmonic_e100_20260731`](artifacts/eval/libero_goal/vtpf_paced_harmonic_e100_20260731)。
当前 commit 的额外 1-episode smoke 已 `exit=0`；退出时仍会出现 robosuite 的已知 EGL 析构警告，但 summary
在此前已经完整写盘。

## 7. 当前标准工作流

固定分工：

```text
3090：生成离线数据；Minimal 默认两卡训练
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

若本机尚无其它三个子集，优先通过镜像下载。模型与数据可以并行下载；日志中的 `EXIT_CODE=0` 才表示完整：

```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download openvla/openvla-7b-finetuned-libero-object \
  --local-dir /data/wulin/hf_files/openvla-7b-finetuned-libero-object --max-workers 2
huggingface-cli download openvla/openvla-7b-finetuned-libero-spatial \
  --local-dir /data/wulin/hf_files/openvla-7b-finetuned-libero-spatial --max-workers 2
huggingface-cli download openvla/openvla-7b-finetuned-libero-10 \
  --local-dir /data/wulin/hf_files/openvla-7b-finetuned-libero-10 --max-workers 2

huggingface-cli download openvla/modified_libero_rlds --repo-type dataset \
  --include 'libero_object_no_noops/**' 'libero_spatial_no_noops/**' 'libero_10_no_noops/**' \
  --local-dir /data/wulin/c/datasets/modified_libero_rlds --max-workers 4
```

其它子集先逐一 smoke；下面三条可分别使用 GPU 4、5、7：

```bash
TASK_SUITE_NAME=libero_object GPU_ID=4 \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh smoke
TASK_SUITE_NAME=libero_spatial GPU_ID=5 \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh smoke
TASK_SUITE_NAME=libero_10 GPU_ID=7 \
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

默认 `KEEP_RAW=False`：脚本只会在 packed 文件完整关闭并标记成功后删除中间 v1，最终留下一个训练文件；
若要做逐 bit 格式审计，可显式设 `KEEP_RAW=True`。打包失败时 raw 会保留，`.partial` 不会被训练入口接受。

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

三个子集正式生成时可以三卡并行。每个进程独立生成 raw v1；脚本默认通过同一把 `flock` 锁串行执行最终
packed v2 大文件重排，防止三个高吞吐打包任务同时打满共享磁盘。`KEEP_RAW=False` 会在各自打包成功后删除
中间文件：

```bash
TASK_SUITE_NAME=libero_object GPU_ID=4 KEEP_RAW=False \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh full
TASK_SUITE_NAME=libero_spatial GPU_ID=5 KEEP_RAW=False \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh full
TASK_SUITE_NAME=libero_10 GPU_ID=7 KEEP_RAW=False \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh full
```

### 7.2 3090 两卡、100-epoch Minimal 训练

当前四个子集统一使用 `minimal`。默认每卡 batch 32、梯度累积 1、global batch 64、`NUM_WORKERS=1`、
100 epoch；只需选择两张可用卡和子集：

```bash
TASK_SUITE_NAME=libero_goal CUDA_VISIBLE_DEVICES=0,1 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh minimal

TASK_SUITE_NAME=libero_object CUDA_VISIBLE_DEVICES=0,1 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh minimal

TASK_SUITE_NAME=libero_spatial CUDA_VISIBLE_DEVICES=0,1 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh minimal

TASK_SUITE_NAME=libero_10 CUDA_VISIBLE_DEVICES=0,1 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh minimal
```

`CUDA_VISIBLE_DEVICES` 可以换成任意两张空闲卡；脚本会检查卡数必须与默认 `NPROC_PER_NODE=2` 一致。
两卡 `32x1` 表示每个 DDP rank 每次读取 32 条、每个 micro-batch 立即更新，所以 global batch 为
`32*1*2=64`。每个 rank 使用一个 worker，因此系统中共有两个只读 worker；`prefetch_factor=1`
和低优先级 IO 继续保护共享磁盘。

四个默认数据/输出会按 `TASK_SUITE_NAME` 自动绑定：

| suite | packed v2 | 默认输出目录 |
| --- | --- | --- |
| `libero_goal` | `dflash_goal_dataset_envfix_20260714_packed_v2.h5` | `ckpt_goal_dflash_minimal_1layer_hidden_soft_b32x1_2gpu_packedv2` |
| `libero_object` | `dflash_object_dataset_packed_v2.h5` | `ckpt_object_dflash_minimal_1layer_hidden_soft_b32x1_2gpu_packedv2` |
| `libero_spatial` | `dflash_spatial_dataset_packed_v2.h5` | `ckpt_spatial_dflash_minimal_1layer_hidden_soft_b32x1_2gpu_packedv2` |
| `libero_10` | `dflash_10_dataset_packed_v2.h5` | `ckpt_10_dflash_minimal_1layer_hidden_soft_b32x1_2gpu_packedv2` |

Goal 已有权重已整理到 `Draft_checkpoint/goal/epoch_100_step_044800`，继续作为正式 e100，不需要按新目录重训。其它子集的
新 HDF5 会携带 `task_suite_name`；训练入口同时检查数据元数据与 OpenVLA `norm_stats`，错配会在训练前报错。

`minimal` 在 Python 参数层还有显式配方校验：一旦误开 RNN、跨 Anchor、hard CE、L1、Prefix、CAD 或任一
课程，会在加载训练前报错，避免一个目录名叫“minimal”而实际混入旧模块。

已固化的复杂版 Golden 仍可显式复现，但不是新子集默认训练协议：

```bash
TASK_SUITE_NAME=libero_goal CUDA_VISIBLE_DEVICES=0,1 NUM_EPOCHS=200 \
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

当前 Goal 权重不能用于 Object、Spatial 或 Long。strict 先运行 VTPF；relaxed 再从保护档和速度档中选择：

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

Golden epoch 200 的 VTPF 时序多候选 Prefill 树使用独立入口。先做 1-episode smoke，再做正式 relaxed：

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=1 MAX_EVAL_TASKS=1 \
  bash openvla/specdecoding/decode-scripts/run_dflash_temporal_prefill_tree_goal_eval.sh relaxed

CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_temporal_prefill_tree_goal_eval.sh relaxed
```

把最后一个参数改为 `strict` 即执行精确 token 校验。该脚本默认仍解析 golden 目录
`ckpt_goal_dflash_joint_domino_1layer_b16x1_4gpu_packedv2`；不会误读 minimal 输出。若以后要把同一推理机制用于
简化权重，必须显式覆盖 `DFLASH_OUTPUT_DIR`，并先做 strict smoke。

`prefill` 默认使用 `stable_actions>=3`，未进入 VTPF 的动作继续使用 `route`；`route` 默认使用
`cosine=0.990`、`stop_on_reject=True`、`fuse_verify=True`。两者的所有历史 proposal 均由 target 逐位置
判定，并在配置层使用独立的 `verify_skip_mode=route`，因此不会因门槛取值或浮点饱和误触发免校验。
`cascade` 才使用 `verify_skip_mode=active`，额外加入 `cosine=0.998`、`stable_actions>=4`、`max_consecutive=1` 的 approximate
免校验门，不能与 strict route 混成一个结果。

当前推荐的 VTPF-TD relaxed 两档均默认关闭 Action-RNN、DDTree 和动作组接受，不修改 checkpoint：

```bash
# 保护档：当前图像相对 L2 <= 0.03 才保持一帧；验证点为 24/30、约 1.55x
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_guarded_bypass_goal_eval.sh

# 速度档：target -> hold -> target；正式 Goal 结果为 377/500、2.608x
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_temporal_decimation_goal_eval.sh
```

两者都必须在同机 paper AR 下计算 Speedup。速度档已完成 50 trials/task；保护档可作为 SR-Speed Pareto
对照，尚未跑满 500 episodes。

风险受限自适应档是独立的单组入口，必须显式指定 checkpoint，避免解析到旧 golden 目录。下面正是 Minimal
epoch 100 的 50 trials/task 命令；它不会顺带运行线性、VTPF strict 或旧 TD-Fast：

```bash
SPEC_CKPT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/Draft_checkpoint/goal/epoch_100_step_044800 \
EVAL_EPOCH=100 CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_adaptive_decimation_goal_eval.sh
```

默认预注册参数为 `min_verified_run=2`、`anchor_pixel_relative_l2<=0.03`、`max_consecutive=2`。如需后续消融，
只能显式覆盖 `DFLASH_TEMPORAL_ADAPTIVE_MIN_VERIFIED_RUN` 或
`DFLASH_TEMPORAL_ADAPTIVE_MAX_ANCHOR_PIXEL_RELATIVE_L2`，并在 run id 中保留参数；不得用正式结果反向挑阈值。

VisualBudget 使用独立入口，不会覆盖旧 Adaptive。下面命令复现已经完成的 `0.15` aggressive 正式结果；
它不是当前默认 balanced 方案，运行和汇报时必须同时保留 SR：

```bash
SPEC_CKPT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/Draft_checkpoint/goal/epoch_100_step_044800 \
EVAL_EPOCH=100 CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
DFLASH_TEMPORAL_VISUAL_BUDGET=0.15 \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_visual_budget_goal_eval.sh
```

当前同时追求 3x 与成功率的主入口是 VTPF-PacedHarmonic。它固定使用 `stable_actions=1`、
`T-H-H,T-H` 节拍和第二 hold 的 `0.5` 连续动作缩放；不启用视觉特征缓存、树或动作组 relaxed 校验：

```bash
SPEC_CKPT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/Draft_checkpoint/goal/epoch_100_step_044800 \
EVAL_EPOCH=100 CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
TRIAL_START_INDEX=0 SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
LOG_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs/paced_harmonic_formal \
RUN_ID_NOTE=dflash-vtpf-paced-harmonic-goal-e100-s7-formal \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_paced_harmonic_goal_eval.sh
```

正式论文结果只能使用默认 `TRIAL_START_INDEX=0` 的完整 500 episodes。`DFLASH_PROFILE_STAGES=True` 或
`DFLASH_DEBUG_COMPARE_TARGET_AR=True` 只用于诊断，会额外同步 CUDA 或串行运行 AR，严禁用于速度表。

四个 suite 的论文主表统一使用 `run_dflash_minimal_suite_main_3way_eval.sh`。它要求显式提供 suite-specific
Draft，随后串行运行 `DFlash strict`、`DFlash + VTPF strict` 和
`DFlash + VTPF + PacedHarmonic`；三路共享 checkpoint、seed 和计时协议。Spatial e100 正式命令为：

```bash
TASK_SUITE_NAME=libero_spatial \
SPEC_CKPT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/Draft_checkpoint/spatial/epoch_100_step_062200 \
EVAL_EPOCH=100 CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
TRIAL_START_INDEX=0 SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_minimal_suite_main_3way_eval.sh
```

默认日志根目录为 `specvla-data/eval_logs/spatial/`。若前一组已经完成，可附加 `START_CASE=2` 或
`START_CASE=3` 从对应组继续。脚本会根据 `TASK_SUITE_NAME` 自动选择 spatial OpenVLA 权重，但不会自动
猜测 Draft；这项限制用于防止把 Goal Draft 误用于其它 suite。

仅做不重复的诊断批次时可附加 `TRIAL_START_INDEX=3`；论文正式 50-trial/task 必须保持默认 0。
PrefixCert 仅用于固定成本消融：

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=10 \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_prefix_cert_goal_eval.sh
```

Minimal Draft 的中途/最终 checkpoint 可用三路入口串行比较：线性 DFlash strict、VTPF strict、
VTPF-TD-Fast。它会强制三路使用同一个 checkpoint、RNN-off、seed 和计时协议，并把日志写入三个带 epoch
标识的独立目录；中断后可用 `START_CASE=2` 或 `3` 续跑。e100 示例：

```bash
SPEC_CKPT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/Draft_checkpoint/goal/epoch_100_step_044800 \
EVAL_EPOCH=100 CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla/specdecoding/decode-scripts/run_dflash_minimal_goal_3way_eval.sh
```

它没有 Action-RNN 权重，因此线性 strict 和 VTPF strict 都必须关闭 RNN；三路入口已经统一处理。若只想单独
运行正式 e200 的 VTPF/VTPF-TD，可使用下面两条细粒度命令。先确认 checkpoint 目录名再执行：

```bash
MINIMAL_CKPT_ROOT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/ckpt_goal_dflash_minimal_1layer_hidden_soft_b16x1_4gpu_packedv2

# Minimal + VTPF strict
LOG_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs/dflash_strict/简化版Draft+VTPF \
DFLASH_OUTPUT_DIR="${MINIMAL_CKPT_ROOT}" DFLASH_USE_CAUSAL_RESIDUAL_SAMPLING=False \
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_temporal_cascade_goal_eval.sh prefill

# Minimal + VTPF-TD-Fast
LOG_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs/dflash_relaxed/简化版Draft+VTPF-TD \
DFLASH_OUTPUT_DIR="${MINIMAL_CKPT_ROOT}" \
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_temporal_decimation_goal_eval.sh
```

这两组与 Golden 正式结果的唯一模型变量应是 Draft checkpoint；目标模型、seed、trial 数、计时口径和 VTPF
阈值必须保持一致。若 Minimal 最佳 checkpoint 不是 epoch 200，只改 `EVAL_EPOCH`，并在结果目录名和
`RUN_ID_NOTE` 中同步写明，禁止悄悄挑权重。

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

4090 当前按“模型结构 + 推理机制”整理正式文件，禁止再把所有实验堆在 `eval_logs` 根目录：

```text
eval_logs/
├── baseline/{openvla_ar,specvla_strict,specvla_relaxed}/
├── dflash_strict/
│   ├── 复杂版Draft/
│   ├── 复杂版Draft+VTPF/
│   └── 复杂版去掉RNN的Draft+VTPF/
└── dflash_relaxed/
    └── 复杂版去掉RNN的Draft+VTPF-TD/
```

现有正式日志已按上面目录人工归档并同步到
[`artifacts/eval/curated_20260720_20260728`](artifacts/eval/curated_20260720_20260728)。新实验可在启动时直接
覆盖 `LOG_DIR`，避免事后移动，例如简化版 VTPF：

```bash
LOG_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs/dflash_strict/简化版Draft+VTPF \
DFLASH_OUTPUT_DIR=/绝对路径/ckpt_goal_dflash_minimal_1layer_hidden_soft_b16x1_4gpu_packedv2 \
CUDA_VISIBLE_DEVICES=0 EVAL_EPOCH=200 NUM_TRIALS_PER_TASK=50 \
SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_temporal_cascade_goal_eval.sh prefill
```

每个实验目录只放同一模型、同一机制、同一正式协议产生的三个文件；pilot、不同 seed 或不同阈值要在目录名或
`RUN_ID_NOTE` 中显式区分。不要重命名 JSON 内部的 `run_id`，否则外部文件名和内部实验身份会失去对应关系。

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
| `run_dflash_data_goal.sh` | `TASK_SUITE_NAME` 选择四个 LIBERO 子集，`smoke/full` 生成 raw v1 并自动无损打包 packed v2 |
| `pack_dflash_hdf5.py` | 将既有 v1 数据一次性迁移为 packed v2；日常无需单独调用 |
| `benchmark_dflash_hdf5.py` | 四进程只读 A/B 测试 legacy v1 与 packed v2 的真实数据吞吐 |
| `run_dflash_train.sh joint` | Golden 兼容入口：200 epoch 高维主导 Base/Final 线性交接 |
| `run_dflash_train.sh minimal` | 当前训练主线：四子集统一 100 epoch、两卡 global batch 64、一层 Draft + multi-anchor + Hidden/Cos/Soft KL |
| `run_dflash_train.sh stage1/stage2` | 仅用于复现已废弃的两阶段消融 |

`train_dflash_libero_goal.py` 是底层训练实现，不建议日常手写几十个 CLI 参数。历史单次课程仍保留在 Python
兼容参数中用于解释旧 checkpoint；`joint` 启用 `domino_linear_curriculum`，`minimal` 不启用课程或 RNN。

### 8.2 推理入口

| 脚本 | 用法 |
| --- | --- |
| `run_specvla_paper_ar_eval.sh` | 一个 suite 的论文 AR 分母 |
| `run_specvla_eval.sh` | 一个 suite 的 strict/relaxed |
| `run_specvla_goal_upstream_compatible_eval.sh` | Goal AR+strict+relaxed 一键复现 |
| `run_specvla_main_table_eval.sh` | 四 suite strict/relaxed 自动续跑与汇总 |
| `run_dflash_goal_eval.sh` | DFlash 单项 strict/relaxed；默认 Goal，也可由 `TASK_SUITE_NAME` 显式选择 suite |
| `run_dflash_minimal_goal_3way_eval.sh` | 同一 Minimal checkpoint 的线性 strict、VTPF strict、VTPF-TD-Fast 串行评测；支持断点续跑 |
| `run_dflash_minimal_suite_main_3way_eval.sh` | 四 suite 主表入口：DFlash strict、VTPF strict、PacedHarmonic 串行评测；必须显式传 suite-specific Draft |
| `run_dflash_temporal_cascade_goal_eval.sh` | strict VTPF 主线：`shadow`、严格 `route`、严格 `prefill`、旧 approximate `cascade` |
| `run_dflash_temporal_prefill_tree_goal_eval.sh` | golden e200 的时序多候选 prefill 树，参数为 `strict` 或 `relaxed` |
| `run_dflash_vtpf_temporal_decimation_goal_eval.sh` | VTPF-TD 速度档：target 与单步 hold 交替 |
| `run_dflash_vtpf_guarded_bypass_goal_eval.sh` | VTPF-TD 保护档：图像变化门控的单步 hold |
| `run_dflash_vtpf_adaptive_decimation_goal_eval.sh` | 单组 VTPF-TD 自适应档：双重证据才扩展第二次 hold，随后强制 target |
| `run_dflash_vtpf_visual_budget_goal_eval.sh` | 单组 VTPF-TD 视觉预算档：累计视觉漂移控制第二次 hold；仅作为 aggressive 速度上界 |
| `run_dflash_vtpf_paced_budget_goal_eval.sh` | `T-H-H,T-H` 固定节拍预算消融，不改变 hold 动作幅度 |
| `run_dflash_vtpf_age_decayed_goal_eval.sh` | 第二 hold 连续动作按 `1/hold_depth` 衰减的单模块消融 |
| `run_dflash_vtpf_paced_harmonic_goal_eval.sh` | 当前主线：节拍预算 + 谐波保持 + `stable_actions=1` 严格 prefill 候选 |
| `run_dflash_vtpf_prefix_cert_goal_eval.sh` | PrefixCert 固定成本消融，不是推荐主线 |
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
mujoco 3.3.2
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

1. 以 Minimal e100 为其它三个 LIBERO 子集的干净训练基线；保留 Golden tag 和权重，只作回退与消融。
2. 为 object/spatial/10 分别生成同格式教师数据、训练独立 Draft，禁止跨子集混用 Draft 或动作统计量。
3. 保存 e60/e80/e100，并用小规模在线 VTPF strict 筛选；不要用离线 loss 直接决定早停。
4. 固定 TD-Fast 为 relaxed 主方案；Adaptive 已完成正式消融但没有净胜 Golden TD-Fast，不进入主表。
5. 在其它 suite 的各自权重和真实机械臂上验证短时保持的稳定性。

### 11.2 论文需要的完整证据

- 主表：paper AR、SpecVLA strict/relaxed、DFlash strict/relaxed。
- 训练到在线的诊断图：teacher-forced、self-rollout、online hit rate。
- 前缀图：条件接受概率与 expected prefix length。
- 速度分解：target prefill、DFlash transformer、target verify、保持帧和环境外开销。
- 消融：Minimal/Golden、Action-RNN、跨 anchor、树、动作组、PrefixCert、VTPF-TD Guard/Fast。
- 鲁棒性：checkpoint、seed、硬件、任务长度。
- 真机：ALICIA-D 上比较成功率、动作延迟、控制频率和失败类型。

P0 证据脚手架已经独立于正式 50-trial 主表落地。Goal 和 Spatial 分别显式传入各自 Draft。成本与 VTPF
审计使用 DFlash 路径；时序动机数据来自 SpecVLA 论文口径的 wrapped AR 分母，不再用 Draft 轨迹自证。各输入
都记录 task、initial-state index、seed、配置和 SHA-256：

```bash
# Goal：逐阶段成本、时序动作持久性、VTPF fused verifier 审计
CUDA_VISIBLE_DEVICES=0 \
SPEC_CKPT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/Draft_checkpoint/goal/epoch_100_step_044800 \
P0_TRIALS=6 P0_TASKS=1 \
bash openvla/specdecoding/decode-scripts/run_dflash_p0_evidence.sh goal

# Spatial：替换成 suite 对应的 Draft，禁止跨 suite 混用
CUDA_VISIBLE_DEVICES=0 \
SPEC_CKPT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/Draft_checkpoint/spatial/epoch_100_step_062200 \
P0_TRIALS=6 P0_TASKS=1 \
bash openvla/specdecoding/decode-scripts/run_dflash_p0_evidence.sh spatial

# 不依赖 Draft 的同状态反事实恢复；正式证据必须使用完整恢复 horizon
CUDA_VISIBLE_DEVICES=0 P0_REFERENCE_EPISODES=3 P0_FORKS_PER_EPISODE=3 \
P0_MAX_RECOVERY_STEPS=0 \
bash openvla/specdecoding/decode-scripts/run_dflash_p0_counterfactual.sh goal
```

完整原始运行写入 `specvla-data/evidence/p0/<suite>/<UTC stamp>`；可提交的证据包写入
`artifacts/evidence/p0/<suite>/<UTC stamp>`。证据包包含压缩原始 JSON/TXT、逐文件 SHA-256、Git 状态、CSV、
矢量 PDF 和 PNG。图宽严格采用 ICLR 2026 模板的 `5.5 in` 正文宽度。阶段 profiler 显式同步 CUDA，只用于
诊断成本组成，不得冒充 paper-style 端到端延迟；时序重复率只是描述性证据，不能代替闭环可恢复性；同状态
实验通过相同 seed、初始状态和完整历史动作重放来恢复 simulator/controller 历史；只有 fork-state 差异不超过
`1e-8` 且 `current_target_path` 正对照全部通过时才有效。

2026-08-02 的 P0 pilot 得到以下结果。它们是机制证据，不是正式总体置信区间：

| 证据 | Goal task-0 | Spatial task-0 | 当前含义 |
|---|---:|---:|---|
| wrapped-AR episode（成功数） | 6（4） | 6（5） | 同时保留成功与失败轨迹；样本仍小 |
| lag-1 完整 7 维动作重复率（成功轨迹） | 12.68% | 13.85% | 不能声称“大多数动作完全相同” |
| 相邻动作 L2 中位数 | 0.151 | 0.150 | 短时间差动作局部集中 |
| 同 episode 半轨迹间隔动作 L2 中位数 | 0.734 | 2.136 | 分别是相邻距离的 4.85x、14.24x |
| DFlash target prefill / anchor / verify | 52.60 / 63.54 / 38.14 ms | 53.09 / 53.23 / 43.48 ms | target 固定工作是主耗时 |
| DFlash 并行 Draft | 4.85 ms/action | 4.11 ms/action | 继续增大 Draft 很难单独突破速度下界 |
| fused-vs-serial 共同因果位置 top-1 分歧 | 39/2321 | 9/824 | BF16 计算形状会改变舍入路径 |
| verifier 接受 token 中相对 serial AR 的分歧 | 40/1723 | 0/555 | Goal 上 strict verifier 不是位级 AR 等价 |

因此，论文的可辩护观察应写成“target 动作具有显著的短时连续空间局部性”，而不是“相邻动作大多完全相同”。
VTPF 应准确称为 `target-verifier strict`；它对 fused 序列执行 target 前缀裁决，但不能写成逐 token KV-cache
AR 的 bitwise 等价。正式主张仍需扩展到更多 task、seed，并补齐同 target 预算的 Paced 消融、同 schedule 的
Harmonic 消融和独立 calibration/test 风险上界。

Spatial task-0 还完成了一轮确定性同状态分叉 pilot：选取 1 条 target 成功轨迹、2 个冻结状态，对
6 种候选和深度 1–3 共执行 36 个分支。`current_target_path` 的 6 个正对照全部通过，重放到分叉点的
simulator-state 最大误差为 0。滞后一步 target 动作在该轨迹的所有分支上仍可恢复，但这只是存在性
证据，不是群体风险上界。更关键的是，`1, 1/2, 1/3` 逆年龄缩放没有降低位姿偏差：深度 3 时，
常幅滞后动作的末端位置/旋转偏差为 `0.00592/0.02231`，谐波缩放为 `0.00838/0.02875`。
因此当前 P0 支持“短时可恢复域值得研究”，却暂不支持“Harmonic 必然降低物理偏移”。该负证据已原样
固化在 `artifacts/evidence/p0/spatial/20260802T_cf_spatial_physics_n2/counterfactual/`。

本轮两个主证据包分别位于
`artifacts/evidence/p0/goal/20260802T_p0_goal_ar_n6b/` 和
`artifacts/evidence/p0/spatial/20260802T_p0_spatial_ar_n6b/`。每个目录中的 `manifest.json` 记录源文件哈希、配置、
Git commit 和有效性边界；`raw/`、`tables/`、`figures/` 分别保存压缩原始证据、作图数据和论文图。

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
- Golden Action-RNN 引入轻量顺序步骤；Minimal 和默认 RNN-off 推理才保持纯块并行 proposal。
- VTPF-TD 已完成 Goal 单 seed 500-episode 正式评测；跨论文比较 HeiSD 前仍需严格统一硬件、计时和 AL 定义。
- BF16 fused verification 与逐 token AR 可能因内核形状和舍入路径不同而产生 top-1 分歧；`strict` 指 target
  verifier 的精确前缀规则，不自动等同于逐 token AR 的 bitwise 输出。

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
