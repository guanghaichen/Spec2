# SpecVLA-DFLASH

本仓库研究冻结 OpenVLA-7B 的低延迟动作解码。当前唯一正式方法已经收敛为：

> **Minimal DFlash + Verified Temporal Prefill Fusion (VTPF) + PacedHarmonic**

它沿两个互补维度降低推理成本：DFlash 与 VTPF 减少一次目标帧内部的动作 token 解码开销，
PacedHarmonic 在闭环控制时间轴上减少完整目标模型调用。Action-RNN、跨 Anchor 蒸馏、Domino
交接、动作组宽松校验、DDTree、RAES/恢复风险校准均不属于当前正式方法，只作为历史研究或消融保留。

代码源自 [SpecVLA](https://github.com/PineTreeWss/SpecVLA) 和
[OpenVLA](https://github.com/openvla/openvla)，块并行 Draft 的灵感来自
[DFlash](https://arxiv.org/abs/2602.06036)。本仓库迁移的是“目标 hidden 条件下的轻量块并行
Draft”思想，不声称复现 DFlash 原论文的完整训练系统。

## 1. 当前方法

### 1.1 Minimal DFlash：动作 token 轴

OpenVLA 把一个动作量化为 7 个 token。目标模型产生第一个动作 token `t0` 后，一层非因果 Draft
以完整 prompt hidden、已知动作前缀和动作维度 embedding 为条件，一次并行提出后续动作块；OpenVLA
随后在一次目标 forward 中逐位置校验，并部分接受、部分纠正。

当前训练配方刻意保持简单：

- 一层 Draft；
- 5 个等间隔目标层 hidden，包含首层与最终层；
- 完整 prompt/prefix hidden；
- multi-anchor 覆盖不同真实动作前缀；
- `Hidden Smooth-L1 = 1.0`、`Cosine = 0.05`、`Soft KL = 0.05`；
- 100 epochs、两卡、global batch 64；
- 不创建 Action-RNN，不使用 hard CE、跨 Anchor KL、L1、Prefix Survival 或训练课程交接。

multi-anchor 是训练覆盖手段，不是独立的在线推理模块。正式在线结果表明，复杂 Golden Draft 中的
Action-RNN/CAD/课程组件并非当前性能的必要条件。

### 1.2 VTPF：目标帧内部

VTPF（Verified Temporal Prefill Fusion）把上一环境步已经执行的完整动作作为当前目标模型 prefill
中的候选前缀。目标模型在这次必需的多模态 prefill 中直接校验该候选：

- 匹配的共同前缀立即接受；
- 第一个不匹配位置由目标模型纠正；
- 未被 VTPF 覆盖的后缀回退到普通 DFlash。

因此 VTPF strict 不跳过目标模型，也不改变目标模型答案，只把“prefill + 单独 verify”融合成一个
因果正确的目标 forward。单独评测 VTPF strict 时使用 `stable_actions >= 3`；完整 PacedHarmonic
路径使用最近一次已确认动作作为候选，门槛为 1。

### 1.3 PacedHarmonic：控制时间轴

PacedHarmonic 允许短时复用最近一次目标帧产生并执行过的动作：

- `T`：调用 OpenVLA，并执行当前目标动作；
- `H1`：第一次 Hold，复用最近目标动作，连续六维保持原幅度；
- `H2`：第二次 Hold，仅在当前图像相对最近目标锚点的归一化变化不超过 `0.15` 时放行；连续六维
  按 `1 / hold_depth = 1/2` 缩放，夹爪维度不缩放；
- 最大连续 Hold 深度为 2，绝不出现 `H3`；
- 使用一次 `H2` 后产生扩展债务，下一个目标区间最多只允许 `H1`，随后必须重新 grounding。

Pace 约束开放环提交的长期频率，Harmonic 限制陈旧增量控制的累计权限。该路径会真实执行近似动作，
不再具有 strict token 等价性，因此必须把成功率与 Speedup 同时报告，不能只报告 Length。

## 2. 四子集正式结果

统一协议：RTX 4090、BF16、batch size 1、seed 7、每个 LIBERO 子集 10 个任务、每任务 50 个初始状态，
共 500 episodes；`SYNC_CUDA_TIMING=False`、`TIMING_SCOPE=last_task`。Speedup 均以同机、同子集
OpenVLA AR 为分母。

单元格为 **SR / Speedup**：

| 方法 | Goal | Spatial | Object | LIBERO-10 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenVLA AR | 74.2 / 1.00x | 87.0 / 1.00x | 88.4 / 1.00x | 51.4 / 1.00x | 75.3 / 1.00x |
| SpecVLA strict | 76.8 / 1.02x | 85.0 / 1.00x | 87.6 / 1.10x | 54.4 / 0.99x | 76.0 / 1.03x |
| SpecVLA relaxed | 73.4 / 1.29x | 86.2 / 1.15x | 85.0 / 1.31x | 49.8 / 1.11x | 73.6 / 1.21x |
| Minimal DFlash strict | 79.2 / 1.14x | 85.8 / 1.15x | 88.2 / 1.18x | 52.8 / 1.18x | 76.5 / 1.16x |
| + VTPF strict | 77.6 / 1.30x | 86.4 / 1.17x | 87.2 / 1.19x | 50.0 / 1.29x | 75.3 / 1.24x |
| **+ PacedHarmonic** | **74.6 / 3.48x** | **80.2 / 2.92x** | **84.4 / 3.08x** | **48.8 / 3.48x** | **72.0 / 3.24x** |

Object 使用独立训练的 epoch 60 checkpoint；其它三个子集使用 epoch 100。该差异必须保留在复现记录中，
不能把 e60 改写为 e100。

### Goal：Pace × Harmonic 消融

四个条件固定同一 Minimal Draft、VTPF、视觉界和最大 Hold 深度：

| Pace | Harmonic | SR | Speedup | Target rate | H2 次数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 关闭 | 关闭 | 67.2 | 3.68x | 34.33% | 3,362 |
| 开启 | 关闭 | 70.6 | 3.20x | 40.47% | 2,167 |
| 关闭 | 开启 | 74.0 | 3.46x | 34.19% | 3,427 |
| **开启** | **开启** | **74.6** | **3.48x** | **40.46%** | **2,243** |

这张表说明 Harmonic 是恢复成功率的主要来源，Pace 进一步限制连续扩展并提高完整组合的闭环稳定性。
Pace 会主动改变目标调用率，因此不能把闭环 2×2 当作“相同计算预算”的纯因果实验；匹配 target 预算的
机制实验应单独报告。

## 3. 仓库地图

| 环节 | 当前入口 | 作用 |
| --- | --- | --- |
| 数据生成 | `openvla/specdecoding/train-scripts/run_dflash_data_goal.sh` | 四子集统一生成并打包为单个 packed-v2 HDF5 |
| Draft 训练 | `openvla/specdecoding/train-scripts/run_dflash_train.sh minimal` | 当前一层 Minimal Draft 配方 |
| Draft 模型 | `openvla/specdecoding/model/dflash.py` | 块并行主干与兼容的历史结构 |
| 时序控制 | `openvla/specdecoding/model/temporal_hold.py` | Paced 信用状态与 Harmonic 权限缩放 |
| 在线解码 | `openvla/prismatic/extern/hf/modeling_speculation.py` | DFlash 提案、VTPF、目标校验与 Hold 执行 |
| LIBERO 评测 | `openvla/experiments/robot/libero/` | strict/relaxed rollout 与统一指标 |
| 三路正式评测 | `openvla/specdecoding/decode-scripts/run_dflash_minimal_suite_main_3way_eval.sh` | DFlash strict、VTPF strict、完整方法 |
| PacedHarmonic | `openvla/specdecoding/decode-scripts/run_dflash_vtpf_paced_harmonic_goal_eval.sh` | 完整时序方法入口，支持四子集 |
| SpecVLA 基线 | `openvla/specdecoding/decode-scripts/run_specvla_main_table_eval.sh` | 四子集 AR/strict/relaxed |
| 论文结果索引 | `openvla/specdecoding/evidence/organize_paper_results.py` | 构建规范结果目录、指标清单和 SHA-256 |

推理脚本的正式、消融和历史分级见
[`openvla/specdecoding/decode-scripts/README.md`](openvla/specdecoding/decode-scripts/README.md)。
完整研究历程已经移至
[`docs/research_history_legacy_zh.md`](docs/research_history_legacy_zh.md)，其中包含失败实验和旧服务器说明，
不得当作当前运行指南。

## 4. 标准工作流

### 4.1 环境与目录

建议使用独立 Conda 环境：

```text
Python 3.10
torch 2.2.0+cu121
transformers 4.40.1
mujoco 3.3.2
robosuite 1.4.1
numpy 1.26.4
accelerate 1.9.0
```

训练机负责数据生成和两卡训练；RTX 4090 负责单卡正式推理。不要在 `.bashrc` 中固定某个子集的
`VLA_PATH`，公共脚本会依据 `TASK_SUITE_NAME` 选择对应模型。

四个子集名称：

```text
libero_goal
libero_spatial
libero_object
libero_10
```

每台机器应准备：

```text
hf_files/openvla-7b-finetuned-libero-{goal,spatial,object,10}
datasets/modified_libero_rlds/libero_{goal,spatial,object,10}_no_noops
LIBERO/
specvla-data/
```

### 4.2 生成数据

先做 smoke，再做 full。`full` 会生成临时 raw-v1，完成后无损打包为一个 packed-v2 HDF5，并默认删除
中间文件，避免数万个小文件造成随机 I/O：

```bash
cd /data/wulin/c/SpecVLA-DFLASH
source /data/wulin/miniconda3/etc/profile.d/conda.sh
conda activate specvla

TASK_SUITE_NAME=libero_goal GPU_ID=0 \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh smoke

TASK_SUITE_NAME=libero_goal GPU_ID=0 \
  bash openvla/specdecoding/train-scripts/run_dflash_data_goal.sh full
```

更换子集只改 `TASK_SUITE_NAME`。训练读取的是：

```text
goal:    specvla-data/dflash_goal_dataset_envfix_20260714_packed_v2.h5
spatial: specvla-data/dflash_spatial_dataset_packed_v2.h5
object:  specvla-data/dflash_object_dataset_packed_v2.h5
10:      specvla-data/dflash_10_dataset_packed_v2.h5
```

### 4.3 训练 Minimal Draft

正式默认是两卡、每卡 batch 32、一个 worker、100 epochs：

```bash
cd /data/wulin/c/SpecVLA-DFLASH
source /data/wulin/miniconda3/etc/profile.d/conda.sh
conda activate specvla

CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
TASK_SUITE_NAME=libero_goal NUM_WORKERS=1 \
  bash openvla/specdecoding/train-scripts/run_dflash_train.sh minimal
```

更换子集只改 `TASK_SUITE_NAME`。默认输出目录形如：

```text
specvla-data/ckpt_<suite>_dflash_minimal_1layer_hidden_soft_b32x1_2gpu_packedv2/
```

训练会每 10 epochs 保存一次轻量 checkpoint，不保存 optimizer state。SwanLab 记录训练指标，但训练准确率
不能替代在线 LIBERO 接受率、SR 和 Speedup。

### 4.4 把 checkpoint 传到 4090

建议统一放置：

```text
specvla-data/Draft_checkpoint/
├── goal/epoch_100_step_044800
├── spatial/epoch_100_step_062200
├── object/epoch_060_step_058440
└── 10/epoch_100_step_138200
```

每个目录至少应包含：

```text
pytorch_model.bin
dflash_config.json
```

### 4.5 一键评测一个子集

以下命令按顺序运行 DFlash strict、VTPF strict、完整 PacedHarmonic：

```bash
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH
source /home/asus/miniconda3/etc/profile.d/conda.sh
conda activate specvla

CUDA_VISIBLE_DEVICES=0 \
TASK_SUITE_NAME=libero_goal \
SPEC_CKPT=/absolute/path/to/goal/epoch_100_step_xxxxxx \
EVAL_EPOCH=100 NUM_TRIALS_PER_TASK=50 TRIAL_START_INDEX=0 \
SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_minimal_suite_main_3way_eval.sh
```

中断后使用 `START_CASE=2` 或 `START_CASE=3` 从对应方法继续。脚本会依据 `TASK_SUITE_NAME` 自动选择
目标模型，但不会猜测 Draft checkpoint，禁止跨子集使用 Draft。

### 4.6 只运行完整方法

```bash
CUDA_VISIBLE_DEVICES=0 TASK_SUITE_NAME=libero_goal \
SPEC_CKPT=/absolute/path/to/goal/checkpoint EVAL_EPOCH=100 \
NUM_TRIALS_PER_TASK=50 SEED=7 \
SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_paced_harmonic_goal_eval.sh
```

### 4.7 输出与指标

每次评测产生：

```text
*.txt
*_timing.json
*_summary.json
```

| 指标 | 含义 |
| --- | --- |
| `success_rate` | 500 个闭环 episode 的任务成功率 |
| `timing.mean` | 当前计时范围内平均动作模型耗时 |
| `speedup` | 同机同子集 AR mean / 当前 mean |
| `generation.length` | 每个 speculative block 的平均推进长度 |
| `avg_accept_length` | 每块平均接受的 Draft token 数，不等于 Length |
| `per_position` | 各 proposal 位置的在线命中率 |
| `temporal_hold.target_prefill_rate` | 实际调用完整目标 prefill 的环境步比例 |
| `base_holds / extended_holds` | H1 / H2 次数 |

正式速度表必须固定同一机器、同一 GPU、同一子集和同一计时口径。`Length` 大不代表端到端一定更快，
因为 Draft、prefill、校验和模拟器开销共同决定延迟。

## 5. 正式入口与历史代码

### 当前正式入口

- `run_dflash_data_goal.sh smoke/full`
- `run_dflash_train.sh minimal`
- `run_dflash_minimal_suite_main_3way_eval.sh`
- `run_dflash_vtpf_paced_harmonic_goal_eval.sh`
- `run_specvla_main_table_eval.sh`
- `organize_paper_results.py`

### 当前论文消融

- `run_dflash_vtpf_visual_budget_goal_eval.sh`：无 Pace、无 Harmonic；
- `run_dflash_vtpf_paced_budget_goal_eval.sh`：仅 Pace；
- `run_dflash_vtpf_age_decayed_goal_eval.sh`：仅 Harmonic；
- `run_dflash_p0_temporal_2x2.sh`：匹配 target 预算的机制实验。

### 历史研究分支

以下内容保留用于审计或复现实验历程，但不是当前推荐方案：

- Golden Action-RNN、Markov/CAD、Domino 式交接；
- 两阶段训练；
- DDTree、动作组接受、Prefill 树；
- PrefixCert、GuardedBypass、Adaptive/VisualBudget 单独主张；
- RAES、上下文风险校准、冻结 profile 和恢复实验。

旧分支可能仍由底层兼容代码支持。不要根据文件仍然存在就把它写成当前方法，也不要删除其历史证据后再
声称从未做过这些实验。

## 6. 结果管理

论文数字的规范入口是服务器上的：

```text
specvla-data/paper_results/
├── main_table/<suite>/<method>/<run_tag>/
├── ablation/<suite>/<method>/<run_tag>/
├── manifest/runs.csv
└── manifest/runs.json
```

运行：

```bash
python openvla/specdecoding/evidence/organize_paper_results.py
```

整理器不会改写原始日志，而是建立硬链接、紧凑指标、源路径和 SHA-256。详细规则见
[`docs/paper_results_zh.md`](docs/paper_results_zh.md)。
Git 中冻结的紧凑四子集清单见
[`artifacts/eval/paced_harmonic_main_20260808/`](artifacts/eval/paced_harmonic_main_20260808/README.md)；
其它既有评测目录的历史属性见 [`artifacts/eval/README.md`](artifacts/eval/README.md)。

## 7. Git 与多机规则

- 4090 是主要代码维护与正式推理机器；
- 3090 是数据生成与训练机器；
- 先在主开发副本提交并推送 GitHub，再在另一台机器快进同步；
- checkpoint、HDF5、原始 rollout 和 SwanLab 日志不进入 Git；
- 不覆盖已有正式结果，不把不同 seed、阈值或 checkpoint 写进同一目录；
- 当前仓库为私有仓库，远程访问依赖各机器自己的 deploy key。

常规同步：

```bash
git status --short
git pull --ff-only origin main
git push origin main
```

## 8. 研究边界

1. DFlash strict 与 VTPF strict 保持目标 token 校验；PacedHarmonic 不具有 strict 等价性。
2. `0.15` 是当前四子集共用的视觉漂移界，不应描述成形式化安全证书。
3. 当前证据来自 LIBERO 和 OpenVLA-7B；跨 VLA 架构与真机泛化仍需实验。
4. 单 seed 的 SR 差异存在采样波动；关键比较应补充配对统计或多 seed 区间。
5. 训练指标只用于诊断 Draft 学习，论文结论以闭环 SR、延迟和目标调用率为准。

## 9. 参考

- OpenVLA: https://github.com/openvla/openvla
- SpecVLA: https://github.com/PineTreeWss/SpecVLA
- DFlash: https://arxiv.org/abs/2602.06036
- SpecForge DFlash implementation: https://github.com/sgl-project/SpecForge
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
