# OpenVLA-OFT 层早退扩展

本目录是 `SpecVLA-DFLASH` 中独立维护的 OpenVLA-OFT 实验支线。上游代码固定于
[`moojink/openvla-oft`](https://github.com/moojink/openvla-oft) 的 `e4287e94541f459edc4feabc4e181f537cd569a8`。
它不替代 DFlash：DFlash 面向自回归 OpenVLA 的动作 token 投机解码；本实验面向 OpenVLA-OFT 的并行 action-chunk 推理。

## 研究目标

OpenVLA-OFT 一次 forward 在序列末尾放置 `8 x 7 = 56` 个动作占位 token，并用最终 LLaMA 层的 56 个 hidden 送入冻结的 OFT L1 action head，直接输出未来 8 个连续动作。因此它已经不是自回归 token 解码，不能机械套用 DFlash 的 draft/verify 流程。

本支线实现 **层早退残差蒸馏**：

1. 完整 OFT 是 teacher。对同一图像、腕部图像、proprio 和语言指令，保存某个早层 `l` 的 56 个 action hidden 与最终层 56 个 action hidden。
2. 训练轻量 `LayerExitResidualAdapter`：输入早层 hidden；先在低维 bottleneck 中做 56 个 action token 的自注意力混合，再预测残差，得到逼近最终层的 refined hidden。
3. 训练目标由 hidden Smooth-L1、hidden cosine、以及经过冻结官方 L1 action head 后的 action loss 组成。最后一项直接约束机器人真正执行的连续动作。
4. 推理时通过 decoder-layer forward hook 在第 `l` 层后中断原始 LLaMA forward，后续层和不需要的 LM vocabulary head 均不执行。adapter 输出 refined hidden，复用官方 action head。

这里没有伪造因果前缀，也不会把 OFT 改成 AR：adapter 只处理 OFT 本来就并行存在的 56 个动作槽。它的 chunk mixer 用来恢复动作 chunk 内的关联，残差形式则让训练从 `H_l` 的恒等映射稳定起步。

## 代码布局

- `experiments/early_exit/adapter.py`：轻量跨 action-token 残差 adapter 与 checkpoint 格式。
- `experiments/early_exit/runtime.py`：在指定 LLaMA 层截断的 hook 实现；保留 OFT 原始 attention、RoPE 和 multimodal embedding 路径。
- `experiments/early_exit/feature_store.py`：单个 HDF5 teacher feature 文件，避免海量小文件。
- `experiments/early_exit/train_layer_exit_adapter.py`：可由 `torchrun` 四卡启动的 adapter 训练器。
- `experiments/robot/libero/run_libero_eval.py`：在不传早退参数时维持官方 OFT 基线；传入参数后可收集 teacher feature、记录同步 CUDA policy-query 时间、或启用早退。
- `scripts/`：不同机器共用的环境脚本、4090 准备脚本、3090 采集/训练脚本、4090 评测脚本。

## 机器分工

| 环节 | 机器 | 目录 |
|---|---|---|
| OFT 代码与最终评测 | `ssh 4090` | `/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH/openvla-oft` |
| Teacher feature 生成与 adapter 训练 | `ssh 3090_wulin` | `/data/wulin/c/SpecVLA-DFLASH/openvla-oft` |
| Goal OFT 权重 | 各机器 `hf_files/` | `openvla-7b-oft-finetuned-libero-goal` |
| 统一版本源 | 私有 GitHub | `guanghaichen/SpecVLA-DFLASH` |

3090 中以前的 `/data/wulin/c/openvla-oft` 仅是早期冒烟复现用的独立 clone；后续以本仓库的 `openvla-oft/` 为唯一权威代码。

## 环境与模型

OFT 论文推荐 Python 3.10.14、PyTorch 2.2.0、其 custom Transformers 4.40.1 fork。现有 3090 OFT 环境已满足核心版本；4090 从 `specvla` clone 出隔离的 `oft` 环境即可，因为其 Torch/CUDA/Mujoco/Robosuite 版本已经匹配。

在 4090 后台准备环境和 Goal 权重：

```bash
screen -S oft_setup_4090
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH
bash openvla-oft/scripts/setup_oft_4090_eval.sh
```

脚本使用清华 PyPI 镜像与 `HF_ENDPOINT=https://hf-mirror.com` 下载模型。OFT 基线和本 adapter 实验不需要 FlashAttention：adapter 训练不回传 7B OFT，推理使用 custom Transformers 的普通 attention 路径；因此不安装 FlashAttention，避免额外 CUDA 编译和环境污染。

第一次载入本地 checkpoint 时，上游 OFT 会把本目录的 `modeling_prismatic.py` 与 `configuration_prismatic.py` 同步入 checkpoint，并留下时间戳 backup。这是上游加载逻辑，不是模型权重重训或覆盖。

## 实验流程

### 1. 4090：复现 OFT Goal 基线

等待 DFlash 评测释放 GPU 后：

```bash
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH
CUDA_VISIBLE_DEVICES=0 NUM_TRIALS_PER_TASK=50 \
  bash openvla-oft/scripts/run_oft_goal_baseline_eval.sh
```

输出包含成功率，以及同步 CUDA 的 `mean_policy_query_seconds`。OFT 每次 query 输出固定 8 个动作，因此速度指标比较的是每个 **policy query**，不是 DFlash/SpecVLA 的接受长度。

### 2. 3090：生成单文件 teacher features

先从 GitHub 拉到同一 commit，再用一张空卡采样。建议开始时收集 4096 个 policy query：

```bash
cd /data/wulin/c/SpecVLA-DFLASH
CUDA_VISIBLE_DEVICES=0 TEACHER_EARLY_EXIT_LAYER=16 TEACHER_FEATURE_LIMIT=4096 \
  bash openvla-oft/scripts/run_oft_goal_collect_teacher_3090.sh
```

生成 `specvla-data/oft_runs/teacher_features/libero_goal_layer16.h5`。这个 HDF5 内含 `early_hidden`、`final_hidden` 与 teacher action chunk；不再生成数千个散文件。

### 3. 3090：四卡训练 adapter

```bash
cd /data/wulin/c/SpecVLA-DFLASH
GPUS=0,1,2,3 EARLY_EXIT_LAYER=16 EPOCHS=40 BATCH_SIZE=16 \
  bash openvla-oft/scripts/run_oft_layer_exit_train_4gpu.sh
```

输出在 `specvla-data/oft_runs/checkpoints/layer_exit_goal_l16/`。每个 epoch 与 `latest/` 都有标准 adapter checkpoint，不包含 7B OFT 权重。

### 4. 本地传权重到 4090，再做真实早退评测

```bash
scp -r 3090_wulin:/data/wulin/c/specvla-data/oft_runs/checkpoints/layer_exit_goal_l16 \
  4090:/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/oft_runs/checkpoints/
```

```bash
cd /media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/SpecVLA-DFLASH
CUDA_VISIBLE_DEVICES=0 EARLY_EXIT_LAYER=16 \
  bash openvla-oft/scripts/run_oft_goal_early_exit_eval.sh
```

## 必做实验

1. OFT full-depth baseline：Goal 50 trials/task，成功率和同步 policy-query latency。
2. Early exit 深度扫描：至少 `l = 8, 12, 16, 20, 24`。汇报 success-rate/latency Pareto 曲线。
3. Adapter 消融：identity early hidden、仅 token-wise MLP、完整 chunk mixer。证明跨 action-token mixing 的必要性。
4. Loss 消融：hidden-only 与 hidden+action loss。验证 action-level supervision 对真实控制成功率的作用。
5. 与 DFlash 主实验并列：强调两者分别覆盖 AR VLA 与 parallel-decoding VLA，而非宣称同一加速器适用于所有 VLA。

## 当前状态

- 3090 已完成 Goal 1 trial/task 的 OFT smoke：10 条 rollout 中 9 条成功，证明官方模型、LIBERO、EGL、两图+proprio输入和 action chunk 推理可运行。
- 4090 的 OFT 环境/权重尚待后台准备；在 DFlash 主表评测占用 GPU 时不要启动 OFT GPU 任务。
- Adapter、teacher HDF5、训练器、早退入口与脚本已纳入本私有仓库；下一步是 4090 baseline 与 3090 teacher feature collection。
