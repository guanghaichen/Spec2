# 评测环境

```text
GPU: NVIDIA GeForce RTX 4090, 24564 MiB
Driver: 595.84
Python: 3.10.20
PyTorch: 2.2.0+cu121
Transformers: 4.40.1
MuJoCo: 3.3.2
robosuite: 1.4.1
NumPy: 1.26.4
SciPy: 1.15.3
Timing scope: last_task
CUDA synchronization around every action: disabled
```

代码仓库：`git@github.com:guanghaichen/SpecVLA-DFLASH.git`

实现 commit：`82cdffe35a64338ccbfd4c5e513e26883b1b646b`

正式评测机器上的 checkpoint 绝对路径：

```text
/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/Draft_checkpoint/goal/epoch_100_step_044800
```

LIBERO 协议：`libero_goal`，seed 7，`trial_start_index=0`，每个 task 50 次，共 500 episodes。
