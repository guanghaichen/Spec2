# 论文实验结果唯一入口

`paper_results/` 是论文数字、表格与复现证据的规范入口。历史目录
`eval_logs/`、`main_table/`、`evidence/` 和 `paper_archive/` 只保存原始运行产物，
不得再从多个目录手工拼数字。

```text
paper_results/
├── main_table/<suite>/<method>/<run_tag>/
├── ablation/<suite>/<method>/<run_tag>/
├── pace_harmonic_ablation/goal/<method>/<run_tag>/
├── evidence/p0
└── manifest/
    ├── runs.csv
    └── runs.json
```

## 当前方法名

- `openvla_ar`：同机 OpenVLA 自回归分母；
- `specvla_strict`、`specvla_relaxed`：本地复现的 SpecVLA 基线；
- `dflash_strict`：Minimal DFlash 并行动作块，逐 token 严格校验；
- `vtpf_strict`：Minimal DFlash + VTPF，仍保持目标 token 严格校验；
- `paced_harmonic`：Minimal DFlash + VTPF + PacedHarmonic，当前完整方法。

RAES、上下文风险校准、Action-RNN、跨 Anchor 蒸馏与树验证均为历史研究分支，
不得注册为当前论文主方法。

## 当前四子集结果

下表为 RTX 4090、seed 7、每任务 50 episodes、`TIMING_SCOPE=last_task`、
`SYNC_CUDA_TIMING=False` 的正式结果。单元格为 `SR / Speedup`，Speedup 以同子集
OpenVLA AR 的平均动作延迟为分母。

| 方法 | Goal | Spatial | Object | LIBERO-10 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenVLA AR | 74.2 / 1.00x | 87.0 / 1.00x | 88.4 / 1.00x | 51.4 / 1.00x | 75.3 / 1.00x |
| SpecVLA strict | 76.8 / 1.02x | 85.0 / 1.00x | 87.6 / 1.10x | 54.4 / 0.99x | 76.0 / 1.03x |
| SpecVLA relaxed | 73.4 / 1.29x | 86.2 / 1.15x | 85.0 / 1.31x | 49.8 / 1.11x | 73.6 / 1.21x |
| DFlash strict | 79.2 / 1.14x | 85.8 / 1.15x | 88.2 / 1.18x | 52.8 / 1.18x | 76.5 / 1.16x |
| + VTPF strict | 77.6 / 1.30x | 86.4 / 1.17x | 87.2 / 1.19x | 50.0 / 1.29x | 75.3 / 1.24x |
| **+ PacedHarmonic** | **74.6 / 3.48x** | **80.2 / 2.92x** | **84.4 / 3.08x** | **48.8 / 3.48x** | **72.0 / 3.24x** |

Object 的 Draft 来自真实的 epoch 60 checkpoint；清单与 `run_tag` 必须保留该事实，
不得改写为 epoch 100。

## 文件语义

每个 run 目录包含原始日志三件套的硬链接，并附带：

- `metrics.json`：SR、平均延迟、相对 AR Speedup、Length、Target prefill rate 和 checkpoint；
- `SOURCE.txt`：原始日志目录；
- `MANIFEST.sha256`：原始文件校验和。

硬链接不重复占用大文件空间。`manifest/runs.csv` 是论文表格的首选机器可读输入，
`runs.json` 保留完整类型与来源信息。

## 论文取数约束

1. 主表只读取 `table=main_table` 的行；机制消融读取各自表名。
2. Speedup 只能使用同 suite、同机器、同计时协议的 `openvla_ar` 分母。
3. 正式结果固定 500 episodes、seed 7、`last_task`、异步 CUDA 计时。
4. smoke、选择集、调参日志、旧 RAES 与旧 TD-Fast 结果不得进入主表。
5. VTPF 的 token `Length` 与 PacedHarmonic 的跨环境步 Hold 深度不得混写。
6. 新增或替换正式结果后，先更新 `organize_paper_results.py` 的注册表，再重建清单。
