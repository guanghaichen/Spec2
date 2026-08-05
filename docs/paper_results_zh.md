# 论文实验结果唯一入口

本目录是论文数字、表格和复现实验的唯一入口。`eval_logs/`、`main_table/`、
`evidence/` 与 `paper_archive/` 是历史原始目录，不再直接从中手抄论文数字。

```text
paper_results/
├── main_table/<suite>/<method>/<run_tag>/   # 正式 500-episode 结果
├── ablation/<suite>/<method>/<run_tag>/     # 与主表共享同一原始文件
├── evidence/p0                              # 机制实验入口
├── evidence/calibration                     # 校准实验入口
├── reproducibility/                         # 代码快照、环境和校验和
└── manifest/
    ├── runs.csv                             # 论文表格的首选输入
    └── runs.json                            # 保留类型的完整清单
```

## 命名规则

- `goal`、`spatial`、`object`、`libero_10`：LIBERO 子集。
- `openvla_ar`：同机 AR 分母。
- `specvla_strict`、`specvla_relaxed`：本地复现基线。
- `dflash_strict`：一层 Minimal DFlash 并行动作块，逐 token 严格校验。
- `vtpf_strict`：并行动作块加 VTPF，仍保持当前 target 严格校验。
- `raes_rho040`：完整 RAES，公共目标密度预算为 0.40。
- `run_tag` 必须显式写出 Draft epoch、episode 数和 seed；不能把 e60 记成 e100。

## 文件语义

每个正式 run 包含原始 `summary.json`、`timing.json` 和文本日志的硬链接，同时包含：

- `metrics.json`：用于论文的紧凑指标，包括 SR、延迟、相对 AR Speedup、Length、
  target-frame ratio 和 checkpoint。
- `SOURCE.txt`：历史原始目录，便于审计。
- `MANIFEST.sha256`：原始日志校验和。

硬链接不重复占用磁盘。删除历史路径中的同名目录条目也不会使本目录的数据失效；但不要手工修改
任何正式日志。新增正式结果后，应更新归档脚本的 run registry，再重新生成清单。

## 论文取数约束

1. 主表只读取 `manifest/runs.csv` 中 `table=main_table` 的行。
2. Speedup 统一以同一子集、同一 RTX 4090 协议的 `openvla_ar` 平均动作延迟为分母。
3. `timing_scope=last_task`、`sync_cuda_timing=false`、500 episodes 和 seed 7 才能进入正式主表。
4. smoke、profile、校准选择集和旧 PacedHarmonic 结果不能进入主表。
5. VTPF 的 token Length 与跨环境步 Hold 深度是不同指标，不能混写。
