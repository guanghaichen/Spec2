# 四子集历史结果，2026-08-08

> **Legacy:** 本目录只对 H2 使用视觉约束，已经被 H1/H2 共用 Target 锚点的
> `../dual_anchor_main_20260809/` 取代。保留本目录仅用于历史审计，不能作为当前主表来源。

本目录对应当前冻结方法：

```text
Minimal DFlash + VTPF + PacedHarmonic
```

协议为 RTX 4090、seed 7、每任务 50 episodes、`TIMING_SCOPE=last_task`、
`SYNC_CUDA_TIMING=False`。`runs.csv` 由服务器规范结果整理器生成，包含原始日志路径、
checkpoint、SR、延迟、相对 AR Speedup、Length 和时序调用率。

注意：Object 的 Draft 是真实 epoch 60 checkpoint；其它三个子集为 epoch 100。
大体积逐 episode summary 保留在服务器 `specvla-data/paper_results/`，通过清单中的来源路径和
服务器侧 `MANIFEST.sha256` 审计，不在 Git 中重复存储。
