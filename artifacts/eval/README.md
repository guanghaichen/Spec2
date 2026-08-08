# 评测产物索引

本目录保存可进入 Git 的紧凑评测证据。当前正式方法固定为：

```text
Minimal DFlash + VTPF + PacedHarmonic
```

## 当前规范结果

- `paced_harmonic_main_20260808/`：四个 LIBERO 子集的主表、三阶段机制消融与
  Goal Pace × Harmonic 消融；
- 服务器上的完整规范目录由
  `openvla/specdecoding/evidence/organize_paper_results.py` 构建；
- 论文数字以该目录的 `manifest/runs.csv` 为准，原始多 MB rollout summary 不重复提交。

## 历史归档

其余既有目录记录早期 SpecVLA 基线、复杂 Draft、Action-RNN、树验证、TD-Fast、
RAES 和中间诊断实验。它们用于追溯研究历程，不代表当前推荐方法，也不能替代最新主表。

新增正式结果时，不要覆盖旧日志；应先更新结果整理器的显式注册表，再生成新的带日期归档。
