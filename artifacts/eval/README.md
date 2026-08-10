# 评测产物索引

本目录保存可进入 Git 的紧凑评测证据。当前正式方法固定为：

```text
Minimal DFlash + VTPF + PacedHarmonic
```

## 当前规范结果

- `dual_anchor_main_20260809/`：当前最终方法的四子集 500-episode 冻结清单；H1/H2
  都由最近 Target 图像锚点约束；
- 服务器上的完整规范目录由
  `openvla/specdecoding/evidence/organize_paper_results.py` 构建；
- 当前论文完整方法数字以 schema-v2 `paced_harmonic_runs.json` 为准；基线与 strict
  数字继续由服务器规范目录的 `manifest/runs.csv` 提供。

## 历史归档

`paced_harmonic_main_20260808/` 是只约束 H2 的旧结果。其余既有目录记录早期 SpecVLA
基线、复杂 Draft、Action-RNN、树验证、TD-Fast、RAES 和中间诊断实验。它们仅用于追溯
研究历程，不代表当前推荐方法，也不能替代最新主表。

新增正式结果时，不要覆盖旧日志；应先更新结果整理器的显式注册表，再生成新的带日期归档。
