# VTPF-PacedHarmonic Goal e100 正式证据

本目录固化 2026-07-31 在 RTX 4090 上完成的 LIBERO-Goal 正式评测，以及决定该方案前使用的设计集、
留出集和正确性诊断。正式协议为 seed 7、每个 task 50 条初始状态、共 500 episodes；速度分母是同机、
同评测口径的 paper-wrapped OpenVLA AR。

## 正式结果

| 指标 | 数值 |
| --- | ---: |
| 成功率 | **373/500 = 0.746** |
| 各 task 成功数 | `[32, 44, 43, 23, 47, 38, 29, 48, 41, 28]` |
| 最后 task 的 mean step | **0.0524479 s** |
| 相对 AR `0.1827176 s` 的 Speedup | **3.4838x** |
| Length / Table-1 Length | 3.6144 |
| avg_accept_length | 0.9523 |
| draft overall_hit_rate | 0.3039 |
| target prefill 比例 | 0.4046 |
| hold 比例 | 0.5954 |

与相同 500 个初始状态上的 AR 配对比较，PacedHarmonic 为 `+0.4` 个百分点；两者不一致的轨迹为
`73` 条 PacedHarmonic 独赢、`71` 条 AR 独赢，精确 McNemar `p=0.9336`，配对 bootstrap 95% CI
为 `[-4.4, +5.2]` 个百分点。因此严谨结论是：**本次评测未观察到成功率下降，同时达到 3.484x**；
这不是统计意义上的等价性证明。

相比 VisualBudget `p=0.15`，PacedHarmonic 在保留 3x 以上速度的同时把成功率从 `0.672` 提高到
`0.746`。配对差值为 `+7.4` 个百分点，bootstrap 95% CI `[+2.8, +12.0]`，McNemar
`p=0.00215`。这支持“第二次 hold 的谐波缩放修复了完整陈旧增量重复造成的过冲”这一机制解释。

完整计算机可读结果在 `paired_analysis.json`；正式文本、summary 和逐动作 timing 位于 `formal/`。

## 方法身份

VTPF-PacedHarmonic 组合三个互补约束：

1. 每个 target 帧都允许上一动作作为 VTPF prefill 候选，但候选仍由当前 target 逐 token 校验和纠正。
2. target/hold 使用均匀的 `T-H-H, T-H` 节拍；第二次 hold 后产生 temporal debt，使下一周期最多一次 hold。
3. 第一次 hold 原幅执行；第二次 hold 的 6 个连续控制维度乘 `1/2`，gripper 保持离散值，随后强制 target。

它不启用视觉特征缓存、动作组 relaxed 校验、候选树或任务阶段规则。`p=0.15` 只作为第二次 hold 的
低频视觉漂移上限，不是安全证书。

实现 commit：`82cdffe35a64338ccbfd4c5e513e26883b1b646b`。

checkpoint：`epoch_100_step_044800`。

checkpoint SHA-256：

```text
pytorch_model.bin  f9975d3776f5c7e5f84b7caca5efdf53650e7a97524d7c9e9f463dc82f73f41a
dflash_config.json 10c03e7e0c081ee0f75aede461c1265ba64037a52d545024e2467f29ac1ca68f
```

## 正式复现命令

```bash
CUDA_VISIBLE_DEVICES=0 \
SPEC_CKPT=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/Draft_checkpoint/goal/epoch_100_step_044800 \
EVAL_EPOCH=100 \
NUM_TRIALS_PER_TASK=50 \
TRIAL_START_INDEX=0 \
SEED=7 \
TIMING_SCOPE=last_task \
SYNC_CUDA_TIMING=False \
LOG_DIR=/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data/eval_logs/paced_harmonic_formal \
RUN_ID_NOTE=dflash-vtpf-paced-harmonic-goal-e100-s7-formal \
bash openvla/specdecoding/decode-scripts/run_dflash_vtpf_paced_harmonic_goal_eval.sh
```

## 筛选与诊断边界

- `screening/age_decay_design/`：状态 0-4 上的谐波保持设计集。
- `screening/age_decay_holdout/`：同一方案在状态 5-9 上的留出复核，因未改善而被否决。
- `screening/paced_stable3_holdout/`：节拍 + 谐波但 `stable=3`，速度未稳定超过 3x。
- `screening/paced_stable1_holdout/`：唯一同时通过留出 SR 与 3x 门槛的候选。
- `screening/paced_no_harmonic_formal/`：500 条无谐波对照，`SR=0.706 / 3.194x`。
- `diagnostics/parity_stable1/` 与 `parity_stable3/`：单轨迹 target-AR 影子诊断，仅排除明显回归，
  不能证明整套 DFlash 与纯 AR 等价。
- `diagnostics/postcommit_smoke/`：commit `82cdffe` 当前 launcher 的 1-episode 冒烟测试，进程 `exit=0`。

正式 500 条运行启动时使用的是实现提交前的同一工作树版本；正式结果写盘后，运行中的 wrapper 曾被同步替换，
因此进程退出阶段出现一次 `unexpected EOF`。该错误发生在 summary/timing 全部写完之后，不影响 500 条轨迹；
随后当前 commit 的 post-commit smoke 已完整退出为 0。正式原始输出目录仍显示旧路由标签 `PacedBudget`，
只是当时 wrapper 的展示字段，当前 launcher 已改为 `PacedHarmonic`，不改变正式运行的策略参数。

## 环境

环境与硬件身份见 `ENVIRONMENT.md`。目录内所有文件的完整性由 `SHA256SUMS` 记录。
