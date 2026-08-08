# 推理脚本索引

本目录同时包含正式入口、论文消融和历史研究脚本。文件仍存在不代表它属于当前方法。

当前正式方法固定为：

```text
Minimal DFlash + VTPF + PacedHarmonic
```

## 正式入口

| 脚本 | 用途 |
| --- | --- |
| `libero_eval_common.sh` | 四子集模型、LIBERO、日志和统一计时配置 |
| `run_dflash_goal_eval.sh` | DFlash strict/relaxed 底层单项入口；通常由上层脚本调用 |
| `run_dflash_minimal_suite_main_3way_eval.sh` | 同一 suite Draft 串行评测 DFlash strict、VTPF strict、完整 PacedHarmonic |
| `run_dflash_temporal_cascade_goal_eval.sh` | VTPF strict 的底层入口 |
| `run_dflash_vtpf_paced_harmonic_goal_eval.sh` | 当前完整时序方法 |
| `run_dflash_vtpf_paced_budget_goal_eval.sh` | PacedHarmonic 的 Pace 依赖入口 |
| `run_dflash_vtpf_adaptive_decimation_goal_eval.sh` | Paced/VisualBudget 共用的底层 Hold 入口 |
| `run_specvla_paper_ar_eval.sh` | OpenVLA AR 分母 |
| `run_specvla_eval.sh` | SpecVLA strict/relaxed 单项入口 |
| `run_specvla_main_table_eval.sh` | 四子集 SpecVLA 基线工作流 |

四子集正式评测示例：

```bash
TASK_SUITE_NAME=libero_goal \
SPEC_CKPT=/absolute/path/to/suite/checkpoint \
EVAL_EPOCH=100 NUM_TRIALS_PER_TASK=50 \
SEED=7 SYNC_CUDA_TIMING=False TIMING_SCOPE=last_task \
  bash openvla/specdecoding/decode-scripts/run_dflash_minimal_suite_main_3way_eval.sh
```

## Paced × Harmonic 消融

| 条件 | 入口 | 动作衰减 |
| --- | --- | --- |
| 无 Pace、无 Harmonic | `run_dflash_vtpf_visual_budget_goal_eval.sh` | `none` |
| 仅 Pace | `run_dflash_vtpf_paced_budget_goal_eval.sh` | `none` |
| 仅 Harmonic | `run_dflash_vtpf_age_decayed_goal_eval.sh` | `inverse_age` |
| 完整方法 | `run_dflash_vtpf_paced_harmonic_goal_eval.sh` | `inverse_age` |

`run_dflash_p0_temporal_2x2.sh` 是匹配 target 调用预算的机制实验，不等同于上述闭环 2×2。

## 其它有效消融

- `run_dflash_minimal_goal_3way_eval.sh`：旧 Minimal 三路入口，第三路是 TD-Fast，不是当前完整方法；
- `run_dflash_vtpf_visual_budget_goal_eval.sh`：aggressive 速度点；
- `run_dflash_vtpf_prefix_cert_goal_eval.sh`：短前缀认证；
- `run_dflash_p0_evidence.sh`：成本、时序持久性和 VTPF 因果审计；
- `run_dflash_p0_counterfactual.sh`：同状态恢复实验；
- `run_dflash_goal_repeat_eval.sh`：多 seed 重复评测。

## 历史入口

以下脚本只用于复现已放弃或未进入当前方法的分支：

- `run_dflash_action_rnn_goal_4way_eval.sh`
- `run_dflash_action_rnn_goal_pair_eval.sh`
- `run_dflash_temporal_prefill_tree_goal_eval.sh`
- `run_dflash_vtpf_temporal_decimation_goal_eval.sh`
- `run_dflash_vtpf_guarded_bypass_goal_eval.sh`
- `run_dflash_calibrated_suite_eval.sh`
- `run_dflash_calibrated_all_suites_eval.sh`
- `run_recoverability_calibration.sh`
- `run_recoverability_all_suites.sh`
- `run_contextual_reference_trace.sh`
- `run_dflash_verify_skip_shadow_goal.sh`

其中部分仍被证据构建器引用，不应在论文冻结前物理删除。当前方法不读取冻结风险 profile，也不运行
RAES/上下文风险选择器。

## 公共约束

正式结果必须固定：

```text
NUM_TRIALS_PER_TASK=50
TRIAL_START_INDEX=0
SEED=7
SYNC_CUDA_TIMING=False
TIMING_SCOPE=last_task
```

每个 suite 必须使用同 suite 的 OpenVLA 和 Draft。`libero_10` 的 SpecVLA relaxed 基线使用 `r=5`；
这与当前 DFlash token strict/PacedHarmonic 配置无关。
