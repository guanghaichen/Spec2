import math
import statistics
from collections import Counter


def parse_tree_calibration_positions(raw: str) -> list[int]:
    positions = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        position = int(item)
        if position not in {0, 2, 3, 4, 5}:
            raise ValueError("Tree calibration positions must be drawn from 0,2,3,4,5.")
        if position not in positions:
            positions.append(position)
    if 0 not in positions:
        positions.insert(0, 0)
    if len(positions) < 2:
        raise ValueError("Tree auto-calibration needs the off baseline and at least one fork.")
    return positions


def one_sided_sign_test_pvalue(paired_differences: list[float]) -> float:
    nonzero = [value for value in paired_differences if value != 0.0]
    if not nonzero:
        return 1.0
    faster_count = sum(value < 0.0 for value in nonzero)
    sample_count = len(nonzero)
    return sum(
        math.comb(sample_count, count) for count in range(faster_count, sample_count + 1)
    ) / (2 ** sample_count)


def select_tree_branch_position(
    times_by_position: dict[int, list[float]],
    triggered_blocks: dict[int, int],
) -> tuple[int, dict]:
    """Choose a fork only when paired full-action timings beat tree-off."""
    baseline_times = times_by_position[0]
    candidate_count = max(len(times_by_position) - 1, 1)
    corrected_alpha = 0.05 / candidate_count
    diagnostics = {
        "baseline_position": 0,
        "samples": len(baseline_times),
        "corrected_alpha": corrected_alpha,
        "candidates": {},
    }
    eligible = []
    for position, candidate_times in times_by_position.items():
        if position == 0:
            continue
        paired = [candidate - baseline for candidate, baseline in zip(candidate_times, baseline_times)]
        median_delta = statistics.median(paired) if paired else float("inf")
        pvalue = one_sided_sign_test_pvalue(paired)
        diagnostics["candidates"][str(position)] = {
            "median_seconds": statistics.median(candidate_times) if candidate_times else None,
            "median_delta_seconds": median_delta,
            "sign_test_pvalue": pvalue,
            "triggered_blocks": int(triggered_blocks.get(position, 0)),
        }
        if (
            paired
            and median_delta < 0.0
            and pvalue < corrected_alpha
            and triggered_blocks.get(position, 0) > 0
        ):
            eligible.append((median_delta, position))

    selected = min(eligible)[1] if eligible else 0
    diagnostics["selected_position"] = selected
    diagnostics["baseline_median_seconds"] = (
        statistics.median(baseline_times) if baseline_times else None
    )
    return selected, diagnostics


def _flatten_step_times(episode_times):
    times = []
    for episode in episode_times:
        for step in episode:
            if step is None:
                continue
            end_t, start_t = step
            times.append(float(end_t) - float(start_t))
    return times


def summarize_timing(episode_times):
    times = _flatten_step_times(episode_times)
    if not times:
        return {
            "steps": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "total": 0.0,
        }
    sorted_times = sorted(times)

    def percentile(p):
        if len(sorted_times) == 1:
            return sorted_times[0]
        rank = (len(sorted_times) - 1) * p / 100.0
        lower = int(rank)
        upper = min(lower + 1, len(sorted_times) - 1)
        weight = rank - lower
        return sorted_times[lower] * (1.0 - weight) + sorted_times[upper] * weight

    return {
        "steps": len(times),
        "mean": sum(times) / len(times),
        "median": statistics.median(times),
        "min": sorted_times[0],
        "max": sorted_times[-1],
        "p95": percentile(95),
        "p99": percentile(99),
        "total": sum(times),
    }


def summarize_generation_stats(step_stats_list):
    valid_stats = [item for item in step_stats_list if item]
    if not valid_stats:
        return None

    total_blocks = sum(int(item.get("num_blocks", 0)) for item in valid_stats)
    total_generated = sum(int(item.get("generated_tokens", 0)) for item in valid_stats)
    total_progressed = sum(
        int(item.get("progressed_tokens", item.get("generated_tokens", 0)))
        for item in valid_stats
    )

    accept_lengths = []
    main_path_accept_lengths = []
    progress_lengths = []
    for item in valid_stats:
        accept_lengths.extend(int(x) for x in item.get("accept_lengths", []))
        main_path_accept_lengths.extend(
            int(x) for x in item.get("main_path_accept_lengths", [])
        )
        progress_lengths.extend(int(x) for x in item.get("progress_lengths", []))

    total_accept_length = sum(accept_lengths)
    total_accepted = sum(int(item.get("accepted_tokens", 0)) for item in valid_stats)
    total_compared = sum(int(item.get("compared_tokens", 0)) for item in valid_stats)
    confidence_truncated_blocks = sum(
        int(item.get("confidence_truncated_blocks", 0)) for item in valid_stats
    )
    action_group_rescued_blocks = sum(
        int(item.get("action_group_rescued_blocks", 0)) for item in valid_stats
    )
    action_group_extra_accepted = sum(
        int(item.get("action_group_extra_accepted", 0)) for item in valid_stats
    )
    tree_triggered_blocks = sum(
        int(item.get("tree_triggered_blocks", 0)) for item in valid_stats
    )
    tree_selected_alternate_blocks = sum(
        int(item.get("tree_selected_alternate_blocks", 0)) for item in valid_stats
    )
    tree_extra_verified_nodes = sum(
        int(item.get("tree_extra_verified_nodes", 0)) for item in valid_stats
    )
    tree_extra_accepted = sum(
        int(item.get("tree_extra_accepted", 0)) for item in valid_stats
    )
    weighted_branch_score = sum(
        float(item.get("tree_mean_branch_score", 0.0) or 0.0)
        * int(item.get("tree_triggered_blocks", 0))
        for item in valid_stats
    )

    position_hits = {}
    position_counts = {}
    for item in valid_stats:
        for position_stat in item.get("per_position", []):
            pos = int(position_stat["position"])
            position_hits[pos] = position_hits.get(pos, 0) + int(position_stat.get("hit_count", 0))
            position_counts[pos] = position_counts.get(pos, 0) + int(position_stat.get("count", 0))

    per_position = []
    for pos in sorted(position_counts):
        count = position_counts[pos]
        hit = position_hits.get(pos, 0)
        reject = count - hit
        per_position.append(
            {
                "position": pos,
                "count": count,
                "hit_count": hit,
                "reject_count": reject,
                "hit_rate": (hit / count) if count > 0 else None,
                "reject_rate": (reject / count) if count > 0 else None,
            }
        )

    avg_progress_length = (total_progressed / total_blocks) if total_blocks > 0 else None
    avg_generated_length = (total_generated / total_blocks) if total_blocks > 0 else None
    avg_accept_length = (total_accept_length / total_blocks) if total_blocks > 0 else None
    # Paper-style speculative "Length" is the number of tokens actually
    # advanced by each verification block.  Some backends still keep
    # generated_tokens as the full draft/action width, so using it here would
    # incorrectly report a fixed action length such as 7 for SpecVLA.
    length = avg_progress_length if avg_progress_length is not None else avg_generated_length

    # 连续前缀条件命中率：第 k 项表示已经连续命中前 k-1 个 token 后，
    # 第 k 个仍被接受的条件概率。它比独立 per-position hit rate 更贴近真实 Length。
    conditional_prefix = []
    previous_survivors = total_blocks
    configured_max_position = max(
        (max(int(item.get("block_size", 1)) - 1, 0) for item in valid_stats),
        default=0,
    )
    max_prefix_position = max(
        max(accept_lengths, default=0),
        max(position_counts, default=0),
        configured_max_position,
    )
    for pos in range(1, max_prefix_position + 1):
        survivors = sum(int(length_value >= pos) for length_value in accept_lengths)
        conditional_prefix.append(
            {
                "position": pos,
                "eligible_blocks": previous_survivors,
                "survived_blocks": survivors,
                "conditional_hit_rate": (
                    survivors / previous_survivors if previous_survivors > 0 else None
                ),
            }
        )
        previous_survivors = survivors

    return {
        "backend": valid_stats[0].get("backend"),
        "use_causal_residual_sampling": valid_stats[0].get("use_causal_residual_sampling"),
        "action_head_type": valid_stats[0].get("action_head_type"),
        "acceptance_mode": valid_stats[0].get("acceptance_mode"),
        "tree_mode": valid_stats[0].get("tree_mode"),
        "tree_branch_position": valid_stats[-1].get("tree_branch_position"),
        "tree_first_anchor_only": valid_stats[0].get("tree_first_anchor_only"),
        "confidence_threshold": valid_stats[0].get("confidence_threshold"),
        "confidence_min_tokens": valid_stats[0].get("confidence_min_tokens"),
        "confidence_truncated_blocks": confidence_truncated_blocks,
        "num_steps": len(valid_stats),
        "num_blocks": total_blocks,
        "generated_tokens": total_generated,
        "progressed_tokens": total_progressed,
        "avg_generated_length": avg_generated_length,
        "length": length,
        "table1_length": length,
        "avg_progress_length": avg_progress_length,
        "avg_accept_length": avg_accept_length,
        "avg_main_path_accept_length": (
            sum(main_path_accept_lengths) / len(main_path_accept_lengths)
            if main_path_accept_lengths
            else None
        ),
        "accepted_tokens": total_accepted,
        "compared_tokens": total_compared,
        "overall_hit_rate": (total_accepted / total_compared) if total_compared > 0 else None,
        "action_group_rescued_blocks": action_group_rescued_blocks,
        "action_group_extra_accepted": action_group_extra_accepted,
        "tree_triggered_blocks": tree_triggered_blocks,
        "tree_selected_alternate_blocks": tree_selected_alternate_blocks,
        "tree_extra_verified_nodes": tree_extra_verified_nodes,
        "tree_extra_accepted": tree_extra_accepted,
        "tree_mean_branch_score": (
            weighted_branch_score / tree_triggered_blocks
            if tree_triggered_blocks > 0
            else None
        ),
        "accept_length_histogram": dict(sorted(Counter(accept_lengths).items())),
        "progress_length_histogram": dict(sorted(Counter(progress_lengths).items())),
        "per_position": per_position,
        "conditional_prefix": conditional_prefix,
    }


def write_eval_summary(
    path,
    *,
    cfg,
    run_id,
    eval_family,
    total_episodes,
    total_successes,
    episode_times,
    generation_stats=None,
):
    import json

    generation_summary = summarize_generation_stats(generation_stats or [])
    payload = {
        "run_id": run_id,
        "eval_family": eval_family,
        "task_suite_name": cfg.task_suite_name,
        "model_family": cfg.model_family,
        "draft_backend": getattr(cfg, "draft_backend", None),
        "use_spec": getattr(cfg, "use_spec", None),
        "parallel_draft": getattr(cfg, "parallel_draft", None),
        "ar_baseline": (
            "specvla_paper_wrapped_ar"
            if eval_family == "openvla_ar" and getattr(cfg, "use_spec", False)
            else None
        ),
        "accept_threshold": getattr(cfg, "accept_threshold", None),
        "dflash_block_size": getattr(cfg, "dflash_block_size", None),
        "dflash_num_draft_layers": getattr(cfg, "dflash_num_draft_layers", None),
        "dflash_use_causal_residual_sampling": getattr(
            cfg, "dflash_use_causal_residual_sampling", None
        ),
        "dflash_confidence_threshold": getattr(cfg, "dflash_confidence_threshold", None),
        "dflash_confidence_min_tokens": getattr(cfg, "dflash_confidence_min_tokens", None),
        "dflash_acceptance_mode": getattr(cfg, "dflash_acceptance_mode", None),
        "dflash_tree_mode": getattr(cfg, "dflash_tree_mode", None),
        "dflash_tree_branch_position": getattr(cfg, "dflash_tree_branch_position", None),
        "dflash_tree_first_anchor_only": getattr(cfg, "dflash_tree_first_anchor_only", None),
        "dflash_tree_auto_calibrate": getattr(cfg, "dflash_tree_auto_calibrate", None),
        "dflash_tree_calibration_result": getattr(
            cfg, "dflash_tree_calibration_result", None
        ),
        "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
        "spec_checkpoint": str(getattr(cfg, "spec_checkpoint", "")),
        "num_trials_per_task": cfg.num_trials_per_task,
        "seed": cfg.seed,
        "timing_scope": getattr(cfg, "timing_scope", None),
        "sync_cuda_timing": getattr(cfg, "sync_cuda_timing", None),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": (float(total_successes) / float(total_episodes)) if total_episodes else None,
        "timing": summarize_timing(episode_times),
        "generation": generation_summary,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def format_generation_summary(summary, prefix="Speculative stats"):
    if summary is None:
        return f"{prefix}: None"
    parts = [
        f"length={summary['length']:.3f}" if summary.get("length") is not None else "length=None",
        (
            f"avg_accept_length={summary['avg_accept_length']:.3f}"
            if summary.get("avg_accept_length") is not None
            else "avg_accept_length=None"
        ),
        f"num_blocks={summary.get('num_blocks', 0)}",
    ]
    if summary.get("overall_hit_rate") is not None:
        parts.append(f"overall_hit_rate={summary['overall_hit_rate']:.3f}")
    if summary.get("confidence_truncated_blocks"):
        parts.append(f"confidence_truncated={summary['confidence_truncated_blocks']}")
    if summary.get("action_group_rescued_blocks"):
        parts.append(f"group_rescued={summary['action_group_rescued_blocks']}")
        parts.append(f"group_extra_tokens={summary['action_group_extra_accepted']}")
    if summary.get("tree_triggered_blocks"):
        parts.append(f"tree_blocks={summary['tree_triggered_blocks']}")
        parts.append(f"tree_alt_wins={summary['tree_selected_alternate_blocks']}")
        parts.append(f"tree_extra_tokens={summary['tree_extra_accepted']}")
        if summary.get("avg_main_path_accept_length") is not None:
            parts.append(f"main_accept={summary['avg_main_path_accept_length']:.3f}")
    return f"{prefix}: " + ", ".join(parts)


def format_conditional_prefix(summary, prefix="Conditional prefix hit rate"):
    if summary is None or not summary.get("conditional_prefix"):
        return None
    values = []
    for item in summary["conditional_prefix"]:
        rate = item.get("conditional_hit_rate")
        values.append(
            f"p{item['position']}={rate:.3f}" if rate is not None else f"p{item['position']}=None"
        )
    return f"{prefix}: " + ", ".join(values)
