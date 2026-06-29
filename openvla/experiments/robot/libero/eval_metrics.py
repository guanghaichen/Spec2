import statistics
from collections import Counter


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
    progress_lengths = []
    for item in valid_stats:
        accept_lengths.extend(int(x) for x in item.get("accept_lengths", []))
        progress_lengths.extend(int(x) for x in item.get("progress_lengths", []))

    total_accept_length = sum(accept_lengths)
    total_accepted = sum(int(item.get("accepted_tokens", 0)) for item in valid_stats)
    total_compared = sum(int(item.get("compared_tokens", 0)) for item in valid_stats)

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

    length = (total_generated / total_blocks) if total_blocks > 0 else None
    avg_progress_length = (total_progressed / total_blocks) if total_blocks > 0 else None
    avg_accept_length = (total_accept_length / total_blocks) if total_blocks > 0 else None

    return {
        "backend": valid_stats[0].get("backend"),
        "num_steps": len(valid_stats),
        "num_blocks": total_blocks,
        "generated_tokens": total_generated,
        "progressed_tokens": total_progressed,
        "length": length,
        "table1_length": length,
        "avg_progress_length": avg_progress_length,
        "avg_accept_length": avg_accept_length,
        "accepted_tokens": total_accepted,
        "compared_tokens": total_compared,
        "overall_hit_rate": (total_accepted / total_compared) if total_compared > 0 else None,
        "accept_length_histogram": dict(sorted(Counter(accept_lengths).items())),
        "progress_length_histogram": dict(sorted(Counter(progress_lengths).items())),
        "per_position": per_position,
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
        "accept_threshold": getattr(cfg, "accept_threshold", None),
        "dflash_block_size": getattr(cfg, "dflash_block_size", None),
        "dflash_num_draft_layers": getattr(cfg, "dflash_num_draft_layers", None),
        "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
        "spec_checkpoint": str(getattr(cfg, "spec_checkpoint", "")),
        "num_trials_per_task": cfg.num_trials_per_task,
        "seed": cfg.seed,
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
    return f"{prefix}: " + ", ".join(parts)
