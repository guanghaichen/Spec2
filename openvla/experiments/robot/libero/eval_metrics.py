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
    total_verified_accepted = sum(
        int(item.get("verified_accepted_tokens", item.get("accepted_tokens", 0)))
        for item in valid_stats
    )
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
    weighted_verified_nodes = sum(
        float(item.get("tree_average_verified_nodes", 0.0) or 0.0)
        * int(item.get("tree_triggered_blocks", 0))
        for item in valid_stats
    )
    weighted_max_depth = sum(
        float(item.get("tree_average_max_depth", 0.0) or 0.0)
        * int(item.get("tree_triggered_blocks", 0))
        for item in valid_stats
    )
    target_logit_shadow_checks = sum(
        int(item.get("target_logit_shadow_checks", 0)) for item in valid_stats
    )
    target_logit_shadow_mismatches = sum(
        int(item.get("target_logit_shadow_mismatches", 0)) for item in valid_stats
    )
    verify_skipped_blocks = sum(
        int(item.get("verify_skipped_blocks", 0)) for item in valid_stats
    )
    verify_skipped_tokens = sum(
        int(item.get("verify_skipped_tokens", 0)) for item in valid_stats
    )
    verify_skipped_actions = sum(
        int(item.get("verify_skipped_actions", 0)) for item in valid_stats
    )
    first_token_early_reject_blocks = sum(
        int(item.get("first_token_early_reject_blocks", 0))
        for item in valid_stats
    )
    temporal_proposal_routed_actions = sum(
        int(item.get("temporal_proposal_routed_actions", 0))
        for item in valid_stats
    )
    temporal_proposal_routed_blocks = sum(
        int(item.get("temporal_proposal_routed_blocks", 0))
        for item in valid_stats
    )
    temporal_proposal_rejected_blocks = sum(
        int(item.get("temporal_proposal_rejected_blocks", 0))
        for item in valid_stats
    )
    temporal_fallback_draft_blocks = sum(
        int(item.get("temporal_fallback_draft_blocks", 0))
        for item in valid_stats
    )
    temporal_fused_verify_blocks = sum(
        int(item.get("temporal_fused_verify_blocks", 0))
        for item in valid_stats
    )
    temporal_prefill_fusion_records = [
        item["temporal_prefill_fusion_record"]
        for item in valid_stats
        if item.get("temporal_prefill_fusion_record") is not None
    ]
    temporal_prefill_accept_lengths = [
        int(record.get("accept_length", 0))
        for record in temporal_prefill_fusion_records
    ]
    temporal_prefill_tree_records = [
        record
        for record in temporal_prefill_fusion_records
        if record.get("mode") == "temporal_tree"
    ]
    temporal_prefix_cert_records = [
        record
        for record in temporal_prefill_fusion_records
        if record.get("mode") == "prefix_cert"
    ]
    temporal_prefill_bypass_records = [
        item["temporal_prefill_bypass_record"]
        for item in valid_stats
        if item.get("temporal_prefill_bypass_record") is not None
    ]
    temporal_prefill_bypass_pixel_l2 = [
        float(record["pixel_relative_l2"])
        for record in temporal_prefill_bypass_records
        if record.get("pixel_relative_l2") is not None
    ]
    temporal_hold_decision_records = [
        item["temporal_hold_decision_record"]
        for item in valid_stats
        if item.get("temporal_hold_decision_record") is not None
    ]
    temporal_hold_allowed_records = [
        record
        for record in temporal_hold_decision_records
        if bool(record.get("allow", False))
    ]
    temporal_hold_extension_candidates = [
        record
        for record in temporal_hold_decision_records
        if int(record.get("consecutive_holds_before", 0)) == 1
    ]
    temporal_hold_extended_records = [
        record
        for record in temporal_hold_allowed_records
        if bool(record.get("adaptive_extension", False))
    ]
    temporal_hold_anchor_pixel_l2 = [
        float(record["anchor_pixel_relative_l2"])
        for record in temporal_hold_decision_records
        if record.get("anchor_pixel_relative_l2") is not None
    ]

    stage_profile_totals = {}
    stage_profile_calls = {}
    for item in valid_stats:
        for stage_name, stage_values in (item.get("stage_profile") or {}).items():
            stage_profile_totals[stage_name] = stage_profile_totals.get(stage_name, 0.0) + float(
                stage_values.get("total_ms", 0.0)
            )
            stage_profile_calls[stage_name] = stage_profile_calls.get(stage_name, 0) + int(
                stage_values.get("calls", 0)
            )
    stage_profile = {
        stage_name: {
            "total_ms": total_ms,
            "calls": stage_profile_calls[stage_name],
            "mean_ms": (
                total_ms / stage_profile_calls[stage_name]
                if stage_profile_calls[stage_name] > 0
                else None
            ),
        }
        for stage_name, total_ms in stage_profile_totals.items()
    }

    vtpf_parity_records = [
        item["vtpf_parity_record"]
        for item in valid_stats
        if item.get("vtpf_parity_record") is not None
    ]
    vtpf_parity_positions = [
        position
        for record in vtpf_parity_records
        for position in record.get("per_position", [])
    ]
    vtpf_parity = None
    if vtpf_parity_records:
        comparable_positions = sum(
            int(record.get("causal_comparable_positions", 0))
            for record in vtpf_parity_records
        )
        vtpf_parity = {
            "num_records": len(vtpf_parity_records),
            "causal_comparable_positions": comparable_positions,
            "top1_mismatches": sum(
                int(record.get("top1_mismatches", 0))
                for record in vtpf_parity_records
            ),
            "serial_divergent_accepted_tokens": sum(
                int(record.get("serial_divergent_accepted_tokens", 0))
                for record in vtpf_parity_records
            ),
            "strict_accepted_tokens": sum(
                int(record.get("strict_accepted_prefix_length", 0))
                for record in vtpf_parity_records
            ),
            "max_abs_logit_difference": max(
                float(record.get("max_abs_logit_difference", 0.0))
                for record in vtpf_parity_records
            ),
            "mean_abs_logit_difference": (
                sum(
                    float(position.get("mean_abs_logit_difference", 0.0))
                    for position in vtpf_parity_positions
                )
                / len(vtpf_parity_positions)
                if vtpf_parity_positions
                else None
            ),
            "records": vtpf_parity_records,
        }

    verify_skip_records = []
    for item in valid_stats:
        verify_skip_records.extend(item.get("verify_skip_records", []))

    def percentile(values, fraction):
        values = sorted(float(value) for value in values)
        if not values:
            return None
        rank = (len(values) - 1) * float(fraction)
        lower = int(rank)
        upper = min(lower + 1, len(values) - 1)
        weight = rank - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    labeled_skip_records = [
        record for record in verify_skip_records if record.get("full_exact_match") is not None
    ]

    def gate_result(name, predicate, records=None):
        source_records = labeled_skip_records if records is None else records
        selected = [record for record in source_records if predicate(record)]
        exact = sum(bool(record["full_exact_match"]) for record in selected)
        result = {
            "name": name,
            "selected_blocks": len(selected),
            "coverage": (
                len(selected) / len(source_records) if source_records else None
            ),
            "full_exact_blocks": exact,
            "full_exact_precision": (exact / len(selected)) if selected else None,
            "unsafe_blocks": len(selected) - exact,
            "mean_exact_accept_length": (
                sum(int(record["exact_accept_length"]) for record in selected) / len(selected)
                if selected
                else None
            ),
        }
        for threshold in (3, 5, 9):
            label_name = f"full_action_group_match_r{threshold}"
            relaxed_selected = [
                record for record in selected if record.get(label_name) is not None
            ]
            relaxed_matches = sum(
                bool(record[label_name]) for record in relaxed_selected
            )
            result[f"action_group_r{threshold}_precision"] = (
                relaxed_matches / len(relaxed_selected) if relaxed_selected else None
            )
        return result

    verify_skip_shadow = None
    if verify_skip_records:
        gate_summaries = [
            gate_result("configured_gate", lambda record: bool(record.get("gate_selected"))),
            gate_result("sentinel_exact", lambda record: bool(record.get("sentinel_match"))),
            gate_result(
                "sentinel+base_agreement_1.0",
                lambda record: bool(record.get("sentinel_match"))
                and float(record.get("base_agreement", 0.0)) >= 1.0 - 1e-6,
            ),
        ]
        for threshold in (0.25, 0.5, 0.7, 0.8, 0.9):
            gate_summaries.append(
                gate_result(
                    f"sentinel+min_top1_prob_{threshold:.2f}",
                    lambda record, threshold=threshold: bool(record.get("sentinel_match"))
                    and float(record.get("min_top1_prob", 0.0)) >= threshold,
                )
            )
        temporal_gate_specs = (
            ("temporal_cos_0.990_delta_9", 0.990, 9.0, 0.0),
            ("temporal_cos_0.995_delta_9", 0.995, 9.0, 0.0),
            ("temporal_cos_0.995_delta_5", 0.995, 5.0, 0.0),
            ("temporal_cos_0.995_delta_9_prob_0.5", 0.995, 9.0, 0.5),
        )
        for name, min_cosine, max_delta, min_probability in temporal_gate_specs:
            gate_summaries.append(
                gate_result(
                    name,
                    lambda record, min_cosine=min_cosine, max_delta=max_delta, min_probability=min_probability: (
                        bool(record.get("sentinel_match"))
                        and record.get("temporal_hidden_cosine") is not None
                        and float(record["temporal_hidden_cosine"]) >= min_cosine
                        and record.get("previous_action_max_token_delta") is not None
                        and float(record["previous_action_max_token_delta"]) <= max_delta
                        and float(record.get("min_top1_prob", 0.0)) >= min_probability
                    ),
                )
            )
        feature_names = (
            "min_top1_prob",
            "mean_top1_prob",
            "min_margin",
            "max_normalized_entropy",
            "base_agreement",
            "temporal_hidden_cosine",
            "previous_action_max_token_delta",
            "previous_action_mean_token_delta",
            "target_tail_max_token_delta",
            "target_tail_mean_token_delta",
        )
        by_q_len = {}
        for q_len in sorted({int(record["q_len"]) for record in labeled_skip_records}):
            q_records = [
                record for record in labeled_skip_records if int(record["q_len"]) == q_len
            ]
            q_gates = [
                gate_result(
                    "sentinel_exact",
                    lambda record: bool(record.get("sentinel_match")),
                    q_records,
                )
            ]
            for threshold in (0.5, 0.7, 0.9):
                q_gates.append(
                    gate_result(
                        f"sentinel+min_top1_prob_{threshold:.2f}",
                        lambda record, threshold=threshold: bool(record.get("sentinel_match"))
                        and float(record.get("min_top1_prob", 0.0)) >= threshold,
                        q_records,
                    )
                )
            by_q_len[str(q_len)] = {
                "num_records": len(q_records),
                "full_exact_base_rate": (
                    sum(bool(record["full_exact_match"]) for record in q_records)
                    / len(q_records)
                    if q_records
                    else None
                ),
                "action_group_base_rates": {
                    f"r{threshold}": (
                        sum(
                            bool(record[f"full_action_group_match_r{threshold}"])
                            for record in q_records
                            if record.get(f"full_action_group_match_r{threshold}") is not None
                        )
                        / sum(
                            record.get(f"full_action_group_match_r{threshold}") is not None
                            for record in q_records
                        )
                        if any(
                            record.get(f"full_action_group_match_r{threshold}") is not None
                            for record in q_records
                        )
                        else None
                    )
                    for threshold in (3, 5, 9)
                },
                "gates": q_gates,
            }
        verify_skip_shadow = {
            "mode": valid_stats[0].get("verify_skip_mode"),
            "num_records": len(verify_skip_records),
            "num_labeled_records": len(labeled_skip_records),
            "full_exact_base_rate": (
                sum(bool(record["full_exact_match"]) for record in labeled_skip_records)
                / len(labeled_skip_records)
                if labeled_skip_records
                else None
            ),
            "feature_quantiles": {
                name: {
                    "p10": percentile(
                        [
                            record[name]
                            for record in labeled_skip_records
                            if record.get(name) is not None
                        ],
                        0.10,
                    ),
                    "p50": percentile(
                        [
                            record[name]
                            for record in labeled_skip_records
                            if record.get(name) is not None
                        ],
                        0.50,
                    ),
                    "p90": percentile(
                        [
                            record[name]
                            for record in labeled_skip_records
                            if record.get(name) is not None
                        ],
                        0.90,
                    ),
                }
                for name in feature_names
            },
            "gates": gate_summaries,
            "by_q_len": by_q_len,
        }

    temporal_action_records = [
        item["temporal_action_skip_record"]
        for item in valid_stats
        if item.get("temporal_action_skip_record") is not None
    ]
    labeled_temporal_records = [
        record
        for record in temporal_action_records
        if record.get("full_exact_match") is not None
    ]

    def temporal_gate_result(name, predicate):
        selected = [record for record in labeled_temporal_records if predicate(record)]
        result = {
            "name": name,
            "selected_actions": len(selected),
            "coverage": (
                len(selected) / len(labeled_temporal_records)
                if labeled_temporal_records
                else None
            ),
            "full_exact_precision": (
                sum(bool(record["full_exact_match"]) for record in selected) / len(selected)
                if selected
                else None
            ),
            "mean_tail_exact_accept_length": (
                sum(int(record.get("tail_exact_accept_length", 0)) for record in selected)
                / len(selected)
                if selected
                else None
            ),
        }
        for threshold in (3, 5, 9):
            label_name = f"full_action_group_match_r{threshold}"
            result[f"action_group_r{threshold}_precision"] = (
                sum(bool(record[label_name]) for record in selected) / len(selected)
                if selected
                else None
            )
            result[f"mean_tail_action_group_accept_length_r{threshold}"] = (
                sum(
                    int(
                        record.get(
                            f"tail_action_group_accept_length_r{threshold}", 0
                        )
                    )
                    for record in selected
                )
                / len(selected)
                if selected
                else None
            )
        return result

    temporal_action_skip = None
    if temporal_action_records:
        temporal_feature_names = (
            "prompt_temporal_cosine",
            "prompt_temporal_min_layer_cosine",
            "prompt_temporal_mean_layer_cosine",
            "prompt_temporal_relative_l2",
            "prompt_pooled_min_layer_cosine",
            "prompt_pooled_mean_layer_cosine",
            "prompt_pooled_relative_l2",
            "first_action_prob_cosine",
            "first_action_prob_total_variation",
            "pixel_temporal_cosine",
            "pixel_temporal_relative_l2",
        )

        def temporal_feature_quantiles(records):
            return {
                name: {
                    "p01": percentile(
                        [record[name] for record in records if record.get(name) is not None],
                        0.01,
                    ),
                    "p10": percentile(
                        [record[name] for record in records if record.get(name) is not None],
                        0.10,
                    ),
                    "p50": percentile(
                        [record[name] for record in records if record.get(name) is not None],
                        0.50,
                    ),
                    "p90": percentile(
                        [record[name] for record in records if record.get(name) is not None],
                        0.90,
                    ),
                    "p99": percentile(
                        [record[name] for record in records if record.get(name) is not None],
                        0.99,
                    ),
                }
                for name in temporal_feature_names
            }

        configured_cosine = float(
            valid_stats[0].get("verify_skip_min_temporal_cosine", 1.0)
        )
        configured_stable_actions = int(
            valid_stats[0].get("verify_skip_min_stable_actions", 4)
        )
        temporal_gates = [
            temporal_gate_result(
                "configured_route_gate",
                lambda record: bool(record.get("first_token_matches_previous"))
                and record.get("prompt_temporal_cosine") is not None
                and float(record["prompt_temporal_cosine"])
                >= float(valid_stats[0].get("temporal_route_min_cosine", 1.0)),
            ),
            temporal_gate_result(
                "configured_gate",
                lambda record: bool(record.get("first_token_matches_previous"))
                and record.get("prompt_temporal_cosine") is not None
                and float(record["prompt_temporal_cosine"]) >= configured_cosine
                and int(record.get("previous_verified_action_run_length", 0))
                >= configured_stable_actions,
            )
        ]
        for threshold in (0.990, 0.995, 0.997, 0.998, 0.999):
            temporal_gates.append(
                temporal_gate_result(
                    f"t0_exact+prompt_cos_{threshold:.3f}",
                    lambda record, threshold=threshold: bool(
                        record.get("first_token_matches_previous")
                    )
                    and record.get("prompt_temporal_cosine") is not None
                    and float(record["prompt_temporal_cosine"]) >= threshold,
                )
            )
            temporal_gates.append(
                temporal_gate_result(
                    f"stable4+t0_exact+prompt_cos_{threshold:.3f}",
                    lambda record, threshold=threshold: bool(
                        record.get("first_token_matches_previous")
                    )
                    and int(
                        record.get("previous_verified_action_run_length", 0)
                    )
                    >= 4
                    and record.get("prompt_temporal_cosine") is not None
                    and float(record["prompt_temporal_cosine"]) >= threshold,
                )
            )
        temporal_action_skip = {
            "num_records": len(temporal_action_records),
            "num_labeled_records": len(labeled_temporal_records),
            "prompt_cosine_quantiles": {
                "p10": percentile(
                    [
                        record["prompt_temporal_cosine"]
                        for record in labeled_temporal_records
                        if record.get("prompt_temporal_cosine") is not None
                    ],
                    0.10,
                ),
                "p50": percentile(
                    [
                        record["prompt_temporal_cosine"]
                        for record in labeled_temporal_records
                        if record.get("prompt_temporal_cosine") is not None
                    ],
                    0.50,
                ),
                "p90": percentile(
                    [
                        record["prompt_temporal_cosine"]
                        for record in labeled_temporal_records
                        if record.get("prompt_temporal_cosine") is not None
                    ],
                    0.90,
                ),
            },
            "previous_action_base_rates": {
                "exact": (
                    sum(
                        bool(record["full_exact_match"])
                        for record in labeled_temporal_records
                    )
                    / len(labeled_temporal_records)
                    if labeled_temporal_records
                    else None
                ),
                **{
                    f"action_group_r{threshold}": (
                        sum(
                            bool(record[f"full_action_group_match_r{threshold}"])
                            for record in labeled_temporal_records
                        )
                        / len(labeled_temporal_records)
                        if labeled_temporal_records
                        else None
                    )
                    for threshold in (3, 5, 9)
                },
            },
            "feature_quantiles": {
                "all": temporal_feature_quantiles(labeled_temporal_records),
                "exact": temporal_feature_quantiles(
                    [
                        record
                        for record in labeled_temporal_records
                        if bool(record["full_exact_match"])
                    ]
                ),
                "mismatch": temporal_feature_quantiles(
                    [
                        record
                        for record in labeled_temporal_records
                        if not bool(record["full_exact_match"])
                    ]
                ),
            },
            "gates": temporal_gates,
            # Shadow mode is explicitly diagnostic. Keeping its compact rows in
            # the summary enables threshold sweeps without rerunning LIBERO.
            "records": (
                temporal_action_records
                if (
                    valid_stats[0].get("verify_skip_mode") == "shadow"
                    or bool(valid_stats[0].get("debug_compare_target_ar"))
                )
                else None
            ),
        }

    action_token_trace = [
        item["final_action_tokens"]
        for item in valid_stats
        if item.get("final_action_tokens") is not None
    ]
    target_ar_reference_trace = [
        item["target_ar_reference_tokens"]
        for item in valid_stats
        if item.get("target_ar_reference_tokens") is not None
    ]
    evidence_trace = [
        item["evidence_trace"]
        for item in valid_stats
        if item.get("evidence_trace") is not None
    ]

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
        "tree_budget": valid_stats[0].get("tree_budget"),
        "target_logits_mode": valid_stats[0].get("target_logits_mode"),
        "target_logit_shadow_checks": target_logit_shadow_checks,
        "target_logit_shadow_mismatches": target_logit_shadow_mismatches,
        "target_logit_shadow_mismatch_rate": (
            target_logit_shadow_mismatches / target_logit_shadow_checks
            if target_logit_shadow_checks > 0
            else None
        ),
        "verify_skip_mode": valid_stats[0].get("verify_skip_mode"),
        "verify_skipped_blocks": verify_skipped_blocks,
        "verify_skipped_tokens": verify_skipped_tokens,
        "verify_skipped_actions": verify_skipped_actions,
        "first_token_early_reject_blocks": first_token_early_reject_blocks,
        "temporal_proposal_routed_actions": temporal_proposal_routed_actions,
        "temporal_proposal_routed_blocks": temporal_proposal_routed_blocks,
        "temporal_proposal_rejected_blocks": temporal_proposal_rejected_blocks,
        "temporal_fallback_draft_blocks": temporal_fallback_draft_blocks,
        "temporal_fused_verify_blocks": temporal_fused_verify_blocks,
        "temporal_prefill_fused_actions": len(temporal_prefill_fusion_records),
        "temporal_prefill_full_match_actions": sum(
            int(bool(record.get("full_match", False)))
            for record in temporal_prefill_fusion_records
        ),
        "temporal_prefill_avg_accept_length": (
            sum(temporal_prefill_accept_lengths)
            / len(temporal_prefill_accept_lengths)
            if temporal_prefill_accept_lengths
            else None
        ),
        "temporal_prefill_accept_histogram": dict(
            sorted(Counter(temporal_prefill_accept_lengths).items())
        ),
        "temporal_prefix_cert_attempts": len(temporal_prefix_cert_records),
        "temporal_prefix_cert_successes": sum(
            int(bool(record.get("prefix_certified", False)))
            for record in temporal_prefix_cert_records
        ),
        "temporal_prefix_cert_fallbacks": sum(
            int(not bool(record.get("prefix_certified", False)))
            for record in temporal_prefix_cert_records
        ),
        "temporal_prefix_cert_trusted_tokens": sum(
            int(record.get("trusted_suffix_length", 0))
            for record in temporal_prefix_cert_records
        ),
        "temporal_prefix_cert_avg_verified_tokens": (
            sum(
                int(record.get("compared_length", 0))
                for record in temporal_prefix_cert_records
            )
            / len(temporal_prefix_cert_records)
            if temporal_prefix_cert_records
            else None
        ),
        "temporal_prefill_bypassed_actions": len(
            temporal_prefill_bypass_records
        ),
        "temporal_prefill_bypass_avg_pixel_relative_l2": (
            sum(temporal_prefill_bypass_pixel_l2)
            / len(temporal_prefill_bypass_pixel_l2)
            if temporal_prefill_bypass_pixel_l2
            else None
        ),
        "temporal_hold": {
            "policy": valid_stats[0].get("temporal_hold_policy", "fixed"),
            "num_decisions": len(temporal_hold_decision_records),
            "allowed_holds": len(temporal_hold_allowed_records),
            "hold_rate": (
                len(temporal_hold_allowed_records)
                / len(temporal_hold_decision_records)
                if temporal_hold_decision_records
                else None
            ),
            "target_prefill_actions": (
                len(temporal_hold_decision_records)
                - len(temporal_hold_allowed_records)
            ),
            "target_prefill_rate": (
                (
                    len(temporal_hold_decision_records)
                    - len(temporal_hold_allowed_records)
                )
                / len(temporal_hold_decision_records)
                if temporal_hold_decision_records
                else None
            ),
            "base_holds": sum(
                int(record.get("hold_depth", 0) == 1)
                for record in temporal_hold_allowed_records
            ),
            "adaptive_extension_candidates": len(
                temporal_hold_extension_candidates
            ),
            "adaptive_extended_holds": len(temporal_hold_extended_records),
            "adaptive_extension_rate": (
                len(temporal_hold_extended_records)
                / len(temporal_hold_extension_candidates)
                if temporal_hold_extension_candidates
                else None
            ),
            # Policy-neutral names for Visual Budget and future schedulers.
            # Keep the adaptive_* aliases above so historical analysis scripts
            # remain readable without a migration.
            "extension_candidates": len(temporal_hold_extension_candidates),
            "extended_holds": len(temporal_hold_extended_records),
            "extension_rate": (
                len(temporal_hold_extended_records)
                / len(temporal_hold_extension_candidates)
                if temporal_hold_extension_candidates
                else None
            ),
            "forced_target_after_hold": sum(
                int(
                    not bool(record.get("allow", False))
                    and int(record.get("consecutive_holds_before", 0)) > 0
                )
                for record in temporal_hold_decision_records
            ),
            "extension_debt_forced_targets": sum(
                int(record.get("reason") == "extension_debt")
                for record in temporal_hold_decision_records
            ),
            "executed_action_scale_histogram": dict(
                sorted(
                    Counter(
                        f"{float(record['executed_continuous_action_scale']):.6g}"
                        for record in temporal_hold_allowed_records
                        if record.get("executed_continuous_action_scale") is not None
                    ).items()
                )
            ),
            "allowed_depth_histogram": dict(
                sorted(
                    Counter(
                        int(record.get("hold_depth", 0))
                        for record in temporal_hold_allowed_records
                    ).items()
                )
            ),
            "reason_histogram": dict(
                sorted(
                    Counter(
                        str(record.get("reason", "unknown"))
                        for record in temporal_hold_decision_records
                    ).items()
                )
            ),
            "avg_anchor_pixel_relative_l2": (
                sum(temporal_hold_anchor_pixel_l2)
                / len(temporal_hold_anchor_pixel_l2)
                if temporal_hold_anchor_pixel_l2
                else None
            ),
            "records": temporal_hold_decision_records,
        },
        "temporal_prefill_tree_actions": len(temporal_prefill_tree_records),
        "temporal_prefill_tree_full_exact_actions": sum(
            int(bool(record.get("full_exact_match", False)))
            for record in temporal_prefill_tree_records
        ),
        "temporal_prefill_tree_selected_alternate_actions": sum(
            int(bool(record.get("selected_alternate", False)))
            for record in temporal_prefill_tree_records
        ),
        "temporal_prefill_tree_extra_accepted": sum(
            int(record.get("extra_accepted_over_hold", 0))
            for record in temporal_prefill_tree_records
        ),
        "temporal_prefill_tree_avg_candidates": (
            sum(int(record.get("candidate_count", 0)) for record in temporal_prefill_tree_records)
            / len(temporal_prefill_tree_records)
            if temporal_prefill_tree_records
            else None
        ),
        "temporal_prefill_tree_avg_verified_nodes": (
            sum(
                int(record.get("verified_node_count", 0))
                for record in temporal_prefill_tree_records
            )
            / len(temporal_prefill_tree_records)
            if temporal_prefill_tree_records
            else None
        ),
        "temporal_prefill_fusion_records": temporal_prefill_fusion_records,
        "verify_skip_shadow": verify_skip_shadow,
        "temporal_action_skip": temporal_action_skip,
        "action_token_trace": action_token_trace or None,
        "target_ar_reference_trace": target_ar_reference_trace or None,
        "evidence_trace": evidence_trace or None,
        "stage_profile": stage_profile or None,
        "vtpf_parity": vtpf_parity,
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
        "verified_accepted_tokens": total_verified_accepted,
        "overall_hit_rate": (
            total_verified_accepted / total_compared if total_compared > 0 else None
        ),
        "action_group_rescued_blocks": action_group_rescued_blocks,
        "action_group_extra_accepted": action_group_extra_accepted,
        "tree_triggered_blocks": tree_triggered_blocks,
        "tree_selected_alternate_blocks": tree_selected_alternate_blocks,
        "tree_extra_verified_nodes": tree_extra_verified_nodes,
        "tree_extra_accepted": tree_extra_accepted,
        "tree_average_verified_nodes": (
            weighted_verified_nodes / tree_triggered_blocks
            if tree_triggered_blocks > 0
            else None
        ),
        "tree_average_max_depth": (
            weighted_max_depth / tree_triggered_blocks
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
    evidence_trace=None,
):
    import json

    generation_summary = summarize_generation_stats(generation_stats or []) or {}
    if evidence_trace:
        generation_summary["evidence_trace"] = evidence_trace
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
        "dflash_tree_budget": getattr(cfg, "dflash_tree_budget", None),
        "dflash_target_logits_mode": getattr(cfg, "dflash_target_logits_mode", None),
        "dflash_verify_skip_mode": getattr(cfg, "dflash_verify_skip_mode", None),
        "dflash_verify_skip_min_top1_prob": getattr(
            cfg, "dflash_verify_skip_min_top1_prob", None
        ),
        "dflash_verify_skip_min_margin": getattr(
            cfg, "dflash_verify_skip_min_margin", None
        ),
        "dflash_verify_skip_min_base_agreement": getattr(
            cfg, "dflash_verify_skip_min_base_agreement", None
        ),
        "dflash_temporal_route_min_cosine": getattr(
            cfg, "dflash_temporal_route_min_cosine", None
        ),
        "dflash_temporal_route_stop_on_reject": getattr(
            cfg, "dflash_temporal_route_stop_on_reject", None
        ),
        "dflash_temporal_fuse_verify": getattr(
            cfg, "dflash_temporal_fuse_verify", None
        ),
        "dflash_temporal_prefill_fusion": getattr(
            cfg, "dflash_temporal_prefill_fusion", None
        ),
        "dflash_temporal_prefill_min_stable_actions": getattr(
            cfg, "dflash_temporal_prefill_min_stable_actions", None
        ),
        "dflash_temporal_prefix_cert_tokens": getattr(
            cfg, "dflash_temporal_prefix_cert_tokens", None
        ),
        "dflash_temporal_bypass_max_pixel_relative_l2": getattr(
            cfg, "dflash_temporal_bypass_max_pixel_relative_l2", None
        ),
        "dflash_temporal_bypass_use_pixel_guard": getattr(
            cfg, "dflash_temporal_bypass_use_pixel_guard", None
        ),
        "dflash_temporal_hold_policy": getattr(
            cfg, "dflash_temporal_hold_policy", None
        ),
        "dflash_temporal_adaptive_min_verified_run": getattr(
            cfg, "dflash_temporal_adaptive_min_verified_run", None
        ),
        "dflash_temporal_adaptive_max_anchor_pixel_relative_l2": getattr(
            cfg,
            "dflash_temporal_adaptive_max_anchor_pixel_relative_l2",
            None,
        ),
        "dflash_temporal_hold_action_decay": getattr(
            cfg, "dflash_temporal_hold_action_decay", None
        ),
        "dflash_temporal_prefill_tree": getattr(
            cfg, "dflash_temporal_prefill_tree", None
        ),
        "dflash_temporal_prefill_tree_max_candidates": getattr(
            cfg, "dflash_temporal_prefill_tree_max_candidates", None
        ),
        "dflash_temporal_prefill_tree_min_history": getattr(
            cfg, "dflash_temporal_prefill_tree_min_history", None
        ),
        "dflash_verify_skip_min_temporal_cosine": getattr(
            cfg, "dflash_verify_skip_min_temporal_cosine", None
        ),
        "dflash_verify_skip_min_stable_actions": getattr(
            cfg, "dflash_verify_skip_min_stable_actions", None
        ),
        "dflash_verify_skip_max_consecutive": getattr(
            cfg, "dflash_verify_skip_max_consecutive", None
        ),
        "dflash_profile_stages": getattr(cfg, "dflash_profile_stages", None),
        "dflash_evidence_trace": getattr(cfg, "dflash_evidence_trace", None),
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
        if summary.get("tree_average_verified_nodes") is not None:
            parts.append(f"tree_nodes={summary['tree_average_verified_nodes']:.2f}")
        if summary.get("tree_average_max_depth") is not None:
            parts.append(f"tree_depth={summary['tree_average_max_depth']:.2f}")
        if summary.get("avg_main_path_accept_length") is not None:
            parts.append(f"main_accept={summary['avg_main_path_accept_length']:.3f}")
    if summary.get("target_logit_shadow_checks"):
        parts.append(
            "action_vocab_mismatch="
            f"{summary['target_logit_shadow_mismatches']}/"
            f"{summary['target_logit_shadow_checks']}"
        )
    if summary.get("verify_skipped_blocks"):
        parts.append(f"verify_skipped_blocks={summary['verify_skipped_blocks']}")
        parts.append(f"verify_skipped_tokens={summary['verify_skipped_tokens']}")
    if summary.get("temporal_proposal_rejected_blocks"):
        parts.append(
            "temporal_route_rejects="
            f"{summary['temporal_proposal_rejected_blocks']}"
        )
        parts.append(
            "temporal_fallback_blocks="
            f"{summary['temporal_fallback_draft_blocks']}"
        )
    if summary.get("temporal_fused_verify_blocks"):
        parts.append(
            "temporal_fused_blocks="
            f"{summary['temporal_fused_verify_blocks']}"
        )
    if summary.get("temporal_prefill_fused_actions"):
        parts.append(
            "prefill_fused_actions="
            f"{summary['temporal_prefill_fused_actions']}"
        )
        parts.append(
            "prefill_full_matches="
            f"{summary['temporal_prefill_full_match_actions']}"
        )
        parts.append(
            "prefill_accept="
            f"{summary['temporal_prefill_avg_accept_length']:.3f}"
        )
    if summary.get("temporal_prefill_tree_actions"):
        parts.append(
            "prefill_tree="
            f"actions={summary['temporal_prefill_tree_actions']}"
            f"/alternate={summary['temporal_prefill_tree_selected_alternate_actions']}"
            f"/extra={summary['temporal_prefill_tree_extra_accepted']}"
        )
    if summary.get("temporal_prefix_cert_attempts"):
        parts.append(
            "prefix_cert="
            f"{summary['temporal_prefix_cert_successes']}/"
            f"{summary['temporal_prefix_cert_attempts']}"
            f"/trusted={summary['temporal_prefix_cert_trusted_tokens']}"
        )
    if summary.get("temporal_prefill_bypassed_actions"):
        pixel_l2 = summary.get("temporal_prefill_bypass_avg_pixel_relative_l2")
        suffix = f"/pixel_l2={pixel_l2:.6f}" if pixel_l2 is not None else ""
        parts.append(
            f"prefill_bypass={summary['temporal_prefill_bypassed_actions']}{suffix}"
        )
    temporal_hold = summary.get("temporal_hold") or {}
    if temporal_hold.get("allowed_holds"):
        parts.append(
            "temporal_hold="
            f"policy={temporal_hold.get('policy')}"
            f"/base={temporal_hold.get('base_holds', 0)}"
            f"/extended={temporal_hold.get('extended_holds', temporal_hold.get('adaptive_extended_holds', 0))}"
            f"/{temporal_hold.get('extension_candidates', temporal_hold.get('adaptive_extension_candidates', 0))}"
            f"/forced={temporal_hold.get('forced_target_after_hold', 0)}"
            f"/debt={temporal_hold.get('extension_debt_forced_targets', 0)}"
        )
    shadow = summary.get("verify_skip_shadow") or {}
    if shadow:
        sentinel = next(
            (item for item in shadow.get("gates", []) if item.get("name") == "sentinel_exact"),
            None,
        )
        if sentinel is not None:
            coverage = sentinel.get("coverage")
            precision = sentinel.get("full_exact_precision")
            coverage_text = f"{coverage:.3f}" if coverage is not None else "None"
            precision_text = f"{precision:.3f}" if precision is not None else "None"
            parts.append(
                f"shadow_sentinel=coverage={coverage_text}/precision={precision_text}"
            )
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
