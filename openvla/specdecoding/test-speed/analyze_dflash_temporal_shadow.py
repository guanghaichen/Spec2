#!/usr/bin/env python3
"""Analyze temporal-route and verify-skip gates from a DFlash shadow summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COSINE_THRESHOLDS = (0.990, 0.995, 0.997, 0.998, 0.9983, 0.9985)
STABLE_ACTION_THRESHOLDS = (1, 2, 4, 6)
VISUAL_RELATIVE_L2_THRESHOLDS = (0.0006, 0.0008, 0.0010, 0.0015, 0.0025)


def wilson_error_upper_bound(errors: int, total: int, z: float = 1.959963984540054) -> float | None:
    if total <= 0:
        return None
    rate = errors / total
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return (center + radius) / denominator


def attach_stable_run_lengths(records: list[dict]) -> list[dict]:
    """Backfill the online-visible run length for older shadow files."""
    run_length = 0
    episode_index = -1
    labeled = []
    for original in records:
        record = dict(original)
        label = record.get("full_exact_match")
        if not record.get("eligible"):
            episode_index += 1
            run_length = 1
            continue
        if label is None:
            run_length = 1
            continue
        record.setdefault("previous_verified_action_run_length", run_length)
        record.setdefault("episode_index", episode_index)
        labeled.append(record)
        run_length = run_length + 1 if bool(label) else 1
    return labeled


def summarize_gate(records: list[dict], *, min_cosine: float, min_stable_actions: int) -> dict:
    selected = [
        record
        for record in records
        if bool(record.get("first_token_matches_previous"))
        and float(record.get("prompt_temporal_cosine", -1.0)) >= min_cosine
        and int(record.get("previous_verified_action_run_length", 0)) >= min_stable_actions
    ]
    errors = sum(not bool(record["full_exact_match"]) for record in selected)
    return {
        "min_cosine": min_cosine,
        "min_stable_actions": min_stable_actions,
        "selected": len(selected),
        "coverage": len(selected) / len(records) if records else None,
        "errors": errors,
        "precision": 1.0 - errors / len(selected) if selected else None,
        "error_rate_95pct_upper": wilson_error_upper_bound(errors, len(selected)),
        "selected_episodes": sorted({int(record["episode_index"]) for record in selected}),
    }


def summarize_visual_gate(
    records: list[dict], *, max_relative_l2: float, min_stable_actions: int
) -> dict:
    selected = [
        record
        for record in records
        if record.get("pixel_temporal_relative_l2") is not None
        and float(record["pixel_temporal_relative_l2"]) <= max_relative_l2
        and int(record.get("previous_verified_action_run_length", 0))
        >= min_stable_actions
    ]
    errors = sum(not bool(record["full_exact_match"]) for record in selected)
    return {
        "max_relative_l2": max_relative_l2,
        "min_stable_actions": min_stable_actions,
        "selected": len(selected),
        "coverage": len(selected) / len(records) if records else None,
        "errors": errors,
        "precision": 1.0 - errors / len(selected) if selected else None,
        "error_rate_95pct_upper": wilson_error_upper_bound(errors, len(selected)),
        "selected_episodes": sorted({int(record.get("episode_index", -1)) for record in selected}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path, help="Shadow *_summary.json")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of TSV")
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text())
    temporal = (payload.get("generation") or {}).get("temporal_action_skip") or {}
    records = temporal.get("records")
    if not records:
        raise SystemExit("No temporal shadow records found; run the shadow launcher with full_suite timing scope.")
    labeled = attach_stable_run_lengths(records)
    rows = [
        summarize_gate(
            labeled,
            min_cosine=min_cosine,
            min_stable_actions=min_stable_actions,
        )
        for min_stable_actions in STABLE_ACTION_THRESHOLDS
        for min_cosine in COSINE_THRESHOLDS
    ]

    visual_rows = [
        summarize_visual_gate(
            labeled,
            max_relative_l2=max_relative_l2,
            min_stable_actions=min_stable_actions,
        )
        for min_stable_actions in STABLE_ACTION_THRESHOLDS
        for max_relative_l2 in VISUAL_RELATIVE_L2_THRESHOLDS
    ]

    result = {
        "summary": str(args.summary),
        "labeled_actions": len(labeled),
        "gates": rows,
        "visual_prefill_gates": visual_rows,
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("stable\tcosine\tselected\tepisodes\tcoverage\terrors\tprecision\terror95_upper")
    for row in rows:
        values = (
            row["min_stable_actions"],
            f'{row["min_cosine"]:.4f}',
            row["selected"],
            len(row["selected_episodes"]),
            f'{row["coverage"]:.4%}' if row["coverage"] is not None else "NA",
            row["errors"],
            f'{row["precision"]:.4%}' if row["precision"] is not None else "NA",
            f'{row["error_rate_95pct_upper"]:.4%}'
            if row["error_rate_95pct_upper"] is not None
            else "NA",
        )
        print("\t".join(str(value) for value in values))

    if any(row["selected"] for row in visual_rows):
        print("\nvisual_prefill_shadow")
        print("stable\tmax_relative_l2\tselected\tepisodes\tcoverage\terrors\tprecision\terror95_upper")
        for row in visual_rows:
            values = (
                row["min_stable_actions"],
                f'{row["max_relative_l2"]:.4g}',
                row["selected"],
                len(row["selected_episodes"]),
                f'{row["coverage"]:.4%}' if row["coverage"] is not None else "NA",
                row["errors"],
                f'{row["precision"]:.4%}' if row["precision"] is not None else "NA",
                f'{row["error_rate_95pct_upper"]:.4%}'
                if row["error_rate_95pct_upper"] is not None
                else "NA",
            )
            print("\t".join(str(value) for value in values))


if __name__ == "__main__":
    main()
