#!/usr/bin/env python3
"""Build tables and ICLR-sized figures for the paired temporal 2x2 study."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from specdecoding.evidence.plot_style import (
    COLORS,
    ICLR_DOUBLE_COLUMN_IN,
    iclr_style,
    save_figure,
)
from specdecoding.evidence.temporal_factorial_design import (
    FACTORIAL_PERIOD,
    FACTORIAL_TARGET_COUNT,
    LOW_DISCREPANCY_INTERVALS,
    MAX_CONCENTRATION_INTERVALS,
)
from specdecoding.evidence.temporal_schedule_design import (
    exact_mcnemar_p,
    power_law_authority_scale,
    target_indicator,
)


ORDER = (
    "low_discrepancy_linear",
    "low_discrepancy_critical",
    "max_concentration_linear",
    "max_concentration_critical",
)
LEGACY_NAMES = {
    "paced_constant": "low_discrepancy_linear",
    "paced_harmonic": "low_discrepancy_critical",
    "clustered_constant": "max_concentration_linear",
    "clustered_harmonic": "max_concentration_critical",
}
LABELS = {
    "low_discrepancy_linear": "Low-discrepancy / linear",
    "low_discrepancy_critical": "Low-discrepancy / critical",
    "max_concentration_linear": "Max-concentration / linear",
    "max_concentration_critical": "Max-concentration / critical",
}
PALETTE = {
    "low_discrepancy_linear": COLORS["blue"],
    "low_discrepancy_critical": COLORS["green"],
    "max_concentration_linear": COLORS["red"],
    "max_concentration_critical": COLORS["orange"],
}


def canonical_condition(name: str) -> str:
    return LEGACY_NAMES.get(name, name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def mean(values):
    return float(np.mean(values)) if values else None


def bootstrap_rate(values, seed=7, draws=10000):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    estimates = np.mean(
        rng.choice(array, size=(draws, array.size), replace=True), axis=1
    )
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def paired_effects(episodes: list[dict], seed=7, draws=20000) -> list[dict]:
    by_episode = defaultdict(dict)
    for row in episodes:
        by_episode[int(row["episode_index"])][row["condition"]] = float(
            row["success"]
        )
    complete = [
        values
        for _, values in sorted(by_episode.items())
        if all(condition in values for condition in ORDER)
    ]
    if not complete:
        return []
    matrix = np.asarray(
        [[values[condition] for condition in ORDER] for values in complete],
        dtype=np.float64,
    )
    # ORDER: low-discrepancy linear/critical, max-concentration linear/critical.
    contrasts = (
        ("critical_at_low_discrepancy", np.asarray([-1.0, 1.0, 0.0, 0.0])),
        ("critical_at_max_concentration", np.asarray([0.0, 0.0, -1.0, 1.0])),
        ("low_discrepancy_at_linear", np.asarray([1.0, 0.0, -1.0, 0.0])),
        ("low_discrepancy_at_critical", np.asarray([0.0, 1.0, 0.0, -1.0])),
        ("marginal_critical_authority", np.asarray([-0.5, 0.5, -0.5, 0.5])),
        ("marginal_low_discrepancy", np.asarray([0.5, 0.5, -0.5, -0.5])),
        ("regularity_authority_interaction", np.asarray([-1.0, 1.0, 1.0, -1.0])),
    )
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, matrix.shape[0], size=(draws, matrix.shape[0])
    )
    sampled_means = matrix[sampled_indices].mean(axis=1)
    rows = []
    for name, weights in contrasts:
        per_episode = matrix @ weights
        bootstrap = sampled_means @ weights
        low, high = np.percentile(bootstrap, [2.5, 97.5])
        row = {
            "contrast": name,
            "paired_episodes": matrix.shape[0],
            "effect": float(per_episode.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "gains": None,
            "losses": None,
            "mcnemar_two_sided_p": None,
        }
        nonzero = np.flatnonzero(weights)
        if len(nonzero) == 2 and set(weights[nonzero]) == {-1.0, 1.0}:
            gains = int(np.sum(per_episode > 0))
            losses = int(np.sum(per_episode < 0))
            row.update(
                {
                    "gains": gains,
                    "losses": losses,
                    "mcnemar_two_sided_p": exact_mcnemar_p(gains, losses),
                }
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    raw_dir = output / "raw"
    table_dir = output / "tables"
    figure_dir = output / "figures"
    for directory in (raw_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = [
        json.loads(line)
        for line in args.records.read_text().splitlines()
        if line.strip()
    ]
    for row in records:
        if "condition" in row:
            row["condition"] = canonical_condition(row["condition"])
    steps = [row for row in records if row["record_type"] == "step"]
    episodes = [row for row in records if row["record_type"] == "episode"]

    episode_rows = []
    summary_rows = []
    for condition in ORDER:
        condition_episodes = [
            row for row in episodes if row["condition"] == condition
        ]
        condition_steps = [row for row in steps if row["condition"] == condition]
        successes = [int(row["success"]) for row in condition_episodes]
        correction = [
            float(row["target_correction_l2"])
            for row in condition_steps
            if row.get("target_correction_l2") is not None
        ]
        correction_h1 = [
            float(row["target_correction_l2"])
            for row in condition_steps
            if row.get("target_correction_l2") is not None
            and int(row.get("previous_hold_depth_at_target", 0)) == 1
        ]
        correction_h2 = [
            float(row["target_correction_l2"])
            for row in condition_steps
            if row.get("target_correction_l2") is not None
            and int(row.get("previous_hold_depth_at_target", 0)) == 2
        ]
        ci_low, ci_high = bootstrap_rate(successes)
        summary_rows.append(
            {
                "condition": condition,
                "condition_label": LABELS[condition],
                "episodes": len(condition_episodes),
                "successes": sum(successes),
                "success_rate": mean(successes),
                "success_ci_low": ci_low,
                "success_ci_high": ci_high,
                "mean_steps": mean([row["steps"] for row in condition_episodes]),
                "mean_target_rate": mean(
                    [row["target_rate"] for row in condition_episodes]
                ),
                "mean_target_calls": mean(
                    [row["target_calls"] for row in condition_episodes]
                ),
                "mean_correction_l2": mean(correction),
                "p95_correction_l2": percentile(correction, 95),
                "mean_h1_correction_l2": mean(correction_h1),
                "mean_h2_correction_l2": mean(correction_h2),
            }
        )
        for row in condition_episodes:
            episode_rows.append(
                {
                    "condition": condition,
                    "episode_index": row["episode_index"],
                    "episode_seed": row["episode_seed"],
                    "success": int(row["success"]),
                    "steps": row["steps"],
                    "target_calls": row["target_calls"],
                    "target_rate": row["target_rate"],
                    "hold_1_count": row["hold_1_count"],
                    "hold_2_count": row["hold_2_count"],
                }
            )

    write_csv(
        table_dir / "condition_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    write_csv(
        table_dir / "paired_episodes.csv",
        episode_rows,
        list(episode_rows[0]),
    )
    correction_rows = [
        {
            "condition": row["condition"],
            "episode_index": row["episode_index"],
            "step": row["step"],
            "previous_hold_depth": row["previous_hold_depth_at_target"],
            "target_correction_l2": row["target_correction_l2"],
            "target_vs_stale_target_l2": row["target_vs_stale_target_l2"],
        }
        for row in steps
        if row.get("target_correction_l2") is not None
    ]
    if correction_rows:
        write_csv(
            table_dir / "regrounding_corrections.csv",
            correction_rows,
            list(correction_rows[0]),
        )
    contrast_rows = paired_effects(episodes)
    write_csv(
        table_dir / "paired_success_contrasts.csv",
        contrast_rows,
        list(contrast_rows[0]),
    )

    with iclr_style():
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(ICLR_DOUBLE_COLUMN_IN, 4.15),
            constrained_layout=True,
        )
        schedules = {
            "Low-discrepancy": LOW_DISCREPANCY_INTERVALS,
            "Max-concentration": MAX_CONCENTRATION_INTERVALS,
        }
        horizon = 60
        density = FACTORIAL_TARGET_COUNT / FACTORIAL_PERIOD
        prefixes = np.arange(1, horizon + 1)
        for label, intervals in schedules.items():
            targets = np.asarray(
                target_indicator(intervals, horizon), dtype=np.float64
            )
            discrepancy = np.cumsum(targets) - density * prefixes
            axes[0, 0].plot(prefixes, discrepancy, label=label)
        axes[0, 0].axhline(0, color=COLORS["gray"], linewidth=0.8)
        axes[0, 0].set_xlabel("Control-step prefix")
        axes[0, 0].set_ylabel("Target-count discrepancy")
        axes[0, 0].set_title(
            "(a) Equal-budget regularity",
            loc="left",
            fontweight="bold",
        )
        axes[0, 0].legend(frameon=False)
        axes[0, 0].grid(True)

        hold_depth = np.arange(1, 21)
        linear_authority = np.cumsum(
            [
                power_law_authority_scale(exponent=0.0, hold_depth=d)
                for d in hold_depth
            ]
        )
        critical_authority = np.cumsum(
            [
                power_law_authority_scale(exponent=1.0, hold_depth=d)
                for d in hold_depth
            ]
        )
        axes[0, 1].plot(hold_depth, linear_authority, label="Linear authority")
        axes[0, 1].plot(hold_depth, critical_authority, label="Critical authority")
        axes[0, 1].set_xlabel("Open-loop depth")
        axes[0, 1].set_ylabel("Cumulative control authority")
        axes[0, 1].set_title(
            "(b) Cumulative authority", loc="left", fontweight="bold"
        )
        axes[0, 1].legend(frameon=False)
        axes[0, 1].grid(True)

        x = np.arange(len(ORDER))
        rates = [next(row for row in summary_rows if row["condition"] == name)["success_rate"] for name in ORDER]
        lows = [next(row for row in summary_rows if row["condition"] == name)["success_ci_low"] for name in ORDER]
        highs = [next(row for row in summary_rows if row["condition"] == name)["success_ci_high"] for name in ORDER]
        axes[1, 0].bar(x, rates, color=[PALETTE[name] for name in ORDER], width=0.72)
        axes[1, 0].errorbar(
            x,
            rates,
            yerr=[np.asarray(rates) - np.asarray(lows), np.asarray(highs) - np.asarray(rates)],
            fmt="none",
            color=COLORS["black"],
            capsize=2,
        )
        axes[1, 0].set_xticks(
            x, ["LD\nLinear", "LD\nCritical", "MC\nLinear", "MC\nCritical"]
        )
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].set_ylabel("Task success rate")
        axes[1, 0].set_title("(c) Paired outcomes", loc="left", fontweight="bold")
        axes[1, 0].grid(axis="y")

        box_values = []
        box_positions = []
        box_colors = []
        for index, condition in enumerate(ORDER):
            values = [
                float(row["target_correction_l2"])
                for row in correction_rows
                if row["condition"] == condition
            ]
            if values:
                box_values.append(values)
                box_positions.append(index)
                box_colors.append(PALETTE[condition])
        boxes = axes[1, 1].boxplot(
            box_values,
            positions=box_positions,
            widths=0.65,
            showfliers=False,
            patch_artist=True,
        )
        for patch, color in zip(boxes["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axes[1, 1].set_xticks(
            x, ["LD\nLinear", "LD\nCritical", "MC\nLinear", "MC\nCritical"]
        )
        axes[1, 1].set_ylabel("Next-target correction (L2)")
        axes[1, 1].set_title("(d) Re-grounding correction", loc="left", fontweight="bold")
        axes[1, 1].grid(axis="y")
        save_figure(fig, figure_dir / "fig6_temporal_authority_factorial")
        plt.close(fig)

        fig, ax = plt.subplots(
            figsize=(ICLR_DOUBLE_COLUMN_IN, 2.65), constrained_layout=True
        )
        display = (
            ("critical_at_low_discrepancy", "Critical | low discrepancy"),
            ("critical_at_max_concentration", "Critical | max concentration"),
            ("low_discrepancy_at_linear", "Low discrepancy | linear"),
            ("low_discrepancy_at_critical", "Low discrepancy | critical"),
            ("marginal_critical_authority", "Authority main effect"),
            ("marginal_low_discrepancy", "Regularity main effect"),
            ("regularity_authority_interaction", "Interaction"),
        )
        contrast_by_name = {row["contrast"]: row for row in contrast_rows}
        y = np.arange(len(display))[::-1]
        effects = np.asarray(
            [contrast_by_name[name]["effect"] for name, _ in display]
        )
        lows = np.asarray(
            [contrast_by_name[name]["ci_low"] for name, _ in display]
        )
        highs = np.asarray(
            [contrast_by_name[name]["ci_high"] for name, _ in display]
        )
        ax.errorbar(
            effects,
            y,
            xerr=[effects - lows, highs - effects],
            fmt="o",
            color=COLORS["blue"],
            ecolor=COLORS["gray"],
            capsize=2,
        )
        ax.axvline(0, color=COLORS["black"], linewidth=0.8)
        ax.set_yticks(y, [label for _, label in display])
        ax.set_xlabel("Paired success-rate difference")
        ax.set_title(
            "Paired factorial effects",
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="x")
        save_figure(fig, figure_dir / "fig7_paired_factorial_effects")
        plt.close(fig)

    archived = []
    for source in (args.records, args.run_manifest):
        destination = raw_dir / f"{source.name}.gz"
        with source.open("rb") as src, gzip.open(destination, "wb") as dst:
            dst.write(src.read())
        archived.append(
            {
                "source": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": sha256(source),
                "archive": str(destination),
                "archive_bytes": destination.stat().st_size,
                "archive_sha256": sha256(destination),
            }
        )

    run_manifest = json.loads(args.run_manifest.read_text())
    manifest = {
        "schema_version": 1,
        "git_commit": git_value(args.repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": git_value(args.repo_root, "status", "--porcelain"),
        "run_manifest": run_manifest,
        "inputs": archived,
        "counts": {"steps": len(steps), "episodes": len(episodes)},
        "conditions": summary_rows,
        "interpretation": (
            "Temporal regularity and authority growth are paired on identical "
            "initial states. Both schedules share one target-call budget and "
            "one integer gap multiset; they are the minimum-prefix-discrepancy "
            "and maximum-concentration extremizers. Inference latency is "
            "descriptive; the mechanism endpoints are task success and "
            "next-target correction."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    (output / "README.md").write_text(
        "# Paired temporal-factor evidence\n\n"
        "Minimum-prefix-discrepancy and maximum-concentration temporal "
        "arrangements are crossed with linear and critical authority growth "
        "on identical LIBERO initial states. The two arrangements have equal "
        "target-call budgets and identical gap multisets. See `manifest.json` "
        "for exact derivation, configuration, and hashes.\n"
    )


if __name__ == "__main__":
    main()
