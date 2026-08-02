#!/usr/bin/env python3
"""Aggregate same-state LIBERO forks into an auditable P0 evidence pack."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openvla.specdecoding.evidence.plot_style import (
    COLORS,
    ICLR_DOUBLE_COLUMN_IN,
    iclr_style,
    save_figure,
)


CANDIDATE_ORDER = (
    "current_target_path",
    "lag1_target",
    "lag1_harmonic",
    "lag2_target",
    "same_norm_random",
    "zero_motion",
)
CANDIDATE_LABELS = {
    "current_target_path": "Target continuation",
    "lag1_target": "Lag-1 target action",
    "lag1_harmonic": "Lag-1, inverse-age",
    "lag2_target": "Lag-2 target action",
    "same_norm_random": "Same-norm random",
    "zero_motion": "Zero motion",
}
CANDIDATE_COLORS = {
    "current_target_path": COLORS["green"],
    "lag1_target": COLORS["blue"],
    "lag1_harmonic": COLORS["purple"],
    "lag2_target": COLORS["sky"],
    "same_norm_random": COLORS["red"],
    "zero_motion": COLORS["orange"],
}

RECOVERY_CANDIDATE_ORDER = (
    "current_target_path",
    "lag1_target",
    "lag2_target",
    "same_norm_random",
    "zero_motion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive(path: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as source, destination.open("wb") as sink:
        with gzip.GzipFile(fileobj=sink, mode="wb", filename="", mtime=0) as zipped:
            shutil.copyfileobj(source, zipped)
    return {
        "source": str(path.resolve()),
        "source_bytes": path.stat().st_size,
        "source_sha256": sha256(path),
        "archive": str(destination),
        "archive_bytes": destination.stat().st_size,
        "archive_sha256": sha256(destination),
    }


def git_output(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def cluster_bootstrap(
    rows: list[dict],
    field: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    clusters: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            clusters[(int(row["episode_index"]), int(row["fork_step"]))].append(
                float(value)
            )
    cluster_values = np.asarray(
        [np.mean(values) for values in clusters.values()], dtype=np.float64
    )
    if not len(cluster_values):
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.mean(cluster_values))
    if len(cluster_values) == 1:
        return estimate, estimate, estimate
    draws = rng.choice(
        cluster_values,
        size=(samples, len(cluster_values)),
        replace=True,
    ).mean(axis=1)
    lower, upper = percentile_interval(draws)
    return estimate, lower, upper


def summarize(
    rows: list[dict], samples: int, seed: int
) -> tuple[list[dict], list[dict]]:
    branch_rows = [row for row in rows if row.get("record_type") == "counterfactual"]
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in branch_rows:
        grouped[(str(row["candidate"]), int(row["hold_depth"]))].append(row)
    rng = np.random.default_rng(seed)
    summary = []
    for candidate in CANDIDATE_ORDER:
        depths = sorted(depth for name, depth in grouped if name == candidate)
        for depth in depths:
            group = grouped[(candidate, depth)]
            harm, harm_low, harm_high = cluster_bootstrap(
                group, "one_sided_harm", samples, rng
            )
            correction, correction_low, correction_high = cluster_bootstrap(
                group, "correction_l2", samples, rng
            )
            recovery, recovery_low, recovery_high = cluster_bootstrap(
                group, "recovery_steps", samples, rng
            )
            state_l2, state_l2_low, state_l2_high = cluster_bootstrap(
                group, "post_hold_state_l2", samples, rng
            )
            position_l2, position_l2_low, position_l2_high = cluster_bootstrap(
                group, "post_hold_eef_position_l2", samples, rng
            )
            rotation_l2, rotation_l2_low, rotation_l2_high = cluster_bootstrap(
                group, "post_hold_eef_rotation_l2", samples, rng
            )
            summary.append(
                {
                    "candidate": candidate,
                    "candidate_label": CANDIDATE_LABELS[candidate],
                    "hold_depth": depth,
                    "forks": len(group),
                    "branch_success_rate": float(
                        np.mean([bool(row["branch_success"]) for row in group])
                    ),
                    "one_sided_harm_rate": harm,
                    "harm_ci_low": harm_low,
                    "harm_ci_high": harm_high,
                    "mean_first_correction_l2": correction,
                    "correction_ci_low": correction_low,
                    "correction_ci_high": correction_high,
                    "mean_recovery_steps": recovery,
                    "recovery_ci_low": recovery_low,
                    "recovery_ci_high": recovery_high,
                    "mean_post_hold_state_l2": state_l2,
                    "state_l2_ci_low": state_l2_low,
                    "state_l2_ci_high": state_l2_high,
                    "mean_post_hold_eef_position_l2": position_l2,
                    "position_l2_ci_low": position_l2_low,
                    "position_l2_ci_high": position_l2_high,
                    "mean_post_hold_eef_rotation_l2": rotation_l2,
                    "rotation_l2_ci_low": rotation_l2_low,
                    "rotation_l2_ci_high": rotation_l2_high,
                }
            )
    return branch_rows, summary


def draw_panel(
    axes,
    rows: list[dict],
    metric: str,
    low: str,
    high: str,
    ylabel: str,
    candidate_order=CANDIDATE_ORDER,
):
    for candidate in candidate_order:
        group = sorted(
            [row for row in rows if row["candidate"] == candidate],
            key=lambda row: row["hold_depth"],
        )
        if not group:
            continue
        x = np.asarray([row["hold_depth"] for row in group], dtype=float)
        y = np.asarray([row[metric] for row in group], dtype=float)
        lower = np.asarray([row[low] for row in group], dtype=float)
        upper = np.asarray([row[high] for row in group], dtype=float)
        axes.errorbar(
            x,
            y,
            yerr=np.vstack(
                (np.maximum(y - lower, 0.0), np.maximum(upper - y, 0.0))
            ),
            marker="o",
            capsize=2,
            linewidth=1.25,
            markersize=3.5,
            color=CANDIDATE_COLORS[candidate],
            label=CANDIDATE_LABELS[candidate],
        )
    axes.set_xticks(sorted({int(row["hold_depth"]) for row in rows}))
    axes.set_xlabel("Open-loop depth")
    axes.set_ylabel(ylabel)
    axes.grid()


def plot(output: Path, rows: list[dict]) -> None:
    with iclr_style():
        fig, axes_grid = plt.subplots(
            2,
            2,
            figsize=(ICLR_DOUBLE_COLUMN_IN, 4.0),
        )
        axes = axes_grid.ravel()
        draw_panel(
            axes[0],
            rows,
            "one_sided_harm_rate",
            "harm_ci_low",
            "harm_ci_high",
            "One-sided task harm rate",
            RECOVERY_CANDIDATE_ORDER,
        )
        axes[0].set_title("(a) Task harm", loc="left", fontweight="bold")
        draw_panel(
            axes[1],
            rows,
            "mean_first_correction_l2",
            "correction_ci_low",
            "correction_ci_high",
            "First target correction (L2)",
            RECOVERY_CANDIDATE_ORDER,
        )
        axes[1].set_title("(b) Re-grounding correction", loc="left", fontweight="bold")
        draw_panel(
            axes[2],
            rows,
            "mean_recovery_steps",
            "recovery_ci_low",
            "recovery_ci_high",
            "Target recovery steps",
            RECOVERY_CANDIDATE_ORDER,
        )
        axes[2].set_title("(c) Recovery effort", loc="left", fontweight="bold")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[3].axis("off")
        axes[3].legend(
            handles,
            labels,
            loc="upper left",
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
        )
        cluster_count = max((int(row["forks"]) for row in rows), default=0)
        axes[3].text(
            0.0,
            0.08,
            f"95% interval: episode-fork cluster bootstrap\n"
            f"Independent fork clusters per condition: {cluster_count}",
            transform=axes[3].transAxes,
            fontsize=7,
            va="bottom",
        )
        fig.tight_layout(w_pad=1.0, h_pad=1.0)
        save_figure(fig, output)
        plt.close(fig)


def plot_harmonic(output: Path, rows: list[dict]) -> None:
    comparison = ("lag1_target", "lag1_harmonic")
    with iclr_style():
        fig, axes = plt.subplots(1, 2, figsize=(ICLR_DOUBLE_COLUMN_IN, 2.25))
        draw_panel(
            axes[0],
            rows,
            "mean_post_hold_eef_position_l2",
            "position_l2_ci_low",
            "position_l2_ci_high",
            "End-effector position deviation (m)",
            comparison,
        )
        axes[0].set_title("(a) Position deviation", loc="left", fontweight="bold")
        draw_panel(
            axes[1],
            rows,
            "mean_post_hold_eef_rotation_l2",
            "rotation_l2_ci_low",
            "rotation_l2_ci_high",
            "End-effector rotation deviation",
            comparison,
        )
        axes[1].set_title("(b) Rotation deviation", loc="left", fontweight="bold")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
        fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.0)
        save_figure(fig, output)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    tables = output / "tables"
    figures = output / "figures"
    raw = output / "raw"
    for directory in (tables, figures, raw):
        directory.mkdir(parents=True, exist_ok=True)

    run_manifest = json.loads(args.run_manifest.read_text())
    rows = read_jsonl(args.records)
    branch_rows, summary = summarize(
        rows, args.bootstrap_samples, args.bootstrap_seed
    )
    if not summary:
        raise ValueError("Counterfactual run has no completed branch records.")
    flattened = [
        {
            "suite": row["suite"],
            "task_id": row["task_id"],
            "episode_index": row["episode_index"],
            "fork_step": row["fork_step"],
            "candidate": row["candidate"],
            "hold_depth": row["hold_depth"],
            "branch_success": int(bool(row["branch_success"])),
            "one_sided_harm": int(bool(row["one_sided_harm"])),
            "recovery_steps": row["recovery_steps"],
            "correction_l2": row["correction_l2"],
            "minimum_reference_robot_l2": row["minimum_reference_robot_l2"],
            "post_hold_state_l2": row.get("post_hold_state_l2"),
            "post_hold_eef_position_l2": row.get(
                "post_hold_eef_position_l2"
            ),
            "post_hold_eef_rotation_l2": row.get(
                "post_hold_eef_rotation_l2"
            ),
            "post_hold_gripper_l2": row.get("post_hold_gripper_l2"),
            "fork_state_linf": row["fork_state_linf"],
            "mean_action_norm": row["mean_action_norm"],
            "continuous_action_scales": json.dumps(
                row.get("continuous_action_scales", []), separators=(",", ":")
            ),
        }
        for row in branch_rows
    ]
    write_csv(tables / "counterfactual_branches.csv", flattened)
    write_csv(tables / "counterfactual_summary.csv", summary)
    plot(figures / "fig5_same_state_recovery", summary)
    plot_harmonic(figures / "fig8_harmonic_authority", summary)

    positive_controls = [
        row for row in branch_rows if row["candidate"] == "current_target_path"
    ]
    positive_control_failures = sum(
        not bool(row["branch_success"]) for row in positive_controls
    )
    maximum_fork_state_linf = max(
        (float(row["fork_state_linf"]) for row in branch_rows),
        default=float("inf"),
    )
    archived = [
        archive(args.records, raw / f"{args.records.name}.gz"),
        archive(args.run_manifest, raw / f"{args.run_manifest.name}.gz"),
    ]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(args.repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": git_output(args.repo_root, "status", "--short"),
        "run_manifest": run_manifest,
        "inputs": archived,
        "counts": {
            "all_records": len(rows),
            "counterfactual_branches": len(branch_rows),
            "positive_controls": len(positive_controls),
            "positive_control_failures": positive_control_failures,
            "maximum_fork_state_linf": maximum_fork_state_linf,
        },
        "validity": {
            "same_state_positive_control_passed": (
                positive_control_failures == 0 and maximum_fork_state_linf <= 1e-8
            ),
            "interpretation": (
                "Task harm is conditional on a successful frozen-target reference. "
                "This pilot does not establish a population-level risk bound."
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    (output / "README.md").write_text(
        "# P0 same-state counterfactual recovery\n\n"
        "Every branch starts from the same MuJoCo snapshot and returns to the "
        "unchanged target policy after a bounded physical commitment. The target "
        "continuation is a positive control for state restoration. Error bars are "
        "95% fork-cluster bootstrap intervals. This pilot supports mechanism "
        "diagnosis only; formal risk claims require more tasks and held-out trials.\n"
    )


if __name__ == "__main__":
    main()
