#!/usr/bin/env python3
"""Build auditable P0 tables and ICLR-sized figures from eval summaries.

The script never infers a stronger claim than its inputs support. In
particular, VTPF logit parity is evaluated only at positions whose preceding
candidate prefix is identical to serial target AR.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
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


STAGE_LABELS = {
    "target_prefill": "Target prefill",
    "target_prefill_fused": "Fused target prefill",
    "target_anchor": "Target anchor",
    "target_fused_verify": "Fused target verify",
    "draft_backbone": "Parallel draft",
    "action_head": "Draft head",
    "target_verify": "Target verify",
    "verify_prepare": "Verify setup",
    "accept_select": "Accept / correct",
    "cache_commit": "KV commit",
    "target_ar_reference_prefill": "Independent AR prefill",
    "target_ar_reference_tail": "6-step target AR tail",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--profile-summary", type=Path, required=True)
    parser.add_argument("--shadow-summary", type=Path, required=True)
    parser.add_argument("--parity-summary", type=Path, required=True)
    parser.add_argument("--ar-summary", type=Path)
    parser.add_argument(
        "--method-summary",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-run-root", type=Path)
    parser.add_argument("--repo-root", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def archive_input(path: Path, raw_dir: Path, relative_name: Path | None = None) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    relative_name = relative_name or Path(path.name)
    destination = raw_dir / relative_name.parent / f"{relative_name.name}.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as source, destination.open("wb") as sink:
        with gzip.GzipFile(fileobj=sink, mode="wb", filename="", mtime=0) as zipped:
            shutil.copyfileobj(source, zipped)
    return {
        "source": str(path.resolve()),
        "archived": str(destination),
        "source_bytes": path.stat().st_size,
        "source_sha256": sha256(path),
        "archive_bytes": destination.stat().st_size,
        "archive_sha256": sha256(destination),
    }


def common_prefix(left, right) -> int:
    length = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        length += 1
    return length


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return None
    return float(np.dot(left, right) / denominator)


def build_temporal_rows(shadow: dict) -> tuple[list[dict], list[dict]]:
    generation = shadow.get("generation") or {}
    target_tokens = generation.get("target_ar_reference_trace") or []
    draft_tokens = generation.get("action_token_trace") or []
    traces = generation.get("evidence_trace") or []
    temporal = (generation.get("temporal_action_skip") or {}).get("records") or []
    if not traces:
        raise ValueError(
            "Temporal summary lacks evidence_trace; run the P0 launcher with "
            "AR evidence tracing enabled."
        )
    if not target_tokens:
        target_tokens = [trace["environment_action"] for trace in traces]
    if len(target_tokens) != len(traces):
        raise ValueError(
            f"Trace alignment failed: target={len(target_tokens)} evidence={len(traces)}"
        )

    histories: dict[tuple[int, int], list[int]] = defaultdict(list)
    transition_rows: list[dict] = []
    lag_rows: list[dict] = []
    for index, (tokens, trace) in enumerate(zip(target_tokens, traces)):
        key = (int(trace["task_id"]), int(trace["episode_index"]))
        history = histories[key]
        current_action = np.asarray(trace["environment_action"], dtype=np.float64)
        episode_steps = max(int(trace.get("episode_control_steps", 1)), 1)
        phase_value = int(trace.get("control_step", 0)) / episode_steps
        phase = "early" if phase_value < 1 / 3 else "middle" if phase_value < 2 / 3 else "late"
        for lag in range(1, 5):
            if len(history) < lag:
                continue
            previous_index = history[-lag]
            previous_tokens = target_tokens[previous_index]
            previous_action = np.asarray(
                traces[previous_index]["environment_action"], dtype=np.float64
            )
            prefix = common_prefix(previous_tokens, tokens)
            lag_rows.append(
                {
                    "task_id": key[0],
                    "episode_index": key[1],
                    "control_step": int(trace.get("control_step", 0)),
                    "episode_success": bool(trace.get("episode_success", False)),
                    "phase": phase,
                    "lag": lag,
                    "token_exact": int(prefix == len(tokens)),
                    "token_common_prefix": prefix,
                    "continuous_l2": float(np.linalg.norm(current_action - previous_action)),
                    "continuous_cosine": cosine(current_action, previous_action),
                    "action_norm": float(np.linalg.norm(current_action[:6])),
                    "gripper_switch": int(current_action[-1] != previous_action[-1]),
                }
            )

        if history:
            previous_index = history[-1]
            previous_tokens = target_tokens[previous_index]
            previous_draft = (
                draft_tokens[previous_index]
                if len(draft_tokens) == len(target_tokens)
                else None
            )
            record = temporal[index] if len(temporal) == len(target_tokens) else {}
            pixel_relative_l2 = record.get("pixel_temporal_relative_l2")
            if pixel_relative_l2 is None and trace.get("image_signature") is not None:
                current_pixels = np.asarray(
                    trace["image_signature"], dtype=np.float64
                )
                previous_pixels = np.asarray(
                    traces[previous_index]["image_signature"], dtype=np.float64
                )
                pixel_relative_l2 = float(
                    np.linalg.norm(current_pixels - previous_pixels)
                    / max(np.linalg.norm(previous_pixels), 1e-12)
                )
            transition_rows.append(
                {
                    "task_id": key[0],
                    "episode_index": key[1],
                    "control_step": int(trace.get("control_step", 0)),
                    "episode_success": bool(trace.get("episode_success", False)),
                    "phase": phase,
                    "target_lag1_exact": int(
                        common_prefix(previous_tokens, tokens) == len(tokens)
                    ),
                    "target_lag1_prefix": common_prefix(previous_tokens, tokens),
                    "previous_dflash_exact": (
                        int(common_prefix(previous_draft, tokens) == len(tokens))
                        if previous_draft is not None
                        else None
                    ),
                    "previous_dflash_prefix": (
                        common_prefix(previous_draft, tokens)
                        if previous_draft is not None
                        else None
                    ),
                    "continuous_l2": next(
                        row["continuous_l2"]
                        for row in reversed(lag_rows)
                        if row["task_id"] == key[0]
                        and row["episode_index"] == key[1]
                        and row["control_step"] == int(trace.get("control_step", 0))
                        and row["lag"] == 1
                    ),
                    "action_norm": float(np.linalg.norm(current_action[:6])),
                    "pixel_relative_l2": pixel_relative_l2,
                    "prompt_cosine": record.get("prompt_temporal_cosine"),
                    "gripper_switch": int(
                        current_action[-1]
                        != np.asarray(
                            traces[previous_index]["environment_action"],
                            dtype=np.float64,
                        )[-1]
                    ),
                }
            )
        history.append(index)

    trace_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    transition_by_step = {
        (
            int(row["task_id"]),
            int(row["episode_index"]),
            int(row["control_step"]),
        ): row
        for row in transition_rows
    }
    for index, trace in enumerate(traces):
        trace_indices[(int(trace["task_id"]), int(trace["episode_index"]))].append(
            index
        )
    for key, indices in trace_indices.items():
        if len(indices) < 4:
            continue
        offset = max(2, len(indices) // 2)
        for local_index, trace_index in enumerate(indices):
            trace = traces[trace_index]
            row = transition_by_step.get(
                (key[0], key[1], int(trace.get("control_step", 0)))
            )
            if row is None:
                continue
            comparison_index = indices[(local_index + offset) % len(indices)]
            current_action = np.asarray(
                trace["environment_action"], dtype=np.float64
            )
            distant_action = np.asarray(
                traces[comparison_index]["environment_action"], dtype=np.float64
            )
            row["distant_continuous_l2"] = float(
                np.linalg.norm(current_action - distant_action)
            )
    return transition_rows, lag_rows


def aggregate_lags(lag_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, bool], list[dict]] = defaultdict(list)
    for row in lag_rows:
        grouped[(int(row["lag"]), bool(row["episode_success"]))].append(row)
    result = []
    for (lag, success), rows in sorted(grouped.items()):
        result.append(
            {
                "lag": lag,
                "episode_outcome": "success" if success else "failure",
                "transitions": len(rows),
                "exact_rate": float(np.mean([row["token_exact"] for row in rows])),
                "mean_prefix": float(
                    np.mean([row["token_common_prefix"] for row in rows])
                ),
                "median_continuous_l2": float(
                    np.median([row["continuous_l2"] for row in rows])
                ),
            }
        )
    return result


def aggregate_strata(rows: list[dict]) -> list[dict]:
    positive_norms = [float(row["action_norm"]) for row in rows]
    boundary = float(np.median(positive_norms)) if positive_norms else 0.0
    grouped: dict[tuple[str, str, bool], list[dict]] = defaultdict(list)
    for row in rows:
        magnitude = "high-motion" if float(row["action_norm"]) >= boundary else "low-motion"
        grouped[(str(row["phase"]), magnitude, bool(row["episode_success"]))].append(row)
    return [
        {
            "phase": key[0],
            "magnitude": key[1],
            "episode_outcome": "success" if key[2] else "failure",
            "transitions": len(group),
            "exact_rate": float(np.mean([row["target_lag1_exact"] for row in group])),
            "mean_prefix": float(np.mean([row["target_lag1_prefix"] for row in group])),
        }
        for key, group in sorted(grouped.items())
    ]


def stage_rows(summary: dict, source: str) -> list[dict]:
    generation = summary.get("generation") or {}
    steps = max(int(generation.get("num_steps", 0)), 1)
    rows = []
    for stage, values in (generation.get("stage_profile") or {}).items():
        rows.append(
            {
                "source": source,
                "stage": stage,
                "label": STAGE_LABELS.get(stage, stage.replace("_", " ").title()),
                "calls": int(values.get("calls", 0)),
                "mean_ms_per_call": float(values.get("mean_ms", 0.0)),
                "total_ms": float(values.get("total_ms", 0.0)),
                "mean_ms_per_action": float(values.get("total_ms", 0.0)) / steps,
            }
        )
    return rows


def method_point(
    label: str,
    summary: dict,
    ar_mean: float,
    baseline_tail_blocks_per_target: float,
) -> dict:
    generation = summary.get("generation") or {}
    hold = generation.get("temporal_hold") or {}
    steps = max(int(generation.get("num_steps", 0)), 1)
    hold_actions = int(hold.get("allowed_holds", 0) or 0)
    target_frames = max(steps - hold_actions, 1)
    fused = int(generation.get("temporal_prefill_fused_actions", 0) or 0)
    full = int(generation.get("temporal_prefill_full_match_actions", 0) or 0)
    tail_blocks = max(
        int(generation.get("num_blocks", 0)) - fused - hold_actions,
        0,
    )
    tail_retention = tail_blocks / max(
        target_frames * baseline_tail_blocks_per_target,
        1e-9,
    )
    mean = float((summary.get("timing") or {}).get("mean"))
    return {
        "label": label,
        "measured_speedup": ar_mean / mean,
        "target_frame_rate": target_frames / steps,
        "vtpf_full_match_rate": full / fused if fused else 0.0,
        "tail_work_elimination_rate": 1.0 - tail_retention,
        "hold_cost_ms": float((summary.get("timing") or {}).get("median")) * 1000.0,
        "target_frames": target_frames,
        "tail_blocks": tail_blocks,
    }


def plot_cost(
    output: Path,
    profile_rows: list[dict],
    parity_rows: list[dict],
    ar_summary: dict | None,
    method_summaries: list[tuple[str, dict]],
) -> dict:
    profile_by_stage = {row["stage"]: row for row in profile_rows}
    parity_by_stage = {row["stage"]: row for row in parity_rows}
    prefill_ms = float(
        profile_by_stage.get("target_prefill", {}).get("mean_ms_per_action", 0.0)
    )
    tail_ms = sum(
        float(row["mean_ms_per_action"])
        for row in profile_rows
        if row["stage"] != "target_prefill"
    )
    ar_tail_ms = float(
        parity_by_stage.get("target_ar_reference_tail", {}).get(
            "mean_ms_per_action",
            parity_by_stage.get("target_ar_reference", {}).get(
                "mean_ms_per_action", 0.0
            ),
        )
    )
    ar_model_ms = prefill_ms + ar_tail_ms
    if min(prefill_ms, tail_ms, ar_tail_ms) <= 0.0:
        raise ValueError("Profiler did not record prefill, DFlash tail, and AR reference costs.")

    ar_mean_ms = (
        float((ar_summary.get("timing") or {})["mean"]) * 1000.0
        if ar_summary is not None
        else ar_model_ms
    )
    overhead_ms = max(ar_mean_ms - ar_model_ms, 0.0)
    elimination_values = np.linspace(0.0, 1.0, 101)
    rho_values = np.linspace(0.25, 1.0, 101)
    elimination_grid, rho_grid = np.meshgrid(elimination_values, rho_values)
    nominal_hold_ms = 0.4
    predicted_ms = rho_grid * (
        prefill_ms + (1.0 - elimination_grid) * tail_ms
    ) + (1.0 - rho_grid) * nominal_hold_ms
    speedup = ar_model_ms / np.maximum(predicted_ms, 1e-9)

    measured_points = []
    if ar_summary is not None:
        ar_mean = float((ar_summary.get("timing") or {})["mean"])
        baseline_tail_blocks_per_target = (
            float(profile_by_stage["target_anchor"]["calls"])
            / max(int(profile_by_stage["target_prefill"]["calls"]), 1)
        )
        for label, summary in method_summaries:
            point = method_point(
                label,
                summary,
                ar_mean,
                baseline_tail_blocks_per_target,
            )
            point["predicted_speedup"] = ar_model_ms / (
                point["target_frame_rate"]
                * (
                    prefill_ms
                    + (1.0 - point["tail_work_elimination_rate"]) * tail_ms
                )
                + (1.0 - point["target_frame_rate"]) * point["hold_cost_ms"]
            )
            measured_points.append(point)

    with iclr_style():
        fig, axes = plt.subplots(1, 3, figsize=(ICLR_DOUBLE_COLUMN_IN, 2.15))
        components = sorted(profile_rows, key=lambda row: row["mean_ms_per_action"], reverse=True)
        axes[0].barh(
            [row["label"] for row in components][::-1],
            [row["mean_ms_per_action"] for row in components][::-1],
            color=COLORS["blue"],
        )
        axes[0].set_xlabel("Latency per action (ms)")
        axes[0].set_title("(a) Per-action cost", loc="left", fontweight="bold")
        axes[0].grid(axis="x")

        image = axes[1].imshow(
            speedup,
            origin="lower",
            aspect="auto",
            extent=[0.0, 1.0, 0.25, 1.0],
            cmap="viridis",
        )
        axes[1].set_xlabel("Tail work eliminated")
        axes[1].set_ylabel("Target-call rate")
        axes[1].set_title("(b) Analytical speedup", loc="left", fontweight="bold")
        contour = axes[1].contour(
            elimination_grid,
            rho_grid,
            speedup,
            levels=[1, 2, 3, 4],
            colors="white",
            linewidths=0.6,
        )
        axes[1].clabel(contour, inline=True, fontsize=6, fmt="%gx")
        fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03, label="Speedup")

        if measured_points:
            predicted = [point["predicted_speedup"] for point in measured_points]
            measured = [point["measured_speedup"] for point in measured_points]
            limit = max(predicted + measured + [1.0]) * 1.08
            axes[2].plot([0, limit], [0, limit], color=COLORS["gray"], linestyle="--")
            axes[2].scatter(predicted, measured, color=COLORS["red"], zorder=3)
            for point in measured_points:
                axes[2].annotate(
                    point["label"],
                    (point["predicted_speedup"], point["measured_speedup"]),
                    xytext=(3, 2),
                    textcoords="offset points",
                    fontsize=6,
                )
            axes[2].set_xlim(0, limit)
            axes[2].set_ylim(0, limit)
            axes[2].set_xlabel("Predicted speedup")
            axes[2].set_ylabel("Measured speedup")
        else:
            rho_curve = np.linspace(0.25, 1.0, 100)
            for eliminated, color in (
                (0.0, COLORS["gray"]),
                (0.5, COLORS["orange"]),
                (1.0, COLORS["green"]),
            ):
                curve = ar_model_ms / (
                    rho_curve * (prefill_ms + (1.0 - eliminated) * tail_ms)
                    + (1.0 - rho_curve) * nominal_hold_ms
                )
                axes[2].plot(
                    rho_curve,
                    curve,
                    label=f"tail eliminated={eliminated:.1f}",
                    color=color,
                )
            axes[2].set_xlabel("Target-frame rate")
            axes[2].set_ylabel("Predicted speedup")
            axes[2].legend(frameon=False)
        axes[2].set_title("(c) Prediction audit", loc="left", fontweight="bold")
        axes[2].grid()
        fig.tight_layout(w_pad=0.8)
        save_figure(fig, output)
        plt.close(fig)
    return {
        "prefill_ms": prefill_ms,
        "dflash_tail_ms": tail_ms,
        "target_ar_tail_ms": ar_tail_ms,
        "ar_mean_ms": ar_mean_ms,
        "estimated_wrapper_overhead_ms": overhead_ms,
        "measured_points": measured_points,
    }


def plot_temporal(output: Path, transitions: list[dict], lag_rows: list[dict]) -> None:
    with iclr_style():
        fig, axes = plt.subplots(2, 2, figsize=(ICLR_DOUBLE_COLUMN_IN, 4.25))
        for success, color, label in (
            (True, COLORS["blue"], "Successful episodes"),
            (False, COLORS["orange"], "Failed episodes"),
        ):
            exact = []
            for lag in range(1, 5):
                rows = [row for row in lag_rows if row["lag"] == lag and row["episode_success"] == success]
                exact.append(float(np.mean([row["token_exact"] for row in rows])) if rows else np.nan)
            axes[0, 0].plot(range(1, 5), exact, marker="o", color=color, label=label)
        axes[0, 0].set_xticks(range(1, 5))
        axes[0, 0].set_xlabel("Temporal lag")
        axes[0, 0].set_ylabel("7-D action exact-repeat rate")
        axes[0, 0].set_title("(a) Action persistence", loc="left", fontweight="bold")
        axes[0, 0].legend(frameon=False)
        axes[0, 0].grid()

        for success, color, label in (
            (True, COLORS["blue"], "Success"),
            (False, COLORS["orange"], "Failure"),
        ):
            values = sorted(
                int(row["target_lag1_prefix"])
                for row in transitions
                if bool(row["episode_success"]) == success
            )
            if values:
                axes[0, 1].step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=color, label=label)
        axes[0, 1].set_xlim(0, 7)
        axes[0, 1].set_xticks(range(8))
        axes[0, 1].set_xlabel("Lag-1 common prefix length")
        axes[0, 1].set_ylabel("Empirical CDF")
        axes[0, 1].set_title("(b) Strict prefix persistence", loc="left", fontweight="bold")
        axes[0, 1].legend(frameon=False)
        axes[0, 1].grid()

        density_rows = [
            row
            for row in transitions
            if row.get("pixel_relative_l2") is not None
            and math.isfinite(float(row["pixel_relative_l2"]))
        ]
        if density_rows:
            histogram = axes[1, 0].hexbin(
                [float(row["pixel_relative_l2"]) for row in density_rows],
                [float(row["continuous_l2"]) for row in density_rows],
                gridsize=30,
                mincnt=1,
                bins="log",
                cmap="viridis",
            )
            fig.colorbar(histogram, ax=axes[1, 0], fraction=0.046, pad=0.03, label="log count")
        axes[1, 0].set_xlabel("Adjacent-frame visual relative L2")
        axes[1, 0].set_ylabel("Continuous action L2")
        axes[1, 0].set_title("(c) Visual/action deconfounding", loc="left", fontweight="bold")

        adjacent = [float(row["continuous_l2"]) for row in transitions]
        distant = [
            float(row["distant_continuous_l2"])
            for row in transitions
            if row.get("distant_continuous_l2") is not None
        ]
        boxes = axes[1, 1].boxplot(
            [adjacent, distant],
            tick_labels=["Adjacent", "Distant\nwithin episode"],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for box, color in zip(
            boxes["boxes"], (COLORS["blue"], COLORS["gray"])
        ):
            box.set(facecolor=color, alpha=0.7)
        axes[1, 1].set_ylabel("Continuous action distance (L2)")
        axes[1, 1].set_title("(d) Temporal control", loc="left", fontweight="bold")
        axes[1, 1].grid(axis="y")
        fig.tight_layout(w_pad=0.9, h_pad=1.0)
        save_figure(fig, output)
        plt.close(fig)


def parity_rows(summary: dict) -> list[dict]:
    parity = ((summary.get("generation") or {}).get("vtpf_parity") or {})
    rows = []
    for record_index, record in enumerate(parity.get("records") or []):
        for position in record.get("per_position", []):
            rows.append(
                {
                    "record": record_index,
                    "position": int(position["position"]),
                    "top1_match": int(bool(position["top1_match"])),
                    "max_abs_logit_difference": float(position["max_abs_logit_difference"]),
                    "mean_abs_logit_difference": float(position["mean_abs_logit_difference"]),
                    "serial_divergent_accepted_tokens_in_action": int(
                        record.get("serial_divergent_accepted_tokens", 0)
                    ),
                    "strict_accepted_prefix_length": int(
                        record.get("strict_accepted_prefix_length", 0)
                    ),
                    "causal_comparable_positions": int(record.get("causal_comparable_positions", 0)),
                }
            )
    return rows


def plot_parity(output: Path, rows: list[dict], parity: dict) -> None:
    if not rows:
        raise ValueError("Parity summary contains no causally comparable logit records.")
    positions = sorted({int(row["position"]) for row in rows})
    with iclr_style():
        fig, axes = plt.subplots(1, 2, figsize=(ICLR_DOUBLE_COLUMN_IN, 2.15))
        data = [
            [float(row["max_abs_logit_difference"]) for row in rows if row["position"] == position]
            for position in positions
        ]
        boxes = axes[0].boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
        for box in boxes["boxes"]:
            box.set(facecolor=COLORS["sky"], edgecolor=COLORS["blue"], alpha=0.75)
        axes[0].set_xlabel("Action-token position")
        axes[0].set_ylabel("Maximum absolute logit difference")
        axes[0].set_title("(a) Finite-precision target audit", loc="left", fontweight="bold")
        axes[0].grid(axis="y")

        comparable = int(parity.get("causal_comparable_positions", 0))
        top1_mismatches = int(parity.get("top1_mismatches", 0))
        accepted = int(parity.get("strict_accepted_tokens", 0))
        divergent = int(parity.get("serial_divergent_accepted_tokens", 0))
        values = [
            100.0 * top1_mismatches / max(comparable, 1),
            100.0 * divergent / max(accepted, 1),
        ]
        labels = [
            "Comparable-logit\ntop-1 disagreement",
            "Verifier-accepted token\nvs. serial AR",
        ]
        bars = axes[1].bar(
            labels, values, color=[COLORS["orange"], COLORS["red"]]
        )
        axes[1].bar_label(
            bars,
            labels=[f"{top1_mismatches}/{comparable}", f"{divergent}/{accepted}"],
            padding=2,
            fontsize=7,
        )
        axes[1].set_ylabel("Disagreement rate (%)")
        axes[1].set_title(
            "(b) Fused verifier vs. serial AR", loc="left", fontweight="bold"
        )
        axes[1].grid(axis="y")
        fig.tight_layout(w_pad=1.0)
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

    profile = load_json(args.profile_summary)
    shadow = load_json(args.shadow_summary)
    parity_summary = load_json(args.parity_summary)
    ar_summary = load_json(args.ar_summary) if args.ar_summary else None
    method_summaries = []
    method_paths = []
    for value in args.method_summary:
        label, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}")
        path = Path(path_text)
        method_summaries.append((label, load_json(path)))
        method_paths.append(path)

    profile_rows = stage_rows(profile, "plain_dflash")
    parity_stage_rows = stage_rows(parity_summary, "vtpf_parity")
    transitions, lag_rows = build_temporal_rows(shadow)
    lag_summary = aggregate_lags(lag_rows)
    strata = aggregate_strata(transitions)
    parity_table = parity_rows(parity_summary)

    write_csv(tables / "stage_profile.csv", profile_rows + parity_stage_rows)
    write_csv(tables / "temporal_transitions.csv", transitions)
    write_csv(tables / "temporal_lag_rows.csv", lag_rows)
    write_csv(tables / "temporal_lag_summary.csv", lag_summary)
    write_csv(tables / "temporal_strata.csv", strata)
    write_csv(tables / "vtpf_parity.csv", parity_table)

    cost_model = plot_cost(
        figures / "fig2_cost_model",
        profile_rows,
        parity_stage_rows,
        ar_summary,
        method_summaries,
    )
    plot_temporal(figures / "fig3_temporal_persistence", transitions, lag_rows)
    parity_payload = (parity_summary.get("generation") or {}).get("vtpf_parity") or {}
    plot_parity(figures / "fig4_vtpf_parity", parity_table, parity_payload)
    (tables / "cost_model.json").write_text(
        json.dumps(cost_model, indent=2, ensure_ascii=False)
    )

    repo_root = args.repo_root or Path(__file__).resolve().parents[3]
    input_paths = [
        args.profile_summary,
        args.shadow_summary,
        args.parity_summary,
        *([args.ar_summary] if args.ar_summary else []),
        *method_paths,
    ]
    if args.raw_run_root:
        raw_run_root = args.raw_run_root.resolve()
        archive_paths = [path for path in raw_run_root.rglob("*") if path.is_file()]
        archived = [
            archive_input(path, raw, path.relative_to(raw_run_root))
            for path in sorted(archive_paths)
        ]
        archived_sources = {str(path.resolve()) for path in archive_paths}
        for path in input_paths:
            if str(path.resolve()) not in archived_sources:
                archived.append(archive_input(path.resolve(), raw))
    else:
        archived = [archive_input(path.resolve(), raw) for path in input_paths]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "git_commit": git_output(repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": git_output(repo_root, "status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": archived,
        "counts": {
            "temporal_transitions": len(transitions),
            "temporal_lag_rows": len(lag_rows),
            "vtpf_parity_positions": len(parity_table),
        },
        "claim_boundaries": {
            "profile": "Synchronized diagnostic stages; not paper-style end-to-end timing.",
            "temporal": "Descriptive persistence evidence; not a recoverability guarantee.",
            "parity": (
                "Only causally comparable fused/serial positions are tested; "
                "finite-precision disagreements are reported, not hidden."
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    report = f"""# P0 evidence pack: {args.suite}

This directory is generated from immutable evaluation summaries. Raw inputs are
stored as deterministic gzip files with SHA-256 hashes in `manifest.json`.

## Current evidence

- Profiler actions: {(profile.get('generation') or {}).get('num_steps')}
- Temporal transitions: {len(transitions)}
- VTPF causally comparable logit positions: {parity_payload.get('causal_comparable_positions')}
- VTPF top-1 mismatches: {parity_payload.get('top1_mismatches')}
- Fused-verifier accepted tokens that differ from serial AR: {parity_payload.get('serial_divergent_accepted_tokens')} / {parity_payload.get('strict_accepted_tokens')}

## Interpretation boundary

Figure 2 diagnoses the model-call cost structure. Figure 3 establishes temporal
action persistence after outcome/phase stratification but does not establish
closed-loop recoverability. Figure 4 compares a fused verifier with an
independent token-by-token AR run only where both share the same causal prefix.
It explicitly reports finite-precision top-1 and accepted-token divergences;
positions after the first candidate mismatch are excluded by construction.
Counterfactual state-fork evidence remains a separate P0 experiment.
"""
    (output / "README.md").write_text(report)


if __name__ == "__main__":
    main()
