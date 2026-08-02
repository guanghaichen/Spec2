#!/usr/bin/env python3
"""Render cross-suite risk landscapes from frozen calibration profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


SUITE_LABELS = {
    "libero_goal": "Goal",
    "libero_object": "Object",
    "libero_spatial": "Spatial",
    "libero_10": "Long",
}
SCHEDULE_MARKERS = {
    "minimum_prefix_discrepancy": "o",
    "maximum_gap_concentration": "s",
    "legacy": "D",
    "target_reference": "^",
}


def load_profile(path: Path) -> dict:
    profile = json.loads(path.read_text())
    required = {
        "profile_id",
        "task_suite_name",
        "risk_budget",
        "selected_configuration",
        "all_statistics",
    }
    missing = required.difference(profile)
    if missing:
        raise ValueError(f"{path} is missing profile fields: {sorted(missing)}")
    profile["_path"] = str(path)
    return profile


def selected_name(profile: dict) -> str:
    return str(profile["selected_configuration"]["name"])


def plot_profiles(profiles: list[dict], output_path: Path) -> None:
    columns = min(2, len(profiles))
    rows = math.ceil(len(profiles) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5, 2.55 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    exponent_values = [
        float(row["authority_exponent"])
        for profile in profiles
        for row in profile["all_statistics"]
        if not bool(row.get("deterministic_reference", False))
    ]
    norm = Normalize(
        vmin=min(exponent_values, default=0.0),
        vmax=max(exponent_values, default=1.0) or 1.0,
    )
    cmap = plt.get_cmap("viridis")

    for axis, profile in zip(axes.flat, profiles):
        selected = selected_name(profile)
        for row in profile["all_statistics"]:
            reference = bool(row.get("deterministic_reference", False))
            kind = str(row.get("schedule_kind", "legacy"))
            marker = SCHEDULE_MARKERS.get(kind, "D")
            target_rate = float(row["mean_target_rate"])
            upper = float(row["harm_upper_bound"])
            color = "#444444" if reference else cmap(norm(float(row["authority_exponent"])))
            face = color if bool(row["feasible"]) else "none"
            axis.scatter(
                target_rate,
                upper,
                marker=marker,
                s=38,
                facecolors=face,
                edgecolors=color,
                linewidths=1.0,
                zorder=3,
            )
            if str(row["configuration"]) == selected:
                axis.scatter(
                    target_rate,
                    upper,
                    marker="*",
                    s=105,
                    facecolors="#d62728",
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=4,
                )
        axis.axhline(
            float(profile["risk_budget"]),
            color="#b22222",
            linestyle="--",
            linewidth=1.0,
        )
        axis.set_title(SUITE_LABELS.get(profile["task_suite_name"], profile["task_suite_name"]))
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)

    for axis in axes.flat[len(profiles) :]:
        axis.axis("off")
    fig.supxlabel("Target-call rate", y=0.19, fontsize=9)
    fig.supylabel("Simultaneous harm upper bound", x=0.015, fontsize=9)

    legend = [
        Line2D(
            [0], [0], marker="o", color="none", markeredgecolor="#333333",
            markerfacecolor="none", label="Minimum prefix discrepancy"
        ),
        Line2D(
            [0], [0], marker="s", color="none", markeredgecolor="#333333",
            markerfacecolor="none", label="Maximum gap concentration"
        ),
        Line2D(
            [0], [0], marker="*", color="none", markeredgecolor="white",
            markerfacecolor="#d62728", markersize=10, label="Selected profile"
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.44, 0.015),
        ncol=3,
        frameon=False,
        fontsize=6.5,
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        ax=[axis for axis in axes.flat if axis.axison],
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Authority exponent", fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.89, top=0.92, bottom=0.30, hspace=0.30, wspace=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_selected_table(profiles: list[dict], path: Path) -> None:
    rows = []
    for profile in profiles:
        selected = profile["selected_configuration"]
        statistics = profile["selected_statistics"]
        rows.append(
            {
                "task_suite_name": profile["task_suite_name"],
                "profile_id": profile["profile_id"],
                "configuration": selected["name"],
                "target_density": selected["target_density"],
                "schedule_kind": selected.get("schedule_kind", "legacy"),
                "schedule_offsets": ",".join(
                    str(int(value)) for value in selected.get("schedule_offsets", [])
                ),
                "authority_exponent": selected["authority_exponent"],
                "max_consecutive_holds": selected["max_consecutive_holds"],
                "empirical_harm_rate": statistics["empirical_harm_rate"],
                "harm_upper_bound": statistics["harm_upper_bound"],
                "mean_target_rate": statistics["mean_target_rate"],
                "profile_path": profile["_path"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--figure_name", default="cross_suite_recoverability_landscape")
    args = parser.parse_args()

    profiles = [load_profile(path) for path in args.profiles]
    profiles.sort(key=lambda item: item["task_suite_name"])
    plot_profiles(profiles, args.output_dir / args.figure_name)
    write_selected_table(profiles, args.output_dir / "frozen_profiles.csv")


if __name__ == "__main__":
    main()
