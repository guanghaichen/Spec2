#!/usr/bin/env python3
"""Build the immutable paper-result index for the finalized method.

Historical evaluation directories remain untouched. Each canonical run stores
hard links to the original log triplet, compact metrics, provenance, and file
checksums. The registry below is intentionally explicit: adding a paper number
requires naming its exact source run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


DATA_ROOT = Path(
    "/media/asus/1070ecbd-49b3-49fc-a60e-1a5d109d9f55/cgh/specvla-data"
)
OUT_ROOT = DATA_ROOT / "paper_results"


# (table, suite, method, run_tag, summary path relative to DATA_ROOT)
BASELINES = [
    ("goal", "openvla_ar", "official-500-s7", "eval_logs/baseline/openvla_ar/EVAL-libero_goal-openvla-2026_07_20-13_15_47--specvla-paper-ar-goal-openvla_ar_summary.json"),
    ("spatial", "openvla_ar", "official-500-s7", "eval_logs/baseline/openvla_ar/EVAL-libero_spatial-openvla-2026_07_20-22_53_00--specvla-paper-ar-spatial-openvla_ar_summary.json"),
    ("object", "openvla_ar", "official-500-s7", "eval_logs/baseline/openvla_ar/EVAL-libero_object-openvla-2026_07_20-18_04_58--specvla-paper-ar-object-openvla_ar_summary.json"),
    ("libero_10", "openvla_ar", "official-500-s7", "eval_logs/baseline/openvla_ar/EVAL-libero_10-openvla-2026_07_21-02_38_04--specvla-paper-ar-10-openvla_ar_summary.json"),
    ("goal", "specvla_strict", "official-500-s7-r0", "eval_logs/baseline/specvla_strict/EVAL-libero_goal-openvla-2026_07_21-14_20_56--specvla-strict-goal-r0-specvla_strict_summary.json"),
    ("spatial", "specvla_strict", "official-500-s7-r0", "eval_logs/baseline/specvla_strict/EVAL-libero_spatial-openvla-2026_07_22-07_15_20--specvla-strict-spatial-r0-specvla_strict_summary.json"),
    ("object", "specvla_strict", "official-500-s7-r0", "eval_logs/baseline/specvla_strict/EVAL-libero_object-openvla-2026_07_21-22_35_09--specvla-strict-object-r0-specvla_strict_summary.json"),
    ("libero_10", "specvla_strict", "official-500-s7-r0", "eval_logs/baseline/specvla_strict/EVAL-libero_10-openvla-2026_07_22-14_20_15--specvla-strict-10-r0-specvla_strict_summary.json"),
    ("goal", "specvla_relaxed", "official-500-s7-r9", "eval_logs/baseline/specvla_relaxed/EVAL-libero_goal-openvla-2026_07_21-18_45_47--specvla-relaxed-goal-r9-specvla_relaxed_summary.json"),
    ("spatial", "specvla_relaxed", "official-500-s7-r9", "eval_logs/baseline/specvla_relaxed/EVAL-libero_spatial-openvla-2026_07_22-11_01_28--specvla-relaxed-spatial-r9-specvla_relaxed_summary.json"),
    ("object", "specvla_relaxed", "official-500-s7-r9", "eval_logs/baseline/specvla_relaxed/EVAL-libero_object-openvla-2026_07_22-03_09_28--specvla-relaxed-object-r9-specvla_relaxed_summary.json"),
    ("libero_10", "specvla_relaxed", "official-500-s7-r5", "eval_logs/baseline/specvla_relaxed/EVAL-libero_10-openvla-2026_07_23-01_33_12--specvla-relaxed-10-r5-specvla_relaxed_summary.json"),
]

CURRENT_METHODS = [
    ("goal", "dflash_strict", "minimal-e100-500-s7", "main_table/goal/dflash_strict/parallel-block-e100-r7/dflash_strict/EVAL-libero_goal-openvla-2026_08_04-19_19_47--main-table-goal-parallel-block-strict-e100-r7-dflash_strict_summary.json"),
    ("spatial", "dflash_strict", "minimal-e100-500-s7", "eval_logs/spatial/dflash_strict/DFlash-e100/dflash_strict/EVAL-libero_spatial-openvla-2026_08_01-13_13_32--dflash-minimal-spatial-e100-s7-strict-dflash_strict_summary.json"),
    ("object", "dflash_strict", "minimal-e060-500-s7", "eval_logs/object/dflash_strict/DFlash-e60/dflash_strict/EVAL-libero_object-openvla-2026_08_06-17_21_08--dflash-minimal-object-e60-s7-strict-dflash_strict_summary.json"),
    ("libero_10", "dflash_strict", "minimal-e100-500-s7", "eval_logs/10/dflash_strict/DFlash-e100/dflash_strict/EVAL-libero_10-openvla-2026_08_07-19_20_24--dflash-minimal-10-e100-s7-strict-dflash_strict_summary.json"),
    ("goal", "vtpf_strict", "minimal-e100-500-s7", "eval_logs/goal/dflash_strict/简化版Draft+VTPF-e100/EVAL-libero_goal-openvla-2026_07_29-17_58_54--dflash-minimal-goal-e100-s7-vtpf-strict-dflash_strict_summary.json"),
    ("spatial", "vtpf_strict", "minimal-e100-500-s7", "eval_logs/spatial/dflash_strict/DFlash+VTPF-e100/dflash_strict/EVAL-libero_spatial-openvla-2026_08_01-16_33_58--dflash-minimal-spatial-e100-s7-vtpf-strict-dflash_strict_summary.json"),
    ("object", "vtpf_strict", "minimal-e060-500-s7", "eval_logs/object/dflash_strict/DFlash+VTPF-e60/dflash_strict/EVAL-libero_object-openvla-2026_08_06-21_34_54--dflash-minimal-object-e60-s7-vtpf-strict-dflash_strict_summary.json"),
    ("libero_10", "vtpf_strict", "minimal-e100-500-s7", "eval_logs/10/dflash_strict/DFlash+VTPF-e100/dflash_strict/EVAL-libero_10-openvla-2026_08_08-05_18_42--dflash-minimal-10-e100-s7-vtpf-strict-dflash_strict_summary.json"),
    ("goal", "paced_harmonic", "minimal-e100-500-s7", "eval_logs/goal/dflash_relaxed/paced_harmonic_formal/EVAL-libero_goal-openvla-2026_07_31-12_50_22--dflash-vtpf-paced-harmonic-goal-e100-s7-formal-dflash_relaxed_summary.json"),
    ("spatial", "paced_harmonic", "minimal-e100-500-s7", "eval_logs/spatial/dflash_relaxed/简化版Draft+VTPF-TD-PacedHarmonic-e100/dflash_relaxed/EVAL-libero_spatial-openvla-2026_08_01-19_50_33--dflash-minimal-spatial-e100-s7-vtpf-paced-harmonic-dflash_relaxed_summary.json"),
    ("object", "paced_harmonic", "minimal-e060-500-s7", "eval_logs/object/dflash_relaxed/简化版Draft+VTPF-TD-PacedHarmonic-e60/dflash_relaxed/EVAL-libero_object-openvla-2026_08_07-01_40_57--dflash-minimal-object-e60-s7-vtpf-paced-harmonic-dflash_relaxed_summary.json"),
    ("libero_10", "paced_harmonic", "minimal-e100-500-s7", "eval_logs/10/dflash_relaxed/简化版Draft+VTPF-TD-PacedHarmonic-e100/dflash_relaxed/EVAL-libero_10-openvla-2026_08_08-14_29_50--dflash-minimal-10-e100-s7-vtpf-paced-harmonic-dflash_relaxed_summary.json"),
]

PACE_HARMONIC_ABLATION = [
    ("goal", "no_pace_no_harmonic", "minimal-e100-500-s7", "eval_logs/goal/ablation_paced_harmonic_2x2/no_paced_no_harmonic/dflash_relaxed/简化版Draft+VTPF-TD-VisualBudget-e100/dflash_relaxed/EVAL-libero_goal-openvla-2026_08_07-03_52_01--goal-e100-s7-no_paced_no_harmonic-dflash_relaxed_summary.json"),
    ("goal", "pace_only", "minimal-e100-500-s7", "eval_logs/goal/ablation_paced_harmonic_2x2/paced_only/dflash_relaxed/简化版Draft+VTPF-TD-PacedBudget-e100/dflash_relaxed/EVAL-libero_goal-openvla-2026_08_07-15_07_15--goal-e100-s7-paced_only-dflash_relaxed_summary.json"),
    ("goal", "harmonic_only", "minimal-e100-500-s7", "eval_logs/goal/ablation_paced_harmonic_2x2/harmonic_only/dflash_relaxed/简化版Draft+VTPF-TD-VisualBudget-e100/dflash_relaxed/EVAL-libero_goal-openvla-2026_08_07-17_18_35--goal-e100-s7-harmonic_only-dflash_relaxed_summary.json"),
    ("goal", "paced_harmonic", "minimal-e100-500-s7", "eval_logs/goal/dflash_relaxed/paced_harmonic_formal/EVAL-libero_goal-openvla-2026_07_31-12_50_22--dflash-vtpf-paced-harmonic-goal-e100-s7-formal-dflash_relaxed_summary.json"),
]

RUNS = (
    [("main_table", *run) for run in BASELINES]
    + [("main_table", *run) for run in CURRENT_METHODS]
    + [("ablation", suite, method, tag, path) for suite, method, tag, path in CURRENT_METHODS]
    + [("pace_harmonic_ablation", *run) for run in PACE_HARMONIC_ABLATION]
)


def run_prefix(summary: Path) -> str:
    for suffix in (
        "-openvla_ar_summary.json",
        "-specvla_strict_summary.json",
        "-specvla_relaxed_summary.json",
        "-dflash_strict_summary.json",
        "-dflash_relaxed_summary.json",
    ):
        if summary.name.endswith(suffix):
            return summary.name[: -len(suffix)]
    raise ValueError(f"Unrecognized summary suffix: {summary}")


def scalar_metrics(data: dict) -> dict:
    generation = data.get("generation") or {}
    hold = generation.get("temporal_hold") or {}
    fused = generation.get("temporal_prefill_fused_actions") or 0
    full = generation.get("temporal_prefill_full_match_actions") or 0
    return {
        "suite": data.get("task_suite_name"),
        "eval_family": data.get("eval_family"),
        "episodes": data.get("total_episodes"),
        "successes": data.get("total_successes"),
        "success_rate": data.get("success_rate"),
        "seed": data.get("seed"),
        "mean_latency_s": (data.get("timing") or {}).get("mean"),
        "timed_steps": (data.get("timing") or {}).get("steps"),
        "length": generation.get("length"),
        "avg_accept_length": generation.get("avg_accept_length"),
        "overall_hit_rate": generation.get("overall_hit_rate"),
        "target_prefill_rate": hold.get("target_prefill_rate"),
        "vtpf_full_match_rate": full / fused if fused else None,
        "checkpoint": data.get("spec_checkpoint"),
        "target_checkpoint": data.get("pretrained_checkpoint"),
        "timing_scope": data.get("timing_scope"),
        "sync_cuda_timing": data.get("sync_cuda_timing"),
    }


def hardlink_run(summary: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    prefix = run_prefix(summary)
    sources = sorted(path for path in summary.parent.glob(prefix + "*") if path.is_file())
    if not sources:
        raise FileNotFoundError(f"No run files found for {summary}")
    linked = []
    for source in sources:
        target = destination / source.name
        if target.exists():
            if os.path.samefile(source, target):
                linked.append(target)
                continue
            raise FileExistsError(f"Refusing to replace unrelated file: {target}")
        os.link(source, target)
        linked.append(target)
    return linked


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    loaded = []
    for table, suite, method, tag, relative_summary in RUNS:
        summary = DATA_ROOT / relative_summary
        if not summary.is_file():
            raise FileNotFoundError(summary)
        data = json.loads(summary.read_text(encoding="utf-8"))
        loaded.append((table, suite, method, tag, summary, scalar_metrics(data)))

    ar_latency = {
        suite: metrics["mean_latency_s"]
        for table, suite, method, _, _, metrics in loaded
        if table == "main_table" and method == "openvla_ar"
    }
    rows = []
    for table, suite, method, tag, summary, metrics in loaded:
        destination = OUT_ROOT / table / suite / method / tag
        linked = hardlink_run(summary, destination)
        mean = metrics["mean_latency_s"]
        metrics["speedup_vs_ar"] = ar_latency[suite] / mean if mean else None
        metrics.update(
            {
                "table": table,
                "suite_slug": suite,
                "method": method,
                "run_tag": tag,
                "source_summary": str(summary),
                "canonical_directory": str(destination),
            }
        )
        write_text(destination / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
        write_text(destination / "SOURCE.txt", str(summary.parent) + "\n")
        checksums = [
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            for path in linked
        ]
        write_text(destination / "MANIFEST.sha256", "\n".join(checksums) + "\n")
        rows.append(metrics)

    manifest_dir = OUT_ROOT / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "table", "suite_slug", "method", "run_tag", "episodes", "successes",
        "success_rate", "seed", "mean_latency_s", "speedup_vs_ar", "timed_steps",
        "length", "avg_accept_length", "overall_hit_rate", "target_prefill_rate",
        "vtpf_full_match_rate", "checkpoint", "target_checkpoint", "timing_scope",
        "sync_cuda_timing", "source_summary", "canonical_directory",
    ]
    with (manifest_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    write_text(manifest_dir / "runs.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")

    evidence_link = OUT_ROOT / "evidence" / "p0"
    evidence_source = DATA_ROOT / "evidence" / "p0"
    evidence_link.parent.mkdir(parents=True, exist_ok=True)
    if evidence_link.is_symlink():
        if evidence_link.resolve() != evidence_source.resolve():
            raise RuntimeError(f"Unexpected link target: {evidence_link}")
    elif evidence_link.exists():
        raise FileExistsError(evidence_link)
    else:
        evidence_link.symlink_to(evidence_source, target_is_directory=True)


if __name__ == "__main__":
    main()
