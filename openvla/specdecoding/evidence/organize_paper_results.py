#!/usr/bin/env python3
"""Build a canonical, non-destructive index of paper evaluation results.

The historical evaluation directories remain immutable.  Each canonical run
contains hard links to the original log triplet, a compact metrics file, and a
SOURCE.txt pointer.  Hard links avoid copying large summary JSON files while
surviving accidental removal of an old directory entry.
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


RUNS = [
    # Same-machine baselines.
    ("main_table", "goal", "openvla_ar", "official-500-s7", "eval_logs/baseline/openvla_ar/EVAL-libero_goal-openvla-2026_07_20-13_15_47--specvla-paper-ar-goal-openvla_ar_summary.json"),
    ("main_table", "spatial", "openvla_ar", "official-500-s7", "eval_logs/baseline/openvla_ar/EVAL-libero_spatial-openvla-2026_07_20-22_53_00--specvla-paper-ar-spatial-openvla_ar_summary.json"),
    ("main_table", "object", "openvla_ar", "official-500-s7", "eval_logs/baseline/openvla_ar/EVAL-libero_object-openvla-2026_07_20-18_04_58--specvla-paper-ar-object-openvla_ar_summary.json"),
    ("main_table", "libero_10", "openvla_ar", "official-500-s7", "eval_logs/baseline/openvla_ar/EVAL-libero_10-openvla-2026_07_21-02_38_04--specvla-paper-ar-10-openvla_ar_summary.json"),
    ("main_table", "goal", "specvla_strict", "official-500-s7-r0", "eval_logs/baseline/specvla_strict/EVAL-libero_goal-openvla-2026_07_21-14_20_56--specvla-strict-goal-r0-specvla_strict_summary.json"),
    ("main_table", "spatial", "specvla_strict", "official-500-s7-r0", "eval_logs/baseline/specvla_strict/EVAL-libero_spatial-openvla-2026_07_22-07_15_20--specvla-strict-spatial-r0-specvla_strict_summary.json"),
    ("main_table", "object", "specvla_strict", "official-500-s7-r0", "eval_logs/baseline/specvla_strict/EVAL-libero_object-openvla-2026_07_21-22_35_09--specvla-strict-object-r0-specvla_strict_summary.json"),
    ("main_table", "libero_10", "specvla_strict", "official-500-s7-r0", "eval_logs/baseline/specvla_strict/EVAL-libero_10-openvla-2026_07_22-14_20_15--specvla-strict-10-r0-specvla_strict_summary.json"),
    ("main_table", "goal", "specvla_relaxed", "official-500-s7-r9", "eval_logs/baseline/specvla_relaxed/EVAL-libero_goal-openvla-2026_07_21-18_45_47--specvla-relaxed-goal-r9-specvla_relaxed_summary.json"),
    ("main_table", "spatial", "specvla_relaxed", "official-500-s7-r9", "eval_logs/baseline/specvla_relaxed/EVAL-libero_spatial-openvla-2026_07_22-11_01_28--specvla-relaxed-spatial-r9-specvla_relaxed_summary.json"),
    ("main_table", "object", "specvla_relaxed", "official-500-s7-r9", "eval_logs/baseline/specvla_relaxed/EVAL-libero_object-openvla-2026_07_22-03_09_28--specvla-relaxed-object-r9-specvla_relaxed_summary.json"),
    ("main_table", "libero_10", "specvla_relaxed", "official-500-s7-r5", "eval_logs/baseline/specvla_relaxed/EVAL-libero_10-openvla-2026_07_23-01_33_12--specvla-relaxed-10-r5-specvla_relaxed_summary.json"),
    # Paper method: parallel block, strict temporal fusion, and RAES.
    ("main_table", "goal", "dflash_strict", "minimal-e100-500-s7", "main_table/goal/dflash_strict/parallel-block-e100-r7/dflash_strict/EVAL-libero_goal-openvla-2026_08_04-19_19_47--main-table-goal-parallel-block-strict-e100-r7-dflash_strict_summary.json"),
    ("main_table", "spatial", "dflash_strict", "minimal-e100-500-s7", "eval_logs/spatial/dflash_strict/DFlash-e100/dflash_strict/EVAL-libero_spatial-openvla-2026_08_01-13_13_32--dflash-minimal-spatial-e100-s7-strict-dflash_strict_summary.json"),
    ("main_table", "object", "dflash_strict", "minimal-e060-500-s7", "main_table/object/dflash_strict/parallel-block-e60-r7/dflash_strict/EVAL-libero_object-openvla-2026_08_05-02_37_46--main-table-object-parallel-block-strict-e60-r7-dflash_strict_summary.json"),
    ("main_table", "goal", "vtpf_strict", "minimal-e100-500-s7", "eval_logs/goal/dflash_strict/简化版Draft+VTPF-e100/EVAL-libero_goal-openvla-2026_07_29-17_58_54--dflash-minimal-goal-e100-s7-vtpf-strict-dflash_strict_summary.json"),
    ("main_table", "spatial", "vtpf_strict", "minimal-e100-500-s7", "eval_logs/spatial/dflash_strict/DFlash+VTPF-e100/dflash_strict/EVAL-libero_spatial-openvla-2026_08_01-16_33_58--dflash-minimal-spatial-e100-s7-vtpf-strict-dflash_strict_summary.json"),
    ("main_table", "goal", "raes_rho040", "minimal-e100-500-s7", "main_table/online_local_rho400_full500_goal_4090/dflash_relaxed/简化版Draft+VTPF-TD-RevealingBudget-e100/dflash_relaxed/EVAL-libero_goal-openvla-2026_08_04-14_22_47--local400-main-table-full500-goal-e100-dflash_relaxed_summary.json"),
    ("main_table", "spatial", "raes_rho040", "minimal-e100-500-s7", "main_table/online_local_rho400_full500_spatial_4090/dflash_relaxed/简化版Draft+VTPF-TD-RevealingBudget-e100/dflash_relaxed/EVAL-libero_spatial-openvla-2026_08_04-23_22_54--local400-main-table-full500-spatial-e100-dflash_relaxed_summary.json"),
]


def run_prefix(summary: Path) -> str:
    name = summary.name
    for suffix in (
        "-openvla_ar_summary.json",
        "-specvla_strict_summary.json",
        "-specvla_relaxed_summary.json",
        "-dflash_strict_summary.json",
        "-dflash_relaxed_summary.json",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
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
    sources = sorted(p for p in summary.parent.glob(prefix + "*") if p.is_file())
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
    rows = []
    loaded = []

    for table, suite, method, tag, relative_summary in RUNS:
        summary = DATA_ROOT / relative_summary
        if not summary.is_file():
            raise FileNotFoundError(summary)
        data = json.loads(summary.read_text(encoding="utf-8"))
        metrics = scalar_metrics(data)
        loaded.append((table, suite, method, tag, summary, metrics))

    ar_latency = {
        suite: metrics["mean_latency_s"]
        for _, suite, method, _, _, metrics in loaded
        if method == "openvla_ar"
    }

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
        checksums = []
        for path in linked:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checksums.append(f"{digest}  {path.name}")
        write_text(destination / "MANIFEST.sha256", "\n".join(checksums) + "\n")
        rows.append(metrics)

    # The ablation view reuses the exact same immutable runs.
    for suite in ("goal", "spatial"):
        for method in ("openvla_ar", "dflash_strict", "vtpf_strict", "raes_rho040"):
            matches = [r for r in rows if r["suite_slug"] == suite and r["method"] == method]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {suite}/{method} run, got {len(matches)}")
            source_dir = Path(matches[0]["canonical_directory"])
            target = OUT_ROOT / "ablation" / suite / method / matches[0]["run_tag"]
            target.mkdir(parents=True, exist_ok=True)
            for source in source_dir.iterdir():
                if not source.is_file():
                    continue
                destination = target / source.name
                if destination.exists():
                    if os.path.samefile(source, destination):
                        continue
                    raise FileExistsError(destination)
                os.link(source, destination)

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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_text(manifest_dir / "runs.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")

    # Evidence and reproducibility remain directory-level links because they contain
    # many small paired artifacts rather than one formal evaluation triplet.
    links = {
        OUT_ROOT / "evidence" / "p0": DATA_ROOT / "evidence" / "p0",
        OUT_ROOT / "evidence" / "calibration": DATA_ROOT / "calibration",
        OUT_ROOT / "reproducibility" / "goal_raes_rho040_e100_s7":
            DATA_ROOT / "paper_archive" / "goal_raes_rho0400_full500_20260804",
    }
    for link, source in links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != source.resolve():
                raise RuntimeError(f"Unexpected link target: {link}")
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(source, target_is_directory=True)


if __name__ == "__main__":
    main()
