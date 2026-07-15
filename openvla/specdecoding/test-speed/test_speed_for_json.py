#!/usr/bin/env python3
"""Measure SpecVLA speedup from timing JSON with a guarded paper-AR denominator."""

import argparse
import json
from pathlib import Path

import numpy as np


def extract_times(data):
    """Return all per-step generation latencies."""
    return np.asarray(
        [float(end_time) - float(start_time) for episode in data for end_time, start_time in episode],
        dtype=np.float64,
    )


def analyze(path):
    with Path(path).open("r") as handle:
        times = extract_times(json.load(handle))
    if times.size == 0:
        raise ValueError(f"Timing JSON has no steps: {path}")
    return {
        "steps": int(times.size),
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "std": float(np.std(times)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "max": float(np.max(times)),
        "min": float(np.min(times)),
        "total": float(np.sum(times)),
    }


def require_matching_paper_ar(ar_timing_path, ar_summary_path, ar_stats):
    ar_timing_path = Path(ar_timing_path).resolve()
    ar_summary_path = Path(ar_summary_path).resolve()
    with ar_summary_path.open("r") as handle:
        summary = json.load(handle)

    if summary.get("eval_family") != "openvla_ar":
        raise ValueError(f"AR summary has wrong eval_family: {ar_summary_path}")
    if summary.get("ar_baseline") != "specvla_paper_wrapped_ar" or summary.get("use_spec") is not True:
        raise ValueError(
            "Refusing non-paper AR denominator. Re-run "
            "run_openvla_ar_libero_goal_eval.sh; paper reproduction requires use_spec=True."
        )

    expected_timing_path = Path(str(ar_summary_path).replace("_summary.json", "_timing.json"))
    if expected_timing_path != ar_timing_path:
        raise ValueError(
            f"AR timing/summary do not belong to the same run: {ar_timing_path} vs {ar_summary_path}"
        )

    summary_mean = (summary.get("timing") or {}).get("mean")
    if summary_mean is None or not np.isclose(float(summary_mean), ar_stats["mean"], rtol=0.0, atol=1e-12):
        raise ValueError("AR timing.mean does not match its guarded summary JSON")


def print_stats(name, stats):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    for key, value in stats.items():
        if key == "steps":
            print(f"{key:10s}: {value}")
        else:
            print(f"{key:10s}: {value:.6f} s")


def main():
    parser = argparse.ArgumentParser(
        description="Compute strict/relaxed speedup using only a verified SpecVLA paper wrapped-AR denominator."
    )
    parser.add_argument("--ar", required=True, help="Paper AR *_timing.json")
    parser.add_argument("--ar-summary", required=True, help="Matching paper AR *_summary.json")
    parser.add_argument("--spec", required=True, help="SpecVLA strict *_timing.json")
    parser.add_argument("--relaxed", required=True, help="SpecVLA relaxed *_timing.json")
    args = parser.parse_args()

    results = {
        "PaperAR": analyze(args.ar),
        "Spec": analyze(args.spec),
        "SpecRelaxed": analyze(args.relaxed),
    }
    require_matching_paper_ar(args.ar, args.ar_summary, results["PaperAR"])

    for name, stats in results.items():
        print_stats(name, stats)

    print("\n" + "=" * 60)
    print("Speedup vs SpecVLA paper AR")
    print("=" * 60)
    ar_mean = results["PaperAR"]["mean"]
    for name in ("Spec", "SpecRelaxed"):
        print(f"{name:12s}: {ar_mean / results[name]['mean']:.3f}x")


if __name__ == "__main__":
    main()
