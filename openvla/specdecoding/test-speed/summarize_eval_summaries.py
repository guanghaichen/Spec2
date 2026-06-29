import argparse
import json
from pathlib import Path


def load_summary(path):
    with open(path, "r") as f:
        payload = json.load(f)
    payload["_path"] = str(path)
    return payload


def method_name(summary):
    family = summary.get("eval_family") or "unknown"
    note = summary.get("run_id") or ""
    return family if not note else f"{family} ({note})"


def format_float(value, digits=3, suffix=""):
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Summarize LIBERO eval summary JSON files.")
    parser.add_argument("--ar-summary", required=True, help="OpenVLA AR *_summary.json file.")
    parser.add_argument("summaries", nargs="+", help="SpecVLA/DFlash *_summary.json files.")
    args = parser.parse_args()

    ar = load_summary(Path(args.ar_summary))
    ar_mean = ar.get("timing", {}).get("mean")
    if ar_mean is None:
        raise ValueError(f"AR summary has no timing.mean: {args.ar_summary}")

    rows = []
    for summary_path in args.summaries:
        summary = load_summary(Path(summary_path))
        timing_mean = summary.get("timing", {}).get("mean")
        generation = summary.get("generation") or {}
        speedup = (ar_mean / timing_mean) if timing_mean else None
        rows.append(
            {
                "method": method_name(summary),
                "sr": summary.get("success_rate"),
                "length": generation.get("length"),
                "avg_accept_length": generation.get("avg_accept_length"),
                "mean_time": timing_mean,
                "speedup": speedup,
                "path": summary["_path"],
            }
        )

    header = f"{'Method':38s} {'SR':>8s} {'Length':>8s} {'Accept':>8s} {'Time(s)':>9s} {'Speedup':>9s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method'][:38]:38s} "
            f"{format_float(row['sr'], 3):>8s} "
            f"{format_float(row['length'], 3):>8s} "
            f"{format_float(row['avg_accept_length'], 3):>8s} "
            f"{format_float(row['mean_time'], 4):>9s} "
            f"{format_float(row['speedup'], 3, 'x'):>9s}"
        )


if __name__ == "__main__":
    main()
