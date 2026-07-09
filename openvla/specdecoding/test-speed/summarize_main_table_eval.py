#!/usr/bin/env python3
"""Summarize LIBERO main-table eval summaries with speedup vs OpenVLA AR.

The script scans *_summary.json files produced by the LIBERO evaluators and
builds a compact table for:
  - OpenVLA AR
  - SpecVLA strict / relaxed
  - DFlash CAD-head strict / relaxed

DFlash rows are included only when dflash_use_causal_residual_sampling=True,
so old plain-DFlash runs do not get mixed into CAD-head results.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SUITE_ORDER = ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"]
METHOD_ORDER = {
    "OpenVLA AR": 0,
    "SpecVLA": 1,
    "SpecVLA-Relaxed": 2,
    "DFlash CAD Head Strict": 3,
    "DFlash CAD Head Relaxed": 4,
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    payload["_path"] = str(path)
    payload["_mtime"] = path.stat().st_mtime
    return payload


def iter_summaries(root: Path) -> Iterable[Dict[str, Any]]:
    if not root.exists():
        return []
    return (load_json(path) for path in root.rglob("*_summary.json"))


def classify_method(summary: Dict[str, Any], include_plain_dflash: bool = False) -> Optional[str]:
    family = summary.get("eval_family")
    if family == "openvla_ar":
        return "OpenVLA AR"
    if family == "specvla_strict":
        return "SpecVLA"
    if family == "specvla_relaxed":
        return "SpecVLA-Relaxed"
    if family in {"dflash_strict", "dflash_relaxed"}:
        use_cad = bool(summary.get("dflash_use_causal_residual_sampling"))
        if not use_cad and not include_plain_dflash:
            return None
        if family == "dflash_strict":
            return "DFlash CAD Head Strict" if use_cad else "DFlash Strict"
        return "DFlash CAD Head Relaxed" if use_cad else "DFlash Relaxed"
    return None


def extract_epoch(summary: Dict[str, Any]) -> str:
    text = " ".join(str(summary.get(key, "")) for key in ("spec_checkpoint", "run_id", "_path"))
    match = re.search(r"epoch[_-]?(\d{1,4})", text)
    if match:
        return str(int(match.group(1)))
    match = re.search(r"-e(\d{1,4})(?:-|$)", text)
    if match:
        return str(int(match.group(1)))
    return ""


def metric(summary: Optional[Dict[str, Any]], dotted_key: str) -> Any:
    value: Any = summary or {}
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def suite_rank(suite: str) -> int:
    try:
        return SUITE_ORDER.index(suite)
    except ValueError:
        return len(SUITE_ORDER)


def collect_latest_ar(ar_dir: Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for summary in iter_summaries(ar_dir):
        if summary.get("eval_family") != "openvla_ar":
            continue
        suite = summary.get("task_suite_name")
        if not suite:
            continue
        if suite not in latest or summary["_mtime"] > latest[suite]["_mtime"]:
            latest[suite] = summary
    return latest


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    ar_by_suite = collect_latest_ar(Path(args.ar_dir))
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for summary in iter_summaries(Path(args.log_dir)):
        method = classify_method(summary, include_plain_dflash=args.include_plain_dflash)
        if method is None:
            continue
        suite = summary.get("task_suite_name") or "unknown"
        epoch = extract_epoch(summary)
        if method in {"OpenVLA AR", "SpecVLA", "SpecVLA-Relaxed"}:
            epoch = ""
        key = (suite, method, epoch)
        if key not in grouped or summary["_mtime"] > grouped[key]["_mtime"]:
            grouped[key] = summary

    rows: List[Dict[str, Any]] = []
    for (suite, method, epoch), summary in grouped.items():
        ar = ar_by_suite.get(suite)
        ar_mean = metric(ar, "timing.mean")
        mean_time = metric(summary, "timing.mean")
        speedup = (float(ar_mean) / float(mean_time)) if ar_mean and mean_time else None
        generation = summary.get("generation") or {}
        rows.append(
            {
                "suite": suite,
                "method": method,
                "checkpoint_epoch": epoch,
                "success_rate": summary.get("success_rate"),
                "length": generation.get("length"),
                "avg_accept_length": generation.get("avg_accept_length"),
                "speedup_vs_ar": speedup,
                "mean_step_time": mean_time,
                "ar_mean_step_time": ar_mean,
                "timing_scope": summary.get("timing_scope"),
                "sync_cuda_timing": summary.get("sync_cuda_timing"),
                "summary_path": summary.get("_path"),
            }
        )

    return sorted(
        rows,
        key=lambda r: (
            suite_rank(r["suite"]),
            METHOD_ORDER.get(r["method"], 99),
            int(r["checkpoint_epoch"] or 0),
        ),
    )


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "suite",
        "method",
        "checkpoint_epoch",
        "success_rate",
        "length",
        "avg_accept_length",
        "speedup_vs_ar",
        "mean_step_time",
        "ar_mean_step_time",
        "timing_scope",
        "sync_cuda_timing",
        "summary_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Suite | Method | Epoch | SR | Length | Avg Accept | Speedup vs AR | Mean Step |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {suite} | {method} | {epoch} | {sr} | {length} | {accept} | {speedup} | {mean} |".format(
                suite=row["suite"],
                method=row["method"],
                epoch=row["checkpoint_epoch"] or "-",
                sr=fmt(row["success_rate"]),
                length=fmt(row["length"]),
                accept=fmt(row["avg_accept_length"]),
                speedup=(fmt(row["speedup_vs_ar"]) + "x") if row["speedup_vs_ar"] is not None else "",
                mean=fmt(row["mean_step_time"], 4),
            )
        )
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, help="Root eval_logs directory.")
    parser.add_argument("--ar-dir", required=True, help="Directory containing OpenVLA AR *_summary.json files.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown output path.")
    parser.add_argument("--include-plain-dflash", action="store_true", help="Also include non-CAD DFlash rows.")
    args = parser.parse_args()

    rows = build_rows(args)
    if args.output_csv:
        write_csv(rows, Path(args.output_csv))
    if args.output_md:
        text = write_markdown(rows, Path(args.output_md))
    else:
        text = write_markdown(rows, Path("/tmp/main_table_eval.md"))
    print(text)


if __name__ == "__main__":
    main()
