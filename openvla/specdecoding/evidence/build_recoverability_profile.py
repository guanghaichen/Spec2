#!/usr/bin/env python3
"""Select and freeze one suite-specific configuration from paired rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from specdecoding.evidence.recoverability_calibration import (
    select_lowest_cost_feasible,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty calibration table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--run_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--risk_budget", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.records.read_text().splitlines()
        if line.strip()
    ]
    manifest = json.loads(args.run_manifest.read_text())
    if not records:
        raise ValueError("Calibration records are empty.")
    configurations = {
        row["name"]: row for row in manifest["configurations"]
    }
    reference = {
        (int(row["task_id"]), int(row["episode_index"])): bool(row["success"])
        for row in records
        if row["configuration"] == "target_reference"
    }
    grouped = defaultdict(list)
    for row in records:
        grouped[row["configuration"]].append(row)

    summary_rows = []
    for name, episodes in sorted(grouped.items()):
        configuration = configurations[name]
        paired = [
            row
            for row in episodes
            if (int(row["task_id"]), int(row["episode_index"])) in reference
        ]
        if not paired:
            raise ValueError(
                f"Configuration {name!r} has no episodes paired with the reference."
            )
        harms = sum(
            reference[(int(row["task_id"]), int(row["episode_index"]))]
            and not bool(row["success"])
            for row in paired
        )
        summary_rows.append(
            {
                "configuration": name,
                "schedule_kind": configuration.get("schedule_kind", "legacy"),
                "schedule_period": int(configuration["schedule_period"]),
                "schedule_target_count": int(
                    configuration["schedule_target_count"]
                ),
                "schedule_offsets": ",".join(
                    str(int(value))
                    for value in configuration.get("schedule_offsets", [])
                ),
                "target_density": float(configuration["target_density"]),
                "authority_exponent": float(
                    configuration["authority_exponent"]
                ),
                "max_consecutive_holds": int(
                    configuration["max_consecutive_holds"]
                ),
                "paired_episodes": len(paired),
                "successes": sum(bool(row["success"]) for row in paired),
                "success_rate": float(np.mean([row["success"] for row in paired])),
                "harm_count": int(harms),
                "empirical_harm_rate": harms / len(paired),
                "mean_target_rate": float(
                    np.mean([row["target_rate"] for row in paired])
                ),
                "deterministic_reference": name == "target_reference",
            }
        )

    selected, evaluated = select_lowest_cost_feasible(
        summary_rows, risk_budget=args.risk_budget, alpha=args.alpha
    )
    selected_configuration = configurations[selected["configuration"]]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "calibration_summary.csv", evaluated)

    suite = manifest["config"]["task_suite_name"]
    identity = hashlib.sha256(
        json.dumps(
            {
                "suite": suite,
                "selected": selected_configuration,
                "risk_budget": args.risk_budget,
                "alpha": args.alpha,
                "records_sha256": sha256(args.records),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    profile = {
        "schema_version": 1,
        "profile_id": f"{suite}-{identity}",
        "task_suite_name": suite,
        "selection_rule": (
            "minimum target rate under a simultaneous exact one-sided harm bound"
        ),
        "risk_budget": args.risk_budget,
        "family_wise_alpha": args.alpha,
        "selected_configuration": selected_configuration,
        "selected_statistics": selected,
        "all_statistics": evaluated,
        "candidate_family": list(configurations.values()),
        "evidence": {
            "records": str(args.records),
            "records_sha256": sha256(args.records),
            "run_manifest": str(args.run_manifest),
            "run_manifest_sha256": sha256(args.run_manifest),
        },
    }
    profile_path = output_dir / f"{profile['profile_id']}.json"
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    print(profile_path)


if __name__ == "__main__":
    main()
