#!/usr/bin/env python3
"""Run paired target-policy rollouts for recoverability calibration."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from specdecoding.evidence.recoverability_calibration import (
    RecoveryConfiguration,
    enumerate_temporal_control_family,
)
from specdecoding.evidence.run_temporal_schedule_p0 import (
    robot_state,
    suite_horizon,
    target_action,
)
from specdecoding.model.temporal_hold import (
    mechanical_target_due,
    periodic_target_due,
    temporal_hold_action_scale,
)


@dataclass
class CalibrationRunConfig:
    pretrained_checkpoint: Union[str, Path]
    output_dir: Union[str, Path]
    task_suite_name: str = "libero_spatial"
    task_ids: str = "0"
    trial_start_index: int = 0
    num_trials: int = 10
    num_steps_wait: int = 10
    seed: int = 7
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    model_family: str = "openvla"
    use_spec: bool = False
    accept_threshold: int = 0
    schedule_resolution: int = 10
    max_hold_depth: int = 2
    min_target_density: float = 0.30
    max_target_density: float = 0.50
    min_authority_exponent: float = 0.0
    max_authority_exponent: float = 1.0
    num_authority_exponents: int = 3
    reference_only: bool = False
    context_trace: bool = False
    context_trace_max_lag: int = 3
    run_id_note: Optional[str] = None
    resume: bool = False


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed:
        raise ValueError("At least one task id is required.")
    return parsed


def load_completed_records(path: Path) -> tuple[list[dict], set[tuple[int, str, int]]]:
    """Load a crash-safe JSONL prefix and reject duplicates or partial lines."""
    records = []
    completed = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at {path}:{line_number}; "
                    "refusing an ambiguous resume."
                ) from exc
            key = (
                int(payload["task_id"]),
                str(payload["configuration"]),
                int(payload["episode_index"]),
            )
            if key in completed:
                raise ValueError(f"Duplicate calibration record {key} in {path}.")
            completed.add(key)
            records.append(payload)
    return records, completed


def validate_resume_manifest(
    manifest: dict,
    cfg_payload: dict,
    configurations: tuple[RecoveryConfiguration, ...],
) -> None:
    """Ensure a resumed run cannot silently change the experimental protocol."""
    previous = manifest.get("config", {})
    immutable_fields = (
        "pretrained_checkpoint",
        "task_suite_name",
        "task_ids",
        "trial_start_index",
        "num_trials",
        "num_steps_wait",
        "seed",
        "center_crop",
        "schedule_resolution",
        "max_hold_depth",
        "min_target_density",
        "max_target_density",
        "min_authority_exponent",
        "max_authority_exponent",
        "num_authority_exponents",
        "reference_only",
        "context_trace",
        "context_trace_max_lag",
        "run_id_note",
    )
    mismatches = {
        field: (previous.get(field), cfg_payload.get(field))
        for field in immutable_fields
        if field in previous and previous.get(field) != cfg_payload.get(field)
    }
    expected_family = json.loads(
        json.dumps([item.as_dict() for item in configurations])
    )
    if manifest.get("configurations") != expected_family:
        mismatches["configurations"] = ("stored family", "current family")
    if mismatches:
        raise ValueError(f"Resume protocol mismatch: {mismatches}")


def authority_exponent_grid(
    minimum: float, maximum: float, count: int
) -> tuple[float, ...]:
    """Construct a resolution-controlled exponent grid over one interval."""
    if count < 1 or minimum < 0.0 or maximum < minimum:
        raise ValueError("Invalid authority-exponent interval or resolution.")
    if count == 1:
        return (float(minimum),)
    return tuple(float(value) for value in np.linspace(minimum, maximum, count))


def compact_image_signature(obs: dict, resize_size: int) -> np.ndarray:
    """Return a cheap image signature available before the target forward."""
    image = np.asarray(get_libero_image(obs, resize_size), dtype=np.float32)
    y_index = np.linspace(0, image.shape[0] - 1, 16).round().astype(np.int64)
    x_index = np.linspace(0, image.shape[1] - 1, 16).round().astype(np.int64)
    return image[np.ix_(y_index, x_index)].reshape(-1)


def relative_l2(current: np.ndarray, anchor: np.ndarray) -> float:
    return float(
        np.linalg.norm(current - anchor) / max(float(np.linalg.norm(anchor)), 1e-6)
    )


def build_context_lag_records(
    *,
    current_signature: np.ndarray,
    current_state: np.ndarray,
    current_action: np.ndarray,
    history: list[dict],
    max_lag: int,
) -> list[dict]:
    """Measure target-free context drift and its target-action consequence."""
    lag_records = []
    for lag in range(1, min(int(max_lag), len(history)) + 1):
        anchor = history[-lag]
        anchor_action = anchor["action"]
        inverse_age_action = anchor_action.copy()
        inverse_age_action[:6] *= lag ** -1
        anchor_action_change_l2 = 0.0
        if len(history) > lag:
            previous_anchor_action = history[-lag - 1]["action"]
            anchor_action_change_l2 = float(
                np.linalg.norm(anchor_action[:6] - previous_anchor_action[:6])
            )
        state_delta = current_state - anchor["state"]
        lag_records.append(
            {
                "lag": lag,
                "image_relative_l2": relative_l2(
                    current_signature, anchor["image_signature"]
                ),
                "eef_position_l2": float(np.linalg.norm(state_delta[:3])),
                "eef_rotation_l2": float(np.linalg.norm(state_delta[3:6])),
                "gripper_state_l2": float(np.linalg.norm(state_delta[6:])),
                "anchor_action_norm": float(np.linalg.norm(anchor_action[:6])),
                "anchor_action_change_l2": anchor_action_change_l2,
                "stale_action_correction_l2": float(
                    np.linalg.norm(current_action[:6] - anchor_action[:6])
                ),
                "inverse_age_action_correction_l2": float(
                    np.linalg.norm(current_action[:6] - inverse_age_action[:6])
                ),
                "gripper_switch": bool(current_action[6] != anchor_action[6]),
            }
        )
    return lag_records


@draccus.wrap()
def run(cfg: CalibrationRunConfig) -> None:
    set_seed_everywhere(cfg.seed)
    if cfg.context_trace and not cfg.reference_only:
        raise ValueError("context_trace requires reference_only=True.")
    if cfg.context_trace_max_lag < 1:
        raise ValueError("context_trace_max_lag must be positive.")
    cfg.unnorm_key = cfg.task_suite_name
    reference = RecoveryConfiguration(
        name="target_reference",
        schedule_kind="target_reference",
        schedule_period=1,
        schedule_target_count=1,
        schedule_offsets=(0,),
        schedule_phase=-1.0,
        authority_exponent=0.0,
        max_consecutive_holds=0,
    )
    family = enumerate_temporal_control_family(
        schedule_resolution=cfg.schedule_resolution,
        max_hold_depth=cfg.max_hold_depth,
        min_target_density=cfg.min_target_density,
        max_target_density=cfg.max_target_density,
        authority_exponents=authority_exponent_grid(
            cfg.min_authority_exponent,
            cfg.max_authority_exponent,
            cfg.num_authority_exponents,
        ),
    )
    configurations = (reference,) if cfg.reference_only else (reference,) + family
    task_ids = parse_ints(cfg.task_ids)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    note = cfg.run_id_note or f"{cfg.task_suite_name}-seed{cfg.seed}"
    records_path = output_dir / f"recoverability-calibration-{note}.jsonl"
    manifest_path = output_dir / f"recoverability-calibration-{note}-manifest.json"
    cfg_payload = json.loads(json.dumps(asdict(cfg), default=str))
    if records_path.exists() or manifest_path.exists():
        if not cfg.resume:
            raise FileExistsError(f"Refusing to append to existing run: {output_dir}")
        if not records_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                "Resume requires both the JSONL records and run manifest."
            )
        manifest = json.loads(manifest_path.read_text())
        validate_resume_manifest(manifest, cfg_payload, configurations)
        existing_records, completed_keys = load_completed_records(records_path)
        records_written = len(existing_records)
        if int(manifest.get("records", -1)) != records_written:
            raise ValueError(
                "Manifest/JSONL record-count mismatch; refusing an ambiguous resume."
            )
        if manifest.get("status") == "complete":
            print(f"Calibration already complete: {output_dir}", flush=True)
            return
        manifest.update(
            {
                "status": "running",
                "resumed_at_unix": time.time(),
                "resume_count": int(manifest.get("resume_count", 0)) + 1,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(
            f"Resuming {cfg.task_suite_name} after {records_written} records.",
            flush=True,
        )
    else:
        manifest = {
            "schema_version": 1,
            "config": cfg_payload,
            "configurations": [item.as_dict() for item in configurations],
            "status": "running",
            "started_at_unix": time.time(),
            "records": 0,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        records_written = 0
        completed_keys = set()

    model = get_model(cfg)
    if (
        cfg.unnorm_key not in model.norm_stats
        and f"{cfg.unnorm_key}_no_noops" in model.norm_stats
    ):
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    if cfg.unnorm_key not in model.norm_stats:
        raise KeyError(f"Missing action statistics for {cfg.unnorm_key}")
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    horizon = suite_horizon(cfg.task_suite_name)

    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        episode_stop = min(
            cfg.trial_start_index + cfg.num_trials, len(initial_states)
        )
        if episode_stop <= cfg.trial_start_index:
            raise ValueError(f"No initial states selected for task {task_id}.")
        env, task_description = get_libero_env(
            task, cfg.model_family, resolution=256
        )
        try:
            for configuration in configurations:
                for episode_index in range(cfg.trial_start_index, episode_stop):
                    record_key = (task_id, configuration.name, episode_index)
                    if record_key in completed_keys:
                        continue
                    episode_seed = cfg.seed + task_id * 100_003 + episode_index * 1009
                    env.seed(episode_seed)
                    env.reset()
                    obs = env.set_init_state(initial_states[episode_index])
                    for _ in range(cfg.num_steps_wait):
                        obs, _, _, _ = env.step(
                            get_libero_dummy_action(cfg.model_family)
                        )

                    success = False
                    last_target_action = None
                    last_target_step = None
                    target_calls = 0
                    hold_counts = {}
                    context_history = []
                    context_steps = []
                    for step in range(horizon):
                        if configuration.schedule_offsets:
                            target_due = periodic_target_due(
                                control_step=step,
                                period=configuration.schedule_period,
                                target_offsets=configuration.schedule_offsets,
                            )
                        else:
                            target_due = mechanical_target_due(
                                control_step=step,
                                period=configuration.schedule_period,
                                target_count=configuration.schedule_target_count,
                                phase=configuration.schedule_phase,
                            )
                        if target_due or last_target_action is None:
                            current_signature = None
                            current_state = None
                            if cfg.context_trace:
                                current_signature = compact_image_signature(
                                    obs, resize_size
                                )
                                current_state = robot_state(obs)
                            action = target_action(
                                cfg,
                                model,
                                processor,
                                obs,
                                task_description,
                                resize_size,
                            )
                            last_target_action = action.copy()
                            last_target_step = step
                            target_calls += 1
                            if cfg.context_trace:
                                context_steps.append(
                                    {
                                        "step": step,
                                        "current_action_norm": float(
                                            np.linalg.norm(action[:6])
                                        ),
                                        "lag_records": build_context_lag_records(
                                            current_signature=current_signature,
                                            current_state=current_state,
                                            current_action=action,
                                            history=context_history,
                                            max_lag=cfg.context_trace_max_lag,
                                        ),
                                    }
                                )
                                context_history.append(
                                    {
                                        "image_signature": current_signature,
                                        "state": current_state,
                                        "action": action.copy(),
                                    }
                                )
                                if len(context_history) > cfg.context_trace_max_lag + 1:
                                    context_history.pop(0)
                        else:
                            hold_depth = step - last_target_step
                            action = last_target_action.copy()
                            action[:6] *= temporal_hold_action_scale(
                                "power_law",
                                hold_depth,
                                exponent=configuration.authority_exponent,
                            )
                            hold_counts[hold_depth] = hold_counts.get(hold_depth, 0) + 1
                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            success = True
                            break

                    steps = step + 1
                    payload = {
                        "record_type": "episode",
                        "task_suite_name": cfg.task_suite_name,
                        "task_id": task_id,
                        "task_description": task_description,
                        "episode_index": episode_index,
                        "episode_seed": episode_seed,
                        "configuration": configuration.name,
                        "success": success,
                        "steps": steps,
                        "target_calls": target_calls,
                        "target_rate": target_calls / steps,
                        "hold_counts": hold_counts,
                        "final_robot_state": robot_state(obs).tolist(),
                    }
                    if cfg.context_trace:
                        payload["context_trace_schema_version"] = 1
                        payload["context_steps"] = context_steps
                    append_jsonl(records_path, payload)
                    completed_keys.add(record_key)
                    records_written += 1
                    manifest.update({"records": records_written, "last_record": payload})
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False)
                    )
                    print(
                        f"[{configuration.name}] task={task_id} "
                        f"episode={episode_index} success={success} "
                        f"steps={steps} target_rate={payload['target_rate']:.4f}",
                        flush=True,
                    )
        finally:
            env.close()

    manifest.update(
        {
            "status": "complete",
            "finished_at_unix": time.time(),
            "records": records_written,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
