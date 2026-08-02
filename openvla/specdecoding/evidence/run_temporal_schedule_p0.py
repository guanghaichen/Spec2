#!/usr/bin/env python3
"""Paired factorial study of temporal regularity and action authority.

All conditions use the same frozen OpenVLA target and the same LIBERO initial
states. Under an identical target-call budget and gap multiset, the only
interventions are extremal prefix regularity and the critical authority law.
"""

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
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from specdecoding.evidence.temporal_factorial_design import PAIRED_CONDITIONS
from specdecoding.evidence.temporal_schedule_design import (
    power_law_authority_scale,
    target_indicator,
)


@dataclass
class TemporalScheduleConfig:
    pretrained_checkpoint: Union[str, Path]
    output_dir: Union[str, Path]
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
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
    run_id_note: Optional[str] = None


def suite_horizon(suite: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }[suite]


def robot_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float64)


def target_action(cfg, model, processor, obs, description, resize_size):
    image = get_libero_image(obs, resize_size)
    observation = {"full_image": image, "state": robot_state(obs)}
    action = np.asarray(
        get_action(cfg, model, observation, description, processor=processor),
        dtype=np.float64,
    ).copy()
    action = normalize_gripper_action(action, binarize=True)
    return invert_gripper_action(action)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


@draccus.wrap()
def run(cfg: TemporalScheduleConfig) -> None:
    if cfg.task_suite_name not in {
        "libero_goal",
        "libero_spatial",
        "libero_object",
        "libero_10",
    }:
        raise ValueError("Unsupported LIBERO suite.")
    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name
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
    task = suite.get_task(cfg.task_id)
    initial_states = suite.get_task_init_states(cfg.task_id)
    env, task_description = get_libero_env(
        task, cfg.model_family, resolution=256
    )
    episode_stop = min(
        cfg.trial_start_index + cfg.num_trials, len(initial_states)
    )
    if episode_stop <= cfg.trial_start_index:
        raise ValueError("No LIBERO initial states selected.")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    note = cfg.run_id_note or (
        f"{cfg.task_suite_name}-task{cfg.task_id}-seed{cfg.seed}"
    )
    output_jsonl = output_dir / f"temporal-2x2-{note}.jsonl"
    manifest_path = output_dir / f"temporal-2x2-{note}-manifest.json"
    if output_jsonl.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to append to existing run: {output_dir}")

    horizon = suite_horizon(cfg.task_suite_name)
    manifest = {
        "schema_version": 1,
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
        "task_description": task_description,
        "conditions": [asdict(condition) for condition in PAIRED_CONDITIONS],
        "output_jsonl": str(output_jsonl),
        "status": "running",
        "started_at_unix": time.time(),
        "records": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    records_written = 0

    try:
        for condition in PAIRED_CONDITIONS:
            target_mask = target_indicator(condition.intervals, horizon)
            for episode_index in range(cfg.trial_start_index, episode_stop):
                episode_seed = cfg.seed + episode_index * 1009
                env.seed(episode_seed)
                env.reset()
                obs = env.set_init_state(initial_states[episode_index])
                for _ in range(cfg.num_steps_wait):
                    obs, _, _, _ = env.step(
                        get_libero_dummy_action(cfg.model_family)
                    )

                success = False
                last_target_action = None
                last_executed_action = None
                last_target_step = None
                target_calls = 0
                hold_counts = {1: 0, 2: 0}
                episode_records = 0
                for step in range(horizon):
                    is_target = target_mask[step]
                    state_before = robot_state(obs)
                    correction_l2 = None
                    stale_target_l2 = None
                    open_loop_depth_before = (
                        max(0, step - last_target_step - 1)
                        if last_target_step is not None
                        else 0
                    )
                    inference_seconds = 0.0

                    if is_target or last_target_action is None:
                        started = time.perf_counter()
                        action = target_action(
                            cfg,
                            model,
                            processor,
                            obs,
                            task_description,
                            resize_size,
                        )
                        inference_seconds = time.perf_counter() - started
                        if last_executed_action is not None and last_target_step is not None:
                            prior_depth = step - last_target_step - 1
                            if prior_depth > 0:
                                correction_l2 = float(
                                    np.linalg.norm(
                                        action[:6] - last_executed_action[:6]
                                    )
                                )
                                stale_target_l2 = float(
                                    np.linalg.norm(
                                        action[:6] - last_target_action[:6]
                                    )
                                )
                        last_target_action = action.copy()
                        last_target_step = step
                        target_calls += 1
                        frame_type = "target"
                        hold_depth = 0
                        scale = 1.0
                    else:
                        hold_depth = step - last_target_step
                        scale = power_law_authority_scale(
                            exponent=condition.authority_exponent,
                            hold_depth=hold_depth,
                        )
                        action = last_target_action.copy()
                        action[:6] *= scale
                        hold_counts[hold_depth] = hold_counts.get(hold_depth, 0) + 1
                        frame_type = f"hold_{hold_depth}"

                    obs, _, done, _ = env.step(action.tolist())
                    state_after = robot_state(obs)
                    payload = {
                        "record_type": "step",
                        "suite": cfg.task_suite_name,
                        "task_id": cfg.task_id,
                        "episode_index": episode_index,
                        "episode_seed": episode_seed,
                        "condition": condition.name,
                        "intervals": list(condition.intervals),
                        "harmonic": condition.harmonic,
                        "step": step,
                        "frame_type": frame_type,
                        "is_target": is_target,
                        "hold_depth": hold_depth,
                        "authority_scale": scale,
                        "previous_hold_depth_at_target": (
                            open_loop_depth_before if is_target else 0
                        ),
                        "action": action.tolist(),
                        "continuous_action_norm": float(
                            np.linalg.norm(action[:6])
                        ),
                        "eef_position_step_l2": float(
                            np.linalg.norm(state_after[:3] - state_before[:3])
                        ),
                        "eef_rotation_step_l2": float(
                            np.linalg.norm(state_after[3:6] - state_before[3:6])
                        ),
                        "target_correction_l2": correction_l2,
                        "target_vs_stale_target_l2": stale_target_l2,
                        "target_inference_seconds": inference_seconds,
                        "done": bool(done),
                    }
                    append_jsonl(output_jsonl, payload)
                    records_written += 1
                    episode_records += 1
                    last_executed_action = action.copy()
                    if done:
                        success = True
                        break

                episode_payload = {
                    "record_type": "episode",
                    "suite": cfg.task_suite_name,
                    "task_id": cfg.task_id,
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "condition": condition.name,
                    "intervals": list(condition.intervals),
                    "harmonic": condition.harmonic,
                    "success": success,
                    "steps": episode_records,
                    "target_calls": target_calls,
                    "target_rate": (
                        target_calls / episode_records if episode_records else 0.0
                    ),
                    "hold_1_count": hold_counts.get(1, 0),
                    "hold_2_count": hold_counts.get(2, 0),
                }
                append_jsonl(output_jsonl, episode_payload)
                records_written += 1
                manifest.update(
                    {"records": records_written, "last_record": episode_payload}
                )
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False)
                )
                print(
                    f"[{condition.name}] episode={episode_index} "
                    f"success={success} steps={episode_records} "
                    f"target_rate={episode_payload['target_rate']:.4f}",
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
