#!/usr/bin/env python3
"""Same-state LIBERO forks for recoverability evidence.

Each intervention starts from an identical MuJoCo snapshot. A stale
target-grounded action, a same-norm random action, or a zero-motion action is
committed for a bounded depth; the unmodified target policy then resumes.
Records are flushed one branch at a time so interrupted studies remain usable.
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


@dataclass
class CounterfactualConfig:
    pretrained_checkpoint: Union[str, Path]
    output_dir: Union[str, Path]
    task_suite_name: str = "libero_goal"
    task_id: int = 0
    trial_start_index: int = 0
    num_reference_episodes: int = 1
    max_reference_attempts: int = 10
    require_reference_success: bool = True
    num_steps_wait: int = 10
    num_forks_per_episode: int = 2
    hold_depths: tuple[int, ...] = (1, 2, 3)
    max_recovery_steps: int = 0
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
        "libero_90": 400,
    }[suite]


def robot_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float64)


def policy_action(cfg, model, processor, obs, task_description, resize_size):
    image = get_libero_image(obs, resize_size)
    observation = {"full_image": image, "state": robot_state(obs)}
    action = get_action(
        cfg,
        model,
        observation,
        task_description,
        processor=processor,
    )
    action = np.asarray(action, dtype=np.float64).copy()
    action = normalize_gripper_action(action, binarize=True)
    return invert_gripper_action(action)


def choose_fork_steps(length: int, count: int) -> list[int]:
    if length < 4 or count <= 0:
        return []
    lower = max(2, int(round(0.20 * (length - 1))))
    upper = max(lower, int(round(0.80 * (length - 1))))
    if count == 1:
        return [int(round((lower + upper) / 2))]
    return sorted(
        {
            int(round(value))
            for value in np.linspace(lower, upper, num=min(count, upper - lower + 1))
        }
    )


def same_norm_random(action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = np.asarray(action, dtype=np.float64).copy()
    direction = rng.normal(size=6)
    norm = float(np.linalg.norm(direction))
    action_norm = float(np.linalg.norm(result[:6]))
    result[:6] = direction / max(norm, 1e-12) * action_norm
    result[:6] = np.clip(result[:6], -1.0, 1.0)
    return result


def harmonic_action_sequence(action: np.ndarray, depth: int) -> list[np.ndarray]:
    """Age a target-grounded delta command while preserving gripper state."""
    sequence = []
    for age in range(1, depth + 1):
        aged = np.asarray(action, dtype=np.float64).copy()
        aged[:6] *= 1.0 / age
        sequence.append(aged)
    return sequence


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def replay_to_fork(
    env,
    cfg,
    episode_seed: int,
    initial_state: np.ndarray,
    reference_actions: list[np.ndarray],
    fork_step: int,
):
    """Reconstruct the full simulator/controller history before a state fork."""
    env.seed(episode_seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    for action in reference_actions[:fork_step]:
        obs, _, done, _ = env.step(action.tolist())
        if done:
            raise RuntimeError("Reference replay terminated before the fork step.")
    return obs, env.get_sim_state().copy()


@draccus.wrap()
def run(cfg: CounterfactualConfig) -> None:
    if cfg.task_suite_name not in {
        "libero_goal",
        "libero_spatial",
        "libero_object",
        "libero_10",
    }:
        raise ValueError("Unsupported LIBERO suite for counterfactual recovery.")
    if any(depth <= 0 for depth in cfg.hold_depths):
        raise ValueError("All hold depths must be positive.")
    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name
    model = get_model(cfg)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    if cfg.unnorm_key not in model.norm_stats:
        raise KeyError(f"Missing action statistics for {cfg.unnorm_key}")
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = suite.get_task(cfg.task_id)
    initial_states = suite.get_task_init_states(cfg.task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    note = cfg.run_id_note or f"{cfg.task_suite_name}-task{cfg.task_id}-seed{cfg.seed}"
    output_jsonl = output_dir / f"counterfactual-{note}.jsonl"
    manifest_path = output_dir / f"counterfactual-{note}-manifest.json"
    if output_jsonl.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to append to an existing counterfactual run: {output_dir}"
        )
    manifest = {
        "schema_version": 1,
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
        "task_description": task_description,
        "output_jsonl": str(output_jsonl),
        "status": "running",
        "started_at_unix": time.time(),
        "records": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    horizon = suite_horizon(cfg.task_suite_name)
    rng = np.random.default_rng(cfg.seed)
    records_written = 0
    selected_reference_episodes = 0
    attempted_reference_episodes = 0
    max_attempts = min(
        cfg.max_reference_attempts,
        len(initial_states) - cfg.trial_start_index,
    )
    while (
        attempted_reference_episodes < max_attempts
        and selected_reference_episodes < cfg.num_reference_episodes
    ):
        episode_index = cfg.trial_start_index + attempted_reference_episodes
        episode_seed = cfg.seed + episode_index * 1009
        attempted_reference_episodes += 1
        env.seed(episode_seed)
        env.reset()
        obs = env.set_init_state(initial_states[episode_index])
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

        reference_states = []
        reference_robot_states = []
        reference_actions = []
        reference_success = False
        for _ in range(horizon):
            reference_states.append(env.get_sim_state().copy())
            reference_robot_states.append(robot_state(obs))
            action = policy_action(
                cfg, model, processor, obs, task_description, resize_size
            )
            reference_actions.append(action.copy())
            obs, _, done, _ = env.step(action.tolist())
            if done:
                reference_success = True
                break

        episode_row = {
            "record_type": "reference_episode",
            "suite": cfg.task_suite_name,
            "task_id": cfg.task_id,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "reference_success": reference_success,
            "reference_trajectory_steps": len(reference_actions),
            "selected_for_forks": bool(
                reference_success or not cfg.require_reference_success
            ),
        }
        append_jsonl(output_jsonl, episode_row)
        records_written += 1
        print(
            f"[reference] episode={episode_index} success={reference_success} "
            f"steps={len(reference_actions)} selected={episode_row['selected_for_forks']}",
            flush=True,
        )
        if cfg.require_reference_success and not reference_success:
            manifest.update(
                {
                    "records": records_written,
                    "attempted_reference_episodes": attempted_reference_episodes,
                    "selected_reference_episodes": selected_reference_episodes,
                    "last_record": episode_row,
                }
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False)
            )
            continue
        selected_reference_episodes += 1

        fork_steps = choose_fork_steps(
            len(reference_actions), cfg.num_forks_per_episode
        )
        for fork_step in fork_steps:
            reference_row = {
                "record_type": "reference",
                "suite": cfg.task_suite_name,
                "task_id": cfg.task_id,
                "episode_index": episode_index,
                "fork_step": fork_step,
                "candidate": "target_policy",
                "hold_depth": 0,
                "reference_success": reference_success,
                "branch_success": reference_success,
                "one_sided_harm": False,
                "reference_trajectory_steps": len(reference_actions),
                "action_norm": float(np.linalg.norm(reference_actions[fork_step][:6])),
            }
            append_jsonl(output_jsonl, reference_row)
            records_written += 1

            candidates = {
                "lag1_target": reference_actions[fork_step - 1].copy(),
                "zero_motion": np.concatenate(
                    [np.zeros(6), reference_actions[fork_step - 1][-1:]]
                ),
                "same_norm_random": same_norm_random(
                    reference_actions[fork_step - 1], rng
                ),
            }
            if fork_step >= 2:
                candidates["lag2_target"] = reference_actions[fork_step - 2].copy()

            for hold_depth in cfg.hold_depths:
                current_target_path = [
                    action.copy()
                    for action in reference_actions[
                        fork_step : fork_step + hold_depth
                    ]
                ]
                if len(current_target_path) != hold_depth:
                    continue
                branch_candidates = {
                    "current_target_path": current_target_path,
                    **{
                        name: [action.copy() for _ in range(hold_depth)]
                        for name, action in candidates.items()
                    },
                    "lag1_harmonic": harmonic_action_sequence(
                        reference_actions[fork_step - 1], hold_depth
                    ),
                }
                for candidate_name, candidate_actions in branch_candidates.items():
                    obs, replayed_state = replay_to_fork(
                        env,
                        cfg,
                        episode_seed,
                        initial_states[episode_index],
                        reference_actions,
                        fork_step,
                    )
                    fork_state_linf = float(
                        np.max(
                            np.abs(
                                replayed_state
                                - reference_states[fork_step]
                            )
                        )
                    )
                    branch_success = False
                    executed_holds = 0
                    for candidate_action in candidate_actions:
                        obs, _, done, _ = env.step(candidate_action.tolist())
                        executed_holds += 1
                        if done:
                            branch_success = True
                            break

                    post_hold_robot_state = robot_state(obs)
                    post_hold_reference_index = fork_step + executed_holds
                    post_hold_reference_state = (
                        reference_robot_states[post_hold_reference_index]
                        if post_hold_reference_index < len(reference_robot_states)
                        else None
                    )
                    post_hold_state_l2 = (
                        float(
                            np.linalg.norm(
                                post_hold_robot_state - post_hold_reference_state
                            )
                        )
                        if post_hold_reference_state is not None
                        else None
                    )
                    post_hold_state_delta = (
                        post_hold_robot_state - post_hold_reference_state
                        if post_hold_reference_state is not None
                        else None
                    )
                    post_hold_eef_position_l2 = (
                        float(np.linalg.norm(post_hold_state_delta[:3]))
                        if post_hold_state_delta is not None
                        else None
                    )
                    post_hold_eef_rotation_l2 = (
                        float(np.linalg.norm(post_hold_state_delta[3:6]))
                        if post_hold_state_delta is not None
                        else None
                    )
                    post_hold_gripper_l2 = (
                        float(np.linalg.norm(post_hold_state_delta[6:]))
                        if post_hold_state_delta is not None
                        else None
                    )

                    correction_l2 = None
                    recovery_steps = 0
                    minimum_reference_robot_l2 = None
                    remaining = max(horizon - fork_step - executed_holds, 0)
                    if cfg.max_recovery_steps > 0:
                        remaining = min(remaining, cfg.max_recovery_steps)
                    while not branch_success and recovery_steps < remaining:
                        recovery_action = policy_action(
                            cfg, model, processor, obs, task_description, resize_size
                        )
                        if correction_l2 is None:
                            correction_l2 = float(
                                np.linalg.norm(
                                    recovery_action[:6]
                                    - candidate_actions[executed_holds - 1][:6]
                                )
                            )
                        obs, _, done, _ = env.step(recovery_action.tolist())
                        recovery_steps += 1
                        reference_index = fork_step + executed_holds + recovery_steps
                        if reference_index < len(reference_robot_states):
                            distance = float(
                                np.linalg.norm(
                                    robot_state(obs)
                                    - reference_robot_states[reference_index]
                                )
                            )
                            minimum_reference_robot_l2 = (
                                distance
                                if minimum_reference_robot_l2 is None
                                else min(minimum_reference_robot_l2, distance)
                            )
                        if done:
                            branch_success = True

                    payload = {
                        "record_type": "counterfactual",
                        "suite": cfg.task_suite_name,
                        "task_id": cfg.task_id,
                        "episode_index": episode_index,
                        "episode_seed": episode_seed,
                        "fork_step": fork_step,
                        "candidate": candidate_name,
                        "hold_depth": int(hold_depth),
                        "executed_holds": executed_holds,
                        "reference_success": reference_success,
                        "branch_success": branch_success,
                        "one_sided_harm": bool(
                            reference_success and not branch_success
                        ),
                        "recovery_steps": recovery_steps,
                        "correction_l2": correction_l2,
                        "minimum_reference_robot_l2": minimum_reference_robot_l2,
                        "post_hold_state_l2": post_hold_state_l2,
                        "post_hold_eef_position_l2": post_hold_eef_position_l2,
                        "post_hold_eef_rotation_l2": post_hold_eef_rotation_l2,
                        "post_hold_gripper_l2": post_hold_gripper_l2,
                        "post_hold_robot_state": post_hold_robot_state.tolist(),
                        "post_hold_reference_robot_state": (
                            post_hold_reference_state.tolist()
                            if post_hold_reference_state is not None
                            else None
                        ),
                        "fork_state_linf": fork_state_linf,
                        "candidate_actions": [
                            action.tolist() for action in candidate_actions
                        ],
                        "continuous_action_scales": (
                            [1.0 / age for age in range(1, hold_depth + 1)]
                            if candidate_name == "lag1_harmonic"
                            else [1.0 for _ in range(hold_depth)]
                        ),
                        "mean_action_norm": float(
                            np.mean(
                                [
                                    np.linalg.norm(action[:6])
                                    for action in candidate_actions
                                ]
                            )
                        ),
                        "reference_trajectory_steps": len(reference_actions),
                    }
                    append_jsonl(output_jsonl, payload)
                    records_written += 1
                    print(
                        f"[branch] episode={episode_index} fork={fork_step} "
                        f"candidate={candidate_name} depth={hold_depth} "
                        f"success={branch_success} recovery_steps={recovery_steps}",
                        flush=True,
                    )
                    manifest.update(
                        {"records": records_written, "last_record": payload}
                    )
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False)
                    )

    env.close()
    if selected_reference_episodes < cfg.num_reference_episodes:
        manifest.update(
            {
                "status": "incomplete",
                "finished_at_unix": time.time(),
                "records": records_written,
                "attempted_reference_episodes": attempted_reference_episodes,
                "selected_reference_episodes": selected_reference_episodes,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        raise RuntimeError(
            "Could not collect the requested number of successful target "
            f"references: selected={selected_reference_episodes} "
            f"requested={cfg.num_reference_episodes}."
        )
    manifest.update(
        {
            "status": "complete",
            "finished_at_unix": time.time(),
            "records": records_written,
            "attempted_reference_episodes": attempted_reference_episodes,
            "selected_reference_episodes": selected_reference_episodes,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
