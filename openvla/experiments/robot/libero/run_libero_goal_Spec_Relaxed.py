"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[4]
OPENVLA_ROOT = REPO_ROOT / "openvla"
for path in (str(OPENVLA_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb
import json


def summarize_dflash_stats(step_stats_list):
    valid_stats = [item for item in step_stats_list if item is not None]
    if not valid_stats:
        return None

    total_blocks = sum(item.get("num_blocks", 0) for item in valid_stats)
    total_accept_length = sum(sum(item.get("accept_lengths", [])) for item in valid_stats)
    total_accepted = sum(item.get("accepted_tokens", 0) for item in valid_stats)
    total_compared = sum(item.get("compared_tokens", 0) for item in valid_stats)

    position_hits = {}
    position_counts = {}
    for item in valid_stats:
        for position_stat in item.get("per_position", []):
            pos = position_stat["position"]
            position_hits[pos] = position_hits.get(pos, 0) + position_stat.get("hit_count", 0)
            position_counts[pos] = position_counts.get(pos, 0) + position_stat.get("count", 0)

    per_position = []
    for pos in sorted(position_counts):
        count = position_counts[pos]
        hit = position_hits.get(pos, 0)
        reject = count - hit
        per_position.append(
            {
                "position": pos,
                "count": count,
                "hit_count": hit,
                "reject_count": reject,
                "hit_rate": (hit / count) if count > 0 else None,
                "reject_rate": (reject / count) if count > 0 else None,
            }
        )

    return {
        "num_steps": len(valid_stats),
        "num_blocks": total_blocks,
        "avg_accept_length": (total_accept_length / total_blocks) if total_blocks > 0 else 0.0,
        "overall_hit_rate": (total_accepted / total_compared) if total_compared > 0 else None,
        "accepted_tokens": total_accepted,
        "compared_tokens": total_compared,
        "per_position": per_position,
    }

# Append current directory so that interpreter can find experiments.robot
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero.eval_metrics import (
    format_conditional_prefix,
    format_generation_summary,
    summarize_generation_stats,
    write_eval_summary,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    use_spec: bool = True
    parallel_draft: bool = False
    draft_backend: str = "dflash"                    # eagle | dflash，当切为dflash时，generate_mode自动切到dflash
    accept_threshold: int = 9
    dflash_block_size: int = 7                      # OpenVLA action token 只有 7 个，block 取小一些更合适
    dflash_num_draft_layers: int = 5
    dflash_target_layer_ids: Optional[list[int]] = None
    dflash_mask_token_id: Optional[int] = None
    dflash_use_causal_residual_sampling: bool = False
    dflash_confidence_threshold: float = 0.0
    dflash_confidence_min_tokens: int = 1
    dflash_acceptance_mode: str = "action_group"
    dflash_tree_mode: str = "off"
    dflash_tree_budget: int = 0
    dflash_target_logits_mode: str = "full"
    dflash_verify_skip_mode: str = "off"
    dflash_verify_skip_min_top1_prob: float = 1.0
    dflash_verify_skip_min_margin: float = 1.0
    dflash_verify_skip_min_base_agreement: float = 1.0
    dflash_temporal_route_min_cosine: float = 1.0
    dflash_temporal_route_stop_on_reject: bool = True
    dflash_temporal_fuse_verify: bool = True
    dflash_temporal_prefill_fusion: bool = False
    dflash_temporal_prefill_min_stable_actions: int = 3
    dflash_temporal_prefix_cert_tokens: int = 0
    dflash_temporal_bypass_max_pixel_relative_l2: float = 0.0
    dflash_temporal_bypass_use_pixel_guard: bool = True
    dflash_temporal_hold_policy: str = "fixed"
    dflash_temporal_adaptive_min_verified_run: int = 2
    dflash_temporal_adaptive_max_anchor_pixel_relative_l2: float = 0.03
    dflash_temporal_prefill_tree: bool = False
    dflash_temporal_prefill_tree_max_candidates: int = 3
    dflash_temporal_prefill_tree_min_history: int = 2
    dflash_verify_skip_min_temporal_cosine: float = 1.0
    dflash_verify_skip_min_stable_actions: int = 4
    dflash_verify_skip_max_consecutive: int = 1
    dflash_profile_stages: bool = False
    dflash_debug_compare_target_ar: bool = False
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    spec_checkpoint: Union[str, Path] = "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_goal_dflash"
    task_suite_name: str = "libero_goal"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    trial_start_index: int = 0                       # Diagnostic offset into LIBERO's fixed initial-state list.
    max_eval_tasks: Optional[int] = None              # Diagnostic smoke limit; None evaluates all tasks.

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    sync_cuda_timing: bool = False                   # False matches upstream SpecVLA paper-style timing.
    timing_scope: str = "last_task"                  # last_task matches upstream timing JSON; full_suite is less noisy.

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    if cfg.timing_scope not in {"last_task", "full_suite"}:
        raise ValueError("cfg.timing_scope must be one of: last_task, full_suite")

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    eval_family = "dflash_relaxed" if cfg.draft_backend == "dflash" else "specvla_relaxed"
    target_dir = Path(cfg.local_log_dir) / eval_family
    target_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = str(target_dir / f"{run_id}.txt")
    log_file = open(local_log_filepath, "w")
    local_log_timefilepath = str(target_dir / f"{run_id}-{eval_family}_timing.json")
    local_log_summaryfilepath = str(target_dir / f"{run_id}-{eval_family}_summary.json")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if cfg.max_eval_tasks is not None:
        if cfg.max_eval_tasks <= 0:
            raise ValueError("max_eval_tasks must be positive when provided.")
        num_tasks_in_suite = min(num_tasks_in_suite, cfg.max_eval_tasks)
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    total_episode_time = []
    total_generation_step_stats = []
    last_task_episode_time = []
    last_task_generation_step_stats = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        trial_stop_index = cfg.trial_start_index + cfg.num_trials_per_task
        if cfg.trial_start_index < 0 or trial_stop_index > len(initial_states):
            raise ValueError(
                "Requested LIBERO initial-state range is out of bounds: "
                f"start={cfg.trial_start_index}, stop={trial_stop_index}, "
                f"available={len(initial_states)}."
            )

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_episode_time = []
        task_generation_step_stats = []
        episode_indices = range(cfg.trial_start_index, trial_stop_index)
        for episode_idx in tqdm.tqdm(episode_indices):
            total_time = []
            episode_generation_stats = []
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            reset_runtime_state = getattr(model, "reset_dflash_runtime_state", None)
            if reset_runtime_state is not None:
                reset_runtime_state()
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            print(f"LIBERO initial-state index: {episode_idx}")
            log_file.write(f"LIBERO initial-state index: {episode_idx}\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    action,time,generation_stats = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        return_time=True,
                        return_generation_stats=True,
                        generate_mode=("dflash" if cfg.draft_backend == "dflash" else "speculative")
                    )
                    episode_generation_stats.append(generation_stats)
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    # Match upstream SpecVLA timing: successful terminal
                    # actions are not included in the timing JSON.
                    total_time.append(time)

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break
            #exit()
            task_episodes += 1
            total_episodes += 1
            total_episode_time.append(total_time)
            task_episode_time.append(total_time)
            task_generation_step_stats.extend(episode_generation_stats)
            total_generation_step_stats.extend(episode_generation_stats)

            # Save a replay video of the episode
            # save_rollout_video(
            #     replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            # )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            episode_generation_summary = summarize_generation_stats(episode_generation_stats)
            if episode_generation_summary is not None:
                episode_summary_line = format_generation_summary(episode_generation_summary)
                print(episode_summary_line)
                log_file.write(f"{episode_summary_line}\n")
                if episode_generation_summary["per_position"]:
                    per_position_str = ", ".join(
                        f"p{item['position']}={item['hit_rate']:.3f}" if item["hit_rate"] is not None else f"p{item['position']}=None"
                        for item in episode_generation_summary["per_position"]
                    )
                    print(f"DFlash per-position hit rate: {per_position_str}")
                    log_file.write(f"DFlash per-position hit rate: {per_position_str}\n")
                if cfg.use_wandb:
                    episode_log_payload = {
                        f"spec/length/{task_description}": episode_generation_summary["length"],
                        f"spec/avg_accept_length/{task_description}": episode_generation_summary["avg_accept_length"],
                    }
                    if episode_generation_summary["overall_hit_rate"] is not None:
                        episode_log_payload[f"spec/overall_hit_rate/{task_description}"] = episode_generation_summary["overall_hit_rate"]
                    for item in episode_generation_summary["per_position"]:
                        if item["hit_rate"] is not None:
                            episode_log_payload[f"dflash/hit_rate_p{item['position']}/{task_description}"] = item["hit_rate"]
                    wandb.log(episode_log_payload)
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        task_generation_summary = summarize_generation_stats(task_generation_step_stats)
        if task_generation_summary is not None:
            task_summary_line = format_generation_summary(task_generation_summary, prefix="Current task speculative stats")
            print(task_summary_line)
            log_file.write(f"{task_summary_line}\n")
        log_file.flush()
        if cfg.use_wandb:
            log_payload = {
                f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                f"num_episodes/{task_description}": task_episodes,
            }
            if task_generation_summary is not None:
                log_payload[f"spec/length/{task_description}"] = task_generation_summary["length"]
                log_payload[f"spec/avg_accept_length/{task_description}"] = task_generation_summary["avg_accept_length"]
                if task_generation_summary["overall_hit_rate"] is not None:
                    log_payload[f"spec/overall_hit_rate/{task_description}"] = task_generation_summary["overall_hit_rate"]
            wandb.log(log_payload)
        #exit()
        last_task_episode_time = task_episode_time
        last_task_generation_step_stats = task_generation_step_stats
    timing_episode_time = last_task_episode_time if cfg.timing_scope == "last_task" else total_episode_time
    timing_generation_stats = (
        last_task_generation_step_stats if cfg.timing_scope == "last_task" else total_generation_step_stats
    )
    with open(local_log_timefilepath,mode='w') as f:
        json.dump(timing_episode_time,f)
    summary_payload = write_eval_summary(
        local_log_summaryfilepath,
        cfg=cfg,
        run_id=run_id,
        eval_family=eval_family,
        total_episodes=total_episodes,
        total_successes=total_successes,
        episode_times=timing_episode_time,
        generation_stats=timing_generation_stats,
    )
    print(f"Saved eval summary to: {local_log_summaryfilepath}")
    log_file.write(f"Saved eval summary to: {local_log_summaryfilepath}\n")
    total_generation_summary = summary_payload.get("generation")
    if total_generation_summary is not None:
        total_summary_line = format_generation_summary(total_generation_summary, prefix="Total speculative stats")
        print(total_summary_line)
        log_file.write(f"{total_summary_line}\n")
        conditional_line = format_conditional_prefix(total_generation_summary)
        if conditional_line is not None:
            print(conditional_line)
            log_file.write(f"{conditional_line}\n")
    # Save local log file
    log_file.close()
   # print('total time')
   # print(sum([sum(item) for item in total_episode_time]))

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        total_log_payload = {
            "success_rate/total": float(total_successes) / float(total_episodes),
            "num_episodes/total": total_episodes,
        }
        if total_generation_summary is not None:
            total_log_payload["spec/length/total"] = total_generation_summary["length"]
            total_log_payload["spec/avg_accept_length/total"] = total_generation_summary["avg_accept_length"]
            if total_generation_summary["overall_hit_rate"] is not None:
                total_log_payload["spec/overall_hit_rate/total"] = total_generation_summary["overall_hit_rate"]
        wandb.log(total_log_payload)
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
