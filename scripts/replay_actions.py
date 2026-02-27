"""Replay recorded dataset actions in MuJoCo to verify correctness."""

import argparse
import json
import os
import sys
import time

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mujoco_manip.constants import ACTION_REPEAT, TASK_SETS
from mujoco_manip.gym_env import PickPlaceGymEnv


def main() -> None:
    """Parse CLI args and replay dataset actions in MuJoCo."""
    parser = argparse.ArgumentParser(
        description="Replay dataset actions in MuJoCo viewer"
    )
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo ID")
    parser.add_argument(
        "--root",
        type=str,
        default="./datasets",
        help="Parent directory containing datasets (default: ./datasets)",
    )
    parser.add_argument(
        "--episode-index", type=int, default=0, help="Episode index to replay"
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="action.ee.pos_quat_g",
        help="Action key to replay (default: action.ee.pos_quat_g)",
    )
    parser.add_argument(
        "--slow",
        type=float,
        default=1.0,
        help="Slow-motion multiplier (e.g. 2 = half speed)",
    )
    args = parser.parse_args()

    dataset_root = os.path.join(args.root, args.repo_id)
    dataset = LeRobotDataset(
        args.repo_id,
        episodes=[args.episode_index],
        root=dataset_root,
    )
    num_frames: int = len(dataset)
    print(f"Loaded episode {args.episode_index}: {num_frames} frames")
    print(f"Action key: {args.action_key}")

    first_frame = dataset[0]
    if args.action_key not in first_frame:
        available = [k for k in first_frame if k.startswith("action.")]
        print(f"Error: '{args.action_key}' not found. Available: {available}")
        sys.exit(1)

    # Derive action_mode from the action key
    # "action.ee.pos_quat_g" → "ee_pos_quat_g"
    action_mode = args.action_key.replace("action.", "").replace(".", "_")

    # Read generation metadata to restore randomization and task
    metadata_path = os.path.join(dataset_root, "metadata.json")
    has_randomization = False
    spawn_x_range = (-0.20, 0.20)
    spawn_y_range = (0.30, 0.45)
    episode_seed: int | None = None
    task: tuple[str, str] | None = None

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Restore randomization settings
        episode_seeds = metadata.get("episode_seeds")
        if episode_seeds and args.episode_index < len(episode_seeds):
            has_randomization = True
            episode_seed = episode_seeds[args.episode_index]
        if "spawn_x_range" in metadata:
            spawn_x_range = tuple(metadata["spawn_x_range"])
        if "spawn_y_range" in metadata:
            spawn_y_range = tuple(metadata["spawn_y_range"])

        # Reconstruct task from metadata
        meta_task = metadata.get("task")
        meta_tasks = metadata.get("tasks", "all")
        if meta_task is not None:
            task_list = [tuple(meta_task)]
        elif meta_tasks in TASK_SETS:
            task_list = TASK_SETS[meta_tasks]
        else:
            task_list = TASK_SETS["all"]
        task = task_list[args.episode_index % len(task_list)]

    # Create gym env with human render mode
    gym_env = PickPlaceGymEnv(
        action_mode=action_mode,
        render_mode="human",
        reward_type="staged",
        randomize_objects=has_randomization,
        spawn_x_range=spawn_x_range,
        spawn_y_range=spawn_y_range,
    )

    reset_kwargs: dict = {}
    if episode_seed is not None:
        reset_kwargs["seed"] = episode_seed
    if task is not None:
        reset_kwargs["options"] = {"task": task}
    obs, info = gym_env.reset(**reset_kwargs)

    if has_randomization:
        print(f"Restored object randomization for episode {args.episode_index}")
    if task is not None:
        print(f"Task: {task[0]} → {task[1]}")

    step_time: float = (
        gym_env.pick_place_env.model.opt.timestep * ACTION_REPEAT * args.slow
    )

    print(f"\nReplaying {num_frames} frames (ACTION_REPEAT={ACTION_REPEAT})...")
    print(f"{'Frame':>6}  {'Action XYZ':>30}  {'EE XYZ':>30}  {'Error':>8}")
    print("-" * 82)

    for i in range(num_frames):
        if not gym_env.pick_place_env.is_running():
            print("\nViewer closed.")
            break

        frame = dataset[i]
        action = frame[args.action_key].numpy()

        t_start: float = time.monotonic()
        obs, reward, terminated, truncated, info = gym_env.step(action)
        gym_env.render()

        # Compute target position for error reporting
        target_xyz, _ = gym_env.decode_action(action)
        ee_pos: np.ndarray = gym_env.robot.ee_pos
        err: float = float(np.linalg.norm(ee_pos - target_xyz))
        print(
            f"{i:>6}  {target_xyz[0]:>9.4f} {target_xyz[1]:>9.4f} {target_xyz[2]:>9.4f}"
            f"  {ee_pos[0]:>9.4f} {ee_pos[1]:>9.4f} {ee_pos[2]:>9.4f}"
            f"  {err:>8.4f}"
        )

        elapsed: float = time.monotonic() - t_start
        sleep_time: float = step_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\nReplay finished. Viewer remains open — close window to exit.")
    while gym_env.pick_place_env.is_running():
        gym_env.pick_place_env.sync()
        time.sleep(0.05)

    gym_env.close()


if __name__ == "__main__":
    main()
