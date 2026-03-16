"""Replay recorded bottle-packing dataset actions in MuJoCo to verify correctness."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mujoco_manip.tasks.bottle_packing.constants import (
    ACTION_REPEAT,
    NUM_WELLS,
    well_row_col,
)
from mujoco_manip.tasks.bottle_packing.gym_env import BottlePackingGymEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay bottle-packing dataset actions in MuJoCo viewer"
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

    dataset_root = Path(args.root) / args.repo_id
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
    action_mode = args.action_key.replace("action.", "").replace(".", "_")

    # Read generation metadata to reconstruct the well schedule
    metadata_path = dataset_root / "metadata.json"
    metadata: dict = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    task_mode = metadata.get("task", "sequential")
    seed = metadata.get("seed", 0)
    excluded_wells: set[int] = set()
    if metadata.get("excluded_wells") is not None:
        excluded_wells = {int(w) for w in metadata["excluded_wells"]}
    available_wells = NUM_WELLS - len(excluded_wells)
    num_bottles = metadata.get("num_bottles") or available_wells
    num_bottles = min(int(num_bottles), available_wells)

    # Rebuild the well schedule for the run containing this episode
    rng = random.Random(seed)
    ep = args.episode_index
    # Fast-forward RNG through prior runs
    num_prior_runs = ep // num_bottles
    for _ in range(num_prior_runs):
        wells = [w for w in range(NUM_WELLS) if w not in excluded_wells]
        if task_mode == "random":
            rng.shuffle(wells)

    # Build schedule for the current run
    wells = [w for w in range(NUM_WELLS) if w not in excluded_wells]
    if task_mode == "random":
        rng.shuffle(wells)
    well_schedule = wells[:num_bottles]

    step_in_run = ep % num_bottles
    bottle_index = step_in_run
    well_index = well_schedule[step_in_run]

    # Build packed state from prior episodes in this run
    packed: dict[int, int] = {}
    for i in range(step_in_run):
        packed[i] = well_schedule[i]

    # Reconstruct per-episode seed — spawn the same count as generation script
    num_episodes = metadata.get("num_episodes", ep + 1)
    if ep >= num_episodes:
        raise ValueError(
            f"Episode {ep} exceeds num_episodes={num_episodes} from metadata"
        )
    ep_seed = int(
        np.random.SeedSequence(seed).spawn(num_episodes)[ep].generate_state(1)[0]
    )

    gym_env = BottlePackingGymEnv(
        action_mode=action_mode,
        render_mode="human",
        reward_type="staged",
    )

    obs, info = gym_env.reset(
        seed=ep_seed,
        options={
            "well_index": well_index,
            "bottle_index": bottle_index,
            "packed": packed,
        },
    )

    row, col = well_row_col(well_index)
    print(f"Target well: ({row},{col}) [index {well_index}]")
    if packed:
        print(f"Pre-packed: {packed}")

    step_time: float = (
        gym_env.bottle_packing_env.model.opt.timestep * ACTION_REPEAT * args.slow
    )

    print(f"\nReplaying {num_frames} frames (ACTION_REPEAT={ACTION_REPEAT})...")
    print(
        f"{'Frame':>6}  {'Action XYZ':>30}  {'EE XYZ':>30}  {'Error':>8}"
        f"  {'F/T (fx fy fz tx ty tz)':>50}"
    )
    print("-" * 134)

    for i in range(num_frames):
        if not gym_env.bottle_packing_env.is_running():
            print("\nViewer closed.")
            break

        frame = dataset[i]
        action = frame[args.action_key].numpy()

        t_start: float = time.monotonic()
        obs, reward, terminated, truncated, info = gym_env.step(action)
        gym_env.render()

        target_xyz, _ = gym_env.decode_action(action)
        ee_pos: np.ndarray = gym_env.robot.ee_pos
        err: float = float(np.linalg.norm(ee_pos - target_xyz))
        ft: np.ndarray = obs["state.ee.force_torque"]
        print(
            f"{i:>6}  {target_xyz[0]:>9.4f} {target_xyz[1]:>9.4f} {target_xyz[2]:>9.4f}"
            f"  {ee_pos[0]:>9.4f} {ee_pos[1]:>9.4f} {ee_pos[2]:>9.4f}"
            f"  {err:>8.4f}"
            f"  {ft[0]:>7.3f} {ft[1]:>7.3f} {ft[2]:>7.3f}"
            f" {ft[3]:>7.4f} {ft[4]:>7.4f} {ft[5]:>7.4f}"
        )

        elapsed: float = time.monotonic() - t_start
        sleep_time: float = step_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\nReplay finished. Viewer remains open — close window to exit.")
    while gym_env.bottle_packing_env.is_running():
        gym_env.bottle_packing_env.sync()
        time.sleep(0.05)

    gym_env.close()


if __name__ == "__main__":
    main()
