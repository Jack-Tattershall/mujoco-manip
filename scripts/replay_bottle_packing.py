"""Replay recorded bottle-packing dataset actions in MuJoCo to verify correctness."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mujoco_manip.tasks.bottle_packing.constants import (
    ACTION_REPEAT,
    TASK_SETS,
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

    # Read generation metadata to restore task
    metadata_path = dataset_root / "metadata.json"
    well_index: int = 0

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        meta_well = metadata.get("well_index")
        meta_wells = metadata.get("wells", "all")
        if meta_well is not None:
            well_list = [int(meta_well)]
        elif meta_wells in TASK_SETS:
            well_list = TASK_SETS[meta_wells]
        else:
            well_list = TASK_SETS["all"]
        well_index = well_list[args.episode_index % len(well_list)]

    gym_env = BottlePackingGymEnv(
        action_mode=action_mode,
        render_mode="human",
        reward_type="staged",
    )

    obs, info = gym_env.reset(options={"well_index": well_index})

    row, col = well_row_col(well_index)
    print(f"Target well: ({row},{col}) [index {well_index}]")

    step_time: float = (
        gym_env.bottle_packing_env.model.opt.timestep * ACTION_REPEAT * args.slow
    )

    print(f"\nReplaying {num_frames} frames (ACTION_REPEAT={ACTION_REPEAT})...")
    print(f"{'Frame':>6}  {'Action XYZ':>30}  {'EE XYZ':>30}  {'Error':>8}")
    print("-" * 82)

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
    while gym_env.bottle_packing_env.is_running():
        gym_env.bottle_packing_env.sync()
        time.sleep(0.05)

    gym_env.close()


if __name__ == "__main__":
    main()
