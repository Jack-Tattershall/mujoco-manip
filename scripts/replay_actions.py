"""Replay recorded dataset actions in MuJoCo to verify correctness."""

import argparse
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from src.constants import ACTION_REPEAT  # noqa: E402
from src.controller import IKController  # noqa: E402
from src.env import PickPlaceEnv  # noqa: E402
from src.pose_utils import pos_rotmat_to_se3, se3_from_8dof, se3_from_10dof  # noqa: E402
from src.robot import PandaRobot  # noqa: E402

SCENE_XML = os.path.join(_PROJECT_ROOT, "pick_and_place_scene.xml")


def main() -> None:
    """Parse CLI args and replay dataset actions in MuJoCo."""
    parser = argparse.ArgumentParser(
        description="Replay dataset actions in MuJoCo viewer"
    )
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo ID")
    parser.add_argument(
        "--root", type=str, default=None, help="Local dataset root directory"
    )
    parser.add_argument(
        "--episode-index", type=int, default=0, help="Episode index to replay"
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="action.ee.8dof",
        help="Action key to replay (default: action.ee.8dof)",
    )
    parser.add_argument(
        "--slow",
        type=float,
        default=1.0,
        help="Slow-motion multiplier (e.g. 2 = half speed)",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.repo_id,
        episodes=[args.episode_index],
        root=args.root,
    )
    num_frames: int = len(dataset)
    print(f"Loaded episode {args.episode_index}: {num_frames} frames")
    print(f"Action key: {args.action_key}")

    first_frame = dataset[0]
    if args.action_key not in first_frame:
        available = [k for k in first_frame if k.startswith("action.")]
        print(f"Error: '{args.action_key}' not found. Available: {available}")
        sys.exit(1)

    is_relative: bool = args.action_key.endswith("_rel")
    is_10dof: bool = "10dof" in args.action_key

    env = PickPlaceEnv(SCENE_XML, add_wrist_camera=False)
    robot = PandaRobot(env.model, env.data)
    controller = IKController(env.model, env.data, robot)

    env.launch_viewer()
    env.reset_to_keyframe("scene_start")

    T_initial: np.ndarray | None = None
    if is_relative:
        T_initial = pos_rotmat_to_se3(robot.ee_pos, robot.ee_xmat)
        print(f"T_initial position: {T_initial[:3, 3]}")

    step_time: float = env.model.opt.timestep * ACTION_REPEAT * args.slow

    print(f"\nReplaying {num_frames} frames (ACTION_REPEAT={ACTION_REPEAT})...")
    print(f"{'Frame':>6}  {'Action XYZ':>30}  {'EE XYZ':>30}  {'Error':>8}")
    print("-" * 82)

    for i in range(num_frames):
        if not env.is_running():
            print("\nViewer closed.")
            break

        frame = dataset[i]
        action = frame[args.action_key].numpy()

        if is_relative:
            if is_10dof:
                T_abs = T_initial @ se3_from_10dof(action)
            else:
                T_abs = T_initial @ se3_from_8dof(action)
            target_xyz = T_abs[:3, 3]
        else:
            target_xyz = action[:3]

        gripper_val: float = action[-1]
        robot.data.ctrl[7] = gripper_val * 255

        t_start: float = time.monotonic()
        for _ in range(ACTION_REPEAT):
            q_target = controller.compute(target_xyz)
            robot.set_arm_ctrl(q_target)
            env.step()
        env.sync()

        ee_pos: np.ndarray = robot.ee_pos
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
    while env.is_running():
        env.sync()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
