"""Generate a LeRobot v3.0 dataset from expert FSM demonstrations."""

import argparse
import os
import sys

import mujoco
import numpy as np

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.cameras import CameraRenderer, compute_keypoints  # noqa: E402
from src.constants import ACTION_REPEAT, BINS, CONTROL_FPS, IMAGE_SIZE, TASK_SETS  # noqa: E402
from src.controller import IKController, TARGET_ORI  # noqa: E402
from src.env import PickPlaceEnv  # noqa: E402
from src.pick_and_place import PickAndPlaceTask  # noqa: E402
from src.pose_utils import pos_rotmat_to_se3, se3_to_8dof, se3_to_10dof  # noqa: E402
from src.robot import PandaRobot  # noqa: E402

SCENE_XML = os.path.join(_PROJECT_ROOT, "pick_and_place_scene.xml")

FEATURES = {
    "observation.images.overhead": {
        "dtype": "image",
        "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist": {
        "dtype": "image",
        "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (11,),
        "names": None,
    },
    "observation.state.ee.8dof": {
        "dtype": "float32",
        "shape": (8,),
        "names": None,
    },
    "observation.state.ee.10dof": {
        "dtype": "float32",
        "shape": (10,),
        "names": None,
    },
    "observation.state.ee.8dof_rel": {
        "dtype": "float32",
        "shape": (8,),
        "names": None,
    },
    "observation.state.ee.10dof_rel": {
        "dtype": "float32",
        "shape": (10,),
        "names": None,
    },
    "action.ee.8dof": {
        "dtype": "float32",
        "shape": (8,),
        "names": None,
    },
    "action.ee.10dof": {
        "dtype": "float32",
        "shape": (10,),
        "names": None,
    },
    "action.ee.8dof_rel": {
        "dtype": "float32",
        "shape": (8,),
        "names": None,
    },
    "action.ee.10dof_rel": {
        "dtype": "float32",
        "shape": (10,),
        "names": None,
    },
    "observation.target_bin_onehot": {
        "dtype": "float32",
        "shape": (3,),
        "names": None,
    },
    "observation.keypoints_overhead": {
        "dtype": "float32",
        "shape": (14,),
        "names": None,
    },
    "observation.keypoints_wrist": {
        "dtype": "float32",
        "shape": (14,),
        "names": None,
    },
}


def make_task_string(obj_name: str, bin_name: str) -> str:
    """Create a human-readable task description."""
    obj_color = obj_name.replace("obj_", "")
    bin_color = bin_name.replace("bin_", "")
    return f"Pick {obj_color} object and place in {bin_color} bin"


def get_state(robot: PandaRobot) -> np.ndarray:
    """Get state vector: [ee_xyz(3), gripper_normalized(1), arm_qpos(7)]."""
    gripper_norm = robot.gripper_ctrl / PandaRobot.GRIPPER_OPEN
    return np.concatenate(
        [
            robot.ee_pos.astype(np.float32),
            np.array([gripper_norm], dtype=np.float32),
            robot.arm_qpos.astype(np.float32),
        ]
    )


def get_bin_onehot(bin_name: str) -> np.ndarray:
    """Get one-hot encoding for target bin."""
    idx = BINS.index(bin_name)
    onehot = np.zeros(3, dtype=np.float32)
    onehot[idx] = 1.0
    return onehot


def get_actions(
    task: PickAndPlaceTask,
    initial_se3_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute FSM's commanded SE(3) in both absolute and relative frames.

    The FSM always commands a position target with downward orientation (TARGET_ORI).
    We build the absolute target SE(3), then compute T_rel = T_init_inv @ T_target.

    Returns:
        (action_8dof, action_10dof, action_8dof_rel, action_10dof_rel)
    """
    target_pos = task._target_pos if task._target_pos is not None else task.robot.ee_pos
    gripper = 1.0 if task.robot.gripper_ctrl == PandaRobot.GRIPPER_OPEN else 0.0

    # Absolute target SE(3): commanded position + fixed downward orientation
    T_target = pos_rotmat_to_se3(target_pos, TARGET_ORI)

    # Relative to initial
    T_rel = initial_se3_inv @ T_target

    return (
        se3_to_8dof(T_target, gripper),
        se3_to_10dof(T_target, gripper),
        se3_to_8dof(T_rel, gripper),
        se3_to_10dof(T_rel, gripper),
    )


def run_episode(
    env: PickPlaceEnv,
    robot: PandaRobot,
    controller: IKController,
    renderer: CameraRenderer,
    obj_name: str,
    bin_name: str,
) -> list[dict]:
    """Run one expert FSM episode and collect frames."""
    env.reset_to_keyframe("scene_start")

    # Capture initial EE SE(3) and its inverse
    initial_se3 = pos_rotmat_to_se3(robot.ee_pos, robot.ee_xmat)
    initial_se3_inv = np.linalg.inv(initial_se3)

    task = PickAndPlaceTask(env, robot, controller, tasks=[(obj_name, bin_name)])

    frames = []
    physics_step = 0
    task_str = make_task_string(obj_name, bin_name)
    bin_onehot = get_bin_onehot(bin_name)

    while not task.is_done:
        task.update()
        env.step()
        physics_step += 1

        # Record a frame every ACTION_REPEAT physics steps
        if physics_step % ACTION_REPEAT == 0:
            mujoco.mj_forward(env.model, env.data)

            img_overhead = renderer.render(env.data, "overhead")
            img_wrist = renderer.render(env.data, "wrist")
            kp_overhead = compute_keypoints(env.model, env.data, "overhead").flatten()
            kp_wrist = compute_keypoints(env.model, env.data, "wrist").flatten()

            action_8dof, action_10dof, action_8dof_rel, action_10dof_rel = get_actions(
                task, initial_se3_inv
            )

            # Current EE pose (absolute and relative to initial)
            T_current = pos_rotmat_to_se3(robot.ee_pos, robot.ee_xmat)
            T_rel_obs = initial_se3_inv @ T_current
            gripper_val = robot.gripper_ctrl / PandaRobot.GRIPPER_OPEN
            obs_state_8dof = se3_to_8dof(T_current, gripper_val)
            obs_state_10dof = se3_to_10dof(T_current, gripper_val)
            obs_state_8dof_rel = se3_to_8dof(T_rel_obs, gripper_val)
            obs_state_10dof_rel = se3_to_10dof(T_rel_obs, gripper_val)

            frame = {
                "task": task_str,
                "observation.images.overhead": img_overhead,
                "observation.images.wrist": img_wrist,
                "observation.state": get_state(robot),
                "observation.state.ee.8dof": obs_state_8dof,
                "observation.state.ee.10dof": obs_state_10dof,
                "observation.state.ee.8dof_rel": obs_state_8dof_rel,
                "observation.state.ee.10dof_rel": obs_state_10dof_rel,
                "action.ee.8dof": action_8dof,
                "action.ee.10dof": action_10dof,
                "action.ee.8dof_rel": action_8dof_rel,
                "action.ee.10dof_rel": action_10dof_rel,
                "observation.target_bin_onehot": bin_onehot.copy(),
                "observation.keypoints_overhead": kp_overhead,
                "observation.keypoints_wrist": kp_wrist,
            }
            frames.append(frame)

    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Generate LeRobot dataset from expert FSM"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repo ID (e.g. user/pick-place)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Total number of episodes"
    )
    parser.add_argument(
        "--root", type=str, default="./datasets", help="Local dataset root directory"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        choices=list(TASK_SETS.keys()),
        help="Task set: 'all' (9 combos), 'match' (color-matched), 'cross' (cross-color)",
    )
    args = parser.parse_args()

    task_list = TASK_SETS[args.tasks]

    # Import LeRobot (handle both old and new import paths)
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Load environment with wrist camera
    print("Loading scene...")
    env = PickPlaceEnv(SCENE_XML, add_wrist_camera=True)
    robot = PandaRobot(env.model, env.data)
    controller = IKController(env.model, env.data, robot)
    renderer = CameraRenderer(env.model, IMAGE_SIZE, IMAGE_SIZE)

    # Create dataset
    print(f"Creating dataset: {args.repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=CONTROL_FPS,
        features=FEATURES,
        root=args.root,
        robot_type="franka_panda",
        use_videos=False,
        image_writer_threads=4,
    )

    # Generate episodes, cycling through tasks
    print(f"Task set '{args.tasks}': {len(task_list)} task(s)")
    for ep_idx in range(args.num_episodes):
        task_idx = ep_idx % len(task_list)
        obj_name, bin_name = task_list[task_idx]

        print(
            f"Episode {ep_idx + 1}/{args.num_episodes}: {obj_name} → {bin_name}",
            end="",
            flush=True,
        )

        frames = run_episode(env, robot, controller, renderer, obj_name, bin_name)

        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()

        print(f" ({len(frames)} frames)")

    # Finalize
    dataset.finalize()
    renderer.close()
    print(f"\nDataset saved to {args.root}/{args.repo_id}")
    print(f"Total episodes: {args.num_episodes}")


if __name__ == "__main__":
    main()
