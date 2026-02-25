"""Generate a LeRobot v3.0 dataset from expert FSM demonstrations."""

import json
import os
import sys
from pathlib import Path

import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from mujoco_manip.cameras import CameraRenderer, compute_keypoints, project_3d_to_2d  # noqa: E402
from mujoco_manip.constants import (  # noqa: E402
    ACTION_REPEAT,
    BINS,
    CONTROL_FPS,
    IMAGE_SIZE,
    OBJECTS,
    TASK_SETS,
)
from mujoco_manip.controller import IKController, TARGET_ORI  # noqa: E402
from mujoco_manip.env import PickPlaceEnv  # noqa: E402
from mujoco_manip.features import FEATURES  # noqa: E402
from mujoco_manip.pick_and_place import PickAndPlaceTask  # noqa: E402
from mujoco_manip.pose_utils import pos_rotmat_to_se3, se3_to_8dof, se3_to_10dof  # noqa: E402
from mujoco_manip.robot import PandaRobot  # noqa: E402

SCENE_XML = os.path.join(_PROJECT_ROOT, "pick_and_place_scene.xml")


def make_task_string(obj_name: str, bin_name: str) -> str:
    """Create a human-readable task description.

    Args:
        obj_name: Object body name (e.g. ``"obj_red"``).
        bin_name: Bin body name (e.g. ``"bin_red"``).

    Returns:
        Description string like ``"Pick red object and place in red bin"``.
    """
    obj_color = obj_name.replace("obj_", "")
    bin_color = bin_name.replace("bin_", "")
    return f"Pick {obj_color} object and place in {bin_color} bin"


def get_state(robot: PandaRobot) -> np.ndarray:
    """Build the state vector [ee_xyz(3), gripper_norm(1), arm_qpos(7)].

    Args:
        robot: Robot interface.

    Returns:
        State array (11,), dtype float32.
    """
    gripper_norm = robot.gripper_ctrl / PandaRobot.GRIPPER_OPEN
    return np.concatenate(
        [
            robot.ee_pos.astype(np.float32),
            np.array([gripper_norm], dtype=np.float32),
            robot.arm_qpos.astype(np.float32),
        ]
    )


def get_bin_onehot(bin_name: str) -> np.ndarray:
    """Return one-hot encoding (3,) for the target bin.

    Args:
        bin_name: Bin body name (e.g. ``"bin_red"``).

    Returns:
        One-hot array (3,), dtype float32.
    """
    idx = BINS.index(bin_name)
    onehot = np.zeros(3, dtype=np.float32)
    onehot[idx] = 1.0
    return onehot


def get_obj_onehot(obj_name: str) -> np.ndarray:
    """Return one-hot encoding (3,) for the target object.

    Args:
        obj_name: Object body name (e.g. ``"obj_red"``).

    Returns:
        One-hot array (3,), dtype float32.
    """
    idx = OBJECTS.index(obj_name)
    onehot = np.zeros(3, dtype=np.float32)
    onehot[idx] = 1.0
    return onehot


def get_actions(
    task: PickAndPlaceTask,
    initial_se3_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the FSM's commanded SE(3) in absolute and relative frames.

    The FSM always commands a position target with downward orientation
    (``TARGET_ORI``). We build the absolute target SE(3), then compute
    ``T_rel = T_init_inv @ T_target``.

    Args:
        task: Current pick-and-place task state machine.
        initial_se3_inv: Inverse of the initial EE SE(3) (4, 4).

    Returns:
        Tuple of (action_8dof, action_10dof, action_8dof_rel, action_10dof_rel).
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
    feature_keys: set[str],
    rng: np.random.Generator | None = None,
    randomize_kwargs: dict | None = None,
) -> list[dict]:
    """Run one expert FSM episode and collect frames.

    Args:
        env: MuJoCo environment wrapper.
        robot: Robot control interface.
        controller: IK controller.
        renderer: Camera renderer for capturing images.
        obj_name: Object body name.
        bin_name: Target bin body name.
        feature_keys: Set of feature keys to include in each frame.
        rng: If provided, randomize object positions before the episode.
        randomize_kwargs: Extra keyword arguments for ``env.randomize_objects``.

    Returns:
        List of frame dicts with only the requested features.
    """
    if randomize_kwargs is None:
        randomize_kwargs = {}
    env.reset_to_keyframe("scene_start")
    if rng is not None:
        env.randomize_objects(rng, **randomize_kwargs)

    # Check which feature groups are needed
    need_images = bool(
        feature_keys & {"observation.images.overhead", "observation.images.wrist"}
    )
    need_keypoints = bool(
        feature_keys & {"observation.keypoints_overhead", "observation.keypoints_wrist"}
    )
    need_target_obj_kp = "observation.target_obj_keypoints_overhead" in feature_keys
    need_target_bin_kp = "observation.target_bin_keypoints_overhead" in feature_keys
    need_state = bool(
        feature_keys
        & {
            "observation.state",
            "observation.state.ee.8dof",
            "observation.state.ee.10dof",
            "observation.state.ee.8dof_rel",
            "observation.state.ee.10dof_rel",
        }
    )
    need_actions = bool(
        feature_keys
        & {
            "action.ee.8dof",
            "action.ee.10dof",
            "action.ee.8dof_rel",
            "action.ee.10dof_rel",
        }
    )
    need_bin_onehot = "observation.target_bin_onehot" in feature_keys
    need_obj_onehot = "observation.target_obj_onehot" in feature_keys

    # Capture initial EE SE(3) and its inverse (needed for actions and relative states)
    need_initial_se3 = need_actions or (
        need_state
        and feature_keys
        & {"observation.state.ee.8dof_rel", "observation.state.ee.10dof_rel"}
    )
    initial_se3_inv = None
    if need_initial_se3:
        initial_se3 = pos_rotmat_to_se3(robot.ee_pos, robot.ee_xmat)
        initial_se3_inv = np.linalg.inv(initial_se3)

    task = PickAndPlaceTask(env, robot, controller, tasks=[(obj_name, bin_name)])

    frames = []
    physics_step = 0
    task_str = make_task_string(obj_name, bin_name)
    bin_onehot = get_bin_onehot(bin_name) if need_bin_onehot else None
    obj_onehot = get_obj_onehot(obj_name) if need_obj_onehot else None

    while not task.is_done:
        task.update()
        env.step()
        physics_step += 1

        # Record a frame every ACTION_REPEAT physics steps
        if physics_step % ACTION_REPEAT == 0:
            mujoco.mj_forward(env.model, env.data)

            frame: dict = {"task": task_str}

            # Images
            if need_images:
                if "observation.images.overhead" in feature_keys:
                    frame["observation.images.overhead"] = renderer.render(
                        env.data, "overhead"
                    )
                if "observation.images.wrist" in feature_keys:
                    frame["observation.images.wrist"] = renderer.render(
                        env.data, "wrist"
                    )

            # Keypoints
            if need_keypoints:
                if "observation.keypoints_overhead" in feature_keys:
                    frame["observation.keypoints_overhead"] = compute_keypoints(
                        env.model, env.data, "overhead"
                    ).flatten()
                if "observation.keypoints_wrist" in feature_keys:
                    frame["observation.keypoints_wrist"] = compute_keypoints(
                        env.model, env.data, "wrist"
                    ).flatten()

            # Target keypoints (obj and bin projected separately)
            if need_target_obj_kp:
                obj_3d = env.data.xpos[
                    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                ][np.newaxis]
                frame["observation.target_obj_keypoints_overhead"] = project_3d_to_2d(
                    env.model, env.data, "overhead", obj_3d, IMAGE_SIZE
                ).flatten()
            if need_target_bin_kp:
                bin_3d = env.data.xpos[
                    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, bin_name)
                ][np.newaxis]
                frame["observation.target_bin_keypoints_overhead"] = project_3d_to_2d(
                    env.model, env.data, "overhead", bin_3d, IMAGE_SIZE
                ).flatten()

            # State variants
            if need_state:
                if "observation.state" in feature_keys:
                    frame["observation.state"] = get_state(robot)

                # SE3-derived states share some computation
                need_ee = feature_keys & {
                    "observation.state.ee.8dof",
                    "observation.state.ee.10dof",
                    "observation.state.ee.8dof_rel",
                    "observation.state.ee.10dof_rel",
                }
                if need_ee:
                    T_current = pos_rotmat_to_se3(robot.ee_pos, robot.ee_xmat)
                    gripper_val = robot.gripper_ctrl / PandaRobot.GRIPPER_OPEN

                    if "observation.state.ee.8dof" in feature_keys:
                        frame["observation.state.ee.8dof"] = se3_to_8dof(
                            T_current, gripper_val
                        )
                    if "observation.state.ee.10dof" in feature_keys:
                        frame["observation.state.ee.10dof"] = se3_to_10dof(
                            T_current, gripper_val
                        )

                    if feature_keys & {
                        "observation.state.ee.8dof_rel",
                        "observation.state.ee.10dof_rel",
                    }:
                        T_rel_obs = initial_se3_inv @ T_current
                        if "observation.state.ee.8dof_rel" in feature_keys:
                            frame["observation.state.ee.8dof_rel"] = se3_to_8dof(
                                T_rel_obs, gripper_val
                            )
                        if "observation.state.ee.10dof_rel" in feature_keys:
                            frame["observation.state.ee.10dof_rel"] = se3_to_10dof(
                                T_rel_obs, gripper_val
                            )

            # Actions
            if need_actions:
                action_8dof, action_10dof, action_8dof_rel, action_10dof_rel = (
                    get_actions(task, initial_se3_inv)
                )
                if "action.ee.8dof" in feature_keys:
                    frame["action.ee.8dof"] = action_8dof
                if "action.ee.10dof" in feature_keys:
                    frame["action.ee.10dof"] = action_10dof
                if "action.ee.8dof_rel" in feature_keys:
                    frame["action.ee.8dof_rel"] = action_8dof_rel
                if "action.ee.10dof_rel" in feature_keys:
                    frame["action.ee.10dof_rel"] = action_10dof_rel

            # Task context
            if need_bin_onehot:
                frame["observation.target_bin_onehot"] = bin_onehot.copy()
            if need_obj_onehot:
                frame["observation.target_obj_onehot"] = obj_onehot.copy()

            frames.append(frame)

    return frames


@hydra.main(config_path="../configs", config_name="generate", version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate a LeRobot dataset from expert FSM episodes."""
    if not cfg.repo_id:
        raise ValueError("repo_id is required (e.g. repo_id=user/pick-place)")

    if cfg.tasks not in TASK_SETS:
        raise ValueError(
            f"Unknown task set '{cfg.tasks}'. Choose from: {list(TASK_SETS.keys())}"
        )

    task_list = TASK_SETS[cfg.tasks]

    # Filter features if a subset is specified
    features = FEATURES
    if cfg.features is not None:
        requested = list(cfg.features)
        unknown = [k for k in requested if k not in FEATURES]
        if unknown:
            raise ValueError(
                f"Unknown feature keys: {unknown}. Valid keys: {list(FEATURES.keys())}"
            )
        features = {k: FEATURES[k] for k in requested}
    feature_keys = set(features)

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

    # Create dataset — nest under root/repo_id so multiple datasets coexist
    dataset_path = Path(cfg.root) / cfg.repo_id
    print(f"Creating dataset: {cfg.repo_id} → {dataset_path}")
    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=CONTROL_FPS,
        features=features,
        root=dataset_path,
        robot_type="franka_panda",
        use_videos=False,
        image_writer_threads=4,
    )

    # Per-episode seeds for object position randomization
    episode_seeds = None
    randomize_kwargs = {}
    if cfg.randomize_objects:
        ss = np.random.SeedSequence(cfg.seed)
        child_seeds = ss.spawn(cfg.num_episodes)
        episode_seeds = [int(cs.entropy) for cs in child_seeds]
        randomize_kwargs["x_range"] = tuple(cfg.spawn_x_range)
        randomize_kwargs["y_range"] = tuple(cfg.spawn_y_range)

    # Generate episodes, cycling through tasks
    print(f"Task set '{cfg.tasks}': {len(task_list)} task(s)")
    if episode_seeds is not None:
        print(f"Object randomization enabled (seed={cfg.seed}, per-episode seeds)")
    for ep_idx in range(cfg.num_episodes):
        task_idx = ep_idx % len(task_list)
        obj_name, bin_name = task_list[task_idx]

        print(
            f"Episode {ep_idx + 1}/{cfg.num_episodes}: {obj_name} → {bin_name}",
            end="",
            flush=True,
        )

        # Create a fresh RNG per episode for independent reproducibility
        rng = (
            np.random.default_rng(episode_seeds[ep_idx])
            if episode_seeds is not None
            else None
        )

        frames = run_episode(
            env,
            robot,
            controller,
            renderer,
            obj_name,
            bin_name,
            feature_keys,
            rng=rng,
            randomize_kwargs=randomize_kwargs,
        )

        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()

        print(f" ({len(frames)} frames)")

    # Save generation config as metadata
    generation_config = OmegaConf.to_container(cfg, resolve=True)
    # Remove hydra internals from the saved config
    generation_config.pop("hydra", None)
    # Store per-episode seeds for O(1) replay of any episode
    if episode_seeds is not None:
        generation_config["episode_seeds"] = episode_seeds

    # Write standalone metadata.json
    metadata_path = dataset_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(generation_config, f, indent=2)

    # Also store in LeRobot info.json
    try:
        from lerobot.datasets.utils import write_info

        dataset.meta.info["generation_config"] = generation_config
        write_info(dataset.meta.info, dataset.meta.root)
    except (ImportError, AttributeError):
        pass

    # Finalize
    dataset.finalize()
    renderer.close()
    print(f"\nDataset saved to {dataset_path}")
    print(f"Total episodes: {cfg.num_episodes}")


if __name__ == "__main__":
    main()
