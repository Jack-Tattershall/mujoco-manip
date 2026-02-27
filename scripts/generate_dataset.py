"""Generate a LeRobot v3.0 dataset from expert FSM demonstrations."""

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mujoco_manip.constants import (
    ACTION_REPEAT,
    CONTROL_FPS,
    TASK_SETS,
)
from mujoco_manip.controller import TARGET_ORI
from mujoco_manip.features import FEATURES
from mujoco_manip.gym_env import PickPlaceGymEnv
from mujoco_manip.pick_and_place import PickAndPlaceTask
from mujoco_manip.pose_utils import (
    pos_rotmat_to_se3,
    se3_to_pos_quat_g,
    se3_to_pos_rot6d_g,
)

# Map gym obs keys → dataset feature keys
_OBS_TO_FEATURE = {
    "image_overhead": "observation.images.overhead",
    "image_wrist": "observation.images.wrist",
    "state": "observation.state",
    "state.ee.pos_quat_g": "observation.state.ee.pos_quat_g",
    "state.ee.pos_rot6d_g": "observation.state.ee.pos_rot6d_g",
    "state.ee.pos_quat_g_rel": "observation.state.ee.pos_quat_g_rel",
    "state.ee.pos_rot6d_g_rel": "observation.state.ee.pos_rot6d_g_rel",
    "target_bin_onehot": "observation.target_bin_onehot",
    "target_obj_onehot": "observation.target_obj_onehot",
    "target_obj_keypoints_overhead": "observation.target_obj_keypoints_overhead",
    "target_bin_keypoints_overhead": "observation.target_bin_keypoints_overhead",
}


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


def get_actions(
    target_pos: np.ndarray,
    gripper_val: float,
    initial_se3_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the FSM's commanded SE(3) in absolute and relative frames.

    Args:
        target_pos: EE target position (3,) in world frame.
        gripper_val: Gripper value (1.0 open, 0.0 closed).
        initial_se3_inv: Inverse of the initial EE SE(3) (4, 4).

    Returns:
        Tuple of (action_pos_quat_g, action_pos_rot6d_g,
        action_pos_quat_g_rel, action_pos_rot6d_g_rel).
    """
    T_target = pos_rotmat_to_se3(target_pos, TARGET_ORI)
    T_rel = initial_se3_inv @ T_target

    return (
        se3_to_pos_quat_g(T_target, gripper_val),
        se3_to_pos_rot6d_g(T_target, gripper_val),
        se3_to_pos_quat_g(T_rel, gripper_val),
        se3_to_pos_rot6d_g(T_rel, gripper_val),
    )


def run_episode(
    gym_env: PickPlaceGymEnv,
    obj_name: str,
    bin_name: str,
    feature_keys: set[str],
    reward_type: str = "staged",
    episode_seed: int | None = None,
) -> list[dict]:
    """Run one expert FSM episode via the gym env and collect frames.

    Args:
        gym_env: Gymnasium pick-and-place environment.
        obj_name: Object body name.
        bin_name: Target bin body name.
        feature_keys: Set of feature keys to include in each frame.
        reward_type: Reward type string.
        episode_seed: Seed for this episode's reset.

    Returns:
        List of frame dicts with only the requested features.
    """
    # Reset the gym env with task override and optional seed
    reset_kwargs: dict = {}
    if episode_seed is not None:
        reset_kwargs["seed"] = episode_seed
    reset_kwargs["options"] = {"task": (obj_name, bin_name)}
    obs, info = gym_env.reset(**reset_kwargs)

    # Create FSM sharing the gym env's internals
    fsm = PickAndPlaceTask(
        gym_env.pick_place_env,
        gym_env.robot,
        gym_env.controller,
        tasks=[(obj_name, bin_name)],
    )

    # Check which feature groups are needed
    need_actions = bool(
        feature_keys
        & {
            "action.ee.pos_quat_g",
            "action.ee.pos_rot6d_g",
            "action.ee.pos_quat_g_rel",
            "action.ee.pos_rot6d_g_rel",
        }
    )
    need_phase_desc = "observation.phase_description" in feature_keys
    need_reward = "next.reward" in feature_keys and reward_type == "staged"

    # Initial SE(3) inverse for computing relative actions
    initial_se3_inv = None
    if need_actions:
        initial_se3_inv = np.linalg.inv(gym_env.initial_ee_se3)

    task_str = make_task_string(obj_name, bin_name)
    frames = []

    while not fsm.is_done:
        # Plan at gym-step level (ACTION_REPEAT physics steps per gym step)
        fsm.plan(n_steps=ACTION_REPEAT)

        # Build abs_pos action from FSM target
        target_pos = (
            fsm.target_pos if fsm.target_pos is not None else gym_env.robot.ee_pos
        )
        action = np.array([*target_pos, fsm.gripper_val], dtype=np.float32)

        # Step the gym env
        obs, reward, terminated, truncated, info = gym_env.step(action)

        # Build frame from gym obs
        frame: dict = {"task": task_str}

        for obs_key, feat_key in _OBS_TO_FEATURE.items():
            if feat_key in feature_keys and obs_key in obs:
                frame[feat_key] = obs[obs_key]

        # Keypoints need flattening (gym returns (N, 2), dataset expects (14,))
        if (
            "observation.keypoints_overhead" in feature_keys
            and "keypoints_overhead" in obs
        ):
            frame["observation.keypoints_overhead"] = obs[
                "keypoints_overhead"
            ].flatten()
        if "observation.keypoints_wrist" in feature_keys and "keypoints_wrist" in obs:
            frame["observation.keypoints_wrist"] = obs["keypoints_wrist"].flatten()

        # Action variants computed from FSM target
        if need_actions:
            (
                action_pos_quat_g,
                action_pos_rot6d_g,
                action_pos_quat_g_rel,
                action_pos_rot6d_g_rel,
            ) = get_actions(target_pos, fsm.gripper_val, initial_se3_inv)
            if "action.ee.pos_quat_g" in feature_keys:
                frame["action.ee.pos_quat_g"] = action_pos_quat_g
            if "action.ee.pos_rot6d_g" in feature_keys:
                frame["action.ee.pos_rot6d_g"] = action_pos_rot6d_g
            if "action.ee.pos_quat_g_rel" in feature_keys:
                frame["action.ee.pos_quat_g_rel"] = action_pos_quat_g_rel
            if "action.ee.pos_rot6d_g_rel" in feature_keys:
                frame["action.ee.pos_rot6d_g_rel"] = action_pos_rot6d_g_rel

        # Phase description
        if need_phase_desc:
            frame["observation.phase_description"] = fsm.phase_description

        # Staged reward from info
        if need_reward and "reward_components" in info:
            frame["next.reward"] = info["reward_components"]

        frames.append(frame)

    return frames


@hydra.main(config_path="../configs", config_name="generate", version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate a LeRobot dataset from expert FSM episodes."""
    if not cfg.repo_id:
        raise ValueError("repo_id is required (e.g. repo_id=user/pick-place)")

    if cfg.task is not None:
        task_pair = tuple(cfg.task)
        if len(task_pair) != 2:
            raise ValueError(f"task must be [obj, bin], got {task_pair}")
        task_list = [task_pair]
    elif cfg.tasks in TASK_SETS:
        task_list = TASK_SETS[cfg.tasks]
    else:
        raise ValueError(
            f"Unknown task set '{cfg.tasks}'. Choose from: {list(TASK_SETS.keys())}"
        )

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
    # next.reward only supported for staged reward
    if cfg.reward_type != "staged":
        features.pop("next.reward", None)
    feature_keys = set(features)

    # Import LeRobot (handle both old and new import paths)
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Create gym environment
    print("Loading scene...")
    gym_env = PickPlaceGymEnv(
        action_mode="abs_pos",
        reward_type=cfg.reward_type,
        randomize_objects=cfg.randomize_objects,
        spawn_x_range=tuple(cfg.spawn_x_range),
        spawn_y_range=tuple(cfg.spawn_y_range),
    )

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
    if cfg.randomize_objects:
        ss = np.random.SeedSequence(cfg.seed)
        child_seeds = ss.spawn(cfg.num_episodes)
        episode_seeds = [int(cs.generate_state(1)[0]) for cs in child_seeds]

    # Generate episodes, cycling through tasks
    task_label = str(list(cfg.task)) if cfg.task is not None else cfg.tasks
    print(f"Tasks {task_label}: {len(task_list)} task(s)")
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

        ep_seed = episode_seeds[ep_idx] if episode_seeds is not None else None

        frames = run_episode(
            gym_env,
            obj_name,
            bin_name,
            feature_keys,
            reward_type=cfg.reward_type,
            episode_seed=ep_seed,
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
    gym_env.close()
    print(f"\nDataset saved to {dataset_path}")
    print(f"Total episodes: {cfg.num_episodes}")

    # Push to HF Hub (creates dataset card + v3.0 tag automatically)
    if cfg.push_to_hub:
        print(f"\nPushing to HF Hub: {cfg.repo_id} (private={cfg.private})...")
        dataset.push_to_hub(private=cfg.private, upload_large_folder=True)
        print("Push complete.")


if __name__ == "__main__":
    main()
