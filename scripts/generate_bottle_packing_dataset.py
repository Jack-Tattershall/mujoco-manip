"""Generate a LeRobot v3.0 dataset from expert bottle-packing FSM demonstrations."""

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mujoco_manip.tasks.bottle_packing.constants import (
    ACTION_REPEAT,
    CONTROL_FPS,
    TASK_SETS,
    well_row_col,
)
from mujoco_manip.tasks.bottle_packing.features import FEATURES
from mujoco_manip.tasks.bottle_packing.fsm import BottlePackingTask
from mujoco_manip.tasks.bottle_packing.gym_env import BottlePackingGymEnv
from mujoco_manip.controller import TARGET_ORI
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
    "target_well_onehot": "observation.target_well_onehot",
    "target_bottle_keypoints_overhead": "observation.target_bottle_keypoints_overhead",
    "target_well_keypoints_overhead": "observation.target_well_keypoints_overhead",
}


def make_task_string(well_index: int) -> str:
    """Create a human-readable task description."""
    row, col = well_row_col(well_index)
    return f"Pack bottle into well ({row},{col})"


def get_actions(
    target_pos: np.ndarray,
    gripper_val: float,
    initial_se3_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the FSM's commanded SE(3) in absolute and relative frames."""
    T_target = pos_rotmat_to_se3(target_pos, TARGET_ORI)
    T_rel = initial_se3_inv @ T_target

    return (
        se3_to_pos_quat_g(T_target, gripper_val),
        se3_to_pos_rot6d_g(T_target, gripper_val),
        se3_to_pos_quat_g(T_rel, gripper_val),
        se3_to_pos_rot6d_g(T_rel, gripper_val),
    )


def run_episode(
    gym_env: BottlePackingGymEnv,
    well_index: int,
    feature_keys: set[str],
    reward_type: str = "staged",
) -> list[dict]:
    """Run one expert FSM episode and collect frames."""
    obs, info = gym_env.reset(options={"well_index": well_index})

    fsm = BottlePackingTask(
        gym_env.bottle_packing_env,
        gym_env.robot,
        gym_env.controller,
        well_index=well_index,
    )

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

    initial_se3_inv = None
    if need_actions:
        initial_se3_inv = np.linalg.inv(gym_env.initial_ee_se3)

    task_str = make_task_string(well_index)
    frames = []

    while not fsm.is_done:
        fsm.plan(n_steps=ACTION_REPEAT)

        target_pos = (
            fsm.target_pos if fsm.target_pos is not None else gym_env.robot.ee_pos
        )
        action = np.array([*target_pos, fsm.gripper_val], dtype=np.float32)

        frame: dict = {"task": task_str}

        for obs_key, feat_key in _OBS_TO_FEATURE.items():
            if feat_key in feature_keys and obs_key in obs:
                frame[feat_key] = obs[obs_key]

        if (
            "observation.keypoints_overhead" in feature_keys
            and "keypoints_overhead" in obs
        ):
            frame["observation.keypoints_overhead"] = obs[
                "keypoints_overhead"
            ].flatten()
        if "observation.keypoints_wrist" in feature_keys and "keypoints_wrist" in obs:
            frame["observation.keypoints_wrist"] = obs["keypoints_wrist"].flatten()

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

        if need_phase_desc:
            frame["observation.phase_description"] = fsm.phase_description

        obs, reward, terminated, truncated, info = gym_env.step(action)

        if need_reward and "reward_components" in info:
            frame["next.reward"] = info["reward_components"]

        frames.append(frame)

    return frames


@hydra.main(
    config_path="../configs", config_name="generate_bottle_packing", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Generate a LeRobot dataset from expert bottle-packing FSM episodes."""
    if not cfg.repo_id:
        raise ValueError("repo_id is required (e.g. repo_id=user/bottle-packing)")

    if cfg.well_index is not None:
        well_list = [int(cfg.well_index)]
    elif cfg.wells in TASK_SETS:
        well_list = TASK_SETS[cfg.wells]
    else:
        raise ValueError(
            f"Unknown well set '{cfg.wells}'. Choose from: {list(TASK_SETS.keys())}"
        )

    features = FEATURES
    if cfg.features is not None:
        requested = list(cfg.features)
        unknown = [k for k in requested if k not in FEATURES]
        if unknown:
            raise ValueError(
                f"Unknown feature keys: {unknown}. Valid keys: {list(FEATURES.keys())}"
            )
        features = {k: FEATURES[k] for k in requested}
    if cfg.reward_type != "staged":
        features.pop("next.reward", None)
    feature_keys = set(features)

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("Loading scene...")
    gym_env = BottlePackingGymEnv(
        action_mode="abs_pos",
        reward_type=cfg.reward_type,
    )

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

    well_label = str(cfg.well_index) if cfg.well_index is not None else cfg.wells
    print(f"Wells {well_label}: {len(well_list)} well(s)")
    for ep_idx in range(cfg.num_episodes):
        well_idx = well_list[ep_idx % len(well_list)]
        row, col = well_row_col(well_idx)

        print(
            f"Episode {ep_idx + 1}/{cfg.num_episodes}: well ({row},{col})",
            end="",
            flush=True,
        )

        frames = run_episode(
            gym_env,
            well_idx,
            feature_keys,
            reward_type=cfg.reward_type,
        )

        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()

        print(f" ({len(frames)} frames)")

    generation_config = OmegaConf.to_container(cfg, resolve=True)
    generation_config.pop("hydra", None)

    metadata_path = dataset_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(generation_config, f, indent=2)

    try:
        from lerobot.datasets.utils import write_info

        dataset.meta.info["generation_config"] = generation_config
        write_info(dataset.meta.info, dataset.meta.root)
    except (ImportError, AttributeError):
        pass

    dataset.finalize()
    gym_env.close()
    print(f"\nDataset saved to {dataset_path}")
    print(f"Total episodes: {cfg.num_episodes}")

    if cfg.push_to_hub:
        print(f"\nPushing to HF Hub: {cfg.repo_id} (private={cfg.private})...")
        dataset.push_to_hub(private=cfg.private, upload_large_folder=True)
        print("Push complete.")


if __name__ == "__main__":
    main()
