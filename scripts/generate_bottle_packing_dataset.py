"""Generate a LeRobot v3.0 dataset from expert bottle-packing FSM demonstrations.

Each *run* packs up to 20 bottles into wells (random or sequential order).
Each bottle pick-and-place is stored as one episode, with the scene state
reflecting previously packed bottles — matching ``main_bottle_packing.py``.
"""

import json
import random
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mujoco_manip.tasks.bottle_packing.constants import (
    ACTION_REPEAT,
    CONTROL_FPS,
    NUM_WELLS,
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


def build_well_schedule(
    task: str,
    num_bottles: int,
    rng: random.Random,
) -> list[int]:
    """Return the ordered list of target wells for one run.

    Args:
        task: ``"random"`` or ``"sequential"``.
        num_bottles: How many bottles to pack in this run.
        rng: Random number generator (only used for ``"random"``).

    Returns:
        List of well indices, length ``num_bottles``.
    """
    wells = list(range(NUM_WELLS))
    if task == "random":
        rng.shuffle(wells)
    return wells[:num_bottles]


def run_episode(
    gym_env: BottlePackingGymEnv,
    bottle_index: int,
    well_index: int,
    packed: dict[int, int],
    feature_keys: set[str],
    reward_type: str = "staged",
) -> list[dict]:
    """Run one expert FSM episode and collect frames.

    Args:
        gym_env: The gymnasium bottle-packing environment.
        bottle_index: Which bottle body to spawn on conveyor.
        well_index: Target well for this bottle.
        packed: Mapping of already-packed ``bottle_idx → well_idx``.
        feature_keys: Which dataset features to record.
        reward_type: Reward scheme.
    """
    obs, info = gym_env.reset(
        options={
            "well_index": well_index,
            "bottle_index": bottle_index,
            "packed": packed,
        }
    )

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

    task = cfg.get("task", "random")
    if task not in ("random", "sequential"):
        raise ValueError(f"task must be 'random' or 'sequential', got '{task}'")

    num_bottles = cfg.get("num_bottles") or NUM_WELLS
    num_bottles = min(int(num_bottles), NUM_WELLS)

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

    rng = random.Random(cfg.seed)
    total_episodes = 0

    print(f"Task: {task}, {num_bottles} bottles/run, {cfg.num_episodes} episodes")

    for ep_idx in range(cfg.num_episodes):
        # Start a new run when the previous one is exhausted
        if ep_idx % num_bottles == 0:
            well_schedule = build_well_schedule(task, num_bottles, rng)
            packed: dict[int, int] = {}
            run_num = ep_idx // num_bottles + 1
            print(f"\n--- Run {run_num} (well order: {well_schedule}) ---")

        step_in_run = ep_idx % num_bottles
        bottle_idx = step_in_run
        well_idx = well_schedule[step_in_run]
        row, col = well_row_col(well_idx)

        print(
            f"Episode {ep_idx + 1}/{cfg.num_episodes}: "
            f"bottle {bottle_idx} → well ({row},{col})",
            end="",
            flush=True,
        )

        frames = run_episode(
            gym_env,
            bottle_index=bottle_idx,
            well_index=well_idx,
            packed=packed,
            feature_keys=feature_keys,
            reward_type=cfg.reward_type,
        )

        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()

        # Record this bottle as packed for subsequent episodes
        packed[bottle_idx] = well_idx
        total_episodes += 1

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
    print(f"Total episodes: {total_episodes}")

    if cfg.push_to_hub:
        print(f"\nPushing to HF Hub: {cfg.repo_id} (private={cfg.private})...")
        dataset.push_to_hub(private=cfg.private, upload_large_folder=True)
        print("Push complete.")


if __name__ == "__main__":
    main()
