"""Visualize all features of a LeRobot dataset episode in Rerun."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from mujoco_manip.features import DIM_NAMES  # noqa: E402
from mujoco_manip.pose_utils import (  # noqa: E402
    pos_rotmat_to_se3,
    se3_from_8dof,
    se3_from_10dof,
)

SCENE_XML = os.path.join(_PROJECT_ROOT, "pick_and_place_scene.xml")

SKIP_KEYS: set[str] = {
    "frame_index",
    "episode_index",
    "index",
    "timestamp",
    "task_index",
    "task",
    "observation.state",
}

KEYPOINT_OVERLAYS: dict[str, tuple[str, list[tuple[str, tuple[int, int, int]]]]] = {
    "observation.keypoints_overhead": (
        "observation.images.overhead",
        [
            ("red", (255, 60, 60)),
            ("green", (60, 255, 60)),
            ("blue", (60, 60, 255)),
            ("bin_red", (180, 0, 0)),
            ("bin_green", (0, 180, 0)),
            ("bin_blue", (0, 0, 180)),
            ("hand", (255, 255, 255)),
        ],
    ),
    "observation.keypoints_wrist": (
        "observation.images.wrist",
        [
            ("red", (255, 60, 60)),
            ("green", (60, 255, 60)),
            ("blue", (60, 60, 255)),
            ("bin_red", (180, 0, 0)),
            ("bin_green", (0, 180, 0)),
            ("bin_blue", (0, 0, 180)),
            ("hand", (255, 255, 255)),
        ],
    ),
    "observation.target_obj_keypoints_overhead": (
        "observation.images.overhead",
        [
            ("obj", (255, 255, 0)),
        ],
    ),
    "observation.target_bin_keypoints_overhead": (
        "observation.images.overhead",
        [
            ("bin", (0, 255, 255)),
        ],
    ),
}

SPECIAL_KEYS: set[str] = {
    "observation.target_bin_onehot",
    "observation.target_obj_onehot",
}

MARKER_RADIUS: int = 4


def _draw_marker(img: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Draw a filled circle marker on an image array (HWC, uint8)."""
    h, w = img.shape[:2]
    r = MARKER_RADIUS
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                py, px = y + dy, x + dx
                if 0 <= py < h and 0 <= px < w:
                    img[py, px] = color


def visualize_episode(
    dataset: LeRobotDataset, episode_index: int, save_path: str | None = None
) -> None:
    """Log all frames of an episode to Rerun.

    Args:
        dataset: Loaded LeRobot dataset.
        episode_index: Episode to visualise.
        save_path: If set, save the recording to this ``.rrd`` file path.
    """
    rr.init(f"{dataset.repo_id}/episode_{episode_index}", spawn=(save_path is None))

    sample = dataset[0]
    feature_keys = sorted(k for k in sample if k not in SKIP_KEYS)
    print(f"Dataset features: {feature_keys}")

    state_abs_trail: list[list[float]] = []
    state_rel_trail: list[list[float]] = []
    action_abs_trail: list[list[float]] = []
    action_rel_trail: list[list[float]] = []
    T_initial: np.ndarray | None = None

    first_frame = dataset[0]
    has_state_8dof = "observation.state.ee.8dof" in first_frame
    has_state_rel_8dof = "observation.state.ee.8dof_rel" in first_frame
    has_state_rel_10dof = "observation.state.ee.10dof_rel" in first_frame
    has_action_8dof = "action.ee.8dof" in first_frame
    has_action_rel_8dof = "action.ee.8dof_rel" in first_frame
    has_action_rel_10dof = "action.ee.10dof_rel" in first_frame

    has_any_rel: bool = (
        has_state_rel_8dof
        or has_state_rel_10dof
        or has_action_rel_8dof
        or has_action_rel_10dof
    )
    if has_any_rel:
        from mujoco_manip.env import PickPlaceEnv  # noqa: E402
        from mujoco_manip.robot import PandaRobot  # noqa: E402

        _env = PickPlaceEnv(SCENE_XML, add_wrist_camera=False)
        _robot = PandaRobot(_env.model, _env.data)
        _env.reset_to_keyframe("scene_start")
        T_initial = pos_rotmat_to_se3(_robot.ee_pos, _robot.ee_xmat)
        del _env, _robot

    for i in tqdm.tqdm(range(len(dataset)), desc="Logging frames"):
        frame = dataset[i]
        rr.set_time("frame_index", sequence=frame["frame_index"].item())

        images: dict[str, np.ndarray] = {}
        for key in frame:
            if key in SKIP_KEYS or key in SPECIAL_KEYS:
                continue
            val = frame[key]
            if isinstance(val, torch.Tensor):
                val = val.numpy()
            if not isinstance(val, np.ndarray) or val.ndim != 3:
                continue
            if val.shape[0] in (1, 3, 4) and val.shape[0] < val.shape[1]:
                val = np.transpose(val, (1, 2, 0))
            if val.dtype == np.float32 or val.dtype == np.float64:
                val = (val * 255).clip(0, 255).astype(np.uint8)
            images[key] = val.copy()

        for kp_key, (img_key, point_info) in KEYPOINT_OVERLAYS.items():
            if kp_key not in frame or img_key not in images:
                continue
            kp_val = frame[kp_key]
            if isinstance(kp_val, torch.Tensor):
                kp_val = kp_val.numpy()
            img = images[img_key]
            h, w = img.shape[:2]
            pts = kp_val.reshape(-1, 2)
            for pt_idx, (_, color) in enumerate(point_info):
                if pt_idx >= len(pts):
                    break
                px = int(round((1.0 - pts[pt_idx, 0]) * w))
                py = int(round((1.0 - pts[pt_idx, 1]) * h))
                _draw_marker(img, px, py, color)

        for key, img in images.items():
            rr.log(key, rr.Image(img))

        for key in frame:
            if key in SKIP_KEYS or key in SPECIAL_KEYS:
                continue
            val = frame[key]
            if isinstance(val, torch.Tensor):
                val = val.numpy()
            if not isinstance(val, np.ndarray) or val.ndim != 1:
                continue
            names = DIM_NAMES.get(key)
            for dim_idx, v in enumerate(val):
                name = (
                    names[dim_idx] if names and dim_idx < len(names) else str(dim_idx)
                )
                rr.log(f"{key}/{name}", rr.Scalars(float(v)))

        if has_state_8dof:
            ee_xyz = frame["observation.state.ee.8dof"].numpy()[:3].tolist()
            state_abs_trail.append(ee_xyz)
            rr.log("state_3d/abs", rr.Points3D([ee_xyz], colors=[[0, 200, 0]]))
            rr.log(
                "state_3d/abs_trail",
                rr.LineStrips3D([state_abs_trail], colors=[[0, 200, 0]]),
            )

        if T_initial is not None and (has_state_rel_8dof or has_state_rel_10dof):
            if has_state_rel_8dof:
                rel = frame["observation.state.ee.8dof_rel"].numpy()
                T_abs = T_initial @ se3_from_8dof(rel)
            else:
                rel = frame["observation.state.ee.10dof_rel"].numpy()
                T_abs = T_initial @ se3_from_10dof(rel)
            rel_xyz = T_abs[:3, 3].tolist()
            state_rel_trail.append(rel_xyz)
            rr.log("state_3d/rel", rr.Points3D([rel_xyz], colors=[[0, 100, 255]]))
            rr.log(
                "state_3d/rel_trail",
                rr.LineStrips3D([state_rel_trail], colors=[[0, 100, 255]]),
            )

        if has_action_8dof:
            act_xyz = frame["action.ee.8dof"].numpy()[:3].tolist()
            action_abs_trail.append(act_xyz)
            rr.log("action_3d/abs", rr.Points3D([act_xyz], colors=[[0, 200, 0]]))
            rr.log(
                "action_3d/abs_trail",
                rr.LineStrips3D([action_abs_trail], colors=[[0, 200, 0]]),
            )

        if T_initial is not None and (has_action_rel_8dof or has_action_rel_10dof):
            if has_action_rel_8dof:
                rel = frame["action.ee.8dof_rel"].numpy()
                T_abs = T_initial @ se3_from_8dof(rel)
            else:
                rel = frame["action.ee.10dof_rel"].numpy()
                T_abs = T_initial @ se3_from_10dof(rel)
            rel_xyz = T_abs[:3, 3].tolist()
            action_rel_trail.append(rel_xyz)
            rr.log("action_3d/rel", rr.Points3D([rel_xyz], colors=[[0, 100, 255]]))
            rr.log(
                "action_3d/rel_trail",
                rr.LineStrips3D([action_rel_trail], colors=[[0, 100, 255]]),
            )

        if "observation.target_bin_onehot" in frame:
            val = frame["observation.target_bin_onehot"]
            if isinstance(val, torch.Tensor):
                val = val.numpy()
            rr.log("observation/target_bin_onehot", rr.BarChart(val))

        if "observation.target_obj_onehot" in frame:
            val = frame["observation.target_obj_onehot"]
            if isinstance(val, torch.Tensor):
                val = val.numpy()
            rr.log("observation/target_obj_onehot", rr.BarChart(val))

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save_path))
        print(f"Saved to {save_path}")


def main() -> None:
    """Parse CLI args and launch Rerun visualisation for a dataset episode."""
    parser = argparse.ArgumentParser(
        description="Visualize a LeRobot dataset episode in Rerun"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repo ID (e.g. test/pick-place)",
    )
    parser.add_argument(
        "--root", type=str, default=None, help="Local dataset root directory"
    )
    parser.add_argument(
        "--episode-index", type=int, default=0, help="Episode index to visualize"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save .rrd file (e.g. ./viz/ep0.rrd)",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.repo_id,
        episodes=[args.episode_index],
        root=args.root,
    )
    visualize_episode(dataset, args.episode_index, save_path=args.save)


if __name__ == "__main__":
    main()
