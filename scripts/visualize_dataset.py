"""Visualize all features of a LeRobot dataset episode in Rerun.

Logs images, scalar time series (with named dimensions), and all custom
keys like ee.8dof, ee.10dof, keypoints, and target_bin_onehot.

Examples:

    uv run python scripts/visualize_dataset.py \
        --repo-id test/pick-place --root ./datasets --episode-index 0

    uv run python scripts/visualize_dataset.py \
        --repo-id test/pick-place --root ./datasets --episode-index 0 \
        --save ./viz/ep0.rrd
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import tqdm

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from src.features import DIM_NAMES  # noqa: E402

SKIP_KEYS = {"frame_index", "episode_index", "index", "timestamp", "task_index", "task"}


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

    for i in tqdm.tqdm(range(len(dataset)), desc="Logging frames"):
        frame = dataset[i]
        rr.set_time("frame_index", sequence=frame["frame_index"].item())

        for key in frame:
            if key in SKIP_KEYS:
                continue

            val = frame[key]
            if isinstance(val, torch.Tensor):
                val = val.numpy()

            if not isinstance(val, np.ndarray):
                continue

            if val.ndim == 3:
                if val.shape[0] in (1, 3, 4) and val.shape[0] < val.shape[1]:
                    val = np.transpose(val, (1, 2, 0))  # CHW -> HWC
                if val.dtype == np.float32 or val.dtype == np.float64:
                    val = (val * 255).clip(0, 255).astype(np.uint8)
                rr.log(key, rr.Image(val))

            elif val.ndim == 1:
                names = DIM_NAMES.get(key)
                for dim_idx, v in enumerate(val):
                    name = (
                        names[dim_idx]
                        if names and dim_idx < len(names)
                        else str(dim_idx)
                    )
                    rr.log(f"{key}/{name}", rr.Scalars(float(v)))

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
