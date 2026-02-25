"""Camera rendering and 3D-to-2D keypoint projection."""

import mujoco
import numpy as np

from .constants import IMAGE_SIZE, KEYPOINT_BODIES


class CameraRenderer:
    """Wraps mujoco.Renderer for offscreen image rendering."""

    def __init__(
        self, model: mujoco.MjModel, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE
    ) -> None:
        """Initialise the renderer.

        Args:
            model: MuJoCo model.
            height: Image height in pixels.
            width: Image width in pixels.
        """
        self._renderer: mujoco.Renderer = mujoco.Renderer(model, height, width)

    def render(self, data: mujoco.MjData, camera_name: str) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB image from the named camera.

        Args:
            data: MuJoCo data (current simulation state).
            camera_name: Name of the camera defined in the MJCF.

        Returns:
            RGB image array.
        """
        self._renderer.update_scene(data, camera=camera_name)
        return self._renderer.render()

    def render_all(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Return overhead and wrist camera images as a dict.

        Args:
            data: MuJoCo data (current simulation state).

        Returns:
            Dict with keys ``"overhead"`` and ``"wrist"``.
        """
        return {
            "overhead": self.render(data, "overhead"),
            "wrist": self.render(data, "wrist"),
        }

    def close(self) -> None:
        """Release renderer resources."""
        self._renderer.close()


def project_3d_to_2d(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    points_3d: np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Project (N, 3) world points to (N, 2) normalised [0, 1] pixel coordinates.

    Uses standard pinhole projection with MuJoCo camera intrinsics/extrinsics.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        camera_name: Name of the camera defined in the MJCF.
        points_3d: World-frame points, shape (N, 3).
        image_size: Image resolution (square).

    Returns:
        Normalised pixel coordinates, shape (N, 2), dtype float32.

    Raises:
        ValueError: If the camera is not found.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found")

    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    fovy = model.cam_fovy[cam_id]
    f = (image_size / 2.0) / np.tan(np.radians(fovy) / 2.0)

    # MuJoCo camera axes: x=right, y=up, z=backward (OpenGL convention)
    points = np.atleast_2d(points_3d)
    rel = points - cam_pos
    cam_coords = rel @ cam_mat

    # cam_coords[:,2] is backward; points in front have positive depth
    depth = cam_coords[:, 2]
    depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)

    px = f * cam_coords[:, 0] / depth + image_size / 2.0
    py = -f * cam_coords[:, 1] / depth + image_size / 2.0  # flip y (up -> down)

    px_norm = px / image_size
    py_norm = py / image_size

    return np.stack([px_norm, py_norm], axis=-1).astype(np.float32)


def compute_keypoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Project all KEYPOINT_BODIES to normalised pixel coordinates.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        camera_name: Name of the camera defined in the MJCF.
        image_size: Image resolution (square).

    Returns:
        Normalised pixel coordinates, shape (7, 2), dtype float32.
    """
    points_3d = np.array(
        [
            data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
            for name in KEYPOINT_BODIES
        ]
    )
    return project_3d_to_2d(model, data, camera_name, points_3d, image_size)
