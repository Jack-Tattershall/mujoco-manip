"""Camera rendering and 3D-to-2D keypoint projection."""

import mujoco
import numpy as np

from .constants import IMAGE_SIZE, KEYPOINT_BODIES


class CameraRenderer:
    """Wraps mujoco.Renderer for offscreen image rendering."""

    def __init__(
        self, model: mujoco.MjModel, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE
    ):
        self._renderer = mujoco.Renderer(model, height, width)

    def render(self, data: mujoco.MjData, camera_name: str) -> np.ndarray:
        """Render an RGB image from the named camera. Returns (H, W, 3) uint8."""
        self._renderer.update_scene(data, camera=camera_name)
        return self._renderer.render()

    def render_all(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Render from both overhead and wrist cameras."""
        return {
            "overhead": self.render(data, "overhead"),
            "wrist": self.render(data, "wrist"),
        }

    def close(self):
        self._renderer.close()


def project_3d_to_2d(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    points_3d: np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Project N x 3 world points to N x 2 normalized [0,1] pixel coordinates.

    Uses standard pinhole projection with MuJoCo camera intrinsics/extrinsics.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found")

    # Camera extrinsics
    cam_pos = data.cam_xpos[cam_id]  # (3,)
    cam_mat = data.cam_xmat[cam_id].reshape(
        3, 3
    )  # (3,3) columns are camera axes in world frame

    # Camera intrinsics from fovy
    fovy = model.cam_fovy[cam_id]
    f = (image_size / 2.0) / np.tan(np.radians(fovy) / 2.0)

    # Transform to camera frame
    # MuJoCo camera: x=right, y=up, z=backward (OpenGL convention)
    points = np.atleast_2d(points_3d)  # (N, 3)
    rel = points - cam_pos  # (N, 3) vectors from camera to points in world frame
    cam_coords = rel @ cam_mat  # (N, 3) in camera frame [right, up, backward]

    # Perspective divide (z is backward, so depth = -cam_coords[:, 2] would be wrong;
    # in MuJoCo's OpenGL convention, the camera looks along -z, so depth = cam_coords[:, 2])
    # Actually: cam_mat columns are [right, up, -forward], so cam_coords[:,2] is the
    # backward direction. Points in front of the camera have positive cam_coords[:,2].
    depth = cam_coords[:, 2]
    depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)

    # Pixel coordinates (origin at top-left)
    px = f * cam_coords[:, 0] / depth + image_size / 2.0  # x: right
    py = -f * cam_coords[:, 1] / depth + image_size / 2.0  # y: down (flip up)

    # Normalize to [0, 1]
    px_norm = px / image_size
    py_norm = py / image_size

    return np.stack([px_norm, py_norm], axis=-1).astype(np.float32)


def compute_keypoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Project all KEYPOINT_BODIES to (7, 2) float32 normalized pixel coords."""
    points_3d = np.array(
        [
            data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
            for name in KEYPOINT_BODIES
        ]
    )
    return project_3d_to_2d(model, data, camera_name, points_3d, image_size)
