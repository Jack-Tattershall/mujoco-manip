"""SE(3) pose utilities for relative-to-initial action representations.

Conventions:
- Quaternions are (qx, qy, qz, qw) — matching ROS / scipy convention.
- 6D rotation uses the first two rows of the rotation matrix (Zhou et al. 2019).
- 8DOF = [x, y, z, qx, qy, qz, qw, gripper]
- 10DOF = [x, y, z, r11, r12, r13, r21, r22, r23, gripper]
  where r_ij uses 1-indexed row/col of the rotation matrix.
"""

import numpy as np


def pos_rotmat_to_se3(pos: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from position (3,) and rotation (3,3)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotmat
    T[:3, 3] = pos
    return T


def se3_to_pos_rotmat(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract position (3,) and rotation (3,3) from a 4x4 SE(3) matrix."""
    return T[:3, 3].copy(), T[:3, :3].copy()


# ---------------------------------------------------------------------------
# Quaternion <-> rotation matrix
# ---------------------------------------------------------------------------


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (qx, qy, qz, qw)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (qx, qy, qz, qw) to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# ---------------------------------------------------------------------------
# 6D rotation representation (Zhou et al. CVPR 2019)
# ---------------------------------------------------------------------------


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """Extract 6D rotation: first two rows of the rotation matrix, flattened."""
    return R[:2, :].flatten().astype(np.float32)


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-12)


def rotmat_from_6d(d6: np.ndarray) -> np.ndarray:
    """Recover full 3x3 rotation from 6D representation via Gram-Schmidt."""
    a1, a2 = d6[:3], d6[3:6]
    b1 = _normalise(a1)
    b2 = _normalise(a2 - np.dot(b1, a2) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


# ---------------------------------------------------------------------------
# 8DOF / 10DOF ↔ SE(3) conversion
# ---------------------------------------------------------------------------


def se3_to_8dof(T: np.ndarray, gripper: float) -> np.ndarray:
    """SE(3) + gripper → 8DOF [x,y,z, qx,qy,qz,qw, gripper]."""
    pos, R = se3_to_pos_rotmat(T)
    q = rotmat_to_quat_xyzw(R)
    return np.array([*pos, *q, gripper], dtype=np.float32)


def se3_to_10dof(T: np.ndarray, gripper: float) -> np.ndarray:
    """SE(3) + gripper → 10DOF [x,y,z, r11..r23, gripper]."""
    pos, R = se3_to_pos_rotmat(T)
    rot6 = rotmat_to_6d(R)
    return np.array([*pos, *rot6, gripper], dtype=np.float32)


def se3_from_8dof(dof8: np.ndarray) -> np.ndarray:
    """Parse 8DOF [x,y,z, qx,qy,qz,qw, gripper] → 4x4 SE(3). Gripper ignored."""
    pos = dof8[:3]
    R = quat_xyzw_to_rotmat(dof8[3:7])
    return pos_rotmat_to_se3(pos, R)


def se3_from_10dof(dof10: np.ndarray) -> np.ndarray:
    """Parse 10DOF [x,y,z, r11..r23, gripper] → 4x4 SE(3). Gripper ignored."""
    pos = dof10[:3]
    R = rotmat_from_6d(dof10[3:9])
    return pos_rotmat_to_se3(pos, R)
