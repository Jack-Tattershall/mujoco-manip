"""Object position randomization for diverse demonstrations."""

import mujoco
import numpy as np

OBJ_JOINT_NAMES = ("obj_red_jnt", "obj_green_jnt", "obj_blue_jnt")
MIN_OBJ_SEPARATION = 0.08
_MAX_REJECTION_ATTEMPTS = 1000


def randomize_object_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    x_range: tuple[float, float] = (-0.20, 0.20),
    y_range: tuple[float, float] = (0.30, 0.45),
    obj_z: float = 0.26,
    min_separation: float = MIN_OBJ_SEPARATION,
    randomize_yaw: bool = False,
) -> dict[str, np.ndarray]:
    """Randomize XY positions (and optionally yaw) of all free-joint objects.

    Uses rejection sampling to ensure pairwise minimum separation between
    all objects. Does NOT call ``mj_forward`` — the caller is responsible.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (qpos is modified in-place).
        rng: NumPy random generator for reproducibility.
        x_range: (min_x, max_x) spawn bounds.
        y_range: (min_y, max_y) spawn bounds.
        obj_z: Fixed Z height for all objects.
        min_separation: Minimum pairwise XY distance between objects.
        randomize_yaw: If True, randomize Z-axis rotation.

    Returns:
        Dict mapping joint name to the new (x, y, z) position.
    """
    # Resolve qpos addresses for each free joint
    addrs = []
    for name in OBJ_JOINT_NAMES:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            raise ValueError(f"Joint '{name}' not found in model")
        addrs.append(model.jnt_qposadr[jnt_id])

    # Rejection-sample positions with pairwise separation check
    positions = _sample_separated_positions(
        rng, len(addrs), x_range, y_range, min_separation
    )

    result = {}
    for addr, (x, y), name in zip(addrs, positions, OBJ_JOINT_NAMES):
        data.qpos[addr : addr + 3] = [x, y, obj_z]
        if randomize_yaw:
            theta = rng.uniform(0, 2 * np.pi)
            data.qpos[addr + 3 : addr + 7] = [
                np.cos(theta / 2),
                0.0,
                0.0,
                np.sin(theta / 2),
            ]
        else:
            data.qpos[addr + 3 : addr + 7] = [1.0, 0.0, 0.0, 0.0]
        result[name] = np.array([x, y, obj_z])

    return result


def _sample_separated_positions(
    rng: np.random.Generator,
    n: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    min_separation: float,
) -> list[tuple[float, float]]:
    """Sample *n* XY positions with pairwise distance >= *min_separation*."""
    for _ in range(_MAX_REJECTION_ATTEMPTS):
        xs = rng.uniform(x_range[0], x_range[1], size=n)
        ys = rng.uniform(y_range[0], y_range[1], size=n)
        positions = list(zip(xs.tolist(), ys.tolist()))
        if _all_separated(positions, min_separation):
            return positions
    raise RuntimeError(
        f"Failed to sample {n} positions with min_separation={min_separation} "
        f"in {_MAX_REJECTION_ATTEMPTS} attempts"
    )


def _all_separated(positions: list[tuple[float, float]], min_sep: float) -> bool:
    """Return True if all pairwise XY distances >= min_sep."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if dx * dx + dy * dy < min_sep * min_sep:
                return False
    return True
