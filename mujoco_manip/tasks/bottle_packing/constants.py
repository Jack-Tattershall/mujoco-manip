"""Shared constants for the bottle packing simulation."""

import numpy as np

# Well grid: 5 columns (X) x 4 rows (Y) = 20 wells
WELL_ROWS = 4
WELL_COLS = 5
NUM_WELLS = WELL_ROWS * WELL_COLS

# Well spacing (center-to-centre) in metres
WELL_SPACING = 0.055

# Crate body position in the scene (world frame)
CRATE_POS = np.array([0.25, 0.40, 0.24])

# Bottle pickup position (conveyor end, world frame)
BOTTLE_PICKUP_POS = np.array([-0.05, 0.40, 0.29])

# Bottle start position — far back of the belt (belt runs from x=-0.76 to x=0.06).
BOTTLE_CONVEYOR_START = np.array([-0.72, 0.40, 0.29])

# Number of animation frames for conveyor roll (blocking animation, used by gym env)
CONVEYOR_ANIM_STEPS = 300

# Conveyor belt speed (m per physics step) — 0.05 m/s at dt=0.002
CONVEYOR_SPEED = 0.0001

# Spacing between bottle centres on conveyor belt (metres)
CONVEYOR_BOTTLE_SPACING = 0.12

# Maximum bottles visible on belt
MAX_BELT_BOTTLES = 2

# Random Y offset range for bottles on belt (±metres from belt centre)
BELT_Y_NOISE = 0.04

# 10 distinct bottle colours, cycled for 20 bottles
BOTTLE_COLORS = np.array(
    [
        [0.85, 0.15, 0.15, 1.0],  # red
        [0.15, 0.55, 0.85, 1.0],  # blue
        [0.95, 0.75, 0.10, 1.0],  # yellow
        [0.15, 0.75, 0.30, 1.0],  # green
        [0.75, 0.25, 0.75, 1.0],  # purple
        [0.95, 0.50, 0.10, 1.0],  # orange
        [0.10, 0.75, 0.75, 1.0],  # teal
        [0.85, 0.35, 0.55, 1.0],  # pink
        [0.50, 0.70, 0.15, 1.0],  # lime
        [0.60, 0.40, 0.20, 1.0],  # brown
    ],
    dtype=np.float32,
)

# Body names
BOTTLE_BODIES = [f"bottle_{i:02d}" for i in range(NUM_WELLS)]
CRATE_BODY = "crate"
CRATE_JOINT_NAMES = ["crate_slide_x", "crate_slide_y", "crate_hinge_z"]

# Hidden position (underground, out of sight)
BOTTLE_HIDDEN_POS = np.array([0.0, 0.0, -1.0])

# Bottle half-height (for computing center height in wells)
BOTTLE_HALF_HEIGHT = 0.05

# Height of wall tops above crate body origin (metres).
# Walls are 104mm tall above the floor surface (z=0.002 local).
WELL_WALL_HEIGHT = 0.106

# All well indices (0–19)
ALL_WELLS = list(range(NUM_WELLS))

TASK_SETS = {
    "all": ALL_WELLS,
}

# Height thresholds for conveyor/grasp logic
CONVEYOR_RESUME_Z = 0.34  # resume belt when bottle clears this height
GRASP_VERIFY_Z = 0.35  # bottle must exceed this to confirm grasp

IMAGE_SIZE = 224
CONTROL_FPS = 30
PHYSICS_DT = 0.002
ACTION_REPEAT = 16  # int(1/30 / 0.002) ≈ 16 → ~31 Hz control
MAX_EPISODE_STEPS = 500


def well_row_col(well_index: int) -> tuple[int, int]:
    """Convert linear well index to (row, col).

    Args:
        well_index: Linear index 0–19.

    Returns:
        Tuple of (row, col) with row in [0, 3] and col in [0, 4].
    """
    return divmod(well_index, WELL_COLS)


def _compute_well_positions() -> np.ndarray:
    """Pre-compute 3D world positions for all wells. Shape (NUM_WELLS, 3)."""
    positions = np.empty((NUM_WELLS, 3))
    for idx in range(NUM_WELLS):
        row, col = well_row_col(idx)
        positions[idx, 0] = CRATE_POS[0] + (col - (WELL_COLS - 1) / 2) * WELL_SPACING
        positions[idx, 1] = CRATE_POS[1] + (row - (WELL_ROWS - 1) / 2) * WELL_SPACING
        positions[idx, 2] = CRATE_POS[2]
    return positions


WELL_POSITIONS: np.ndarray = _compute_well_positions()


def well_position(well_index: int) -> np.ndarray:
    """Return the 3D world position of a well center.

    The position lies on the crate floor surface.  The bottle center
    should be placed at this position + (0, 0, bottle_half_height).

    Args:
        well_index: Linear index 0–19.

    Returns:
        World-frame position (3,).
    """
    return WELL_POSITIONS[well_index].copy()
