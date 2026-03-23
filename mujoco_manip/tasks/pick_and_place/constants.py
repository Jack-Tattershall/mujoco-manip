"""Shared constants for the pick-and-place simulation."""

import numpy as np

OBJECTS = ["obj_red", "obj_green", "obj_blue"]
BINS = ["bin_red", "bin_green", "bin_blue"]

# Task sets: each task is an (object_body, bin_body) tuple
MATCH_TASKS = [
    ("obj_red", "bin_red"),
    ("obj_green", "bin_green"),
    ("obj_blue", "bin_blue"),
]
CROSS_TASKS = [
    (o, b) for o in OBJECTS for b in BINS if o.split("_")[1] != b.split("_")[1]
]
ALL_TASKS = [(o, b) for o in OBJECTS for b in BINS]

TASK_SETS = {
    "all": ALL_TASKS,
    "match": MATCH_TASKS,
    "cross": CROSS_TASKS,
}

IMAGE_SIZE = 224
CONTROL_FPS = 30
PHYSICS_DT = 0.002
ACTION_REPEAT = 16  # int(1/30 / 0.002) ≈ 16 → ~31 Hz control
MAX_EPISODE_STEPS = 500

# Workspace bounds for EE target clamping (prevents VLA from commanding
# extreme / unreachable poses that can cause simulation divergence).
WORKSPACE_MIN = np.array([-0.35, 0.10, 0.24])
WORKSPACE_MAX = np.array([0.35, 0.65, 0.60])

KEYPOINT_BODIES = [
    "obj_red",
    "obj_green",
    "obj_blue",
    "bin_red",
    "bin_green",
    "bin_blue",
    "hand",
]
