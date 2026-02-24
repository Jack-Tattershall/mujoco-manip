"""Shared constants for the pick-and-place simulation."""

OBJECTS = ["obj_red", "obj_green", "obj_blue"]
BINS = ["bin_red", "bin_green", "bin_blue"]

# Task sets: each task is an (object_body, bin_body) tuple
MATCH_TASKS = [("obj_red", "bin_red"), ("obj_green", "bin_green"), ("obj_blue", "bin_blue")]
CROSS_TASKS = [
    (o, b) for o in OBJECTS for b in BINS if o.split("_")[1] != b.split("_")[1]
]
ALL_TASKS = [(o, b) for o in OBJECTS for b in BINS]

TASK_SETS = {
    "all": ALL_TASKS,
    "match": MATCH_TASKS,
    "cross": CROSS_TASKS,
}

# Backward compat alias
SINGLE_TASKS = MATCH_TASKS

IMAGE_SIZE = 224
CONTROL_FPS = 30
PHYSICS_DT = 0.002
ACTION_REPEAT = 16  # int(1/30 / 0.002) ≈ 16 → ~31 Hz control
MAX_EPISODE_STEPS = 500

KEYPOINT_BODIES = ["obj_red", "obj_green", "obj_blue", "bin_red", "bin_green", "bin_blue", "hand"]
