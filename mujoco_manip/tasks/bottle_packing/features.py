"""Shared dataset feature schema for bottle packing.

Used by ``generate_bottle_packing_dataset.py`` and ``visualize_dataset.py``.
"""

from .constants import IMAGE_SIZE, NUM_WELLS

FEATURES = {
    "observation.images.overhead": {
        "dtype": "image",
        "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist": {
        "dtype": "image",
        "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (11,),
        "names": None,
    },
    "observation.state.ee.pos_quat_g": {
        "dtype": "float32",
        "shape": (8,),
        "names": None,
    },
    "observation.state.ee.pos_rot6d_g": {
        "dtype": "float32",
        "shape": (10,),
        "names": None,
    },
    "observation.state.ee.pos_quat_g_rel": {
        "dtype": "float32",
        "shape": (8,),
        "names": None,
    },
    "observation.state.ee.pos_rot6d_g_rel": {
        "dtype": "float32",
        "shape": (10,),
        "names": None,
    },
    "observation.state.ee.force_torque": {
        "dtype": "float32",
        "shape": (6,),
        "names": None,
    },
    "action.ee.pos_quat_g": {
        "dtype": "float32",
        "shape": (8,),
        "names": None,
    },
    "action.ee.pos_rot6d_g": {
        "dtype": "float32",
        "shape": (10,),
        "names": None,
    },
    "action.ee.pos_quat_g_rel": {
        "dtype": "float32",
        "shape": (8,),
        "names": None,
    },
    "action.ee.pos_rot6d_g_rel": {
        "dtype": "float32",
        "shape": (10,),
        "names": None,
    },
    "observation.target_well_onehot": {
        "dtype": "float32",
        "shape": (NUM_WELLS,),
        "names": None,
    },
    "observation.keypoints_overhead": {
        "dtype": "float32",
        "shape": (6,),  # 3 bodies × 2 coords
        "names": None,
    },
    "observation.keypoints_wrist": {
        "dtype": "float32",
        "shape": (6,),  # 3 bodies × 2 coords
        "names": None,
    },
    "observation.target_bottle_keypoints_overhead": {
        "dtype": "float32",
        "shape": (2,),
        "names": None,
    },
    "observation.target_well_keypoints_overhead": {
        "dtype": "float32",
        "shape": (2,),
        "names": None,
    },
    "observation.phase_description": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "next.reward": {
        "dtype": "float32",
        "shape": (6,),
        "names": None,
    },
}

DIM_NAMES: dict[str, list[str]] = {
    "observation.state": [
        "ee_x",
        "ee_y",
        "ee_z",
        "gripper",
        "q0",
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
    ],
    "observation.state.ee.pos_quat_g": [
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "gripper",
    ],
    "observation.state.ee.pos_rot6d_g": [
        "x",
        "y",
        "z",
        "r11",
        "r12",
        "r13",
        "r21",
        "r22",
        "r23",
        "gripper",
    ],
    "observation.state.ee.pos_quat_g_rel": [
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "gripper",
    ],
    "observation.state.ee.pos_rot6d_g_rel": [
        "x",
        "y",
        "z",
        "r11",
        "r12",
        "r13",
        "r21",
        "r22",
        "r23",
        "gripper",
    ],
    "observation.state.ee.force_torque": ["fx", "fy", "fz", "tx", "ty", "tz"],
    "action.ee.pos_quat_g": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
    "action.ee.pos_rot6d_g": [
        "x",
        "y",
        "z",
        "r11",
        "r12",
        "r13",
        "r21",
        "r22",
        "r23",
        "gripper",
    ],
    "action.ee.pos_quat_g_rel": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
    "action.ee.pos_rot6d_g_rel": [
        "x",
        "y",
        "z",
        "r11",
        "r12",
        "r13",
        "r21",
        "r22",
        "r23",
        "gripper",
    ],
    "observation.target_well_onehot": [f"well_{i}" for i in range(NUM_WELLS)],
    "observation.keypoints_overhead": [
        "bottle_u",
        "bottle_v",
        "well_u",
        "well_v",
        "hand_u",
        "hand_v",
    ],
    "observation.keypoints_wrist": [
        "bottle_u",
        "bottle_v",
        "well_u",
        "well_v",
        "hand_u",
        "hand_v",
    ],
    "observation.target_bottle_keypoints_overhead": ["u", "v"],
    "observation.target_well_keypoints_overhead": ["u", "v"],
    "next.reward": [
        "total",
        "reach_bottle",
        "pick_bottle",
        "reach_well",
        "place_bottle",
        "reach_home",
    ],
}
