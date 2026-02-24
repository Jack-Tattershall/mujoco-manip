"""Gymnasium-compatible pick-and-place environment."""

import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .cameras import CameraRenderer, compute_keypoints
from .constants import (
    ACTION_REPEAT,
    BINS,
    IMAGE_SIZE,
    KEYPOINT_BODIES,
    MAX_EPISODE_STEPS,
    TASK_SETS,
)
from .controller import IKController
from .env import PickPlaceEnv
from .pose_utils import (
    pos_rotmat_to_se3,
    se3_from_8dof,
    se3_from_10dof,
    se3_to_8dof,
    se3_to_10dof,
)
from .robot import PandaRobot

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_XML = os.path.join(_PROJECT_ROOT, "pick_and_place_scene.xml")

ACTION_MODES = ("abs_pos", "ee_8dof", "ee_10dof")


class PickPlaceGymEnv(gym.Env):
    """Gymnasium wrapper for the MuJoCo pick-and-place scene.

    Observations include dual camera images, 2D keypoints, robot state,
    and a one-hot encoding of the target bin.

    Action modes:
        "abs_pos"  — 4D: [ee_x, ee_y, ee_z, gripper] in world frame
        "ee_8dof"  — 8D: [x, y, z, qx, qy, qz, qw, gripper] relative to initial EE pose
        "ee_10dof" — 10D: [x, y, z, r11, r12, r13, r21, r22, r23, gripper] relative to initial EE pose

    State observations include both absolute (world-frame) and relative
    (initial-EE-frame) EE poses under ``state.ee.*`` and ``state.ee.*_rel``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        xml_path: str = _DEFAULT_XML,
        task: tuple[str, str] | None = None,
        tasks: str | list[tuple[str, str]] = "all",
        action_mode: str = "ee_8dof",
        reward_type: str = "dense",
        image_size: int = IMAGE_SIZE,
        render_mode: str = "rgb_array",
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ):
        """
        Args:
            xml_path: Path to the MuJoCo scene XML.
            task: Fixed (obj, bin) pair for every episode. Overrides `tasks`.
            tasks: Task set to sample from on reset. Either a key
                ("all", "match", "cross") or an explicit list of (obj, bin) tuples.
                Ignored when `task` is set.
            action_mode: One of "abs_pos", "ee_8dof", "ee_10dof".
            reward_type: "dense" or "sparse".
            image_size: Resolution for camera rendering.
            render_mode: Gymnasium render mode.
            max_episode_steps: Truncation limit.
        """
        super().__init__()
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}, got '{action_mode}'")

        self._xml_path = xml_path
        self._fixed_task = task
        if isinstance(tasks, str):
            self._task_pool = TASK_SETS[tasks]
        else:
            self._task_pool = tasks
        self._action_mode = action_mode
        self._reward_type = reward_type
        self._image_size = image_size
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._step_count = 0

        # Load environment with wrist camera
        self._env = PickPlaceEnv(xml_path, add_wrist_camera=True)
        self._robot = PandaRobot(self._env.model, self._env.data)
        self._controller = IKController(self._env.model, self._env.data, self._robot)
        self._renderer = CameraRenderer(self._env.model, image_size, image_size)

        # Current task
        self._obj_name: str = ""
        self._bin_name: str = ""

        # Initial EE pose (set on reset)
        self._initial_ee_se3: np.ndarray | None = None

        # Action space depends on mode
        if action_mode == "abs_pos":
            self.action_space = spaces.Box(
                low=np.array([-0.5, 0.0, 0.24, 0.0], dtype=np.float32),
                high=np.array([0.5, 0.8, 0.60, 1.0], dtype=np.float32),
            )
        elif action_mode == "ee_8dof":
            low = np.full(8, -np.inf, dtype=np.float32)
            high = np.full(8, np.inf, dtype=np.float32)
            low[7] = 0.0; high[7] = 1.0  # gripper
            self.action_space = spaces.Box(low=low, high=high)
        elif action_mode == "ee_10dof":
            low = np.full(10, -np.inf, dtype=np.float32)
            high = np.full(10, np.inf, dtype=np.float32)
            low[9] = 0.0; high[9] = 1.0  # gripper
            self.action_space = spaces.Box(low=low, high=high)

        # Observation space
        self.observation_space = spaces.Dict({
            "image_overhead": spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8),
            "image_wrist": spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (11,), dtype=np.float32),
            "state.ee.8dof": spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
            "state.ee.10dof": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
            "state.ee.8dof_rel": spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
            "state.ee.10dof_rel": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
            "target_bin_onehot": spaces.Box(0.0, 1.0, (3,), dtype=np.float32),
            "keypoints_overhead": spaces.Box(0.0, 1.0, (len(KEYPOINT_BODIES), 2), dtype=np.float32),
            "keypoints_wrist": spaces.Box(0.0, 1.0, (len(KEYPOINT_BODIES), 2), dtype=np.float32),
        })

    def _capture_initial_pose(self):
        """Store the initial EE SE(3) pose after reset."""
        self._initial_ee_se3 = pos_rotmat_to_se3(
            self._robot.ee_pos, self._robot.ee_xmat,
        )

    def _relative_action_to_world_pos(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """Convert a relative-to-initial action to absolute world-frame position + gripper.

        Returns:
            (ee_target_xyz, gripper_cmd)
        """
        if self._action_mode == "abs_pos":
            return action[:3], action[3]

        # Build relative SE(3) from the action
        if self._action_mode == "ee_8dof":
            T_rel = se3_from_8dof(action)
            gripper_cmd = action[7]
        else:  # ee_10dof
            T_rel = se3_from_10dof(action)
            gripper_cmd = action[9]

        # Convert to absolute world frame: T_abs = T_init @ T_rel
        T_abs = self._initial_ee_se3 @ T_rel
        return T_abs[:3, 3], gripper_cmd

    def _get_obs(self) -> dict[str, np.ndarray]:
        model = self._env.model
        data = self._env.data

        # Images
        img_overhead = self._renderer.render(data, "overhead")
        img_wrist = self._renderer.render(data, "wrist")

        # State: [ee_xyz(3), gripper_normalized(1), arm_qpos(7)]
        gripper_norm = np.array([self._robot.gripper_ctrl / PandaRobot.GRIPPER_OPEN], dtype=np.float32)
        state = np.concatenate([
            self._robot.ee_pos.astype(np.float32),
            gripper_norm,
            self._robot.arm_qpos.astype(np.float32),
        ])

        # Target bin one-hot
        bin_idx = BINS.index(self._bin_name)
        onehot = np.zeros(3, dtype=np.float32)
        onehot[bin_idx] = 1.0

        # EE pose: absolute (world-frame) and relative to initial
        T_current = pos_rotmat_to_se3(self._robot.ee_pos, self._robot.ee_xmat)
        T_rel = np.linalg.inv(self._initial_ee_se3) @ T_current
        gripper_val = float(gripper_norm[0])
        state_8dof = se3_to_8dof(T_current, gripper_val)
        state_10dof = se3_to_10dof(T_current, gripper_val)
        state_8dof_rel = se3_to_8dof(T_rel, gripper_val)
        state_10dof_rel = se3_to_10dof(T_rel, gripper_val)

        # Keypoints
        kp_overhead = compute_keypoints(model, data, "overhead", self._image_size)
        kp_wrist = compute_keypoints(model, data, "wrist", self._image_size)

        return {
            "image_overhead": img_overhead,
            "image_wrist": img_wrist,
            "state": state,
            "state.ee.8dof": state_8dof,
            "state.ee.10dof": state_10dof,
            "state.ee.8dof_rel": state_8dof_rel,
            "state.ee.10dof_rel": state_10dof_rel,
            "target_bin_onehot": onehot,
            "keypoints_overhead": kp_overhead,
            "keypoints_wrist": kp_wrist,
        }

    def _compute_reward(self) -> tuple[float, bool]:
        """Compute reward and check termination."""
        obj_pos = self._env.get_body_pos(self._obj_name)
        bin_pos = self._env.get_body_pos(self._bin_name)
        ee_pos = self._robot.ee_pos

        # Success: object center within 0.05m of bin center in XY and z < bin_z + 0.06
        xy_dist = np.linalg.norm(obj_pos[:2] - bin_pos[:2])
        success = xy_dist < 0.05 and obj_pos[2] < bin_pos[2] + 0.06

        if self._reward_type == "sparse":
            return (1.0 if success else 0.0), success

        # Dense reward
        reward = 0.0

        # Distance EE to object
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        reward -= dist_ee_obj

        # Grasp bonus: object lifted above z=0.30
        if obj_pos[2] > 0.30:
            reward += 2.0
            # When grasped, penalize distance from object to bin
            dist_obj_bin = np.linalg.norm(obj_pos - bin_pos)
            reward -= dist_obj_bin

        # Placement bonus
        if success:
            reward += 10.0

        return reward, success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._env.reset_to_keyframe("scene_start")
        self._step_count = 0

        # Capture initial EE pose (at keyframe home config)
        self._capture_initial_pose()

        # Pick task
        if self._fixed_task is not None:
            self._obj_name, self._bin_name = self._fixed_task
        else:
            idx = self.np_random.integers(len(self._task_pool))
            self._obj_name, self._bin_name = self._task_pool[idx]

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        ee_target, gripper_cmd = self._relative_action_to_world_pos(action)

        # Set gripper
        if gripper_cmd > 0.5:
            self._robot.open_gripper()
        else:
            self._robot.close_gripper()

        # Run IK + physics for ACTION_REPEAT steps
        for _ in range(ACTION_REPEAT):
            q_target = self._controller.compute(ee_target)
            self._robot.set_arm_ctrl(q_target)
            self._env.step()

        # Forward to update derived quantities
        mujoco.mj_forward(self._env.model, self._env.data)

        self._step_count += 1
        reward, success = self._compute_reward()
        terminated = success
        truncated = self._step_count >= self._max_episode_steps

        obs = self._get_obs()
        info = {"success": success}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._renderer.render(self._env.data, "overhead")
        return None

    def close(self):
        self._renderer.close()
