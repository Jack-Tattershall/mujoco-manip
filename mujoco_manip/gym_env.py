"""Gymnasium-compatible pick-and-place environment."""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .cameras import CameraRenderer, compute_keypoints, project_3d_to_2d
from .constants import (
    ACTION_REPEAT,
    BINS,
    IMAGE_SIZE,
    KEYPOINT_BODIES,
    MAX_EPISODE_STEPS,
    OBJECTS,
    TASK_SETS,
)
from .controller import IKController
from .env import PickPlaceEnv
from .pose_utils import (
    pos_rotmat_to_se3,
    se3_from_pos_quat_g,
    se3_from_pos_rot6d_g,
    se3_to_pos_quat_g,
    se3_to_pos_rot6d_g,
)
from .data import SCENE_XML as _DEFAULT_XML
from .robot import PandaRobot

ACTION_MODES = (
    "abs_pos",
    "ee_pos_quat_g",
    "ee_pos_rot6d_g",
    "ee_pos_quat_g_rel",
    "ee_pos_rot6d_g_rel",
)


class PickPlaceGymEnv(gym.Env):
    """Gymnasium wrapper for the MuJoCo pick-and-place scene.

    Observations include dual camera images, 2D keypoints, robot state,
    and a one-hot encoding of the target bin.

    Action modes:
        ``"abs_pos"``      — 4D: [ee_x, ee_y, ee_z, gripper] in world frame.
        ``"ee_pos_quat_g"``      — 8D: [x, y, z, qx, qy, qz, qw, gripper] in world
            frame (absolute SE(3)).
        ``"ee_pos_rot6d_g"``     — 10D: [x, y, z, r11, r12, r13, r21, r22, r23,
            gripper] in world frame (absolute SE(3)).
        ``"ee_pos_quat_g_rel"``  — 8D: [x, y, z, qx, qy, qz, qw, gripper] relative
            to initial EE pose.
        ``"ee_pos_rot6d_g_rel"`` — 10D: [x, y, z, r11, r12, r13, r21, r22, r23,
            gripper] relative to initial EE pose.

    State observations include both absolute (world-frame) and relative
    (initial-EE-frame) EE poses under ``state.ee.*`` and ``state.ee.*_rel``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        xml_path: str = _DEFAULT_XML,
        task: tuple[str, str] | None = None,
        tasks: str | list[tuple[str, str]] = "all",
        action_mode: str = "ee_pos_quat_g_rel",
        reward_type: str = "dense",
        image_size: int = IMAGE_SIZE,
        render_mode: str = "rgb_array",
        max_episode_steps: int = MAX_EPISODE_STEPS,
        randomize_objects: bool = False,
        spawn_x_range: tuple[float, float] = (-0.20, 0.20),
        spawn_y_range: tuple[float, float] = (0.30, 0.45),
    ) -> None:
        """Initialise the environment.

        Args:
            xml_path: Path to the MuJoCo scene XML.
            task: Fixed (obj, bin) pair for every episode. Overrides *tasks*.
            tasks: Task set to sample from on reset. Either a key
                (``"all"``, ``"match"``, ``"cross"``) or an explicit list of
                ``(obj, bin)`` tuples. Ignored when *task* is set.
            action_mode: One of ``"abs_pos"``, ``"ee_pos_quat_g"``, ``"ee_pos_rot6d_g"``,
                ``"ee_pos_quat_g_rel"``, ``"ee_pos_rot6d_g_rel"``.
            reward_type: ``"dense"``, ``"sparse"``, or ``"staged"``.
            image_size: Resolution for camera rendering.
            render_mode: Gymnasium render mode.
            max_episode_steps: Truncation limit.
            randomize_objects: If True, randomize object positions on reset.
            spawn_x_range: X-axis range for object randomization.
            spawn_y_range: Y-axis range for object randomization.
        """
        super().__init__()
        if action_mode not in ACTION_MODES:
            raise ValueError(
                f"action_mode must be one of {ACTION_MODES}, got '{action_mode}'"
            )

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
        self._randomize_objects = randomize_objects
        self._spawn_x_range = tuple(spawn_x_range)
        self._spawn_y_range = tuple(spawn_y_range)

        self._env = PickPlaceEnv(xml_path, add_wrist_camera=True)
        self._robot = PandaRobot(self._env.model, self._env.data)
        self._controller = IKController(self._env.model, self._env.data, self._robot)
        self._renderer = CameraRenderer(self._env.model, image_size, image_size)

        self._obj_name: str = ""
        self._bin_name: str = ""

        self._initial_ee_se3: np.ndarray | None = None
        self._target_obj_kp_overhead: np.ndarray | None = None
        self._target_bin_kp_overhead: np.ndarray | None = None

        # Staged reward state (only used when reward_type == "staged")
        self._has_grasped = False
        self._has_lifted = False
        self._above_target = False
        self._has_placed = False
        self._reward_hwm: np.ndarray | None = None
        self._robot_geom_ids: set[int] = set()
        self._obstacle_geom_ids: set[int] = set()

        if reward_type == "staged":
            robot_bodies = self._robot.BODY_NAMES
            object_bodies = {"obj_red", "obj_green", "obj_blue"}
            for i in range(self._env.model.ngeom):
                body_id = self._env.model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(
                    self._env.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )
                if body_name in robot_bodies:
                    self._robot_geom_ids.add(i)
                elif (
                    body_name
                    and body_name != "world"
                    and body_name not in object_bodies
                ):
                    self._obstacle_geom_ids.add(i)

        if action_mode == "abs_pos":
            self.action_space = spaces.Box(
                low=np.array([-0.5, 0.0, 0.24, 0.0], dtype=np.float32),
                high=np.array([0.5, 0.8, 0.60, 1.0], dtype=np.float32),
            )
        elif action_mode in ("ee_pos_quat_g", "ee_pos_quat_g_rel"):
            low = np.full(8, -np.inf, dtype=np.float32)
            high = np.full(8, np.inf, dtype=np.float32)
            low[7] = 0.0
            high[7] = 1.0  # gripper
            self.action_space = spaces.Box(low=low, high=high)
        elif action_mode in ("ee_pos_rot6d_g", "ee_pos_rot6d_g_rel"):
            low = np.full(10, -np.inf, dtype=np.float32)
            high = np.full(10, np.inf, dtype=np.float32)
            low[9] = 0.0
            high[9] = 1.0  # gripper
            self.action_space = spaces.Box(low=low, high=high)

        self.observation_space = spaces.Dict(
            {
                "image_overhead": spaces.Box(
                    0, 255, (image_size, image_size, 3), dtype=np.uint8
                ),
                "image_wrist": spaces.Box(
                    0, 255, (image_size, image_size, 3), dtype=np.uint8
                ),
                "state": spaces.Box(-np.inf, np.inf, (11,), dtype=np.float32),
                "state.ee.pos_quat_g": spaces.Box(
                    -np.inf, np.inf, (8,), dtype=np.float32
                ),
                "state.ee.pos_rot6d_g": spaces.Box(
                    -np.inf, np.inf, (10,), dtype=np.float32
                ),
                "state.ee.pos_quat_g_rel": spaces.Box(
                    -np.inf, np.inf, (8,), dtype=np.float32
                ),
                "state.ee.pos_rot6d_g_rel": spaces.Box(
                    -np.inf, np.inf, (10,), dtype=np.float32
                ),
                "target_bin_onehot": spaces.Box(0.0, 1.0, (3,), dtype=np.float32),
                "target_obj_onehot": spaces.Box(0.0, 1.0, (3,), dtype=np.float32),
                "keypoints_overhead": spaces.Box(
                    0.0, 1.0, (len(KEYPOINT_BODIES), 2), dtype=np.float32
                ),
                "keypoints_wrist": spaces.Box(
                    0.0, 1.0, (len(KEYPOINT_BODIES), 2), dtype=np.float32
                ),
                "target_obj_keypoints_overhead": spaces.Box(
                    0.0, 1.0, (2,), dtype=np.float32
                ),
                "target_bin_keypoints_overhead": spaces.Box(
                    0.0, 1.0, (2,), dtype=np.float32
                ),
            }
        )

    def _capture_initial_pose(self) -> None:
        """Store the current EE SE(3) as the episode's initial pose."""
        self._initial_ee_se3 = pos_rotmat_to_se3(
            self._robot.ee_pos,
            self._robot.ee_xmat,
        )

    def _relative_action_to_world_pos(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Convert an action to a world-frame target position.

        Args:
            action: Raw action array from the agent.

        Returns:
            Tuple of (ee_target_xyz (3,), gripper_cmd).
        """
        if self._action_mode == "abs_pos":
            return action[:3], action[3]

        if self._action_mode == "ee_pos_quat_g":
            T_abs = se3_from_pos_quat_g(action)
            return T_abs[:3, 3], action[7]

        if self._action_mode == "ee_pos_rot6d_g":
            T_abs = se3_from_pos_rot6d_g(action)
            return T_abs[:3, 3], action[9]

        if self._action_mode == "ee_pos_quat_g_rel":
            T_rel = se3_from_pos_quat_g(action)
            gripper_cmd = action[7]
        else:  # ee_pos_rot6d_g_rel
            T_rel = se3_from_pos_rot6d_g(action)
            gripper_cmd = action[9]

        # T_abs = T_init @ T_rel
        T_abs = self._initial_ee_se3 @ T_rel
        return T_abs[:3, 3], gripper_cmd

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Build the full observation dictionary for the current state.

        Returns:
            Dict matching ``observation_space``.
        """
        model = self._env.model
        data = self._env.data

        img_overhead = self._renderer.render(data, "overhead")
        img_wrist = self._renderer.render(data, "wrist")

        gripper_norm = np.array(
            [self._robot.gripper_ctrl / PandaRobot.GRIPPER_OPEN], dtype=np.float32
        )
        state = np.concatenate(
            [
                self._robot.ee_pos.astype(np.float32),
                gripper_norm,
                self._robot.arm_qpos.astype(np.float32),
            ]
        )

        bin_idx = BINS.index(self._bin_name)
        bin_onehot = np.zeros(3, dtype=np.float32)
        bin_onehot[bin_idx] = 1.0

        obj_idx = OBJECTS.index(self._obj_name)
        obj_onehot = np.zeros(3, dtype=np.float32)
        obj_onehot[obj_idx] = 1.0

        T_current = pos_rotmat_to_se3(self._robot.ee_pos, self._robot.ee_xmat)
        T_rel = np.linalg.inv(self._initial_ee_se3) @ T_current
        gripper_val = float(gripper_norm[0])
        state_pos_quat_g = se3_to_pos_quat_g(T_current, gripper_val)
        state_pos_rot6d_g = se3_to_pos_rot6d_g(T_current, gripper_val)
        state_pos_quat_g_rel = se3_to_pos_quat_g(T_rel, gripper_val)
        state_pos_rot6d_g_rel = se3_to_pos_rot6d_g(T_rel, gripper_val)

        kp_overhead = compute_keypoints(model, data, "overhead", self._image_size)
        kp_wrist = compute_keypoints(model, data, "wrist", self._image_size)

        return {
            "image_overhead": img_overhead,
            "image_wrist": img_wrist,
            "state": state,
            "state.ee.pos_quat_g": state_pos_quat_g,
            "state.ee.pos_rot6d_g": state_pos_rot6d_g,
            "state.ee.pos_quat_g_rel": state_pos_quat_g_rel,
            "state.ee.pos_rot6d_g_rel": state_pos_rot6d_g_rel,
            "target_bin_onehot": bin_onehot,
            "target_obj_onehot": obj_onehot,
            "keypoints_overhead": kp_overhead,
            "keypoints_wrist": kp_wrist,
            "target_obj_keypoints_overhead": self._target_obj_kp_overhead,
            "target_bin_keypoints_overhead": self._target_bin_kp_overhead,
        }

    def _check_robot_collision(self) -> bool:
        """Check if any robot geom is in contact with an obstacle geom."""
        for i in range(self._env.data.ncon):
            c = self._env.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self._robot_geom_ids and g2 in self._obstacle_geom_ids) or (
                g2 in self._robot_geom_ids and g1 in self._obstacle_geom_ids
            ):
                return True
        return False

    def _compute_staged_reward(self) -> tuple[float, bool]:
        """Compute staged reward with five sequential phases.

        Phases: reach_object → pick_object → reach_target → place_object →
        reach_home.  Each contributes [0, 0.2] for a total range of [0, 1].
        High-water marks ensure the total is strictly monotonic.

        Returns:
            Tuple of (reward, terminated). Returns -1.0 with
            terminated=True on robot-obstacle collision.
        """
        D_MAX = 0.5
        GRASP_Z = 0.35
        LIFT_Z = 0.42

        obj_pos = self._env.get_body_pos(self._obj_name)
        bin_pos = self._env.get_body_pos(self._bin_name)
        ee_pos = self._robot.ee_pos
        gripper_closed = self._robot.gripper_ctrl == PandaRobot.GRIPPER_CLOSED

        # --- Sticky phase transitions ---
        if not self._has_grasped and obj_pos[2] > GRASP_Z and gripper_closed:
            self._has_grasped = True
        if not self._has_lifted and obj_pos[2] > LIFT_Z and gripper_closed:
            self._has_lifted = True
        xy_dist = np.linalg.norm(obj_pos[:2] - bin_pos[:2])
        if not self._above_target and self._has_lifted and xy_dist < 0.06:
            self._above_target = True
        placed = xy_dist < 0.05 and obj_pos[2] < bin_pos[2] + 0.06
        if not self._has_placed and placed:
            self._has_placed = True

        # --- Phase 1: reach object ---
        if self._has_grasped:
            r0 = 1.0
        else:
            r0 = 1.0 - min(np.linalg.norm(ee_pos - obj_pos) / D_MAX, 1.0)

        # --- Phase 2: pick object (lift from grasp height to transit) ---
        if not self._has_grasped:
            r1 = 0.0
        elif self._has_lifted:
            r1 = 1.0
        else:
            r1 = max(0.0, min((obj_pos[2] - 0.30) / (LIFT_Z - 0.30), 1.0))

        # --- Phase 3: reach target (move object above bin XY) ---
        if not self._has_lifted:
            r2 = 0.0
        elif self._above_target:
            r2 = 1.0
        else:
            r2 = 1.0 - min(xy_dist / D_MAX, 1.0)

        # --- Phase 4: place object (lower into bin) ---
        if not self._above_target:
            r3 = 0.0
        elif self._has_placed:
            r3 = 1.0
        else:
            height_above = obj_pos[2] - bin_pos[2]
            r3 = 1.0 - max(0.0, min(height_above / 0.25, 1.0))

        # --- Phase 5: reach home ---
        if not self._has_placed:
            r4 = 0.0
        else:
            init_ee_pos = self._initial_ee_se3[:3, 3]
            r4 = 1.0 - min(np.linalg.norm(ee_pos - init_ee_pos) / D_MAX, 1.0)

        # High-water marks → guarantees monotonicity
        components = np.array([r0, r1, r2, r3, r4])
        if self._reward_hwm is None:
            self._reward_hwm = np.zeros_like(components)
        self._reward_hwm = np.maximum(self._reward_hwm, components)

        # Collision check
        if self._check_robot_collision():
            return -1.0, True

        reward = float(self._reward_hwm.mean())
        done = bool(np.all(self._reward_hwm >= 1.0))
        return reward, done

    def _compute_reward(self) -> tuple[float, bool]:
        """Compute reward and check for success.

        Returns:
            Tuple of (reward, success).
        """
        obj_pos = self._env.get_body_pos(self._obj_name)
        bin_pos = self._env.get_body_pos(self._bin_name)
        ee_pos = self._robot.ee_pos

        # Success: object center within 0.05m of bin center in XY and z < bin_z + 0.06
        xy_dist = np.linalg.norm(obj_pos[:2] - bin_pos[:2])
        success = xy_dist < 0.05 and obj_pos[2] < bin_pos[2] + 0.06

        if self._reward_type == "sparse":
            return (1.0 if success else 0.0), success

        if self._reward_type == "staged":
            return self._compute_staged_reward()

        # Dense reward
        reward = 0.0

        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        reward -= dist_ee_obj

        if obj_pos[2] > 0.30:  # grasp bonus
            reward += 2.0
            dist_obj_bin = np.linalg.norm(obj_pos - bin_pos)
            reward -= dist_obj_bin

        if success:
            reward += 10.0

        return reward, success

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset the environment and return initial observation.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options (unused).

        Returns:
            Tuple of (obs, info).
        """
        super().reset(seed=seed)

        self._env.reset_to_keyframe("scene_start")
        self._step_count = 0

        if self._randomize_objects:
            self._env.randomize_objects(
                self.np_random,
                x_range=self._spawn_x_range,
                y_range=self._spawn_y_range,
            )

        self._capture_initial_pose()

        self._has_grasped = False
        self._has_lifted = False
        self._above_target = False
        self._has_placed = False
        self._reward_hwm = None

        if self._fixed_task is not None:
            self._obj_name, self._bin_name = self._fixed_task
        else:
            idx = self.np_random.integers(len(self._task_pool))
            self._obj_name, self._bin_name = self._task_pool[idx]

        model, data = self._env.model, self._env.data
        obj_3d = data.xpos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._obj_name)
        ][np.newaxis]
        bin_3d = data.xpos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._bin_name)
        ][np.newaxis]
        self._target_obj_kp_overhead = project_3d_to_2d(
            model, data, "overhead", obj_3d, self._image_size
        ).flatten()
        self._target_bin_kp_overhead = project_3d_to_2d(
            model, data, "overhead", bin_3d, self._image_size
        ).flatten()

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Execute one environment step.

        Args:
            action: Action array matching ``action_space``.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        action = np.asarray(action, dtype=np.float32)
        ee_target, gripper_cmd = self._relative_action_to_world_pos(action)

        if gripper_cmd > 0.5:
            self._robot.open_gripper()
        else:
            self._robot.close_gripper()

        for _ in range(ACTION_REPEAT):
            q_target = self._controller.compute(ee_target)
            self._robot.set_arm_ctrl(q_target)
            self._env.step()

        mujoco.mj_forward(self._env.model, self._env.data)

        self._step_count += 1
        reward, success = self._compute_reward()
        if self._reward_type == "staged":
            # collision (reward < 0) is not success; otherwise use the done flag
            terminated = reward < 0 or success
            info = {"success": success and reward >= 0}
        else:
            terminated = success
            info = {"success": success}
        truncated = self._step_count >= self._max_episode_steps

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Return an overhead RGB image if render_mode is ``'rgb_array'``."""
        if self.render_mode == "rgb_array":
            return self._renderer.render(self._env.data, "overhead")
        return None

    def close(self) -> None:
        """Release renderer resources."""
        self._renderer.close()
