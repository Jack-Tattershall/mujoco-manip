"""Gymnasium-compatible bottle packing environment."""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_manip.cameras import CameraRenderer, compute_keypoints, project_3d_to_2d
from mujoco_manip.controller import IKController
from mujoco_manip.data import BOTTLE_PACKING_SCENE_XML as _DEFAULT_XML
from mujoco_manip.pose_utils import (
    pos_rotmat_to_se3,
    se3_from_pos_quat_g,
    se3_from_pos_rot6d_g,
    se3_to_pos_quat_g,
    se3_to_pos_rot6d_g,
)
from mujoco_manip.robot import PandaRobot

from .constants import (
    ACTION_REPEAT,
    BELT_Y_NOISE,
    BOTTLE_BODIES,
    BOTTLE_HALF_HEIGHT,
    CONVEYOR_RESUME_Z,
    IMAGE_SIZE,
    MAX_EPISODE_STEPS,
    NUM_WELLS,
    TASK_SETS,
    WELL_WALL_HEIGHT,
    well_position,
)
from .env import BottlePackingEnv

ACTION_MODES = (
    "abs_pos",
    "ee_pos_quat_g",
    "ee_pos_rot6d_g",
    "ee_pos_quat_g_rel",
    "ee_pos_rot6d_g_rel",
)

# Staged reward thresholds
_REWARD_D_MAX = 0.5  # max distance for normalised distance rewards
_REWARD_GRASP_Z = 0.35  # bottle z threshold for grasp detection
_REWARD_LIFT_Z = 0.42  # bottle z threshold for lift detection
_REWARD_ABOVE_XY = 0.04  # XY distance threshold for "above target"
_REWARD_PLACED_Z = 0.04  # Z tolerance above well target for placement
_REWARD_HOME_DIST = 0.05  # EE distance to home for completion
_REWARD_LIFT_FLOOR_Z = 0.30  # z floor for normalising lift progress
_REWARD_DESCENT_RANGE = 0.25  # max height-above-well for descent reward


class BottlePackingGymEnv(gym.Env):
    """Gymnasium wrapper for the bottle packing scene.

    The robot picks a bottle from the conveyor end and places it into
    one of 20 wells (5 cols x 4 rows) in a crate.

    Observations include dual camera images, 2D keypoints, robot state,
    and a one-hot encoding of the target well.

    Action modes:
        ``"abs_pos"``      — 4D: [ee_x, ee_y, ee_z, gripper] in world frame.
        ``"ee_pos_quat_g"``      — 8D: absolute SE(3) + gripper.
        ``"ee_pos_rot6d_g"``     — 10D: absolute SE(3) (6D rot) + gripper.
        ``"ee_pos_quat_g_rel"``  — 8D: relative to initial EE pose.
        ``"ee_pos_rot6d_g_rel"`` — 10D: relative to initial EE pose.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        xml_path: str = _DEFAULT_XML,
        well_index: int | None = None,
        wells: str | list[int] = "all",
        action_mode: str = "ee_pos_quat_g_rel",
        reward_type: str = "dense",
        image_size: int = IMAGE_SIZE,
        render_mode: str = "rgb_array",
        max_episode_steps: int = MAX_EPISODE_STEPS,
        belt_y_noise: float = BELT_Y_NOISE,
    ) -> None:
        """Initialise the environment.

        Args:
            xml_path: Path to the MuJoCo scene XML.
            well_index: Fixed target well for every episode. Overrides *wells*.
            wells: Well set to sample from on reset. Either ``"all"`` or an
                explicit list of well indices.
            action_mode: One of the supported action modes.
            reward_type: ``"dense"``, ``"sparse"``, or ``"staged"``.
            image_size: Resolution for camera rendering.
            render_mode: Gymnasium render mode.
            max_episode_steps: Truncation limit.
            belt_y_noise: Half-range of uniform Y offset for bottle spawn
                on the conveyor belt (metres). Set to 0 to disable.
        """
        super().__init__()
        if action_mode not in ACTION_MODES:
            raise ValueError(
                f"action_mode must be one of {ACTION_MODES}, got '{action_mode}'"
            )

        self._xml_path = xml_path
        self._fixed_well = well_index
        if isinstance(wells, str):
            self._well_pool = TASK_SETS[wells]
        else:
            self._well_pool = wells
        self._action_mode = action_mode
        self._reward_type = reward_type
        self._image_size = image_size
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._belt_y_noise = belt_y_noise
        self._step_count = 0
        self._conveyor_resumed = False

        self._env = BottlePackingEnv(xml_path, add_wrist_camera=True)
        self._env.colorize_bottles()
        self._robot = PandaRobot(self._env.model, self._env.data)
        self._controller = IKController(self._env.model, self._env.data, self._robot)
        self._renderer = CameraRenderer(self._env.model, image_size, image_size)

        self._well_index: int = 0
        self._initial_ee_se3: np.ndarray | None = None
        self._target_bottle_kp_overhead: np.ndarray | None = None
        self._target_well_kp_overhead: np.ndarray | None = None

        # Peak insertion force (max positive Fz over episode)
        self._peak_fz: float = 0.0

        # Staged reward state
        self._has_grasped = False
        self._has_lifted = False
        self._above_target = False
        self._has_placed = False
        self._reward_hwm: np.ndarray | None = None
        self._robot_geom_ids: frozenset[int] = frozenset()
        self._obstacle_geom_ids: frozenset[int] = frozenset()

        if reward_type == "staged":
            robot_bodies = self._robot.BODY_NAMES
            bottle_body_set = set(BOTTLE_BODIES)
            robot_ids: set[int] = set()
            obstacle_ids: set[int] = set()
            for i in range(self._env.model.ngeom):
                body_id = self._env.model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(
                    self._env.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )
                if body_name in robot_bodies:
                    robot_ids.add(i)
                elif (
                    body_name
                    and body_name != "world"
                    and body_name not in bottle_body_set
                ):
                    obstacle_ids.add(i)
            self._robot_geom_ids = frozenset(robot_ids)
            self._obstacle_geom_ids = frozenset(obstacle_ids)

        if action_mode == "abs_pos":
            self.action_space = spaces.Box(
                low=np.array([-0.5, 0.0, 0.24, 0.0], dtype=np.float32),
                high=np.array([0.5, 0.8, 0.60, 1.0], dtype=np.float32),
            )
        elif action_mode in ("ee_pos_quat_g", "ee_pos_quat_g_rel"):
            low = np.full(8, -np.inf, dtype=np.float32)
            high = np.full(8, np.inf, dtype=np.float32)
            low[7] = 0.0
            high[7] = 1.0
            self.action_space = spaces.Box(low=low, high=high)
        elif action_mode in ("ee_pos_rot6d_g", "ee_pos_rot6d_g_rel"):
            low = np.full(10, -np.inf, dtype=np.float32)
            high = np.full(10, np.inf, dtype=np.float32)
            low[9] = 0.0
            high[9] = 1.0
            self.action_space = spaces.Box(low=low, high=high)

        # Keypoint bodies: active bottle, crate, hand (3 bodies)
        num_kp = 3
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
                "target_well_onehot": spaces.Box(
                    0.0, 1.0, (NUM_WELLS,), dtype=np.float32
                ),
                "keypoints_overhead": spaces.Box(
                    0.0, 1.0, (num_kp, 2), dtype=np.float32
                ),
                "keypoints_wrist": spaces.Box(0.0, 1.0, (num_kp, 2), dtype=np.float32),
                "target_bottle_keypoints_overhead": spaces.Box(
                    0.0, 1.0, (2,), dtype=np.float32
                ),
                "target_well_keypoints_overhead": spaces.Box(
                    0.0, 1.0, (2,), dtype=np.float32
                ),
                "state.ee.force_torque": spaces.Box(
                    -np.inf, np.inf, (6,), dtype=np.float32
                ),
            }
        )

    @property
    def action_mode(self) -> str:
        return self._action_mode

    @property
    def bottle_packing_env(self) -> BottlePackingEnv:
        return self._env

    @property
    def robot(self) -> PandaRobot:
        return self._robot

    @property
    def controller(self) -> IKController:
        return self._controller

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def well_index(self) -> int:
        return self._well_index

    def _capture_initial_pose(self) -> None:
        self._initial_ee_se3 = pos_rotmat_to_se3(
            self._robot.ee_pos,
            self._robot.ee_xmat,
        )

    def decode_action(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """Convert an action to a world-frame target position and gripper cmd."""
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

        T_abs = self._initial_ee_se3 @ T_rel
        return T_abs[:3, 3], gripper_cmd

    def _compute_keypoints(self, camera_name: str) -> np.ndarray:
        """Project keypoint bodies to normalised pixel coordinates (3, 2).

        Keypoint bodies: active bottle, target well, hand.
        The target well keypoint uses the computed well position rather
        than the crate body centre so the keypoint reflects the actual
        slot the bottle should be placed into.
        """
        model, data = self._env.model, self._env.data
        # Bottle and hand from body positions
        body_kps = compute_keypoints(
            model,
            data,
            camera_name,
            [self._env.active_bottle_body, "hand"],
            self._image_size,
        )
        # Target well from computed world position (top of well opening)
        well_pos = well_position(self._well_index).copy()
        well_pos[2] += WELL_WALL_HEIGHT
        well_kp = project_3d_to_2d(
            model, data, camera_name, well_pos[np.newaxis], self._image_size
        )
        # Stack as (bottle, well, hand) — shape (3, 2)
        return np.vstack([body_kps[:1], well_kp, body_kps[1:]])

    def _get_obs(self) -> dict[str, np.ndarray]:
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

        well_onehot = np.zeros(NUM_WELLS, dtype=np.float32)
        well_onehot[self._well_index] = 1.0

        T_current = pos_rotmat_to_se3(self._robot.ee_pos, self._robot.ee_xmat)
        T_rel = np.linalg.inv(self._initial_ee_se3) @ T_current
        gripper_val = float(gripper_norm[0])
        state_pos_quat_g = se3_to_pos_quat_g(T_current, gripper_val)
        state_pos_rot6d_g = se3_to_pos_rot6d_g(T_current, gripper_val)
        state_pos_quat_g_rel = se3_to_pos_quat_g(T_rel, gripper_val)
        state_pos_rot6d_g_rel = se3_to_pos_rot6d_g(T_rel, gripper_val)

        kp_overhead = self._compute_keypoints("overhead")
        kp_wrist = self._compute_keypoints("wrist")

        return {
            "image_overhead": img_overhead,
            "image_wrist": img_wrist,
            "state": state,
            "state.ee.pos_quat_g": state_pos_quat_g,
            "state.ee.pos_rot6d_g": state_pos_rot6d_g,
            "state.ee.pos_quat_g_rel": state_pos_quat_g_rel,
            "state.ee.pos_rot6d_g_rel": state_pos_rot6d_g_rel,
            "state.ee.force_torque": self._robot.ee_force_torque,
            "target_well_onehot": well_onehot,
            "keypoints_overhead": kp_overhead,
            "keypoints_wrist": kp_wrist,
            "target_bottle_keypoints_overhead": self._target_bottle_kp_overhead,
            "target_well_keypoints_overhead": self._target_well_kp_overhead,
        }

    def _check_placement_success(self) -> bool:
        """Return True if the active bottle is placed in the target well."""
        bottle_pos = self._env.get_body_pos(self._env.active_bottle_body)
        well_pos_3d = well_position(self._well_index)
        well_target_z = well_pos_3d[2] + BOTTLE_HALF_HEIGHT
        xy_dist = np.linalg.norm(bottle_pos[:2] - well_pos_3d[:2])
        return (
            xy_dist < _REWARD_ABOVE_XY
            and bottle_pos[2] < well_target_z + _REWARD_PLACED_Z
        )

    def _check_robot_collision(self) -> bool:
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

        Phases: reach_bottle -> pick_bottle -> reach_well -> place_bottle ->
        reach_home.  Each contributes [0, 0.2] for a total range of [0, 1].
        """
        bottle_pos = self._env.get_body_pos(self._env.active_bottle_body)
        well_pos_3d = well_position(self._well_index)
        well_target_z = well_pos_3d[2] + BOTTLE_HALF_HEIGHT
        ee_pos = self._robot.ee_pos
        gripper_closed = self._robot.gripper_ctrl == PandaRobot.GRIPPER_CLOSED

        # Sticky phase transitions
        if not self._has_grasped and bottle_pos[2] > _REWARD_GRASP_Z and gripper_closed:
            self._has_grasped = True
        if not self._has_lifted and bottle_pos[2] > _REWARD_LIFT_Z and gripper_closed:
            self._has_lifted = True
        xy_dist = np.linalg.norm(bottle_pos[:2] - well_pos_3d[:2])
        if not self._above_target and self._has_lifted and xy_dist < _REWARD_ABOVE_XY:
            self._above_target = True
        if not self._has_placed and self._check_placement_success():
            self._has_placed = True

        # Phase 1: reach bottle
        if self._has_grasped:
            r0 = 1.0
        else:
            r0 = 1.0 - min(np.linalg.norm(ee_pos - bottle_pos) / _REWARD_D_MAX, 1.0)

        # Phase 2: pick bottle
        if not self._has_grasped:
            r1 = 0.0
        elif self._has_lifted:
            r1 = 1.0
        else:
            r1 = max(
                0.0,
                min(
                    (bottle_pos[2] - _REWARD_LIFT_FLOOR_Z)
                    / (_REWARD_LIFT_Z - _REWARD_LIFT_FLOOR_Z),
                    1.0,
                ),
            )

        # Phase 3: reach well
        if not self._has_lifted:
            r2 = 0.0
        elif self._above_target:
            r2 = 1.0
        else:
            r2 = 1.0 - min(xy_dist / _REWARD_D_MAX, 1.0)

        # Phase 4: place bottle
        if not self._above_target:
            r3 = 0.0
        elif self._has_placed:
            r3 = 1.0
        else:
            height_above = bottle_pos[2] - well_target_z
            r3 = 1.0 - max(0.0, min(height_above / _REWARD_DESCENT_RANGE, 1.0))

        # Phase 5: reach home
        if not self._has_placed:
            r4 = 0.0
        else:
            init_ee_pos = self._initial_ee_se3[:3, 3]
            dist = np.linalg.norm(ee_pos - init_ee_pos)
            r4 = (
                1.0
                if dist < _REWARD_HOME_DIST
                else 1.0 - min(dist / _REWARD_D_MAX, 1.0)
            )

        components = np.array([r0, r1, r2, r3, r4])
        if self._reward_hwm is None:
            self._reward_hwm = np.zeros_like(components)
        self._reward_hwm = np.maximum(self._reward_hwm, components)

        if self._check_robot_collision():
            return -1.0, True

        reward = float(self._reward_hwm.mean())
        done = bool(np.all(self._reward_hwm >= 0.90))
        return reward, done

    def _compute_reward(self) -> tuple[float, bool]:
        if self._reward_type == "staged":
            return self._compute_staged_reward()

        success = self._check_placement_success()

        if self._reward_type == "sparse":
            return (1.0 if success else 0.0), success

        # Dense reward
        bottle_pos = self._env.get_body_pos(self._env.active_bottle_body)
        well_pos_3d = well_position(self._well_index)
        well_target = np.array(
            [well_pos_3d[0], well_pos_3d[1], well_pos_3d[2] + BOTTLE_HALF_HEIGHT]
        )
        ee_pos = self._robot.ee_pos

        reward = 0.0
        dist_ee_bottle = np.linalg.norm(ee_pos - bottle_pos)
        reward -= dist_ee_bottle

        if bottle_pos[2] > _REWARD_LIFT_FLOOR_Z:
            reward += 2.0
            dist_bottle_well = np.linalg.norm(bottle_pos - well_target)
            reward -= dist_bottle_well

        if success:
            reward += 10.0

        return reward, success

    @property
    def initial_ee_se3(self) -> np.ndarray:
        return self._initial_ee_se3.copy()

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset the environment and return initial observation.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options. Supports:
                ``"well_index"`` — target well for this episode.
                ``"bottle_index"`` — which bottle body to spawn (default =
                    ``well_index`` for sequential, must be set for random).
                ``"packed"`` — dict mapping ``bottle_idx → well_idx`` for
                    bottles already packed in wells from prior episodes.
                ``"belt_bottles"`` — explicit list of bottle indices to
                    place on the belt behind the active bottle.
        """
        super().reset(seed=seed)

        self._env.reset_to_keyframe("scene_start")
        self._step_count = 0
        self._conveyor_resumed = False

        self._peak_fz = 0.0
        self._has_grasped = False
        self._has_lifted = False
        self._above_target = False
        self._has_placed = False
        self._reward_hwm = None

        # Determine target well
        if options and "well_index" in options:
            self._well_index = options["well_index"]
        elif self._fixed_well is not None:
            self._well_index = self._fixed_well
        else:
            self._well_index = self._well_pool[
                int(self.np_random.integers(len(self._well_pool)))
            ]

        # Determine bottle index (defaults to well_index for sequential mode)
        bottle_index = (
            options.get("bottle_index", self._well_index)
            if options
            else self._well_index
        )

        # Place pre-packed bottles in wells, all others hidden
        packed = options.get("packed") if options else None
        self._env.setup_scene(num_prepacked=self._well_index, packed=packed)

        # Capture initial EE pose BEFORE conveyor (robot at home)
        self._capture_initial_pose()

        # Build conveyor queue: active bottle first, then upcoming
        belt_queue = [bottle_index]
        upcoming = options.get("belt_bottles") if options else None
        if upcoming is None:
            packed_set = set(packed.keys()) if packed else set()
            upcoming = [
                i for i in range(NUM_WELLS) if i != bottle_index and i not in packed_set
            ]
        belt_queue.extend(upcoming)

        # Load onto conveyor with Y noise and deliver front bottle
        rng = self.np_random if self._belt_y_noise > 0 else None
        self._env.load_conveyor(belt_queue, rng=rng, y_noise=self._belt_y_noise)

        # Run conveyor until the active bottle arrives at pickup
        while self._env.bottle_at_pickup is None:
            self._env.tick_conveyor()

        # Activate the delivered bottle for the robot to grasp
        self._env.start_pickup()

        model, data = self._env.model, self._env.data

        # Static target keypoints (goal conditioning, computed once at t=0)
        bottle_3d = data.xpos[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, self._env.active_bottle_body
            )
        ][np.newaxis]
        self._target_bottle_kp_overhead = project_3d_to_2d(
            model, data, "overhead", bottle_3d, self._image_size
        ).flatten()

        well_pos_3d = well_position(self._well_index)
        # Project from the top of the well opening (not the floor) so the
        # keypoint aligns with the visible well centre in the overhead image.
        well_top = well_pos_3d.copy()
        well_top[2] += WELL_WALL_HEIGHT
        self._target_well_kp_overhead = project_3d_to_2d(
            model, data, "overhead", well_top[np.newaxis], self._image_size
        ).flatten()

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        ee_target, gripper_cmd = self.decode_action(action)

        if gripper_cmd > 0.5:
            self._robot.open_gripper()
        else:
            self._robot.close_gripper()

        for _ in range(ACTION_REPEAT):
            q_target = self._controller.compute(ee_target)
            self._robot.set_arm_ctrl(q_target)
            self._env.step()
            self._env.tick_conveyor()

        mujoco.mj_forward(self._env.model, self._env.data)

        # Track peak positive Fz (insertion contact force)
        fz = float(self._robot.ee_force_torque[2])
        if fz > self._peak_fz:
            self._peak_fz = fz

        # Resume belt when bottle is lifted clear of the conveyor
        if not self._conveyor_resumed:
            bottle_z = self._env.get_body_pos(self._env.active_bottle_body)[2]
            if bottle_z > CONVEYOR_RESUME_Z:
                self._env.resume_conveyor()
                self._conveyor_resumed = True

        self._step_count += 1
        reward, success = self._compute_reward()
        if self._reward_type == "staged":
            terminated = reward < 0 or success
            info = {"success": success and reward >= 0}
            if self._reward_hwm is not None:
                normed = self._reward_hwm / len(self._reward_hwm)
                info["reward_components"] = np.array(
                    [normed.sum(), *normed], dtype=np.float32
                )
        else:
            terminated = success
            info = {"success": success}
        truncated = self._step_count >= self._max_episode_steps

        if terminated or truncated:
            info["peak_insertion_force"] = self._peak_fz

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._renderer.render(self._env.data, "overhead")
        if self.render_mode == "human":
            if self._env.viewer is None:
                self._env.launch_viewer()
            self._env.sync()
        return None

    def close(self) -> None:
        self._renderer.close()
        if self._env.viewer is not None:
            self._env.viewer.close()
            self._env.viewer = None
