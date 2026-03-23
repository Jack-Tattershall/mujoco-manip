"""MuJoCo environment wrapper for the bottle packing scene."""

from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_manip.data import PANDA_DIR as _DEFAULT_PANDA_DIR
from mujoco_manip.data import BOTTLE_PACKING_SCENE_XML as _DEFAULT_SCENE_XML
from mujoco_manip.scene_loader import load_scene

from .constants import (
    BELT_Y_NOISE,
    BOTTLE_BODIES,
    BOTTLE_COLORS,
    BOTTLE_CONVEYOR_START,
    BOTTLE_HALF_HEIGHT,
    BOTTLE_HIDDEN_POS,
    BOTTLE_PICKUP_POS,
    CONVEYOR_ANIM_STEPS,
    CONVEYOR_BOTTLE_SPACING,
    CONVEYOR_SPEED,
    CRATE_JOINT_NAMES,
    MAX_BELT_BOTTLES,
    NUM_WELLS,
    well_position,
)


class BottlePackingEnv:
    """Loads the bottle-packing MJCF scene, manages simulation stepping and viewer."""

    def __init__(
        self,
        xml_path: str | None = None,
        panda_dir: str | None = None,
        add_wrist_camera: bool = False,
    ) -> None:
        if xml_path is None:
            xml_path = _DEFAULT_SCENE_XML
        if panda_dir is None:
            panda_dir = _DEFAULT_PANDA_DIR
        self.model: mujoco.MjModel = load_scene(
            xml_path, panda_dir, add_wrist_camera=add_wrist_camera
        )
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.viewer = None

        # Cache joint, geom, and body IDs for all 20 bottles
        self._bottle_qpos_adr: list[int] = []
        self._bottle_qvel_adr: list[int] = []
        self._bottle_geom_ids: list[list[int]] = []
        self._bottle_body_ids: list[int] = []
        for name in BOTTLE_BODIES:
            jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_jnt"
            )
            self._bottle_qpos_adr.append(self.model.jnt_qposadr[jnt_id])
            self._bottle_qvel_adr.append(self.model.jnt_dofadr[jnt_id])

            geom_ids = [
                gid
                for s in ("_body", "_neck")
                if (
                    gid := mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}{s}"
                    )
                )
                >= 0
            ]
            self._bottle_geom_ids.append(geom_ids)
            self._bottle_body_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            )

        # Cache crate joint qpos addresses
        self._crate_qpos_adr: list[int] = []
        self._crate_is_dynamic = False
        for jname in CRATE_JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jnt_id >= 0:
                self._crate_qpos_adr.append(self.model.jnt_qposadr[jnt_id])
                self._crate_is_dynamic = True
            else:
                break
        if len(self._crate_qpos_adr) != len(CRATE_JOINT_NAMES):
            self._crate_qpos_adr = []
            self._crate_is_dynamic = False

        # Cache body IDs for name-based lookups (pre-seeded with known bodies)
        self._body_id_cache: dict[str, int] = {
            name: bid for name, bid in zip(BOTTLE_BODIES, self._bottle_body_ids)
        }

        # Index of the active bottle (the one the robot picks)
        self._active_bottle: int = 0
        # Indices of all bottles currently on the conveyor belt
        self._belt_bottle_indices: deque[int] = deque()

        # Tick-based conveyor state
        self._pending_bottles: deque[int] = deque()
        self._conveyor_stopped: bool = True
        self._bottle_at_pickup: int | None = None
        self._belt_y_noise: float = 0.0
        self._belt_rng: np.random.Generator | None = None

    @property
    def active_bottle_body(self) -> str:
        """Body name of the active bottle (the one being packed)."""
        return BOTTLE_BODIES[self._active_bottle]

    def launch_viewer(self) -> None:
        """Open the passive MuJoCo viewer window."""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset_to_keyframe(self, name: str = "scene_start") -> None:
        """Reset simulation state to a named keyframe."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id < 0:
            raise ValueError(f"Keyframe '{name}' not found")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def _set_bottle_pose(
        self, bottle_idx: int, pos: np.ndarray, quat: np.ndarray | None = None
    ) -> None:
        """Set a bottle's qpos (position + quaternion) and zero its velocity/acceleration."""
        adr = self._bottle_qpos_adr[bottle_idx]
        vadr = self._bottle_qvel_adr[bottle_idx]
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat if quat is not None else [1, 0, 0, 0]
        self.data.qvel[vadr : vadr + 6] = 0
        self.data.qacc[vadr : vadr + 6] = 0

    def _disable_bottle_collision(self, idx: int) -> None:
        """Disable collision for a hidden bottle's geoms."""
        for gid in self._bottle_geom_ids[idx]:
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0

    def _enable_bottle_collision(self, idx: int) -> None:
        """Re-enable collision for a bottle's geoms."""
        for gid in self._bottle_geom_ids[idx]:
            self.model.geom_contype[gid] = 1
            self.model.geom_conaffinity[gid] = 1

    def _unfreeze_bottle(self, idx: int) -> None:
        """Restore physics for a single bottle."""
        bid = self._bottle_body_ids[idx]
        self.model.body_gravcomp[bid] = 0.0
        vadr = self._bottle_qvel_adr[idx]
        self.model.dof_damping[vadr : vadr + 6] = 0.0

    def _freeze_bottle(self, idx: int) -> None:
        """Make a single bottle physics-inert (but keep collision geometry).

        Frozen bottles retain their collision properties so the gripper and
        held bottle cannot pass through them.  High damping and gravity
        compensation keep the bottle stationary despite any contact forces.
        """
        bid = self._bottle_body_ids[idx]
        self.model.body_gravcomp[bid] = 1.0
        vadr = self._bottle_qvel_adr[idx]
        self.model.dof_damping[vadr : vadr + 6] = 1e4
        # Zero velocity and acceleration to prevent numerical drift
        self.data.qvel[vadr : vadr + 6] = 0
        self.data.qacc[vadr : vadr + 6] = 0

    def _unfreeze_all_bottles(self) -> None:
        """Restore physics for all bottles (undo any prior freeze)."""
        for idx in range(NUM_WELLS):
            self._unfreeze_bottle(idx)

    def _freeze_inactive_bottles(self) -> None:
        """Make every bottle except the active one physics-inert."""
        for idx in range(NUM_WELLS):
            if idx == self._active_bottle:
                continue
            self._freeze_bottle(idx)

    def setup_scene(
        self,
        num_prepacked: int = 0,
        packed: dict[int, int] | None = None,
    ) -> None:
        """Set up scene with pre-packed bottles in wells, all others hidden.

        There are two modes:

        *Sequential* (``packed`` is ``None``): bottles ``0..num_prepacked-1``
        are placed in wells ``0..num_prepacked-1`` and frozen.

        *Explicit* (``packed`` given): ``packed`` maps
        ``bottle_index → well_index`` for each bottle that should be placed
        in a well.  ``num_prepacked`` is ignored.

        Call after ``reset_to_keyframe`` and before
        ``spawn_bottle_on_conveyor``.

        Args:
            num_prepacked: Number of bottles already packed sequentially.
            packed: Explicit bottle→well mapping (overrides ``num_prepacked``).
        """
        self._unfreeze_all_bottles()
        self._belt_bottle_indices = deque()
        self._pending_bottles = deque()
        self._conveyor_stopped = True
        self._bottle_at_pickup = None

        # Normalise sequential mode into an explicit packed dict
        if packed is None:
            packed = {i: i for i in range(num_prepacked)}

        for i in range(NUM_WELLS):
            if i in packed:
                wp = well_position(packed[i])
                pos = np.array([wp[0], wp[1], wp[2] + BOTTLE_HALF_HEIGHT])
                self._set_bottle_pose(i, pos)
                self._enable_bottle_collision(i)
            else:
                self._set_bottle_pose(i, BOTTLE_HIDDEN_POS)
                self._disable_bottle_collision(i)

        # Placeholder: overwritten by spawn_bottle_on_conveyor / load_conveyor.
        # Uses max(keys)+1 as a safe default for freeze_inactive_bottles.
        self._active_bottle = max(packed.keys()) + 1 if packed else 0

        self._freeze_inactive_bottles()
        mujoco.mj_forward(self.model, self.data)

    def spawn_bottle_on_conveyor(self, bottle_idx: int, y_offset: float = 0.0) -> None:
        """Place a single bottle at the conveyor start and set it as active.

        Freezes all other bottles and unfreezes the new one.

        Args:
            bottle_idx: Bottle index (0–19) to spawn.
            y_offset: Lateral offset from belt centre (metres).
        """
        self._active_bottle = bottle_idx
        self._belt_bottle_indices = deque([bottle_idx])

        # Freeze everything, then unfreeze only the active bottle
        self._freeze_inactive_bottles()
        self._unfreeze_bottle(bottle_idx)
        self._enable_bottle_collision(bottle_idx)
        start = BOTTLE_CONVEYOR_START.copy()
        start[1] += y_offset
        self._set_bottle_pose(bottle_idx, start)
        mujoco.mj_forward(self.model, self.data)

    def setup_bottles(self, target_well: int) -> None:
        """Set up scene and place active bottle on conveyor (legacy helper).

        Equivalent to ``setup_scene(target_well)`` followed by
        ``spawn_bottle_on_conveyor(target_well)``.

        Args:
            target_well: The well index (0–19) that the robot will pack.
        """
        self.setup_scene(num_prepacked=target_well)
        self.spawn_bottle_on_conveyor(target_well)

    def spawn_bottle_at_pickup(self, bottle_idx: int, y_offset: float = 0.0) -> None:
        """Place a bottle directly at the pickup position, ready for grasping.

        Skips the conveyor animation entirely — the robot can pick immediately.

        Args:
            bottle_idx: Bottle index (0–19) to spawn.
            y_offset: Lateral offset from pickup centre (metres).
        """
        self._active_bottle = bottle_idx
        self._belt_bottle_indices = deque()

        self._freeze_inactive_bottles()
        self._unfreeze_bottle(bottle_idx)
        self._enable_bottle_collision(bottle_idx)
        pos = BOTTLE_PICKUP_POS.copy()
        pos[1] += y_offset
        self._set_bottle_pose(bottle_idx, pos)
        mujoco.mj_forward(self.model, self.data)

    def animate_conveyor(self) -> None:
        """Slide all belt bottles forward, delivering the active bottle to pickup.

        Every bottle on the belt advances by the same displacement
        (pickup − start).  Uses kinematic interpolation with smooth
        ease-in-out.  Syncs the viewer each frame when attached.
        """
        travel = BOTTLE_PICKUP_POS - BOTTLE_CONVEYOR_START  # displacement vector
        n = CONVEYOR_ANIM_STEPS

        # Record each belt bottle's starting position
        belt_starts: list[np.ndarray] = []
        for idx in self._belt_bottle_indices:
            adr = self._bottle_qpos_adr[idx]
            belt_starts.append(self.data.qpos[adr : adr + 3].copy())

        for i in range(n):
            t = (i + 1) / n
            t = t * t * (3.0 - 2.0 * t)  # hermite ease-in-out
            offset = travel * t

            for j, idx in enumerate(self._belt_bottle_indices):
                adr = self._bottle_qpos_adr[idx]
                vadr = self._bottle_qvel_adr[idx]
                self.data.qpos[adr : adr + 3] = belt_starts[j] + offset
                self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
                self.data.qvel[vadr : vadr + 6] = 0

            mujoco.mj_forward(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()

        # Zero all belt bottle velocities
        for idx in self._belt_bottle_indices:
            vadr = self._bottle_qvel_adr[idx]
            self.data.qvel[vadr : vadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

        # Freeze all inactive bottles (belt queue + hidden) so they stay put
        self._freeze_inactive_bottles()

    # ------------------------------------------------------------------
    # Bottle colours
    # ------------------------------------------------------------------

    def colorize_bottles(self) -> None:
        """Assign each bottle a unique colour from the palette.

        Clears the material reference so ``geom_rgba`` is used directly.
        """
        for i, geom_ids in enumerate(self._bottle_geom_ids):
            color = BOTTLE_COLORS[i % len(BOTTLE_COLORS)]
            for gid in geom_ids:
                self.model.geom_matid[gid] = -1  # stop using shared material
                self.model.geom_rgba[gid] = color

    # ------------------------------------------------------------------
    # Tick-based conveyor (concurrent with FSM)
    # ------------------------------------------------------------------

    def load_conveyor(
        self,
        bottle_indices: list[int],
        rng: np.random.Generator | None = None,
        y_noise: float = BELT_Y_NOISE,
    ) -> None:
        """Load bottles into the conveyor queue.

        Places the first batch on the belt at staggered positions.
        Call after ``setup_scene``.

        Args:
            bottle_indices: Ordered list of bottle indices to deliver.
            rng: Random generator for Y position noise. If *None* and
                *y_noise* > 0, a default generator is created.
            y_noise: Half-range of uniform Y offset (metres). Set to 0
                to disable.
        """
        self._pending_bottles = deque(bottle_indices)
        self._belt_bottle_indices = deque()
        self._conveyor_stopped = False
        self._bottle_at_pickup = None
        self._belt_y_noise = y_noise
        if y_noise > 0 and rng is None:
            rng = np.random.default_rng()
        self._belt_rng = rng

        # Spawn initial batch spread across the belt, front to back.
        # The first bottle starts one spacing behind the pickup position,
        # subsequent ones are evenly spaced behind it.
        n_initial = min(MAX_BELT_BOTTLES, len(self._pending_bottles))
        front_x = BOTTLE_PICKUP_POS[0] - CONVEYOR_BOTTLE_SPACING
        for i in range(n_initial):
            idx = self._pending_bottles.popleft()
            x = front_x - i * CONVEYOR_BOTTLE_SPACING
            y = BOTTLE_CONVEYOR_START[1]
            if self._belt_y_noise > 0 and self._belt_rng is not None:
                y += self._belt_rng.uniform(-self._belt_y_noise, self._belt_y_noise)
            pos = np.array([x, y, BOTTLE_CONVEYOR_START[2]])
            self._enable_bottle_collision(idx)
            self._set_bottle_pose(idx, pos)
            self._belt_bottle_indices.append(idx)
            self._freeze_bottle(idx)

        mujoco.mj_forward(self.model, self.data)

    def _spawn_next_on_belt(self) -> None:
        """Spawn the next pending bottle behind the last one on the belt."""
        if not self._pending_bottles:
            return
        idx = self._pending_bottles.popleft()

        if self._belt_bottle_indices:
            last = self._belt_bottle_indices[-1]
            adr_last = self._bottle_qpos_adr[last]
            spawn_x = self.data.qpos[adr_last] - CONVEYOR_BOTTLE_SPACING
        else:
            spawn_x = BOTTLE_CONVEYOR_START[0]

        y = BOTTLE_CONVEYOR_START[1]
        if self._belt_y_noise > 0 and self._belt_rng is not None:
            y += self._belt_rng.uniform(-self._belt_y_noise, self._belt_y_noise)

        pos = np.array([spawn_x, y, BOTTLE_CONVEYOR_START[2]])
        self._enable_bottle_collision(idx)
        self._set_bottle_pose(idx, pos)
        self._belt_bottle_indices.append(idx)
        # Belt bottles stay frozen (kinematic positioning)
        self._freeze_bottle(idx)

    def tick_conveyor(self, steps: int = 1) -> int | None:
        """Advance all belt bottles by *steps* ticks.

        Call once per control step, **after** all physics sub-steps.

        Args:
            steps: Number of ticks to advance (typically ``ACTION_REPEAT``).

        Returns:
            The bottle index that just arrived at pickup, or *None*.
        """
        if self._conveyor_stopped or not self._belt_bottle_indices:
            return None

        # Advance every bottle on the belt (clamp to pickup X)
        pickup_x = BOTTLE_PICKUP_POS[0]
        for idx in self._belt_bottle_indices:
            adr = self._bottle_qpos_adr[idx]
            self.data.qpos[adr] = min(
                self.data.qpos[adr] + CONVEYOR_SPEED * steps, pickup_x
            )
            vadr = self._bottle_qvel_adr[idx]
            self.data.qvel[vadr : vadr + 6] = 0
            self.data.qacc[vadr : vadr + 6] = 0

        # Check if front bottle reached pickup
        front = self._belt_bottle_indices[0]
        adr = self._bottle_qpos_adr[front]
        if self.data.qpos[adr] >= pickup_x:
            # Snap Z to pickup surface, keep randomised Y
            self.data.qpos[adr + 2] = BOTTLE_PICKUP_POS[2]
            self._conveyor_stopped = True
            self._bottle_at_pickup = front

            # Remove from belt queue; bottle stays frozen at pickup
            # until caller invokes start_pickup().
            self._belt_bottle_indices.popleft()

            mujoco.mj_forward(self.model, self.data)
            return front

        # Spawn more bottles at the back if there's room
        if self._belt_bottle_indices and self._pending_bottles:
            last = self._belt_bottle_indices[-1]
            adr_last = self._bottle_qpos_adr[last]
            if (
                self.data.qpos[adr_last]
                >= BOTTLE_CONVEYOR_START[0] + CONVEYOR_BOTTLE_SPACING
            ):
                self._spawn_next_on_belt()

        mujoco.mj_forward(self.model, self.data)
        return None

    def resume_conveyor(self) -> None:
        """Resume belt movement (call after robot lifts the bottle clear)."""
        self._conveyor_stopped = False

    def start_pickup(self) -> int | None:
        """Activate the bottle waiting at pickup for the robot to grasp.

        Sets it as the active bottle, unfreezes its physics, and clears
        the waiting state.  Returns the bottle index, or *None* if no
        bottle is waiting.
        """
        idx = self._bottle_at_pickup
        if idx is None:
            return None
        self._bottle_at_pickup = None
        self._active_bottle = idx
        self._unfreeze_bottle(idx)
        mujoco.mj_forward(self.model, self.data)
        return idx

    def mark_bottle_packed(self, bottle_idx: int) -> None:
        """Freeze a placed bottle in its well."""
        self._freeze_bottle(bottle_idx)

    @property
    def conveyor_stopped(self) -> bool:
        """True when the belt is paused (front bottle at pickup)."""
        return self._conveyor_stopped

    @property
    def bottle_at_pickup(self) -> int | None:
        """Index of the bottle waiting at pickup, or None."""
        return self._bottle_at_pickup

    def step(self) -> None:
        """Advance simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

    def sync(self) -> None:
        """Sync viewer with current simulation state."""
        if self.viewer is not None:
            self.viewer.sync()

    def is_running(self) -> bool:
        """Return True if the viewer is open (or no viewer is attached)."""
        if self.viewer is None:
            return True
        return self.viewer.is_running()

    def _resolve_body_id(self, name: str) -> int:
        """Look up body ID by name, caching the result."""
        bid = self._body_id_cache.get(name)
        if bid is None:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body '{name}' not found")
            self._body_id_cache[name] = bid
        return bid

    def get_body_pos(self, name: str) -> np.ndarray:
        """Return world position (3,) of a named body."""
        return self.data.xpos[self._resolve_body_id(name)].copy()

    def get_body_xmat(self, name: str) -> np.ndarray:
        """Return rotation matrix (3, 3) of a named body."""
        return self.data.xmat[self._resolve_body_id(name)].reshape(3, 3).copy()

    @property
    def crate_displacement(self) -> np.ndarray:
        """Return crate displacement [dx, dy, dtheta] from its rest position."""
        if not self._crate_is_dynamic:
            return np.zeros(3, dtype=np.float64)
        return np.array(
            [self.data.qpos[a] for a in self._crate_qpos_adr], dtype=np.float64
        )
