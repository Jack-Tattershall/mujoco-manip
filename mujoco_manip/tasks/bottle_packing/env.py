"""MuJoCo environment wrapper for the bottle packing scene."""

import os
import tempfile

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_manip.data import PANDA_DIR as _DEFAULT_PANDA_DIR
from mujoco_manip.data import BOTTLE_PACKING_SCENE_XML as _DEFAULT_SCENE_XML

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
    MAX_BELT_BOTTLES,
    NUM_WELLS,
    well_position,
)


def _load_scene(
    xml_path: str, panda_dir: str, add_wrist_camera: bool = False
) -> mujoco.MjModel:
    """Load the scene XML, resolving robot meshes from *panda_dir*."""
    with open(xml_path) as f:
        xml = f.read()

    xml = xml.replace('file="franka_emika_panda/panda.xml"', 'file="panda.xml"')
    xml = xml.replace('<compiler angle="radian"/>\n\n', "")

    abs_panda_dir = os.path.abspath(panda_dir)
    fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=abs_panda_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(xml)

        if add_wrist_camera:
            spec = mujoco.MjSpec.from_file(tmp_path)
            hand = spec.body("hand")
            cam = hand.add_camera()
            cam.name = "wrist"
            cam.pos = [-0.07, 0.0, 0.055]
            cam.quat = [
                -0.0616,
                -0.7044,
                0.7044,
                0.0616,
            ]
            cam.fovy = 128.0
            return spec.compile()
        else:
            return mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)


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
        self.model: mujoco.MjModel = _load_scene(
            xml_path, panda_dir, add_wrist_camera=add_wrist_camera
        )
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.viewer = None

        # Cache joint qpos/qvel addresses for all 20 bottles
        self._bottle_qpos_adr: list[int] = []
        self._bottle_qvel_adr: list[int] = []
        for name in BOTTLE_BODIES:
            jnt_name = f"{name}_jnt"
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            self._bottle_qpos_adr.append(self.model.jnt_qposadr[jnt_id])
            self._bottle_qvel_adr.append(self.model.jnt_dofadr[jnt_id])

        # Cache geom and body IDs for all 20 bottles
        self._bottle_geom_ids: list[int] = []
        self._bottle_body_ids: list[int] = []
        for name in BOTTLE_BODIES:
            self._bottle_geom_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
            )
            self._bottle_body_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            )

        # Store default geom collision settings (to restore after freeze)
        self._default_contype: int = self.model.geom_contype[self._bottle_geom_ids[0]]
        self._default_conaffinity: int = self.model.geom_conaffinity[
            self._bottle_geom_ids[0]
        ]

        # Index of the active bottle (the one the robot picks)
        self._active_bottle: int = 0
        # Indices of all bottles currently on the conveyor belt
        self._belt_bottle_indices: list[int] = []

        # Tick-based conveyor state
        self._pending_bottles: list[int] = []
        self._conveyor_stopped: bool = True
        self._bottle_at_pickup: int | None = None

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
        """Set a bottle's qpos (position + quaternion) and zero its velocity."""
        adr = self._bottle_qpos_adr[bottle_idx]
        vadr = self._bottle_qvel_adr[bottle_idx]
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat if quat is not None else [1, 0, 0, 0]
        self.data.qvel[vadr : vadr + 6] = 0

    def _unfreeze_bottle(self, idx: int) -> None:
        """Restore physics for a single bottle."""
        gid = self._bottle_geom_ids[idx]
        self.model.geom_contype[gid] = self._default_contype
        self.model.geom_conaffinity[gid] = self._default_conaffinity
        bid = self._bottle_body_ids[idx]
        self.model.body_gravcomp[bid] = 0.0
        vadr = self._bottle_qvel_adr[idx]
        self.model.dof_damping[vadr : vadr + 6] = 0.0

    def _freeze_bottle(self, idx: int) -> None:
        """Make a single bottle physics-inert."""
        gid = self._bottle_geom_ids[idx]
        self.model.geom_contype[gid] = 0
        self.model.geom_conaffinity[gid] = 0
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

    def setup_scene(self, num_prepacked: int = 0) -> None:
        """Set up scene with pre-packed bottles in wells, all others hidden.

        Bottles 0..num_prepacked-1 are placed upright in their wells and
        frozen.  All remaining bottles are hidden underground.

        Call after ``reset_to_keyframe`` and before
        ``spawn_bottle_on_conveyor``.

        Args:
            num_prepacked: Number of bottles already packed in wells (0–20).
        """
        self._unfreeze_all_bottles()
        self._active_bottle = num_prepacked  # next bottle to pack
        self._belt_bottle_indices = []

        for i in range(NUM_WELLS):
            if i < num_prepacked:
                wp = well_position(i)
                pos = np.array([wp[0], wp[1], wp[2] + BOTTLE_HALF_HEIGHT])
                self._set_bottle_pose(i, pos)
            else:
                self._set_bottle_pose(i, BOTTLE_HIDDEN_POS)

        self._freeze_inactive_bottles()
        mujoco.mj_forward(self.model, self.data)

    def spawn_bottle_on_conveyor(self, bottle_idx: int) -> None:
        """Place a single bottle at the conveyor start and set it as active.

        Freezes all other bottles and unfreezes the new one.

        Args:
            bottle_idx: Bottle index (0–19) to spawn.
        """
        self._active_bottle = bottle_idx
        self._belt_bottle_indices = [bottle_idx]

        # Freeze everything, then unfreeze only the active bottle
        self._freeze_inactive_bottles()
        self._unfreeze_bottle(bottle_idx)
        self._set_bottle_pose(bottle_idx, BOTTLE_CONVEYOR_START.copy())
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
        for i, gid in enumerate(self._bottle_geom_ids):
            self.model.geom_matid[gid] = -1  # stop using shared material
            self.model.geom_rgba[gid] = BOTTLE_COLORS[i % len(BOTTLE_COLORS)]

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
        self._pending_bottles = list(bottle_indices)
        self._belt_bottle_indices = []
        self._conveyor_stopped = False
        self._bottle_at_pickup = None
        self._belt_y_noise = y_noise
        if y_noise > 0 and rng is None:
            rng = np.random.default_rng()
        self._belt_rng = rng

        # Spawn initial batch onto belt
        for _ in range(min(MAX_BELT_BOTTLES, len(self._pending_bottles))):
            self._spawn_next_on_belt()

        mujoco.mj_forward(self.model, self.data)

    def _spawn_next_on_belt(self) -> None:
        """Spawn the next pending bottle at the back of the belt."""
        if not self._pending_bottles:
            return
        idx = self._pending_bottles.pop(0)

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
        self._set_bottle_pose(idx, pos)
        self._belt_bottle_indices.append(idx)
        # Belt bottles stay frozen (kinematic positioning)
        self._freeze_bottle(idx)

    def tick_conveyor(self) -> int | None:
        """Advance all belt bottles by one step.

        Call once per physics step, **after** ``step()``.

        Returns:
            The bottle index that just arrived at pickup, or *None*.
        """
        if self._conveyor_stopped or not self._belt_bottle_indices:
            return None

        # Advance every bottle on the belt
        for idx in self._belt_bottle_indices:
            adr = self._bottle_qpos_adr[idx]
            self.data.qpos[adr] += CONVEYOR_SPEED
            vadr = self._bottle_qvel_adr[idx]
            self.data.qvel[vadr : vadr + 6] = 0

        # Check if front bottle reached pickup
        front = self._belt_bottle_indices[0]
        adr = self._bottle_qpos_adr[front]
        if self.data.qpos[adr] >= BOTTLE_PICKUP_POS[0]:
            # Snap X to pickup, keep randomised Y
            self.data.qpos[adr] = BOTTLE_PICKUP_POS[0]
            self.data.qpos[adr + 2] = BOTTLE_PICKUP_POS[2]
            self._conveyor_stopped = True
            self._bottle_at_pickup = front

            # Remove from belt, unfreeze for robot to pick
            self._belt_bottle_indices.pop(0)
            self._active_bottle = front
            self._unfreeze_bottle(front)

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
        """Consume the bottle waiting at pickup.

        Returns its index and clears the waiting state, or *None* if
        no bottle is waiting.
        """
        idx = self._bottle_at_pickup
        self._bottle_at_pickup = None
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

    def get_body_pos(self, name: str) -> np.ndarray:
        """Return world position (3,) of a named body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found")
        return self.data.xpos[body_id].copy()

    def get_body_xmat(self, name: str) -> np.ndarray:
        """Return rotation matrix (3, 3) of a named body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found")
        return self.data.xmat[body_id].reshape(3, 3).copy()
