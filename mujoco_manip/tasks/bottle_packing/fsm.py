"""Bottle-packing state machine: pick bottle from conveyor, place in crate well."""

from enum import Enum, auto

import numpy as np

from mujoco_manip.controller import IKController
from mujoco_manip.robot import PandaRobot

from .constants import well_position
from .env import BottlePackingEnv


class State(Enum):
    IDLE = auto()
    PRE_GRASP = auto()
    GRASP = auto()
    CLOSE_GRIPPER = auto()
    LIFT = auto()
    MOVE_TO_WELL = auto()
    SETTLE_AT_WELL = auto()
    LOWER_TO_WELL = auto()
    RELEASE = auto()
    RETREAT_UP = auto()
    RETREAT = auto()
    DONE = auto()


class Phase(Enum):
    IDLE = "idle"
    APPROACHING = "approaching"
    GRASPING = "grasping"
    LIFTING = "lifting"
    TRANSPORTING = "transporting"
    PLACING = "placing"
    RETREATING = "retreating"
    DONE = "done"


_STATE_TO_PHASE = {
    State.IDLE: Phase.IDLE,
    State.PRE_GRASP: Phase.APPROACHING,
    State.GRASP: Phase.GRASPING,
    State.CLOSE_GRIPPER: Phase.GRASPING,
    State.LIFT: Phase.LIFTING,
    State.MOVE_TO_WELL: Phase.TRANSPORTING,
    State.SETTLE_AT_WELL: Phase.TRANSPORTING,
    State.LOWER_TO_WELL: Phase.PLACING,
    State.RELEASE: Phase.PLACING,
    State.RETREAT_UP: Phase.RETREATING,
    State.RETREAT: Phase.RETREATING,
    State.DONE: Phase.DONE,
}

# Heights for the 'hand' body frame (finger-pad offset ~0.10m)
# Conveyor surface at z=0.24, bottle center at z=0.28
PRE_GRASP_HEIGHT = 0.46  # hover above bottle
GRASP_HEIGHT = 0.38  # finger pads at bottle mid-height
LIFT_HEIGHT = 0.55
TRANSIT_HEIGHT = 0.55  # lateral move height (clears crate walls)
RELEASE_HEIGHT = 0.44  # lower bottle into well (clears crate walls at ~0.29)

# Settle times (simulation steps)
GRIPPER_SETTLE_STEPS = 150
WELL_SETTLE_STEPS = 100  # let bottle stop swinging after transit

# Maximum EE target speed during transit (m per physics step)
TRANSIT_SPEED = 0.001


class BottlePackingTask:
    """Finite state machine that picks a bottle and places it in a crate well."""

    def __init__(
        self,
        env: BottlePackingEnv,
        robot: PandaRobot,
        controller: IKController,
        well_index: int = 0,
    ) -> None:
        """Initialise the task state machine.

        Args:
            env: MuJoCo environment wrapper.
            robot: Robot control interface.
            controller: IK controller for computing joint targets.
            well_index: Target well index (0–19).
        """
        self.env = env
        self.robot = robot
        self.controller = controller
        self._well_index = well_index
        self.state: State = State.IDLE
        self.settle_counter: int = 0
        self._target_pos: np.ndarray | None = None
        self._transit_end: np.ndarray | None = None
        self._gripper_open: bool = True
        self._initial_ee_pos: np.ndarray = robot.ee_pos.copy()
        self._grasp_retries: int = 0
        self._max_grasp_retries: int = 3

    @property
    def well_index(self) -> int:
        """The target well index."""
        return self._well_index

    @property
    def is_done(self) -> bool:
        """Return True if the task has been completed."""
        return self.state == State.DONE

    @property
    def phase(self) -> Phase:
        """Return the current semantic phase of the FSM."""
        return _STATE_TO_PHASE[self.state]

    @property
    def target_pos(self) -> np.ndarray | None:
        """The current EE target position (3,), or None before first plan()."""
        return self._target_pos

    @property
    def gripper_val(self) -> float:
        """Return 1.0 if gripper is open, 0.0 if closed."""
        return 1.0 if self._gripper_open else 0.0

    @property
    def phase_description(self) -> str:
        """Return a human-readable description of the current phase."""
        phase = self.phase
        if phase in (Phase.IDLE, Phase.DONE):
            return "idle"
        if phase == Phase.RETREATING:
            return "retreating to neutral position"
        row, col = divmod(self._well_index, 5)
        well_str = f"well ({row},{col})"
        match phase:
            case Phase.APPROACHING:
                return "approaching the bottle"
            case Phase.GRASPING:
                return "grasping the bottle"
            case Phase.LIFTING:
                return "lifting the bottle"
            case Phase.TRANSPORTING:
                return f"transporting the bottle to {well_str}"
            case Phase.PLACING:
                return f"placing the bottle in {well_str}"
            case _:
                return "idle"

    def _bottle_xy(self) -> np.ndarray:
        """Return XY position (2,) of the bottle."""
        return self.env.get_body_pos(self.env.active_bottle_body)[:2]

    def _well_xy(self) -> np.ndarray:
        """Return XY position (2,) of the target well."""
        return well_position(self._well_index)[:2]

    def plan(self, n_steps: int = 1) -> str:
        """Advance the state machine decisions without actuating the robot.

        Args:
            n_steps: Number of physics steps to advance timers/interpolation
                by. Use 1 for physics-step-level FSM or ``ACTION_REPEAT`` for
                gym-step-level FSM.

        Returns:
            Human-readable status string.
        """
        if self.state == State.IDLE:
            self._gripper_open = True
            bottle_xy = self._bottle_xy()
            self._target_pos = np.array([bottle_xy[0], bottle_xy[1], PRE_GRASP_HEIGHT])
            self.state = State.PRE_GRASP
            return "Moving to pre-grasp above bottle"

        elif self.state == State.PRE_GRASP:
            if self.controller.reached(self._target_pos):
                bottle_xy = self._bottle_xy()
                self._target_pos = np.array([bottle_xy[0], bottle_xy[1], GRASP_HEIGHT])
                self.state = State.GRASP
                return "Descending to grasp bottle"
            return "Approaching pre-grasp for bottle"

        elif self.state == State.GRASP:
            if self.controller.reached(self._target_pos):
                self._gripper_open = False
                self.settle_counter = GRIPPER_SETTLE_STEPS
                self.state = State.CLOSE_GRIPPER
                return "Closing gripper on bottle"
            return "Descending to bottle"

        elif self.state == State.CLOSE_GRIPPER:
            self.settle_counter -= n_steps
            if self.settle_counter <= 0:
                bottle_xy = self._bottle_xy()
                self._target_pos = np.array([bottle_xy[0], bottle_xy[1], LIFT_HEIGHT])
                self.state = State.LIFT
                return "Lifting bottle"
            return f"Gripping bottle ({self.settle_counter})"

        elif self.state == State.LIFT:
            if self.controller.reached(self._target_pos):
                # Verify bottle was actually lifted
                bottle_z = self.env.get_body_pos(self.env.active_bottle_body)[2]
                if (
                    bottle_z < GRASP_HEIGHT
                    and self._grasp_retries < self._max_grasp_retries
                ):
                    # Bottle still on belt — grasp failed, retry
                    self._grasp_retries += 1
                    self._gripper_open = True
                    self.state = State.IDLE
                    return f"Grasp failed, retry {self._grasp_retries}"

                well_xy = self._well_xy()
                self._transit_end = np.array([well_xy[0], well_xy[1], TRANSIT_HEIGHT])
                self._target_pos = self._target_pos.copy()
                self.state = State.MOVE_TO_WELL
                return "Moving bottle to target well"
            return "Lifting bottle"

        elif self.state == State.MOVE_TO_WELL:
            diff = self._transit_end - self._target_pos
            dist = np.linalg.norm(diff)
            step = TRANSIT_SPEED * n_steps
            if dist > step:
                self._target_pos += diff * (step / dist)
            else:
                self._target_pos = self._transit_end.copy()
            if dist <= self.controller.pos_tolerance:
                self.settle_counter = WELL_SETTLE_STEPS
                self.state = State.SETTLE_AT_WELL
                return "Settling above target well"
            return "Transporting bottle"

        elif self.state == State.SETTLE_AT_WELL:
            self.settle_counter -= n_steps
            if self.settle_counter <= 0:
                well_xy = self._well_xy()
                self._target_pos = np.array([well_xy[0], well_xy[1], RELEASE_HEIGHT])
                self.state = State.LOWER_TO_WELL
                return "Lowering bottle into well"
            return f"Settling above well ({self.settle_counter})"

        elif self.state == State.LOWER_TO_WELL:
            if self.controller.reached(self._target_pos):
                self._gripper_open = True
                self.settle_counter = GRIPPER_SETTLE_STEPS
                self.state = State.RELEASE
                return "Releasing bottle into well"
            return "Lowering to well"

        elif self.state == State.RELEASE:
            self.settle_counter -= n_steps
            if self.settle_counter <= 0:
                ee_xy = self.robot.ee_pos[:2]
                self._target_pos = np.array([ee_xy[0], ee_xy[1], TRANSIT_HEIGHT])
                self.state = State.RETREAT_UP
                return "Lifting above crate"
            return f"Releasing ({self.settle_counter})"

        elif self.state == State.RETREAT_UP:
            if self.controller.reached(self._target_pos):
                self._target_pos = self._initial_ee_pos.copy()
                self.state = State.RETREAT
                return "Retreating to initial position"
            return "Lifting above crate"

        elif self.state == State.RETREAT:
            if self.controller.reached(self._target_pos):
                self.state = State.DONE
                return "Bottle packed!"
            return "Retreating"

        elif self.state == State.DONE:
            return "Bottle packed!"

        return ""

    def _actuate(self) -> None:
        """Send the current target to the controller and robot."""
        if self._gripper_open:
            self.robot.open_gripper()
        else:
            self.robot.close_gripper()

        if self._target_pos is not None:
            q = self.controller.compute(self._target_pos)
            self.robot.set_arm_ctrl(q)

    def update(self) -> str:
        """Advance the state machine by one tick.

        Calls ``plan(1)`` then ``_actuate()``.

        Returns:
            Human-readable status string.
        """
        status = self.plan(1)
        self._actuate()
        return status
