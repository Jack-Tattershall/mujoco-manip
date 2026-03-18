"""Bottle-packing state machine: pick bottle from conveyor, place in crate well."""

from enum import Enum, auto

import numpy as np

from mujoco_manip.controller import IKController
from mujoco_manip.robot import PandaRobot

from .constants import GRASP_VERIFY_Z, well_position
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
    INSERT_INTO_WELL = auto()
    INSERT_SETTLE = auto()
    RELEASE = auto()
    RELEASE_WAIT = auto()
    RELEASE_LIFT = auto()
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
    State.INSERT_INTO_WELL: Phase.PLACING,
    State.INSERT_SETTLE: Phase.PLACING,
    State.RELEASE: Phase.PLACING,
    State.RELEASE_WAIT: Phase.PLACING,
    State.RELEASE_LIFT: Phase.PLACING,
    State.RETREAT_UP: Phase.RETREATING,
    State.RETREAT: Phase.RETREATING,
    State.DONE: Phase.DONE,
}

# Heights for the 'hand' body frame (finger-pad offset ~0.10m)
# Conveyor surface at z=0.24, bottle center at z=0.28
PRE_GRASP_HEIGHT = 0.47  # hover above bottle
GRASP_HEIGHT = 0.42  # finger pads at bottle neck (pads at ~0.317)
LIFT_HEIGHT = 0.55
TRANSIT_HEIGHT = 0.55  # lateral move height (clears crate walls)
APPROACH_WELL_HEIGHT = 0.49  # bottle bottom (hand-0.16) clears outer wall tops (0.322)
INSERT_HEIGHT_LIMIT = 0.38  # safety floor: never descend below this

# F/T sensor: during free descent Fz ≈ −7 to −8 N (bottle weight).
# On wall/floor contact, friction supports the bottle and Fz rises to ~−1 N.
# Trigger when Fz exceeds this threshold (clear load relief = contact).
INSERT_FZ_CONTACT = -2.0  # Newtons
INSERT_FZ_SKIP = 100  # ignore F/T for first N steps (arm dynamics settling)

# Settle times (simulation steps)
GRIPPER_CLOSE_STEPS = 150  # time for gripper to close on bottle
GRIPPER_OPEN_STEPS = 100  # time to open gripper
GRIPPER_OPEN_WAIT = 100  # hold with gripper open before moving
WELL_SETTLE_STEPS = 150  # let bottle stop swinging after transit
INSERT_SETTLE_STEPS = 100  # hold at insertion depth before release

# Maximum EE target speed during transit (m per physics step)
LIFT_SPEED = 0.0003  # slow lift to prevent bottle slipping from grip
TRANSIT_SPEED = 0.0005  # slow transit to reduce bottle swing
INSERTION_SPEED = 0.00015  # very slow descent into well


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
        self._bottle_body: str = env.active_bottle_body  # snapshot at construction
        self.state: State = State.IDLE
        self.settle_counter: int = 0
        self._target_pos: np.ndarray | None = None
        self._transit_end: np.ndarray | None = None
        self._gripper_cmd: float = 1.0  # 0.0 = closed, 1.0 = open
        self._initial_ee_pos: np.ndarray = robot.ee_pos.copy()
        self._grasp_retries: int = 0
        self._max_grasp_retries: int = 3
        # Interpolation targets (set during state transitions)
        self._lift_end: np.ndarray = np.zeros(3)
        self._transit_end: np.ndarray | None = None
        self._insert_end: np.ndarray = np.zeros(3)
        self._insert_step_count: int = 0

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
        """Return gripper command: 0.0 = closed, 1.0 = fully open."""
        return self._gripper_cmd

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
        return self.env.get_body_pos(self._bottle_body)[:2]

    def _well_xy(self) -> np.ndarray:
        """Return XY position (2,) of the target well."""
        return well_position(self._well_index)[:2]

    def _interpolate_toward(
        self, goal: np.ndarray, speed: float, n_steps: int
    ) -> float:
        """Move ``_target_pos`` toward *goal* at *speed* per step.

        Returns the remaining distance after the move.
        """
        diff = goal - self._target_pos
        dist = np.linalg.norm(diff)
        step = speed * n_steps
        if dist > step:
            self._target_pos += diff * (step / dist)
        else:
            self._target_pos = goal.copy()
        return dist

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
            self._gripper_cmd = 1.0
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
                self._gripper_cmd = 0.0
                self.settle_counter = GRIPPER_CLOSE_STEPS
                self.state = State.CLOSE_GRIPPER
                return "Closing gripper on bottle"
            return "Descending to bottle"

        elif self.state == State.CLOSE_GRIPPER:
            self.settle_counter -= n_steps
            if self.settle_counter <= 0:
                bottle_xy = self._bottle_xy()
                self._lift_end = np.array([bottle_xy[0], bottle_xy[1], LIFT_HEIGHT])
                self._target_pos = self._target_pos.copy()
                self.state = State.LIFT
                return "Lifting bottle"
            return f"Gripping bottle ({self.settle_counter})"

        elif self.state == State.LIFT:
            # Interpolated lift to prevent bottle slipping
            dist = self._interpolate_toward(self._lift_end, LIFT_SPEED, n_steps)
            if dist <= self.controller.pos_tolerance:
                # Verify bottle was actually lifted
                bottle_z = self.env.get_body_pos(self._bottle_body)[2]
                if (
                    bottle_z < GRASP_VERIFY_Z
                    and self._grasp_retries < self._max_grasp_retries
                ):
                    # Bottle still near belt — grasp failed, retry
                    self._grasp_retries += 1
                    self._gripper_cmd = 1.0
                    self.state = State.IDLE
                    return f"Grasp failed, retry {self._grasp_retries}"

                well_xy = self._well_xy()
                self._transit_end = np.array([well_xy[0], well_xy[1], TRANSIT_HEIGHT])
                self._target_pos = self._target_pos.copy()
                self.state = State.MOVE_TO_WELL
                return "Moving bottle to target well"
            return "Lifting bottle"

        elif self.state == State.MOVE_TO_WELL:
            dist = self._interpolate_toward(self._transit_end, TRANSIT_SPEED, n_steps)
            if dist <= self.controller.pos_tolerance:
                self.settle_counter = WELL_SETTLE_STEPS
                self.state = State.SETTLE_AT_WELL
                return "Settling above target well"
            return "Transporting bottle"

        elif self.state == State.SETTLE_AT_WELL:
            self.settle_counter -= n_steps
            if self.settle_counter <= 0:
                well_xy = self._well_xy()
                self._target_pos = np.array(
                    [well_xy[0], well_xy[1], APPROACH_WELL_HEIGHT]
                )
                self.state = State.LOWER_TO_WELL
                return "Lowering bottle above well"
            return f"Settling above well ({self.settle_counter})"

        elif self.state == State.LOWER_TO_WELL:
            if self.controller.reached(self._target_pos):
                well_xy = self._well_xy()
                self._insert_end = np.array(
                    [well_xy[0], well_xy[1], INSERT_HEIGHT_LIMIT]
                )
                self._target_pos = self._target_pos.copy()
                self._insert_step_count = 0
                self.state = State.INSERT_INTO_WELL
                return "Inserting bottle into well"
            return "Lowering above well"

        elif self.state == State.INSERT_INTO_WELL:
            self._insert_step_count += n_steps

            # Check F/T sensor for contact after initial settling period.
            # During free descent Fz ≈ −7 N; on floor contact Fz flips to +20 N+.
            if self._insert_step_count > INSERT_FZ_SKIP:
                fz = float(self.robot.ee_force_torque[2])
                if fz > INSERT_FZ_CONTACT:
                    self.settle_counter = INSERT_SETTLE_STEPS
                    self.state = State.INSERT_SETTLE
                    return "Contact detected — holding at insertion depth"

            dist = self._interpolate_toward(self._insert_end, INSERTION_SPEED, n_steps)
            if dist <= self.controller.pos_tolerance:
                # Reached height limit without F/T trigger — release anyway
                self.settle_counter = INSERT_SETTLE_STEPS
                self.state = State.INSERT_SETTLE
                return "Holding at insertion depth"
            return "Inserting into well"

        elif self.state == State.INSERT_SETTLE:
            self.settle_counter -= n_steps
            if self.settle_counter <= 0:
                self.settle_counter = GRIPPER_OPEN_STEPS
                self.state = State.RELEASE
                return "Releasing bottle into well"
            return f"Holding at depth ({self.settle_counter})"

        elif self.state == State.RELEASE:
            # Gradually open gripper over GRIPPER_OPEN_STEPS
            self.settle_counter -= n_steps
            t = 1.0 - max(self.settle_counter, 0) / GRIPPER_OPEN_STEPS
            self._gripper_cmd = t  # ramp 0→1
            if self.settle_counter <= 0:
                self._gripper_cmd = 1.0
                self.settle_counter = GRIPPER_OPEN_WAIT
                self.state = State.RELEASE_WAIT
                return "Waiting for bottle to settle"
            return "Opening gripper"

        elif self.state == State.RELEASE_WAIT:
            self.settle_counter -= n_steps
            if self.settle_counter <= 0:
                ee = self.robot.ee_pos.copy()
                self._target_pos = np.array([ee[0], ee[1], APPROACH_WELL_HEIGHT])
                self.state = State.RELEASE_LIFT
                return "Lifting clear of bottle"
            return f"Waiting ({self.settle_counter})"

        elif self.state == State.RELEASE_LIFT:
            if self.controller.reached(self._target_pos):
                ee_xy = self.robot.ee_pos[:2]
                self._target_pos = np.array([ee_xy[0], ee_xy[1], TRANSIT_HEIGHT])
                self.state = State.RETREAT_UP
                return "Lifting above crate"
            return "Lifting clear of bottle"

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
        # Set gripper to interpolated position (0=closed, 255=open)
        self.robot.data.ctrl[self.robot.NUM_ARM_JOINTS] = (
            self._gripper_cmd * PandaRobot.GRIPPER_OPEN
        )

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
