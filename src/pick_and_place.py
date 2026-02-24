"""Pick-and-place state machine for 3 colored objects."""

from enum import Enum, auto

import numpy as np

from .env import PickPlaceEnv
from .robot import PandaRobot
from .controller import IKController


class State(Enum):
    IDLE = auto()
    PRE_GRASP = auto()
    GRASP = auto()
    CLOSE_GRIPPER = auto()
    LIFT = auto()
    MOVE_TO_BIN = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()


# Task definitions: (object body name, bin body name)
TASKS = [
    ("obj_red", "bin_red"),
    ("obj_green", "bin_green"),
    ("obj_blue", "bin_blue"),
]

# Heights for the 'hand' body frame
# Hand-to-finger-pad offset ~0.10m when pointing down
# Table surface at z=0.24, object center at z=0.26
PRE_GRASP_HEIGHT = 0.44  # hover above object
GRASP_HEIGHT = 0.36  # finger pads at cube center level
LIFT_HEIGHT = 0.50
PLACE_HEIGHT = 0.45
RETREAT_HEIGHT = 0.50

# Gripper settle time (simulation steps)
GRIPPER_SETTLE_STEPS = 150


class PickAndPlaceTask:
    """Finite state machine that picks objects and places them in bins."""

    def __init__(
        self, env: PickPlaceEnv, robot: PandaRobot, controller: IKController, tasks=None
    ):
        self.env = env
        self.robot = robot
        self.controller = controller
        self._tasks = tasks or TASKS
        self.state = State.IDLE
        self.task_index = 0
        self.settle_counter = 0
        self._target_pos = None

    @property
    def is_done(self) -> bool:
        return self.state == State.DONE

    def _obj_name(self) -> str:
        return self._tasks[self.task_index][0]

    def _bin_name(self) -> str:
        return self._tasks[self.task_index][1]

    def _obj_xy(self) -> np.ndarray:
        return self.env.get_body_pos(self._obj_name())[:2]

    def _bin_xy(self) -> np.ndarray:
        return self.env.get_body_pos(self._bin_name())[:2]

    def update(self) -> str:
        """Advance the state machine by one tick. Returns a status string."""

        if self.state == State.IDLE:
            if self.task_index >= len(self._tasks):
                self.state = State.DONE
                return "All objects placed!"
            self.robot.open_gripper()
            obj_xy = self._obj_xy()
            self._target_pos = np.array([obj_xy[0], obj_xy[1], PRE_GRASP_HEIGHT])
            self.state = State.PRE_GRASP
            return f"Moving to pre-grasp above {self._obj_name()}"

        elif self.state == State.PRE_GRASP:
            q = self.controller.compute(self._target_pos)
            self.robot.set_arm_ctrl(q)
            if self.controller.reached(self._target_pos):
                obj_xy = self._obj_xy()
                self._target_pos = np.array([obj_xy[0], obj_xy[1], GRASP_HEIGHT])
                self.state = State.GRASP
                return f"Descending to grasp {self._obj_name()}"
            return f"Approaching pre-grasp for {self._obj_name()}"

        elif self.state == State.GRASP:
            q = self.controller.compute(self._target_pos)
            self.robot.set_arm_ctrl(q)
            if self.controller.reached(self._target_pos):
                self.robot.close_gripper()
                self.settle_counter = GRIPPER_SETTLE_STEPS
                self.state = State.CLOSE_GRIPPER
                return f"Closing gripper on {self._obj_name()}"
            return f"Descending to {self._obj_name()}"

        elif self.state == State.CLOSE_GRIPPER:
            # Keep arm steady while gripper closes
            q = self.controller.compute(self._target_pos)
            self.robot.set_arm_ctrl(q)
            self.settle_counter -= 1
            if self.settle_counter <= 0:
                obj_xy = self._obj_xy()
                self._target_pos = np.array([obj_xy[0], obj_xy[1], LIFT_HEIGHT])
                self.state = State.LIFT
                return f"Lifting {self._obj_name()}"
            return f"Gripping {self._obj_name()} ({self.settle_counter})"

        elif self.state == State.LIFT:
            q = self.controller.compute(self._target_pos)
            self.robot.set_arm_ctrl(q)
            if self.controller.reached(self._target_pos):
                bin_xy = self._bin_xy()
                self._target_pos = np.array([bin_xy[0], bin_xy[1], PLACE_HEIGHT])
                self.state = State.MOVE_TO_BIN
                return f"Moving {self._obj_name()} to {self._bin_name()}"
            return f"Lifting {self._obj_name()}"

        elif self.state == State.MOVE_TO_BIN:
            q = self.controller.compute(self._target_pos)
            self.robot.set_arm_ctrl(q)
            if self.controller.reached(self._target_pos):
                self.robot.open_gripper()
                self.settle_counter = GRIPPER_SETTLE_STEPS
                self.state = State.RELEASE
                return f"Releasing {self._obj_name()} into {self._bin_name()}"
            return f"Transporting {self._obj_name()}"

        elif self.state == State.RELEASE:
            # Keep arm steady while gripper opens
            q = self.controller.compute(self._target_pos)
            self.robot.set_arm_ctrl(q)
            self.settle_counter -= 1
            if self.settle_counter <= 0:
                self._target_pos = np.array([0.0, 0.3, RETREAT_HEIGHT])
                self.state = State.RETREAT
                return "Retreating to neutral position"
            return f"Releasing ({self.settle_counter})"

        elif self.state == State.RETREAT:
            q = self.controller.compute(self._target_pos)
            self.robot.set_arm_ctrl(q)
            if self.controller.reached(self._target_pos):
                self.task_index += 1
                self.state = State.IDLE
                return "Ready for next object"
            return "Retreating"

        elif self.state == State.DONE:
            return "All objects placed!"

        return ""
