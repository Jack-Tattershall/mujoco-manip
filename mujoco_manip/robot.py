"""Panda robot control interface."""

import mujoco
import numpy as np


class PandaRobot:
    """Wraps joint and gripper control for the Franka Panda.

    Attributes:
        NUM_ARM_JOINTS: Number of arm joints.
        GRIPPER_OPEN: Actuator value for fully open gripper.
        GRIPPER_CLOSED: Actuator value for fully closed gripper.
        EE_BODY_NAME: MuJoCo body name for the end-effector.
    """

    NUM_ARM_JOINTS = 7
    GRIPPER_OPEN = 255.0
    GRIPPER_CLOSED = 0.0
    EE_BODY_NAME = "hand"

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialise the robot interface.

        Args:
            model: MuJoCo model.
            data: MuJoCo data.
        """
        self.model = model
        self.data = data
        self._ee_body_id: int = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.EE_BODY_NAME
        )

    @property
    def ee_pos(self) -> np.ndarray:
        """End-effector position (3,)."""
        return self.data.xpos[self._ee_body_id].copy()

    @property
    def ee_xmat(self) -> np.ndarray:
        """End-effector rotation matrix (3, 3)."""
        return self.data.xmat[self._ee_body_id].reshape(3, 3).copy()

    @property
    def arm_qpos(self) -> np.ndarray:
        """Arm joint positions (7,)."""
        return self.data.qpos[: self.NUM_ARM_JOINTS].copy()

    def set_arm_ctrl(self, targets: np.ndarray) -> None:
        """Set arm actuator control targets.

        Args:
            targets: Joint position targets (7,).
        """
        self.data.ctrl[: self.NUM_ARM_JOINTS] = targets

    def open_gripper(self) -> None:
        """Set gripper to fully open."""
        self.data.ctrl[self.NUM_ARM_JOINTS] = self.GRIPPER_OPEN

    def close_gripper(self) -> None:
        """Set gripper to fully closed."""
        self.data.ctrl[self.NUM_ARM_JOINTS] = self.GRIPPER_CLOSED

    @property
    def gripper_ctrl(self) -> float:
        """Current gripper actuator value."""
        return self.data.ctrl[self.NUM_ARM_JOINTS]
