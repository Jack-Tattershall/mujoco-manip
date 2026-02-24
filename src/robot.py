"""Panda robot control interface."""

import mujoco
import numpy as np


class PandaRobot:
    """Wraps joint and gripper control for the Franka Panda."""

    NUM_ARM_JOINTS = 7
    GRIPPER_OPEN = 255.0
    GRIPPER_CLOSED = 0.0
    EE_BODY_NAME = "hand"

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._ee_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.EE_BODY_NAME
        )

    @property
    def ee_pos(self) -> np.ndarray:
        """Current end-effector position."""
        return self.data.xpos[self._ee_body_id].copy()

    @property
    def ee_xmat(self) -> np.ndarray:
        """Current end-effector 3x3 rotation matrix."""
        return self.data.xmat[self._ee_body_id].reshape(3, 3).copy()

    @property
    def arm_qpos(self) -> np.ndarray:
        """Current arm joint positions (7,)."""
        return self.data.qpos[: self.NUM_ARM_JOINTS].copy()

    def set_arm_ctrl(self, targets: np.ndarray):
        """Set arm actuator control targets (7 values)."""
        self.data.ctrl[: self.NUM_ARM_JOINTS] = targets

    def open_gripper(self):
        """Command gripper to open."""
        self.data.ctrl[self.NUM_ARM_JOINTS] = self.GRIPPER_OPEN

    def close_gripper(self):
        """Command gripper to close."""
        self.data.ctrl[self.NUM_ARM_JOINTS] = self.GRIPPER_CLOSED

    @property
    def gripper_ctrl(self) -> float:
        """Current gripper control value."""
        return self.data.ctrl[self.NUM_ARM_JOINTS]
