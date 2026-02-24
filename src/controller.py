"""Jacobian-based IK controller for the Panda arm."""

import mujoco
import numpy as np

from .robot import PandaRobot

_HOME_QPOS = np.array([1.5708, -0.2, 0.0, -2.1, 0.0, 1.8, 0.785])

# Target orientation: gripper pointing straight down (from home pose)
# x=[0,1,0], y=[1,0,0], z=[0,0,-1]
TARGET_ORI = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)


def _orientation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Compute orientation error as an axis-angle 3-vector.

    Args:
        R_current: Current rotation matrix (3, 3).
        R_target: Target rotation matrix (3, 3).

    Returns:
        Axis-angle error vector (3,).
    """
    R_err = R_target @ R_current.T
    trace_val = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
    angle = np.arccos(trace_val)
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    ) / (2 * np.sin(angle))
    return axis * angle


class IKController:
    """Damped least-squares 6-DOF IK with nullspace posture bias."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: PandaRobot,
        damping: float = 1e-3,
        pos_tolerance: float = 0.02,
        max_dq: float = 5.0,
        pos_gain: float = 1.0,
        ori_gain: float = 1.0,
        nullspace_gain: float = 0.5,
    ) -> None:
        """Initialise the IK controller.

        Args:
            model: MuJoCo model.
            data: MuJoCo data.
            robot: Robot interface for reading joint/EE state.
            damping: Damping factor for the pseudo-inverse.
            pos_tolerance: Distance threshold for ``reached()``.
            max_dq: Maximum joint velocity norm clamp.
            pos_gain: Proportional gain on position error.
            ori_gain: Proportional gain on orientation error.
            nullspace_gain: Gain for nullspace posture bias.
        """
        self.model = model
        self.data = data
        self.robot = robot
        self.damping = damping
        self.pos_tolerance = pos_tolerance
        self.max_dq = max_dq
        self.pos_gain = pos_gain
        self.ori_gain = ori_gain
        self.nullspace_gain = nullspace_gain
        self._nv: int = model.nv
        self._jacp: np.ndarray = np.zeros((3, self._nv))
        self._jacr: np.ndarray = np.zeros((3, self._nv))

    def compute(self, target_pos: np.ndarray) -> np.ndarray:
        """Compute joint angle targets to move EE toward *target_pos*.

        Maintains downward orientation via 6-DOF tracking of ``TARGET_ORI``.

        Args:
            target_pos: Desired EE position (3,) in world frame.

        Returns:
            Joint position targets (7,).
        """
        n_arm = PandaRobot.NUM_ARM_JOINTS
        ee_pos = self.robot.ee_pos

        mujoco.mj_jac(
            self.model,
            self.data,
            self._jacp,
            self._jacr,
            ee_pos,
            self.robot._ee_body_id,
        )
        Jp = self._jacp[:, :n_arm]
        Jr = self._jacr[:, :n_arm]

        pos_err = self.pos_gain * (target_pos - ee_pos)
        ori_err = self.ori_gain * _orientation_error(self.robot.ee_xmat, TARGET_ORI)
        error_6d = np.concatenate([pos_err, ori_err])
        J = np.vstack([Jp, Jr])

        JJT = J @ J.T + self.damping * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJT)
        dq = J_pinv @ error_6d

        q_current = self.robot.arm_qpos
        N = np.eye(n_arm) - J_pinv @ J
        dq += N @ (self.nullspace_gain * (_HOME_QPOS - q_current))

        dq_norm = np.linalg.norm(dq)
        if dq_norm > self.max_dq:
            dq *= self.max_dq / dq_norm

        q_target = q_current + dq

        for i in range(n_arm):
            lo = self.model.jnt_range[i, 0]
            hi = self.model.jnt_range[i, 1]
            if lo < hi:
                q_target[i] = np.clip(q_target[i], lo, hi)

        return q_target

    def reached(self, target_pos: np.ndarray) -> bool:
        """Return True if EE is within tolerance of *target_pos*.

        Args:
            target_pos: Desired EE position (3,).
        """
        return np.linalg.norm(self.robot.ee_pos - target_pos) < self.pos_tolerance
