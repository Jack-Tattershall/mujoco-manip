"""Jacobian-based IK controller for the Panda arm."""

import mujoco
import numpy as np

from .robot import PandaRobot

# Home joint config for nullspace bias
_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

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
    """Compute orientation error as a 3-vector (axis * angle)."""
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
    ):
        self.model = model
        self.data = data
        self.robot = robot
        self.damping = damping
        self.pos_tolerance = pos_tolerance
        self.max_dq = max_dq
        self.pos_gain = pos_gain
        self.ori_gain = ori_gain
        self.nullspace_gain = nullspace_gain
        self._nv = model.nv
        self._jacp = np.zeros((3, self._nv))
        self._jacr = np.zeros((3, self._nv))

    def compute(self, target_pos: np.ndarray) -> np.ndarray:
        """Compute joint angle targets to move EE toward target_pos
        while maintaining downward orientation."""
        n_arm = PandaRobot.NUM_ARM_JOINTS
        ee_pos = self.robot.ee_pos

        # Compute Jacobians
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

        # 6-DOF error: position + orientation
        pos_err = self.pos_gain * (target_pos - ee_pos)
        ori_err = self.ori_gain * _orientation_error(self.robot.ee_xmat, TARGET_ORI)
        error_6d = np.concatenate([pos_err, ori_err])
        J = np.vstack([Jp, Jr])

        # Damped pseudo-inverse
        JJT = J @ J.T + self.damping * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJT)
        dq = J_pinv @ error_6d

        # Nullspace posture bias
        q_current = self.robot.arm_qpos
        N = np.eye(n_arm) - J_pinv @ J
        dq += N @ (self.nullspace_gain * (_HOME_QPOS - q_current))

        # Clamp
        dq_norm = np.linalg.norm(dq)
        if dq_norm > self.max_dq:
            dq *= self.max_dq / dq_norm

        q_target = q_current + dq

        # Joint limits
        for i in range(n_arm):
            lo = self.model.jnt_range[i, 0]
            hi = self.model.jnt_range[i, 1]
            if lo < hi:
                q_target[i] = np.clip(q_target[i], lo, hi)

        return q_target

    def reached(self, target_pos: np.ndarray) -> bool:
        """Check if EE is within tolerance of target position."""
        return np.linalg.norm(self.robot.ee_pos - target_pos) < self.pos_tolerance
