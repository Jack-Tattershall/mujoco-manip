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
        FT_FORCE_NOISE_STD: Std dev of Gaussian noise on force channels (N).
        FT_TORQUE_NOISE_STD: Std dev of Gaussian noise on torque channels (Nm).
    """

    NUM_ARM_JOINTS = 7
    GRIPPER_OPEN = 255.0
    GRIPPER_CLOSED = 0.0
    EE_BODY_NAME = "hand"
    FT_FORCE_NOISE_STD = 0.5
    FT_TORQUE_NOISE_STD = 0.02
    BODY_NAMES = frozenset(
        {
            "link0",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "hand",
            "left_finger",
            "right_finger",
        }
    )

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ft_noise: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise the robot interface.

        Args:
            model: MuJoCo model.
            data: MuJoCo data.
            ft_noise: Whether to add simulated sensor noise to F/T readings.
            rng: Random number generator for noise. Uses default if *None*.
        """
        self.model = model
        self.data = data
        self._ft_noise = ft_noise
        self._rng = rng if rng is not None else np.random.default_rng()
        self._ee_body_id: int = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.EE_BODY_NAME
        )
        _force_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_force")
        _torque_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_torque")
        self._ee_force_sensor_id: int | None = _force_id if _force_id >= 0 else None
        self._ee_torque_sensor_id: int | None = _torque_id if _torque_id >= 0 else None

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
    def ee_force_torque(self) -> np.ndarray:
        """End-effector force-torque reading (6,): [fx, fy, fz, tx, ty, tz].

        Returns the simulated sensor value with optional Gaussian noise
        representative of a wrist-mounted 6-axis F/T sensor (e.g. ATI Mini45).
        """
        if self._ee_force_sensor_id is None or self._ee_torque_sensor_id is None:
            raise RuntimeError(
                "F/T sensors not found in model. "
                "Ensure the scene XML includes 'ee_force' and 'ee_torque' sensors."
            )
        force_adr = self.model.sensor_adr[self._ee_force_sensor_id]
        torque_adr = self.model.sensor_adr[self._ee_torque_sensor_id]
        force = self.data.sensordata[force_adr : force_adr + 3].copy()
        torque = self.data.sensordata[torque_adr : torque_adr + 3].copy()
        if self._ft_noise:
            force += self._rng.normal(0.0, self.FT_FORCE_NOISE_STD, size=3)
            torque += self._rng.normal(0.0, self.FT_TORQUE_NOISE_STD, size=3)
        return np.concatenate([force, torque]).astype(np.float32)

    @property
    def gripper_ctrl(self) -> float:
        """Current gripper actuator value."""
        return self.data.ctrl[self.NUM_ARM_JOINTS]
