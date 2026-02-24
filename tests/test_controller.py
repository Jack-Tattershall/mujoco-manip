"""Tests for the IK controller and robot interface."""

import os

import numpy as np
import pytest

from src.controller import IKController, TARGET_ORI
from src.env import PickPlaceEnv
from src.robot import PandaRobot

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(_PROJECT_ROOT, "pick_and_place_scene.xml")


@pytest.fixture
def sim() -> tuple[PickPlaceEnv, PandaRobot, IKController]:
    """Create a sim environment, robot, and controller, reset to keyframe."""
    env = PickPlaceEnv(SCENE_XML, add_wrist_camera=False)
    robot = PandaRobot(env.model, env.data)
    controller = IKController(env.model, env.data, robot)
    env.reset_to_keyframe("scene_start")
    return env, robot, controller


# ---------------------------------------------------------------------------
# IKController.compute
# ---------------------------------------------------------------------------


class TestCompute:
    def test_returns_correct_shape(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        _, _, controller = sim
        q = controller.compute(np.array([0.0, 0.4, 0.4]))
        assert q.shape == (7,)

    def test_within_joint_limits(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        env, _, controller = sim
        q = controller.compute(np.array([0.0, 0.4, 0.4]))
        for i in range(7):
            lo = env.model.jnt_range[i, 0]
            hi = env.model.jnt_range[i, 1]
            if lo < hi:
                assert lo - 1e-6 <= q[i] <= hi + 1e-6, (
                    f"Joint {i}: {q[i]} outside [{lo}, {hi}]"
                )

    def test_finite_output(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        _, _, controller = sim
        q = controller.compute(np.array([-0.2, 0.5, 0.35]))
        assert np.all(np.isfinite(q))


# ---------------------------------------------------------------------------
# IKController.reached
# ---------------------------------------------------------------------------


class TestReached:
    def test_reached_at_current_position(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        _, robot, controller = sim
        assert controller.reached(robot.ee_pos)

    def test_not_reached_far_target(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        _, _, controller = sim
        assert not controller.reached(np.array([10.0, 10.0, 10.0]))


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class TestConvergence:
    @pytest.mark.parametrize(
        "target",
        [
            np.array([0.0, 0.4, 0.4]),
            np.array([-0.15, 0.45, 0.36]),
            np.array([-0.3, 0.55, 0.45]),
        ],
        ids=["center", "left", "far_left"],
    )
    def test_converges_to_target(
        self,
        sim: tuple[PickPlaceEnv, PandaRobot, IKController],
        target: np.ndarray,
    ) -> None:
        """After many IK steps, EE should be close to the target."""
        env, robot, controller = sim
        for _ in range(200):
            q = controller.compute(target)
            robot.set_arm_ctrl(q)
            env.step()
        dist = np.linalg.norm(robot.ee_pos - target)
        assert dist < 0.03, f"EE at {robot.ee_pos}, target {target}, error {dist:.4f}"

    def test_error_decreases_over_steps(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        env, robot, controller = sim
        target = np.array([0.0, 0.4, 0.35])
        initial_dist = np.linalg.norm(robot.ee_pos - target)
        for _ in range(100):
            q = controller.compute(target)
            robot.set_arm_ctrl(q)
            env.step()
        final_dist = np.linalg.norm(robot.ee_pos - target)
        assert final_dist < initial_dist


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


class TestOrientation:
    def test_maintains_downward_orientation(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        """After convergence, EE rotation should be close to TARGET_ORI."""
        env, robot, controller = sim
        target = np.array([0.0, 0.4, 0.4])
        for _ in range(200):
            q = controller.compute(target)
            robot.set_arm_ctrl(q)
            env.step()
        R = robot.ee_xmat
        np.testing.assert_allclose(R, TARGET_ORI, atol=0.1)


# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------


class TestGripper:
    def test_open_sets_ctrl(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        _, robot, _ = sim
        robot.open_gripper()
        assert robot.gripper_ctrl == PandaRobot.GRIPPER_OPEN

    def test_close_sets_ctrl(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        _, robot, _ = sim
        robot.close_gripper()
        assert robot.gripper_ctrl == PandaRobot.GRIPPER_CLOSED

    def test_set_arm_ctrl_does_not_affect_gripper(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        _, robot, controller = sim
        robot.open_gripper()
        q = controller.compute(np.array([0.0, 0.4, 0.4]))
        robot.set_arm_ctrl(q)
        assert robot.gripper_ctrl == PandaRobot.GRIPPER_OPEN


# ---------------------------------------------------------------------------
# Robot properties
# ---------------------------------------------------------------------------


class TestRobotProperties:
    def test_ee_pos_shape(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        _, robot, _ = sim
        assert robot.ee_pos.shape == (3,)

    def test_ee_xmat_shape(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        _, robot, _ = sim
        assert robot.ee_xmat.shape == (3, 3)

    def test_arm_qpos_shape(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        _, robot, _ = sim
        assert robot.arm_qpos.shape == (7,)

    def test_ee_xmat_is_orthogonal(
        self, sim: tuple[PickPlaceEnv, PandaRobot, IKController]
    ) -> None:
        _, robot, _ = sim
        R = robot.ee_xmat
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
