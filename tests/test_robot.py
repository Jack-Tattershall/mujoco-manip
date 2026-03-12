"""Tests for PandaRobot, including the ee_force_torque property."""

import numpy as np
import pytest

from mujoco_manip.robot import PandaRobot
from mujoco_manip.tasks.bottle_packing.env import BottlePackingEnv
from mujoco_manip.tasks.bottle_packing.gym_env import BottlePackingGymEnv


@pytest.fixture
def env():
    """Create a BottlePackingEnv to get a valid model/data pair."""
    return BottlePackingEnv(add_wrist_camera=False)


@pytest.fixture
def robot_noisy(env):
    rng = np.random.default_rng(42)
    return PandaRobot(env.model, env.data, ft_noise=True, rng=rng)


@pytest.fixture
def robot_clean(env):
    return PandaRobot(env.model, env.data, ft_noise=False)


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


class TestBasicProperties:
    def test_ee_pos_shape(self, robot_noisy):
        assert robot_noisy.ee_pos.shape == (3,)

    def test_ee_xmat_shape(self, robot_noisy):
        assert robot_noisy.ee_xmat.shape == (3, 3)

    def test_arm_qpos_shape(self, robot_noisy):
        assert robot_noisy.arm_qpos.shape == (7,)


# ---------------------------------------------------------------------------
# Force-torque sensor
# ---------------------------------------------------------------------------


class TestForceTorque:
    def test_shape(self, robot_noisy):
        ft = robot_noisy.ee_force_torque
        assert ft.shape == (6,)

    def test_dtype(self, robot_noisy):
        ft = robot_noisy.ee_force_torque
        assert ft.dtype == np.float32

    def test_no_noise_deterministic(self, robot_clean):
        """Without noise, successive reads from the same state must be identical."""
        ft1 = robot_clean.ee_force_torque
        ft2 = robot_clean.ee_force_torque
        np.testing.assert_array_equal(ft1, ft2)

    def test_noise_differs_between_reads(self, robot_noisy):
        """With noise enabled, two reads should differ (extremely unlikely to match)."""
        ft1 = robot_noisy.ee_force_torque
        ft2 = robot_noisy.ee_force_torque
        assert not np.array_equal(ft1, ft2)

    def test_noise_reproducible_with_seed(self, env):
        """Same seed + same state must produce the same noisy F/T readings."""
        rng1 = np.random.default_rng(99)
        robot1 = PandaRobot(env.model, env.data, ft_noise=True, rng=rng1)
        ft1 = robot1.ee_force_torque

        rng2 = np.random.default_rng(99)
        robot2 = PandaRobot(env.model, env.data, ft_noise=True, rng=rng2)
        ft2 = robot2.ee_force_torque

        np.testing.assert_array_equal(ft1, ft2)

    def test_noise_magnitude_reasonable(self, env):
        """Noise std should be close to the configured class attributes over many samples."""
        rng = np.random.default_rng(0)
        robot = PandaRobot(env.model, env.data, ft_noise=True, rng=rng)

        # Get the clean baseline
        clean_robot = PandaRobot(env.model, env.data, ft_noise=False)
        baseline = clean_robot.ee_force_torque

        samples = np.array([robot.ee_force_torque for _ in range(2000)])
        noise = samples - baseline

        force_std = noise[:, :3].std(axis=0)
        torque_std = noise[:, 3:].std(axis=0)

        np.testing.assert_allclose(force_std, PandaRobot.FT_FORCE_NOISE_STD, atol=0.05)
        np.testing.assert_allclose(
            torque_std, PandaRobot.FT_TORQUE_NOISE_STD, atol=0.005
        )

    def test_clean_vs_noisy_close(self, env):
        """Clean and noisy readings from the same state should be close (noise is small)."""
        rng = np.random.default_rng(42)
        noisy = PandaRobot(env.model, env.data, ft_noise=True, rng=rng)
        clean = PandaRobot(env.model, env.data, ft_noise=False)

        ft_noisy = noisy.ee_force_torque
        ft_clean = clean.ee_force_torque

        # Noisy reading should be within a few sigma of clean
        np.testing.assert_allclose(ft_noisy[:3], ft_clean[:3], atol=3.0)
        np.testing.assert_allclose(ft_noisy[3:], ft_clean[3:], atol=0.2)


# ---------------------------------------------------------------------------
# F/T changes during interaction
# ---------------------------------------------------------------------------


class TestForceTorqueInteraction:
    def test_ft_changes_when_grasping(self):
        """F/T values should change when the robot grasps a bottle."""
        env = BottlePackingGymEnv(
            action_mode="abs_pos",
            well_index=0,
            max_episode_steps=500,
        )
        try:
            env.reset()
            robot = env.robot

            # Record F/T at rest (no contact)
            ft_rest = robot.ee_force_torque.copy()

            # Move to bottle pickup position and close gripper
            bottle_pos = env.bottle_packing_env.get_body_pos(
                env.bottle_packing_env.active_bottle_body
            )
            # Approach from above
            for _ in range(20):
                action = np.array(
                    [bottle_pos[0], bottle_pos[1], 0.46, 1.0], dtype=np.float32
                )
                env.step(action)

            # Descend to grasp height
            for _ in range(20):
                action = np.array(
                    [bottle_pos[0], bottle_pos[1], 0.38, 1.0], dtype=np.float32
                )
                env.step(action)

            # Close gripper on bottle
            for _ in range(15):
                action = np.array(
                    [bottle_pos[0], bottle_pos[1], 0.38, 0.0], dtype=np.float32
                )
                env.step(action)

            # Lift - this should produce noticeable F/T from bottle weight
            for _ in range(20):
                action = np.array(
                    [bottle_pos[0], bottle_pos[1], 0.55, 0.0], dtype=np.float32
                )
                obs, *_ = env.step(action)

            ft_grasp = obs["state.ee.force_torque"].copy()

            # The F/T readings should differ from rest
            # (gravity of bottle + contact forces)
            assert not np.allclose(ft_rest, ft_grasp, atol=0.1), (
                "F/T should change when holding a bottle"
            )
        finally:
            env.close()
