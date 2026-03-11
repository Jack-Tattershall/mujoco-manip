"""Tests for the pick-and-place base environment."""

import mujoco
import numpy as np
import pytest

from mujoco_manip.data import SCENE_XML
from mujoco_manip.tasks.pick_and_place.env import PickPlaceEnv


@pytest.fixture
def env():
    return PickPlaceEnv(SCENE_XML, add_wrist_camera=False)


@pytest.fixture
def env_wrist():
    return PickPlaceEnv(SCENE_XML, add_wrist_camera=True)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_model_loaded(self, env):
        assert isinstance(env.model, mujoco.MjModel)
        assert isinstance(env.data, mujoco.MjData)

    def test_no_viewer_by_default(self, env):
        assert env.viewer is None

    def test_wrist_camera_injected(self, env_wrist):
        cam_id = mujoco.mj_name2id(env_wrist.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        assert cam_id >= 0

    def test_no_wrist_camera_without_flag(self, env):
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        assert cam_id < 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_to_keyframe(self, env):
        env.reset_to_keyframe("home")
        # After reset, data should be consistent
        mujoco.mj_forward(env.model, env.data)
        assert env.data.time == 0.0

    def test_invalid_keyframe_raises(self, env):
        with pytest.raises(ValueError, match="not found"):
            env.reset_to_keyframe("nonexistent_keyframe")


# ---------------------------------------------------------------------------
# Body queries
# ---------------------------------------------------------------------------


class TestBodyQueries:
    def test_get_body_pos_shape(self, env):
        env.reset_to_keyframe("home")
        pos = env.get_body_pos("obj_red")
        assert pos.shape == (3,)
        assert pos.dtype == np.float64

    def test_get_body_pos_returns_copy(self, env):
        env.reset_to_keyframe("home")
        pos1 = env.get_body_pos("obj_red")
        pos2 = env.get_body_pos("obj_red")
        assert pos1 is not pos2
        np.testing.assert_array_equal(pos1, pos2)

    def test_get_body_xmat_shape(self, env):
        env.reset_to_keyframe("home")
        xmat = env.get_body_xmat("obj_red")
        assert xmat.shape == (3, 3)

    def test_invalid_body_raises(self, env):
        with pytest.raises(ValueError, match="not found"):
            env.get_body_pos("nonexistent_body")


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------


class TestStepping:
    def test_step_advances_time(self, env):
        env.reset_to_keyframe("home")
        t0 = env.data.time
        env.step()
        assert env.data.time > t0

    def test_multiple_steps(self, env):
        env.reset_to_keyframe("home")
        for _ in range(100):
            env.step()
        assert env.data.time > 0

    def test_is_running_without_viewer(self, env):
        assert env.is_running()
