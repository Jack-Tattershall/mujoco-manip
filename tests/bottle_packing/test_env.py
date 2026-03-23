"""Tests for the bottle packing base environment."""

import mujoco
import numpy as np
import pytest

from mujoco_manip.tasks.bottle_packing.constants import (
    BOTTLE_BODIES,
    BOTTLE_CONVEYOR_START,
    BOTTLE_PICKUP_POS,
    NUM_WELLS,
    well_position,
)
from mujoco_manip.tasks.bottle_packing.env import BottlePackingEnv


@pytest.fixture
def env():
    return BottlePackingEnv(add_wrist_camera=False)


@pytest.fixture
def env_wrist():
    return BottlePackingEnv(add_wrist_camera=True)


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

    def test_all_bottle_bodies_exist(self, env):
        for name in BOTTLE_BODIES:
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert body_id >= 0, f"Body {name} not found"

    def test_bottle_joints_cached(self, env):
        assert len(env._bottle_qpos_adr) == NUM_WELLS
        assert len(env._bottle_qvel_adr) == NUM_WELLS


# ---------------------------------------------------------------------------
# Reset & scene setup
# ---------------------------------------------------------------------------


class TestSceneSetup:
    def test_reset_to_keyframe(self, env):
        env.reset_to_keyframe("scene_start")
        assert env.data.time == 0.0

    def test_invalid_keyframe_raises(self, env):
        with pytest.raises(ValueError, match="not found"):
            env.reset_to_keyframe("nonexistent")

    def test_setup_scene_hides_bottles(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=0)
        # All bottles should be hidden underground with collision disabled
        for i in range(NUM_WELLS):
            pos = env.get_body_pos(BOTTLE_BODIES[i])
            assert pos[2] < 0, f"Bottle {i} should be hidden (z={pos[2]})"

    def test_setup_scene_prepacked(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=3)
        # First 3 bottles should be in their wells (above ground)
        for i in range(3):
            pos = env.get_body_pos(BOTTLE_BODIES[i])
            wp = well_position(i)
            np.testing.assert_allclose(pos[:2], wp[:2], atol=0.01)
            assert pos[2] > 0
        # Remaining bottles should be hidden underground
        for i in range(3, NUM_WELLS):
            pos = env.get_body_pos(BOTTLE_BODIES[i])
            assert pos[2] < 0

    def test_active_bottle_after_setup(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=5)
        assert env.active_bottle_body == BOTTLE_BODIES[5]


# ---------------------------------------------------------------------------
# Conveyor (blocking animation)
# ---------------------------------------------------------------------------


class TestConveyorAnimation:
    def test_spawn_bottle_on_conveyor(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=0)
        env.spawn_bottle_on_conveyor(0)
        pos = env.get_body_pos(BOTTLE_BODIES[0])
        np.testing.assert_allclose(pos, BOTTLE_CONVEYOR_START, atol=0.01)

    def test_animate_conveyor_delivers_to_pickup(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=0)
        env.spawn_bottle_on_conveyor(0)
        env.animate_conveyor()
        pos = env.get_body_pos(BOTTLE_BODIES[0])
        np.testing.assert_allclose(pos[:2], BOTTLE_PICKUP_POS[:2], atol=0.02)

    def test_spawn_bottle_at_pickup(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=0)
        env.spawn_bottle_at_pickup(0)
        pos = env.get_body_pos(BOTTLE_BODIES[0])
        np.testing.assert_allclose(pos[:2], BOTTLE_PICKUP_POS[:2], atol=0.01)

    def test_spawn_bottle_at_pickup_with_y_offset(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=0)
        env.spawn_bottle_at_pickup(0, y_offset=0.02)
        pos = env.get_body_pos(BOTTLE_BODIES[0])
        expected_y = BOTTLE_PICKUP_POS[1] + 0.02
        assert abs(pos[1] - expected_y) < 0.01


# ---------------------------------------------------------------------------
# Conveyor (tick-based)
# ---------------------------------------------------------------------------


class TestTickConveyor:
    def test_load_conveyor_places_bottles(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=0)
        env.load_conveyor([0, 1, 2], y_noise=0)
        # At least one bottle should be on the belt (not hidden)
        pos0 = env.get_body_pos(BOTTLE_BODIES[0])
        assert pos0[2] > 0

    def test_tick_advances_bottles(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=0)
        env.load_conveyor([0], y_noise=0)
        pos_before = env.get_body_pos(BOTTLE_BODIES[0])[0]
        env.tick_conveyor()
        pos_after = env.get_body_pos(BOTTLE_BODIES[0])[0]
        assert pos_after > pos_before

    def test_tick_delivers_to_pickup(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=0)
        env.load_conveyor([0], y_noise=0)
        arrived = None
        for _ in range(10000):
            arrived = env.tick_conveyor()
            if arrived is not None:
                break
        assert arrived == 0
        assert env.conveyor_stopped
        assert env.bottle_at_pickup == 0

    def test_start_pickup_clears_waiting(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=0)
        env.load_conveyor([0], y_noise=0)
        for _ in range(10000):
            if env.tick_conveyor() is not None:
                break
        idx = env.start_pickup()
        assert idx == 0
        assert env.bottle_at_pickup is None

    def test_resume_conveyor(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=0)
        env.load_conveyor([0, 1], y_noise=0)
        # Deliver first bottle
        for _ in range(10000):
            if env.tick_conveyor() is not None:
                break
        assert env.conveyor_stopped
        env.resume_conveyor()
        assert not env.conveyor_stopped

    def test_mark_bottle_packed(self, env):
        env.reset_to_keyframe("scene_start")
        env.setup_scene(num_prepacked=0)
        env.spawn_bottle_on_conveyor(0)
        env.animate_conveyor()
        # After marking as packed, bottle should be frozen (high damping)
        # but retain collision geometry
        env.mark_bottle_packed(0)
        bid = env._bottle_body_ids[0]
        assert env.model.body_gravcomp[bid] == 1.0
        vadr = env._bottle_qvel_adr[0]
        assert env.model.dof_damping[vadr] == 1e4


# ---------------------------------------------------------------------------
# Body queries
# ---------------------------------------------------------------------------


class TestCrateDisplacement:
    def test_shape_and_dtype(self, env):
        env.reset_to_keyframe("scene_start")
        d = env.crate_displacement
        assert d.shape == (3,)
        assert d.dtype == np.float64

    def test_starts_at_zero(self, env):
        env.reset_to_keyframe("scene_start")
        np.testing.assert_allclose(env.crate_displacement, 0.0, atol=1e-10)


class TestBodyQueries:
    def test_get_body_pos(self, env):
        env.reset_to_keyframe("scene_start")
        pos = env.get_body_pos("crate")
        assert pos.shape == (3,)

    def test_get_body_xmat(self, env):
        env.reset_to_keyframe("scene_start")
        xmat = env.get_body_xmat("crate")
        assert xmat.shape == (3, 3)

    def test_invalid_body_raises(self, env):
        with pytest.raises(ValueError, match="not found"):
            env.get_body_pos("nonexistent")

    def test_returns_copy(self, env):
        env.reset_to_keyframe("scene_start")
        p1 = env.get_body_pos("crate")
        p2 = env.get_body_pos("crate")
        assert p1 is not p2
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------


class TestStepping:
    def test_step_advances_time(self, env):
        env.reset_to_keyframe("scene_start")
        t0 = env.data.time
        env.step()
        assert env.data.time > t0

    def test_is_running_without_viewer(self, env):
        assert env.is_running()
