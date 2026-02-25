"""Tests for object position randomization."""

import mujoco
import numpy as np
import pytest

from mujoco_manip.env import PickPlaceEnv
from mujoco_manip.randomization import (
    MIN_OBJ_SEPARATION,
    OBJ_JOINT_NAMES,
    randomize_object_positions,
)

SCENE_XML = "pick_and_place_scene.xml"

X_RANGE = (-0.30, 0.30)
Y_RANGE = (0.28, 0.48)
OBJ_Z = 0.26


@pytest.fixture
def env():
    e = PickPlaceEnv(SCENE_XML, add_wrist_camera=False)
    e.reset_to_keyframe("scene_start")
    return e


def _get_obj_positions(env: PickPlaceEnv) -> dict[str, np.ndarray]:
    """Read current XYZ positions of all objects from xpos."""
    positions = {}
    for jnt_name in OBJ_JOINT_NAMES:
        body_name = jnt_name.replace("_jnt", "")
        positions[jnt_name] = env.get_body_pos(body_name)
    return positions


class TestPositionsChange:
    def test_positions_differ_from_keyframe(self, env):
        before = _get_obj_positions(env)
        rng = np.random.default_rng(42)
        env.randomize_objects(rng)
        after = _get_obj_positions(env)

        any_changed = False
        for name in OBJ_JOINT_NAMES:
            if not np.allclose(before[name][:2], after[name][:2]):
                any_changed = True
        assert any_changed, "No object positions changed after randomization"


class TestZPreserved:
    def test_z_coordinate_preserved(self, env):
        rng = np.random.default_rng(123)
        env.randomize_objects(rng)
        for name in OBJ_JOINT_NAMES:
            body_name = name.replace("_jnt", "")
            pos = env.get_body_pos(body_name)
            assert abs(pos[2] - OBJ_Z) < 0.01, (
                f"{name} z={pos[2]:.4f}, expected ~{OBJ_Z}"
            )


class TestWithinBounds:
    def test_positions_within_spawn_bounds(self, env):
        rng = np.random.default_rng(99)
        for _ in range(20):
            env.reset_to_keyframe("scene_start")
            env.randomize_objects(rng)
            for name in OBJ_JOINT_NAMES:
                body_name = name.replace("_jnt", "")
                pos = env.get_body_pos(body_name)
                assert X_RANGE[0] <= pos[0] <= X_RANGE[1], (
                    f"{name} x={pos[0]:.4f} out of range {X_RANGE}"
                )
                assert Y_RANGE[0] <= pos[1] <= Y_RANGE[1], (
                    f"{name} y={pos[1]:.4f} out of range {Y_RANGE}"
                )


class TestMinSeparation:
    def test_pairwise_separation(self, env):
        rng = np.random.default_rng(7)
        for _ in range(20):
            env.reset_to_keyframe("scene_start")
            env.randomize_objects(rng)
            positions = _get_obj_positions(env)
            names = list(positions.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    dist = np.linalg.norm(
                        positions[names[i]][:2] - positions[names[j]][:2]
                    )
                    assert dist >= MIN_OBJ_SEPARATION - 1e-9, (
                        f"{names[i]} and {names[j]} too close: {dist:.4f}m"
                    )


class TestReproducibility:
    def test_same_seed_same_positions(self, env):
        env.reset_to_keyframe("scene_start")
        rng1 = np.random.default_rng(42)
        env.randomize_objects(rng1)
        pos1 = _get_obj_positions(env)

        env.reset_to_keyframe("scene_start")
        rng2 = np.random.default_rng(42)
        env.randomize_objects(rng2)
        pos2 = _get_obj_positions(env)

        for name in OBJ_JOINT_NAMES:
            np.testing.assert_array_equal(pos1[name], pos2[name])

    def test_different_seeds_different_positions(self, env):
        env.reset_to_keyframe("scene_start")
        rng1 = np.random.default_rng(1)
        env.randomize_objects(rng1)
        pos1 = _get_obj_positions(env)

        env.reset_to_keyframe("scene_start")
        rng2 = np.random.default_rng(2)
        env.randomize_objects(rng2)
        pos2 = _get_obj_positions(env)

        any_different = False
        for name in OBJ_JOINT_NAMES:
            if not np.allclose(pos1[name][:2], pos2[name][:2]):
                any_different = True
        assert any_different, "Different seeds produced identical positions"


class TestRobustness:
    def test_50_seeds_no_failures(self, env):
        for seed in range(50):
            env.reset_to_keyframe("scene_start")
            rng = np.random.default_rng(seed)
            result = env.randomize_objects(rng)
            assert len(result) == len(OBJ_JOINT_NAMES)


class TestLowLevelFunction:
    def test_returns_dict_with_all_joints(self, env):
        rng = np.random.default_rng(0)
        result = randomize_object_positions(env.model, env.data, rng)
        assert set(result.keys()) == set(OBJ_JOINT_NAMES)

    def test_does_not_call_mj_forward(self, env):
        """After randomize_object_positions (not env method), xpos should
        still reflect the old keyframe values until mj_forward is called."""
        before = _get_obj_positions(env)
        rng = np.random.default_rng(42)
        randomize_object_positions(env.model, env.data, rng)
        # xpos not yet updated — should still match keyframe
        after = _get_obj_positions(env)
        for name in OBJ_JOINT_NAMES:
            np.testing.assert_array_equal(before[name], after[name])
        # Now forward and check it changed
        mujoco.mj_forward(env.model, env.data)
        updated = _get_obj_positions(env)
        any_changed = any(
            not np.allclose(before[n][:2], updated[n][:2]) for n in OBJ_JOINT_NAMES
        )
        assert any_changed
