"""Tests for BottlePackingGymEnv action modes, observations, and rewards."""

import numpy as np
import pytest

from mujoco_manip.controller import TARGET_ORI
from mujoco_manip.pose_utils import (
    pos_rotmat_to_se3,
    rotmat_to_6d,
    se3_to_pos_quat_g,
    se3_to_pos_rot6d_g,
)
from mujoco_manip.tasks.bottle_packing.constants import NUM_WELLS
from mujoco_manip.tasks.bottle_packing.gym_env import BottlePackingGymEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["ee_pos_quat_g_rel", "ee_pos_rot6d_g_rel"])
def env(request):
    e = BottlePackingGymEnv(
        action_mode=request.param,
        well_index=0,
        max_episode_steps=50,
    )
    yield e
    e.close()


@pytest.fixture
def env_quat_rel():
    e = BottlePackingGymEnv(
        action_mode="ee_pos_quat_g_rel", well_index=0, max_episode_steps=50
    )
    yield e
    e.close()


@pytest.fixture
def env_rot6d_rel():
    e = BottlePackingGymEnv(
        action_mode="ee_pos_rot6d_g_rel", well_index=0, max_episode_steps=50
    )
    yield e
    e.close()


@pytest.fixture
def env_quat_abs():
    e = BottlePackingGymEnv(
        action_mode="ee_pos_quat_g", well_index=0, max_episode_steps=50
    )
    yield e
    e.close()


@pytest.fixture
def env_rot6d_abs():
    e = BottlePackingGymEnv(
        action_mode="ee_pos_rot6d_g", well_index=0, max_episode_steps=50
    )
    yield e
    e.close()


@pytest.fixture
def env_abs():
    e = BottlePackingGymEnv(action_mode="abs_pos", well_index=0, max_episode_steps=50)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_action_mode_raises(self):
        with pytest.raises(ValueError, match="action_mode must be one of"):
            BottlePackingGymEnv(action_mode="invalid")

    def test_action_space_quat_rel(self, env_quat_rel):
        assert env_quat_rel.action_space.shape == (8,)

    def test_action_space_rot6d_rel(self, env_rot6d_rel):
        assert env_rot6d_rel.action_space.shape == (10,)

    def test_action_space_quat_abs(self, env_quat_abs):
        assert env_quat_abs.action_space.shape == (8,)

    def test_action_space_rot6d_abs(self, env_rot6d_abs):
        assert env_rot6d_abs.action_space.shape == (10,)

    def test_action_space_abs_pos(self, env_abs):
        assert env_abs.action_space.shape == (4,)

    def test_gripper_bounds_quat(self, env_quat_rel):
        assert env_quat_rel.action_space.low[7] == 0.0
        assert env_quat_rel.action_space.high[7] == 1.0

    def test_gripper_bounds_rot6d(self, env_rot6d_rel):
        assert env_rot6d_rel.action_space.low[9] == 0.0
        assert env_rot6d_rel.action_space.high[9] == 1.0

    def test_observation_space_keys(self, env):
        expected = {
            "image_overhead",
            "image_wrist",
            "state",
            "state.ee.pos_quat_g",
            "state.ee.pos_rot6d_g",
            "state.ee.pos_quat_g_rel",
            "state.ee.pos_rot6d_g_rel",
            "target_well_onehot",
            "keypoints_overhead",
            "keypoints_wrist",
            "target_bottle_keypoints_overhead",
            "target_well_keypoints_overhead",
        }
        assert set(env.observation_space.spaces.keys()) == expected


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_obs_shapes(self, env):
        obs, _ = env.reset()
        assert obs["image_overhead"].shape == (224, 224, 3)
        assert obs["image_wrist"].shape == (224, 224, 3)
        assert obs["state"].shape == (11,)
        assert obs["state.ee.pos_quat_g"].shape == (8,)
        assert obs["state.ee.pos_rot6d_g"].shape == (10,)
        assert obs["state.ee.pos_quat_g_rel"].shape == (8,)
        assert obs["state.ee.pos_rot6d_g_rel"].shape == (10,)
        assert obs["target_well_onehot"].shape == (NUM_WELLS,)
        assert obs["keypoints_overhead"].shape == (3, 2)
        assert obs["keypoints_wrist"].shape == (3, 2)
        assert obs["target_bottle_keypoints_overhead"].shape == (2,)
        assert obs["target_well_keypoints_overhead"].shape == (2,)

    def test_obs_dtypes(self, env):
        obs, _ = env.reset()
        assert obs["image_overhead"].dtype == np.uint8
        assert obs["image_wrist"].dtype == np.uint8
        assert obs["state"].dtype == np.float32
        assert obs["target_well_onehot"].dtype == np.float32

    def test_well_onehot_valid(self, env):
        obs, _ = env.reset()
        oh = obs["target_well_onehot"]
        assert oh.sum() == pytest.approx(1.0)
        assert set(np.unique(oh)).issubset({0.0, 1.0})

    def test_well_onehot_matches_index(self, env):
        obs, _ = env.reset()
        oh = obs["target_well_onehot"]
        assert np.argmax(oh) == env.well_index

    def test_initial_se3_captured(self, env):
        env.reset()
        T = env.initial_ee_se3
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])

    def test_reset_deterministic_with_seed(self, env):
        obs1, _ = env.reset(seed=42)
        s1 = obs1["state"].copy()
        obs2, _ = env.reset(seed=42)
        s2 = obs2["state"].copy()
        np.testing.assert_array_equal(s1, s2)

    def test_reset_with_well_override(self):
        e = BottlePackingGymEnv(
            action_mode="ee_pos_quat_g_rel", well_index=0, max_episode_steps=10
        )
        try:
            obs, _ = e.reset(options={"well_index": 15})
            assert e.well_index == 15
            assert np.argmax(obs["target_well_onehot"]) == 15
        finally:
            e.close()


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_returns_five_tuple(self, env):
        env.reset()
        result = env.step(env.action_space.sample())
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_obs_shapes_match_reset(self, env):
        obs_reset, _ = env.reset()
        obs_step, *_ = env.step(env.action_space.sample())
        for key in obs_reset:
            assert obs_step[key].shape == obs_reset[key].shape, (
                f"Shape mismatch for {key}"
            )

    def test_step_increments_count(self, env):
        env.reset()
        assert env.step_count == 0
        env.step(env.action_space.sample())
        assert env.step_count == 1

    def test_info_has_success(self, env):
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "success" in info


# ---------------------------------------------------------------------------
# Identity action
# ---------------------------------------------------------------------------


class TestIdentityAction:
    def _identity_quat(self):
        return np.array([0, 0, 0, 0, 0, 0, 1, 1.0], dtype=np.float32)

    def _identity_rot6d(self):
        d6 = rotmat_to_6d(np.eye(3))
        return np.array([0, 0, 0, *d6, 1.0], dtype=np.float32)

    def test_quat_rel_stays_near_initial(self, env_quat_rel):
        obs, _ = env_quat_rel.reset()
        initial_ee = obs["state"][:3].copy()
        for _ in range(5):
            obs, *_ = env_quat_rel.step(self._identity_quat())
        dist = np.linalg.norm(obs["state"][:3] - initial_ee)
        assert dist < 0.05

    def test_rot6d_rel_stays_near_initial(self, env_rot6d_rel):
        obs, _ = env_rot6d_rel.reset()
        initial_ee = obs["state"][:3].copy()
        for _ in range(5):
            obs, *_ = env_rot6d_rel.step(self._identity_rot6d())
        dist = np.linalg.norm(obs["state"][:3] - initial_ee)
        assert dist < 0.05


# ---------------------------------------------------------------------------
# Known displacement
# ---------------------------------------------------------------------------


class TestKnownDisplacement:
    def test_quat_rel_translation(self, env_quat_rel):
        obs, _ = env_quat_rel.reset()
        T_init = env_quat_rel.initial_ee_se3.copy()
        dx = np.array([0.05, 0.0, 0.0])
        action = np.array([*dx, 0, 0, 0, 1, 1.0], dtype=np.float32)
        for _ in range(20):
            obs, *_ = env_quat_rel.step(action)
        expected = (T_init @ np.array([*dx, 1.0]))[:3]
        dist = np.linalg.norm(obs["state"][:3] - expected)
        assert dist < 0.05


# ---------------------------------------------------------------------------
# Gripper control
# ---------------------------------------------------------------------------


class TestGripperControl:
    def test_gripper_opens(self, env_quat_rel):
        env_quat_rel.reset()
        action = np.array([0, 0, 0, 0, 0, 0, 1, 1.0], dtype=np.float32)
        env_quat_rel.step(action)
        assert env_quat_rel.robot.gripper_ctrl == 255.0

    def test_gripper_closes(self, env_quat_rel):
        env_quat_rel.reset()
        action = np.array([0, 0, 0, 0, 0, 0, 1, 0.0], dtype=np.float32)
        env_quat_rel.step(action)
        assert env_quat_rel.robot.gripper_ctrl == 0.0


# ---------------------------------------------------------------------------
# Cross mode parity
# ---------------------------------------------------------------------------


class TestCrossModeParity:
    def test_same_relative_pose_same_position(self, env_quat_rel, env_rot6d_rel):
        obs8, _ = env_quat_rel.reset(seed=0)
        obs10, _ = env_rot6d_rel.reset(seed=0)
        np.testing.assert_allclose(obs8["state"][:3], obs10["state"][:3], atol=1e-5)

        T_rel = pos_rotmat_to_se3(np.array([0.03, -0.02, 0.01]), np.eye(3))
        action_8 = se3_to_pos_quat_g(T_rel, gripper=1.0)
        action_10 = se3_to_pos_rot6d_g(T_rel, gripper=1.0)

        for _ in range(15):
            obs8, *_ = env_quat_rel.step(action_8)
            obs10, *_ = env_rot6d_rel.step(action_10)

        np.testing.assert_allclose(obs8["state"][:3], obs10["state"][:3], atol=0.01)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_truncates_at_max_steps(self, env):
        env.reset()
        for i in range(env._max_episode_steps):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated:
                break
        if not terminated:
            assert truncated

    def test_not_truncated_before_max(self, env):
        env.reset()
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        assert not truncated


# ---------------------------------------------------------------------------
# Reward types
# ---------------------------------------------------------------------------


class TestReward:
    def test_dense_reward_is_float(self, env):
        env.reset()
        _, reward, *_ = env.step(env.action_space.sample())
        assert isinstance(reward, float)

    def test_sparse_reward(self):
        e = BottlePackingGymEnv(
            action_mode="ee_pos_quat_g_rel",
            reward_type="sparse",
            well_index=0,
            max_episode_steps=10,
        )
        try:
            e.reset()
            _, reward, *_ = e.step(e.action_space.sample())
            assert reward in (0.0, 1.0)
        finally:
            e.close()


class TestStagedReward:
    @pytest.fixture
    def staged_env(self):
        e = BottlePackingGymEnv(
            action_mode="abs_pos",
            reward_type="staged",
            well_index=0,
            max_episode_steps=500,
        )
        yield e
        e.close()

    def test_initial_reward_range(self, staged_env):
        staged_env.reset()
        reward, _ = staged_env._compute_reward()
        assert 0.0 <= reward < 1.0

    def test_sticky_flags_reset(self, staged_env):
        staged_env.reset()
        assert staged_env._has_grasped is False
        assert staged_env._has_lifted is False
        assert staged_env._above_target is False
        assert staged_env._has_placed is False
        assert staged_env._reward_hwm is None

    def test_collision_geom_sets_populated(self, staged_env):
        assert len(staged_env._robot_geom_ids) > 0
        assert len(staged_env._obstacle_geom_ids) > 0

    def test_staged_reward_range(self, staged_env):
        staged_env.reset()
        for _ in range(10):
            _, r, term, _, _ = staged_env.step(staged_env.action_space.sample())
            if term and r < 0:
                break
            assert 0.0 <= r <= 1.0

    def test_reward_components_in_info(self, staged_env):
        staged_env.reset()
        _, _, _, _, info = staged_env.step(staged_env.action_space.sample())
        assert "reward_components" in info
        rc = info["reward_components"]
        assert rc.shape == (6,)  # total + 5 phases


# ---------------------------------------------------------------------------
# Well selection
# ---------------------------------------------------------------------------


class TestWellSelection:
    def test_fixed_well(self):
        e = BottlePackingGymEnv(
            action_mode="ee_pos_quat_g_rel", well_index=7, max_episode_steps=10
        )
        try:
            obs, _ = e.reset()
            assert e.well_index == 7
            assert np.argmax(obs["target_well_onehot"]) == 7
        finally:
            e.close()

    def test_random_well_from_pool(self):
        e = BottlePackingGymEnv(
            action_mode="ee_pos_quat_g_rel", wells="all", max_episode_steps=10
        )
        try:
            seen = set()
            for _ in range(50):
                e.reset()
                seen.add(e.well_index)
            # Should see more than 1 well
            assert len(seen) > 1
        finally:
            e.close()

    def test_custom_well_list(self):
        custom = [3, 10, 17]
        e = BottlePackingGymEnv(
            action_mode="ee_pos_quat_g_rel", wells=custom, max_episode_steps=10
        )
        try:
            for _ in range(20):
                e.reset()
                assert e.well_index in custom
        finally:
            e.close()


# ---------------------------------------------------------------------------
# Decode action
# ---------------------------------------------------------------------------


class TestDecodeAction:
    def test_abs_pos_passthrough(self, env_abs):
        env_abs.reset()
        action = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
        pos, gripper = env_abs.decode_action(action)
        np.testing.assert_array_equal(pos, action[:3])
        assert gripper == pytest.approx(0.8)

    def test_identity_maps_to_initial_quat_rel(self, env_quat_rel):
        env_quat_rel.reset()
        initial_pos = env_quat_rel.robot.ee_pos.copy()
        identity = np.array([0, 0, 0, 0, 0, 0, 1, 1.0], dtype=np.float32)
        pos, gripper = env_quat_rel.decode_action(identity)
        np.testing.assert_allclose(pos, initial_pos, atol=1e-6)
        assert gripper == 1.0

    def test_absolute_quat_passthrough(self, env_quat_abs):
        env_quat_abs.reset()
        target = np.array([0.1, 0.4, 0.4])
        T = pos_rotmat_to_se3(target, TARGET_ORI)
        action = se3_to_pos_quat_g(T, gripper=0.5)
        pos, gripper = env_quat_abs.decode_action(action)
        np.testing.assert_allclose(pos, target, atol=1e-5)

    def test_absolute_rot6d_passthrough(self, env_rot6d_abs):
        env_rot6d_abs.reset()
        target = np.array([0.1, 0.4, 0.4])
        T = pos_rotmat_to_se3(target, TARGET_ORI)
        action = se3_to_pos_rot6d_g(T, gripper=0.5)
        pos, gripper = env_rot6d_abs.decode_action(action)
        np.testing.assert_allclose(pos, target, atol=1e-5)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_returns_image(self, env):
        env.reset()
        img = env.render()
        assert img is not None
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# Multiple resets
# ---------------------------------------------------------------------------


class TestMultipleResets:
    def test_step_count_resets(self, env):
        env.reset()
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        assert env.step_count == 2
        env.reset()
        assert env.step_count == 0

    def test_multiple_episodes(self, env):
        for _ in range(3):
            env.reset()
            for _ in range(5):
                _, _, term, trunc, _ = env.step(env.action_space.sample())
                if term or trunc:
                    break


# ---------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------


class TestKeypoints:
    def test_bottle_keypoints_shape(self, env):
        obs, _ = env.reset()
        kp = obs["target_bottle_keypoints_overhead"]
        assert kp.shape == (2,)
        assert kp.dtype == np.float32

    def test_well_keypoints_shape(self, env):
        obs, _ = env.reset()
        kp = obs["target_well_keypoints_overhead"]
        assert kp.shape == (2,)
        assert kp.dtype == np.float32

    def test_keypoints_in_unit_range(self, env):
        obs, _ = env.reset()
        for key in (
            "target_bottle_keypoints_overhead",
            "target_well_keypoints_overhead",
        ):
            kp = obs[key]
            assert np.all(kp >= 0.0) and np.all(kp <= 1.0)

    def test_different_wells_give_different_well_keypoints(self):
        e1 = BottlePackingGymEnv(
            action_mode="ee_pos_quat_g_rel", well_index=0, max_episode_steps=10
        )
        e2 = BottlePackingGymEnv(
            action_mode="ee_pos_quat_g_rel", well_index=19, max_episode_steps=10
        )
        try:
            obs1, _ = e1.reset()
            obs2, _ = e2.reset()
            kp1 = obs1["target_well_keypoints_overhead"]
            kp2 = obs2["target_well_keypoints_overhead"]
            assert not np.allclose(kp1, kp2)
        finally:
            e1.close()
            e2.close()
