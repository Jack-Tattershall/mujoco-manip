"""Tests for PickPlaceGymEnv with ee_8dof and ee_10dof action modes."""

import numpy as np
import pytest

from src.constants import (
    KEYPOINT_BODIES,
)
from src.controller import TARGET_ORI
from src.gym_env import PickPlaceGymEnv
from src.pose_utils import (
    pos_rotmat_to_se3,
    rotmat_to_6d,
    se3_to_8dof,
    se3_to_10dof,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["ee_8dof", "ee_10dof"])
def env(request):
    """Yield a PickPlaceGymEnv for each relative action mode, cleaned up after."""
    e = PickPlaceGymEnv(
        action_mode=request.param,
        task=("obj_red", "bin_red"),
        max_episode_steps=50,
    )
    yield e
    e.close()


@pytest.fixture
def env_8dof():
    e = PickPlaceGymEnv(
        action_mode="ee_8dof", task=("obj_red", "bin_red"), max_episode_steps=50
    )
    yield e
    e.close()


@pytest.fixture
def env_10dof():
    e = PickPlaceGymEnv(
        action_mode="ee_10dof", task=("obj_red", "bin_red"), max_episode_steps=50
    )
    yield e
    e.close()


@pytest.fixture
def env_abs():
    e = PickPlaceGymEnv(
        action_mode="abs_pos", task=("obj_red", "bin_red"), max_episode_steps=50
    )
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_action_mode_raises(self):
        with pytest.raises(ValueError, match="action_mode must be one of"):
            PickPlaceGymEnv(action_mode="invalid")

    def test_action_space_shape_8dof(self, env_8dof):
        assert env_8dof.action_space.shape == (8,)

    def test_action_space_shape_10dof(self, env_10dof):
        assert env_10dof.action_space.shape == (10,)

    def test_action_space_shape_abs(self, env_abs):
        assert env_abs.action_space.shape == (4,)

    def test_gripper_bounds_8dof(self, env_8dof):
        assert env_8dof.action_space.low[7] == 0.0
        assert env_8dof.action_space.high[7] == 1.0

    def test_gripper_bounds_10dof(self, env_10dof):
        assert env_10dof.action_space.low[9] == 0.0
        assert env_10dof.action_space.high[9] == 1.0

    def test_pose_dims_unbounded_8dof(self, env_8dof):
        assert np.all(env_8dof.action_space.low[:7] == -np.inf)
        assert np.all(env_8dof.action_space.high[:7] == np.inf)

    def test_pose_dims_unbounded_10dof(self, env_10dof):
        assert np.all(env_10dof.action_space.low[:9] == -np.inf)
        assert np.all(env_10dof.action_space.high[:9] == np.inf)

    def test_observation_space_keys(self, env):
        expected = {
            "image_overhead",
            "image_wrist",
            "state",
            "state.ee.8dof",
            "state.ee.10dof",
            "state.ee.8dof_rel",
            "state.ee.10dof_rel",
            "target_bin_onehot",
            "keypoints_overhead",
            "keypoints_wrist",
            "target_keypoints_overhead",
        }
        assert set(env.observation_space.spaces.keys()) == expected


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_obs_shapes(self, env):
        obs, _ = env.reset()
        assert obs["image_overhead"].shape == (224, 224, 3)
        assert obs["image_wrist"].shape == (224, 224, 3)
        assert obs["state"].shape == (11,)
        assert obs["state.ee.8dof_rel"].shape == (8,)
        assert obs["state.ee.10dof_rel"].shape == (10,)
        assert obs["target_bin_onehot"].shape == (3,)
        assert obs["keypoints_overhead"].shape == (len(KEYPOINT_BODIES), 2)
        assert obs["keypoints_wrist"].shape == (len(KEYPOINT_BODIES), 2)
        assert obs["target_keypoints_overhead"].shape == (2, 2)

    def test_reset_obs_dtypes(self, env):
        obs, _ = env.reset()
        assert obs["image_overhead"].dtype == np.uint8
        assert obs["image_wrist"].dtype == np.uint8
        assert obs["state"].dtype == np.float32
        assert obs["target_bin_onehot"].dtype == np.float32
        assert obs["keypoints_overhead"].dtype == np.float32
        assert obs["target_keypoints_overhead"].dtype == np.float32

    def test_reset_target_bin_onehot_is_valid(self, env):
        obs, _ = env.reset()
        oh = obs["target_bin_onehot"]
        assert oh.sum() == pytest.approx(1.0)
        assert set(np.unique(oh)).issubset({0.0, 1.0})

    def test_reset_captures_initial_pose(self, env):
        env.reset()
        T = env._initial_ee_se3
        assert T is not None
        assert T.shape == (4, 4)
        # Should be a valid SE(3): bottom row is [0, 0, 0, 1]
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])

    def test_initial_pose_rotation_is_orthogonal(self, env):
        env.reset()
        R = env._initial_ee_se3[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    def test_reset_deterministic_with_seed(self, env):
        obs1, _ = env.reset(seed=42)
        state1 = obs1["state"].copy()
        obs2, _ = env.reset(seed=42)
        state2 = obs2["state"].copy()
        np.testing.assert_array_equal(state1, state2)


# ---------------------------------------------------------------------------
# Step basics
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_returns_five_tuple(self, env):
        env.reset()
        result = env.step(env.action_space.sample())
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_step_obs_shapes_match_reset(self, env):
        obs_reset, _ = env.reset()
        obs_step, *_ = env.step(env.action_space.sample())
        for key in obs_reset:
            assert obs_step[key].shape == obs_reset[key].shape, (
                f"Shape mismatch for {key}"
            )

    def test_step_increments_count(self, env):
        env.reset()
        assert env._step_count == 0
        env.step(env.action_space.sample())
        assert env._step_count == 1
        env.step(env.action_space.sample())
        assert env._step_count == 2

    def test_info_has_success_key(self, env):
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "success" in info


# ---------------------------------------------------------------------------
# Identity action (relative pose = identity → stay at initial position)
# ---------------------------------------------------------------------------


class TestIdentityAction:
    """Sending the identity relative pose should keep the EE near its initial position."""

    def _identity_8dof(self):
        # Identity rotation as quaternion + gripper open
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    def _identity_10dof(self):
        # Identity rotation as 6D (rows of I) + gripper open
        R_id = np.eye(3)
        d6 = rotmat_to_6d(R_id)
        return np.array([0.0, 0.0, 0.0, *d6, 1.0], dtype=np.float32)

    def test_identity_8dof_stays_near_initial(self, env_8dof):
        obs, _ = env_8dof.reset()
        initial_ee = obs["state"][:3].copy()

        # Step with identity action several times
        for _ in range(5):
            obs, *_ = env_8dof.step(self._identity_8dof())

        ee_after = obs["state"][:3]
        dist = np.linalg.norm(ee_after - initial_ee)
        assert dist < 0.05, f"EE drifted {dist:.4f}m from initial with identity action"

    def test_identity_10dof_stays_near_initial(self, env_10dof):
        obs, _ = env_10dof.reset()
        initial_ee = obs["state"][:3].copy()

        for _ in range(5):
            obs, *_ = env_10dof.step(self._identity_10dof())

        ee_after = obs["state"][:3]
        dist = np.linalg.norm(ee_after - initial_ee)
        assert dist < 0.05, f"EE drifted {dist:.4f}m from initial with identity action"


# ---------------------------------------------------------------------------
# Known displacement (relative pose → EE moves to expected world position)
# ---------------------------------------------------------------------------


class TestKnownDisplacement:
    """A relative translation should move the EE by that amount in the initial frame."""

    def test_8dof_translation_moves_ee(self, env_8dof):
        obs, _ = env_8dof.reset()
        T_init = env_8dof._initial_ee_se3.copy()

        # Command: move 0.1m along the initial frame's x-axis, identity rotation, gripper open
        dx_initial = np.array([0.1, 0.0, 0.0])
        action = np.array([*dx_initial, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        # Step many times so IK converges
        for _ in range(20):
            obs, *_ = env_8dof.step(action)

        ee_after = obs["state"][:3]
        # Expected world-frame target: T_init @ [dx; 1]
        expected_world = (T_init @ np.array([*dx_initial, 1.0]))[:3]
        dist = np.linalg.norm(ee_after - expected_world)
        assert dist < 0.05, (
            f"EE at {ee_after}, expected near {expected_world} (error={dist:.4f}m)"
        )

    def test_10dof_translation_moves_ee(self, env_10dof):
        obs, _ = env_10dof.reset()
        T_init = env_10dof._initial_ee_se3.copy()

        dx_initial = np.array([0.1, 0.0, 0.0])
        d6 = rotmat_to_6d(np.eye(3))
        action = np.array([*dx_initial, *d6, 1.0], dtype=np.float32)

        for _ in range(20):
            obs, *_ = env_10dof.step(action)

        ee_after = obs["state"][:3]
        expected_world = (T_init @ np.array([*dx_initial, 1.0]))[:3]
        dist = np.linalg.norm(ee_after - expected_world)
        assert dist < 0.05, (
            f"EE at {ee_after}, expected near {expected_world} (error={dist:.4f}m)"
        )


# ---------------------------------------------------------------------------
# Gripper control
# ---------------------------------------------------------------------------


class TestGripperControl:
    def _make_open_action(self, mode):
        if mode == "ee_8dof":
            return np.array(
                [0, 0, 0, 0, 0, 0, 1, 1.0], dtype=np.float32
            )  # gripper=1 → open
        else:
            d6 = rotmat_to_6d(np.eye(3))
            return np.array([0, 0, 0, *d6, 1.0], dtype=np.float32)

    def _make_close_action(self, mode):
        if mode == "ee_8dof":
            return np.array(
                [0, 0, 0, 0, 0, 0, 1, 0.0], dtype=np.float32
            )  # gripper=0 → close
        else:
            d6 = rotmat_to_6d(np.eye(3))
            return np.array([0, 0, 0, *d6, 0.0], dtype=np.float32)

    def test_gripper_opens(self, env):
        env.reset()
        env.step(self._make_open_action(env._action_mode))
        assert env._robot.gripper_ctrl == 255.0  # PandaRobot.GRIPPER_OPEN

    def test_gripper_closes(self, env):
        env.reset()
        env.step(self._make_close_action(env._action_mode))
        assert env._robot.gripper_ctrl == 0.0  # PandaRobot.GRIPPER_CLOSED

    def test_gripper_reflected_in_state(self, env):
        env.reset()
        obs_open, *_ = env.step(self._make_open_action(env._action_mode))
        gripper_norm_open = obs_open["state"][3]

        obs_close, *_ = env.step(self._make_close_action(env._action_mode))
        gripper_norm_close = obs_close["state"][3]

        assert gripper_norm_open > gripper_norm_close


# ---------------------------------------------------------------------------
# 8dof ↔ 10dof consistency: same relative SE(3) produces same world target
# ---------------------------------------------------------------------------


class TestCrossModeParity:
    """The same relative SE(3) encoded as 8dof or 10dof should yield the same EE target."""

    def test_same_relative_pose_same_ee_position(self, env_8dof, env_10dof):
        obs8, _ = env_8dof.reset(seed=0)
        obs10, _ = env_10dof.reset(seed=0)

        # Both should start at the same place
        np.testing.assert_allclose(obs8["state"][:3], obs10["state"][:3], atol=1e-5)

        # Build a relative SE(3) with a known offset
        T_rel = pos_rotmat_to_se3(np.array([0.05, -0.03, 0.02]), np.eye(3))
        action_8 = se3_to_8dof(T_rel, gripper=1.0)
        action_10 = se3_to_10dof(T_rel, gripper=1.0)

        for _ in range(15):
            obs8, *_ = env_8dof.step(action_8)
            obs10, *_ = env_10dof.step(action_10)

        ee8 = obs8["state"][:3]
        ee10 = obs10["state"][:3]
        np.testing.assert_allclose(
            ee8,
            ee10,
            atol=0.01,
            err_msg="8dof and 10dof should drive EE to the same position",
        )


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
        # If it didn't terminate early, it should have truncated on the last step
        if not terminated:
            assert truncated

    def test_not_truncated_before_max_steps(self, env):
        env.reset()
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        assert not truncated


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


class TestReward:
    def test_dense_reward_is_float(self, env):
        env.reset()
        _, reward, *_ = env.step(env.action_space.sample())
        assert isinstance(reward, float)

    def test_sparse_reward_env(self):
        e = PickPlaceGymEnv(
            action_mode="ee_8dof",
            reward_type="sparse",
            task=("obj_red", "bin_red"),
            max_episode_steps=10,
        )
        try:
            e.reset()
            _, reward, *_ = e.step(e.action_space.sample())
            assert reward in (0.0, 1.0)
        finally:
            e.close()


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


class TestTaskSelection:
    def test_fixed_task(self):
        e = PickPlaceGymEnv(
            action_mode="ee_8dof", task=("obj_blue", "bin_green"), max_episode_steps=10
        )
        try:
            obs, _ = e.reset()
            # bin_green is index 1
            np.testing.assert_array_equal(obs["target_bin_onehot"], [0, 1, 0])
        finally:
            e.close()

    def test_random_task_from_pool(self):
        e = PickPlaceGymEnv(action_mode="ee_8dof", tasks="all", max_episode_steps=10)
        try:
            seen_bins = set()
            for _ in range(30):
                obs, _ = e.reset()
                idx = int(np.argmax(obs["target_bin_onehot"]))
                seen_bins.add(idx)
            # With 30 resets over 3 bins, very likely we see all 3
            assert len(seen_bins) == 3, f"Only saw bin indices {seen_bins}"
        finally:
            e.close()

    def test_custom_task_list(self):
        custom = [("obj_red", "bin_blue"), ("obj_green", "bin_red")]
        e = PickPlaceGymEnv(action_mode="ee_10dof", tasks=custom, max_episode_steps=10)
        try:
            for _ in range(10):
                e.reset()
                assert (e._obj_name, e._bin_name) in custom
        finally:
            e.close()


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
# Relative pose math end-to-end inside the env
# ---------------------------------------------------------------------------


class TestRelativePoseMath:
    """Verify the internal _relative_action_to_world_pos method."""

    def test_identity_maps_to_initial_pos_8dof(self, env_8dof):
        env_8dof.reset()
        initial_pos = env_8dof._robot.ee_pos.copy()
        identity = np.array([0, 0, 0, 0, 0, 0, 1, 1.0], dtype=np.float32)
        world_pos, gripper = env_8dof._relative_action_to_world_pos(identity)
        np.testing.assert_allclose(world_pos, initial_pos, atol=1e-6)
        assert gripper == 1.0

    def test_identity_maps_to_initial_pos_10dof(self, env_10dof):
        env_10dof.reset()
        initial_pos = env_10dof._robot.ee_pos.copy()
        d6 = rotmat_to_6d(np.eye(3))
        identity = np.array([0, 0, 0, *d6, 0.5], dtype=np.float32)
        world_pos, gripper = env_10dof._relative_action_to_world_pos(identity)
        np.testing.assert_allclose(world_pos, initial_pos, atol=1e-6)
        assert gripper == pytest.approx(0.5)

    def test_translation_roundtrip_8dof(self, env_8dof):
        """Build a relative action from a known world target, verify env recovers it."""
        env_8dof.reset()
        T_init = env_8dof._initial_ee_se3.copy()
        T_init_inv = np.linalg.inv(T_init)

        target_world = np.array([-0.1, 0.5, 0.40])
        T_target = pos_rotmat_to_se3(target_world, TARGET_ORI)
        T_rel = T_init_inv @ T_target
        action = se3_to_8dof(T_rel, gripper=1.0)

        world_pos, _ = env_8dof._relative_action_to_world_pos(action)
        np.testing.assert_allclose(world_pos, target_world, atol=1e-5)

    def test_translation_roundtrip_10dof(self, env_10dof):
        env_10dof.reset()
        T_init = env_10dof._initial_ee_se3.copy()
        T_init_inv = np.linalg.inv(T_init)

        target_world = np.array([0.15, 0.55, 0.35])
        T_target = pos_rotmat_to_se3(target_world, TARGET_ORI)
        T_rel = T_init_inv @ T_target
        action = se3_to_10dof(T_rel, gripper=0.0)

        world_pos, gripper = env_10dof._relative_action_to_world_pos(action)
        np.testing.assert_allclose(world_pos, target_world, atol=1e-5)
        assert gripper == pytest.approx(0.0)

    def test_abs_pos_passthrough(self, env_abs):
        env_abs.reset()
        action = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
        world_pos, gripper = env_abs._relative_action_to_world_pos(action)
        np.testing.assert_array_equal(world_pos, action[:3])
        assert gripper == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Target keypoints (overhead)
# ---------------------------------------------------------------------------


class TestTargetKeypointsOverhead:
    def test_shape_and_dtype(self, env):
        obs, _ = env.reset()
        kp = obs["target_keypoints_overhead"]
        assert kp.shape == (2, 2)
        assert kp.dtype == np.float32

    def test_values_in_unit_range(self, env):
        obs, _ = env.reset()
        kp = obs["target_keypoints_overhead"]
        assert np.all(kp >= 0.0) and np.all(kp <= 1.0)

    def test_constant_across_steps(self, env):
        obs, _ = env.reset()
        kp_reset = obs["target_keypoints_overhead"].copy()
        for _ in range(3):
            obs, *_ = env.step(env.action_space.sample())
        kp_step = obs["target_keypoints_overhead"]
        np.testing.assert_array_equal(kp_reset, kp_step)

    def test_different_tasks_give_different_keypoints(self):
        e1 = PickPlaceGymEnv(
            action_mode="ee_8dof", task=("obj_red", "bin_red"), max_episode_steps=10
        )
        e2 = PickPlaceGymEnv(
            action_mode="ee_8dof", task=("obj_blue", "bin_green"), max_episode_steps=10
        )
        try:
            obs1, _ = e1.reset()
            obs2, _ = e2.reset()
            kp1 = obs1["target_keypoints_overhead"]
            kp2 = obs2["target_keypoints_overhead"]
            assert not np.allclose(kp1, kp2), (
                "Different tasks should produce different target keypoints"
            )
        finally:
            e1.close()
            e2.close()


# ---------------------------------------------------------------------------
# Multiple resets don't leak state
# ---------------------------------------------------------------------------


class TestMultipleResets:
    def test_reset_clears_step_count(self, env):
        env.reset()
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        assert env._step_count == 2
        env.reset()
        assert env._step_count == 0

    def test_reset_refreshes_initial_pose(self, env):
        env.reset()
        T1 = env._initial_ee_se3.copy()
        env.step(env.action_space.sample())
        env.reset()
        T2 = env._initial_ee_se3.copy()
        # After reset both should be the same home-config pose
        np.testing.assert_allclose(T1, T2, atol=1e-6)

    def test_multiple_episodes(self, env):
        """Run 3 short episodes to ensure no crash or state leak."""
        for _ in range(3):
            obs, _ = env.reset()
            for _ in range(5):
                obs, r, term, trunc, info = env.step(env.action_space.sample())
                if term or trunc:
                    break
