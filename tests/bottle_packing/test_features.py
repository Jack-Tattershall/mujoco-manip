"""Tests for bottle packing feature schema."""

import pytest

from mujoco_manip.tasks.bottle_packing.constants import IMAGE_SIZE, NUM_WELLS
from mujoco_manip.tasks.bottle_packing.features import FEATURES
from mujoco_manip.tasks.bottle_packing.gym_env import BottlePackingGymEnv


@pytest.fixture(scope="module")
def env_and_obs():
    e = BottlePackingGymEnv(
        action_mode="ee_pos_quat_g_rel",
        well_index=0,
        max_episode_steps=10,
    )
    obs, _ = e.reset()
    yield e, obs
    e.close()


class TestFeatureSchema:
    def test_has_observation_features(self):
        obs_keys = [k for k in FEATURES if k.startswith("observation.")]
        assert len(obs_keys) > 0

    def test_has_action_features(self):
        act_keys = [k for k in FEATURES if k.startswith("action.")]
        assert len(act_keys) > 0

    def test_has_reward_feature(self):
        assert "next.reward" in FEATURES

    def test_image_features_use_correct_size(self):
        for key, spec in FEATURES.items():
            if "image" in key and spec["dtype"] == "image":
                assert spec["shape"][0] == IMAGE_SIZE
                assert spec["shape"][1] == IMAGE_SIZE

    def test_well_onehot_size(self):
        spec = FEATURES["observation.target_well_onehot"]
        assert spec["shape"] == (NUM_WELLS,)
