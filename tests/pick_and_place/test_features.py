"""Tests for pick-and-place feature schema consistency."""

from mujoco_manip.tasks.pick_and_place.constants import IMAGE_SIZE
from mujoco_manip.tasks.pick_and_place.features import DIM_NAMES, FEATURES


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
                assert spec["shape"][2] == 3


class TestDimNames:
    def test_dim_names_keys_subset_of_features(self):
        for key in DIM_NAMES:
            assert key in FEATURES, f"DIM_NAMES has key {key} not in FEATURES"

    def test_dim_names_lengths_match_features(self):
        for key, names in DIM_NAMES.items():
            spec = FEATURES[key]
            expected_len = 1
            for d in spec["shape"]:
                expected_len *= d
            assert len(names) == expected_len, (
                f"{key}: {len(names)} dim names but shape {spec['shape']} "
                f"has {expected_len} elements"
            )
