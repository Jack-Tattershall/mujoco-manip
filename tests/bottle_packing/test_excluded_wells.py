"""Tests for excluded_wells functionality in dataset generation."""

import random
import sys
from pathlib import Path

import pytest

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from generate_bottle_packing_dataset import build_well_schedule

from mujoco_manip.tasks.bottle_packing.constants import NUM_WELLS


class TestBuildWellScheduleExclusion:
    """Verify build_well_schedule correctly filters excluded wells."""

    def test_no_exclusions_returns_all_wells(self):
        sched = build_well_schedule("sequential", NUM_WELLS, random.Random(0))
        assert sched == list(range(NUM_WELLS))

    def test_no_exclusions_with_none(self):
        sched = build_well_schedule(
            "sequential", NUM_WELLS, random.Random(0), excluded_wells=None
        )
        assert sched == list(range(NUM_WELLS))

    def test_no_exclusions_with_empty_set(self):
        sched = build_well_schedule(
            "sequential", NUM_WELLS, random.Random(0), excluded_wells=set()
        )
        assert sched == list(range(NUM_WELLS))

    @pytest.mark.parametrize(
        "excluded",
        [
            {0},
            {19},
            {0, 4, 15, 19},
            {5, 6, 12, 19},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
            set(range(19)),  # exclude all but one
        ],
    )
    def test_excluded_wells_never_in_schedule_sequential(self, excluded):
        available = NUM_WELLS - len(excluded)
        sched = build_well_schedule("sequential", available, random.Random(0), excluded)
        assert not set(sched) & excluded

    @pytest.mark.parametrize(
        "excluded",
        [
            {0},
            {19},
            {0, 4, 15, 19},
            {5, 6, 12, 19},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        ],
    )
    def test_excluded_wells_never_in_schedule_random(self, excluded):
        available = NUM_WELLS - len(excluded)
        for seed in range(10):
            sched = build_well_schedule(
                "random", available, random.Random(seed), excluded
            )
            assert not set(sched) & excluded

    def test_schedule_length_matches_available(self):
        excluded = {0, 4, 15, 19}
        available = NUM_WELLS - len(excluded)
        sched = build_well_schedule("sequential", available, random.Random(0), excluded)
        assert len(sched) == available

    def test_schedule_length_capped_by_num_bottles(self):
        excluded = {0, 4, 15, 19}
        sched = build_well_schedule("sequential", 5, random.Random(0), excluded)
        assert len(sched) == 5
        assert not set(sched) & excluded

    def test_random_schedule_contains_only_valid_wells(self):
        excluded = {5, 6, 12, 19}
        valid = set(range(NUM_WELLS)) - excluded
        sched = build_well_schedule("random", len(valid), random.Random(42), excluded)
        assert set(sched) == valid

    def test_sequential_order_preserved(self):
        excluded = {3, 7, 11}
        expected = [w for w in range(NUM_WELLS) if w not in excluded]
        sched = build_well_schedule(
            "sequential", len(expected), random.Random(0), excluded
        )
        assert sched == expected

    def test_random_is_deterministic_with_same_seed(self):
        excluded = {0, 4, 15, 19}
        available = NUM_WELLS - len(excluded)
        sched1 = build_well_schedule("random", available, random.Random(42), excluded)
        sched2 = build_well_schedule("random", available, random.Random(42), excluded)
        assert sched1 == sched2

    def test_random_differs_with_different_seed(self):
        excluded = {0, 4, 15, 19}
        available = NUM_WELLS - len(excluded)
        sched1 = build_well_schedule("random", available, random.Random(1), excluded)
        sched2 = build_well_schedule("random", available, random.Random(2), excluded)
        assert sched1 != sched2

    def test_no_duplicates_in_schedule(self):
        excluded = {5, 6, 12, 19}
        available = NUM_WELLS - len(excluded)
        for seed in range(10):
            sched = build_well_schedule(
                "random", available, random.Random(seed), excluded
            )
            assert len(sched) == len(set(sched))

    def test_exclude_all_but_one(self):
        excluded = set(range(NUM_WELLS)) - {7}
        sched = build_well_schedule("random", 1, random.Random(0), excluded)
        assert sched == [7]

    def test_num_bottles_zero(self):
        sched = build_well_schedule("sequential", 0, random.Random(0), {0, 1})
        assert sched == []

    def test_num_bottles_exceeds_available_raises(self):
        excluded = {0, 4, 15, 19}
        available = NUM_WELLS - len(excluded)
        with pytest.raises(ValueError, match="exceeds available wells"):
            build_well_schedule("sequential", available + 1, random.Random(0), excluded)

    def test_all_wells_excluded_raises(self):
        with pytest.raises(ValueError, match="exceeds available wells"):
            build_well_schedule(
                "sequential", 1, random.Random(0), set(range(NUM_WELLS))
            )
