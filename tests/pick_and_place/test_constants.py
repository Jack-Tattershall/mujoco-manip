"""Tests for pick-and-place constants."""

from mujoco_manip.tasks.pick_and_place.constants import (
    ALL_TASKS,
    BINS,
    CROSS_TASKS,
    MATCH_TASKS,
    OBJECTS,
    TASK_SETS,
)


class TestTaskSets:
    def test_match_tasks_same_colour(self):
        for obj, bin_ in MATCH_TASKS:
            colour_obj = obj.split("_")[1]
            colour_bin = bin_.split("_")[1]
            assert colour_obj == colour_bin

    def test_cross_tasks_different_colour(self):
        for obj, bin_ in CROSS_TASKS:
            colour_obj = obj.split("_")[1]
            colour_bin = bin_.split("_")[1]
            assert colour_obj != colour_bin

    def test_all_tasks_count(self):
        assert len(ALL_TASKS) == len(OBJECTS) * len(BINS)

    def test_task_sets_keys(self):
        assert "match" in TASK_SETS
        assert "cross" in TASK_SETS
        assert "all" in TASK_SETS

    def test_no_duplicate_tasks(self):
        assert len(ALL_TASKS) == len(set(ALL_TASKS))
        assert len(MATCH_TASKS) == len(set(MATCH_TASKS))
        assert len(CROSS_TASKS) == len(set(CROSS_TASKS))
