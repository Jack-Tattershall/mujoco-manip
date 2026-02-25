"""Tests for the pick-and-place FSM, Phase enum, and phase descriptions."""

import os

import pytest

from mujoco_manip.controller import IKController
from mujoco_manip.env import PickPlaceEnv
from mujoco_manip.pick_and_place import (
    Phase,
    PickAndPlaceTask,
    State,
    _STATE_TO_PHASE,
)
from mujoco_manip.robot import PandaRobot

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(_PROJECT_ROOT, "pick_and_place_scene.xml")


@pytest.fixture
def sim() -> tuple[PickPlaceEnv, PandaRobot, IKController]:
    """Create a sim environment, robot, and controller, reset to keyframe."""
    env = PickPlaceEnv(SCENE_XML, add_wrist_camera=False)
    robot = PandaRobot(env.model, env.data)
    controller = IKController(env.model, env.data, robot)
    env.reset_to_keyframe("scene_start")
    return env, robot, controller


# ---------------------------------------------------------------------------
# Phase enum & mapping
# ---------------------------------------------------------------------------


class TestStateToPhaseMapping:
    def test_every_state_has_a_phase(self):
        for state in State:
            assert state in _STATE_TO_PHASE

    def test_mapping_values_are_phases(self):
        for phase in _STATE_TO_PHASE.values():
            assert isinstance(phase, Phase)

    def test_expected_groupings(self):
        assert _STATE_TO_PHASE[State.PRE_GRASP] == Phase.APPROACHING
        assert _STATE_TO_PHASE[State.GRASP] == Phase.GRASPING
        assert _STATE_TO_PHASE[State.CLOSE_GRIPPER] == Phase.GRASPING
        assert _STATE_TO_PHASE[State.LIFT] == Phase.LIFTING
        assert _STATE_TO_PHASE[State.MOVE_TO_BIN] == Phase.TRANSPORTING
        assert _STATE_TO_PHASE[State.SETTLE_AT_BIN] == Phase.TRANSPORTING
        assert _STATE_TO_PHASE[State.LOWER_TO_BIN] == Phase.PLACING
        assert _STATE_TO_PHASE[State.RELEASE] == Phase.PLACING
        assert _STATE_TO_PHASE[State.RETREAT] == Phase.RETREATING


# ---------------------------------------------------------------------------
# phase property
# ---------------------------------------------------------------------------


class TestPhaseProperty:
    def test_initial_phase_is_idle(self, sim):
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller)
        assert task.phase == Phase.IDLE

    def test_phase_matches_state(self, sim):
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller)
        for state in State:
            task.state = state
            assert task.phase == _STATE_TO_PHASE[state]


# ---------------------------------------------------------------------------
# phase_description property
# ---------------------------------------------------------------------------


class TestPhaseDescription:
    def test_idle_description(self, sim):
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller, tasks=[("obj_red", "bin_blue")])
        task.state = State.IDLE
        assert task.phase_description == "idle"

    def test_approaching_includes_object_color(self, sim):
        env, robot, controller = sim
        task = PickAndPlaceTask(
            env, robot, controller, tasks=[("obj_green", "bin_red")]
        )
        task.state = State.PRE_GRASP
        assert task.phase_description == "approaching the green cube"

    def test_grasping_from_close_gripper(self, sim):
        env, robot, controller = sim
        task = PickAndPlaceTask(
            env, robot, controller, tasks=[("obj_blue", "bin_blue")]
        )
        task.state = State.CLOSE_GRIPPER
        assert task.phase_description == "grasping the blue cube"

    def test_transporting_includes_both_colors(self, sim):
        env, robot, controller = sim
        task = PickAndPlaceTask(
            env, robot, controller, tasks=[("obj_red", "bin_green")]
        )
        task.state = State.MOVE_TO_BIN
        desc = task.phase_description
        assert "red" in desc
        assert "green" in desc
        assert desc == "transporting the red cube to the green bin"

    def test_placing_includes_both_colors(self, sim):
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller, tasks=[("obj_blue", "bin_red")])
        task.state = State.LOWER_TO_BIN
        assert task.phase_description == "placing the blue cube in the red bin"

    def test_retreating_description(self, sim):
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller)
        task.state = State.RETREAT
        assert task.phase_description == "retreating to neutral position"

    def test_done_description(self, sim):
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller)
        task.state = State.DONE
        assert task.phase_description == "idle"

    def test_description_is_always_a_string(self, sim):
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller)
        for state in State:
            task.state = state
            assert isinstance(task.phase_description, str)
            assert len(task.phase_description) > 0


# ---------------------------------------------------------------------------
# Phase descriptions change during a full episode
# ---------------------------------------------------------------------------


class TestPhaseProgression:
    def test_phases_change_during_episode(self, sim):
        """Run the FSM to completion and verify multiple phases appear."""
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller, tasks=[("obj_red", "bin_red")])
        seen_phases: set[Phase] = set()
        seen_descriptions: set[str] = set()
        steps = 0
        max_steps = 20_000
        while not task.is_done and steps < max_steps:
            task.update()
            env.step()
            seen_phases.add(task.phase)
            seen_descriptions.add(task.phase_description)
            steps += 1

        assert task.is_done, f"Episode did not finish in {max_steps} steps"
        # Should have visited at least: IDLE, APPROACHING, GRASPING, LIFTING,
        # TRANSPORTING, PLACING, RETREATING, DONE
        assert len(seen_phases) >= 6
        assert len(seen_descriptions) >= 4
