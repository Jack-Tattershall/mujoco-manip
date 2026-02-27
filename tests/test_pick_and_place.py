"""Tests for the pick-and-place FSM, Phase enum, and phase descriptions."""

import numpy as np
import pytest

from mujoco_manip.constants import ACTION_REPEAT
from mujoco_manip.controller import IKController
from mujoco_manip.data import SCENE_XML
from mujoco_manip.env import PickPlaceEnv
from mujoco_manip.pick_and_place import (
    GRIPPER_SETTLE_STEPS,
    Phase,
    PickAndPlaceTask,
    State,
    _STATE_TO_PHASE,
)
from mujoco_manip.robot import PandaRobot


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


# ---------------------------------------------------------------------------
# plan() / _actuate() split and gripper_val
# ---------------------------------------------------------------------------


class TestPlanDoesNotActuate:
    def test_plan_does_not_move_robot(self, sim):
        """Calling plan() alone should not change robot joint positions."""
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller, tasks=[("obj_red", "bin_red")])

        qpos_before = robot.arm_qpos.copy()
        ctrl_before = env.data.ctrl.copy()

        # plan() should only update FSM state, not actuate
        task.plan(1)

        qpos_after = robot.arm_qpos
        ctrl_after = env.data.ctrl

        np.testing.assert_array_equal(qpos_before, qpos_after)
        np.testing.assert_array_equal(ctrl_before, ctrl_after)


class TestGripperVal:
    def test_initial_gripper_val_is_open(self, sim):
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller)
        assert task.gripper_val == 1.0
        assert task._gripper_open is True

    def test_gripper_closes_at_close_gripper_state(self, sim):
        """After plan() transitions to CLOSE_GRIPPER, gripper_val should be 0."""
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller, tasks=[("obj_red", "bin_red")])

        # Run update() until we hit CLOSE_GRIPPER state
        steps = 0
        max_steps = 10_000
        while task.state != State.CLOSE_GRIPPER and steps < max_steps:
            task.update()
            env.step()
            steps += 1

        assert task.state == State.CLOSE_GRIPPER
        assert task.gripper_val == 0.0
        assert task._gripper_open is False

    def test_gripper_reopens_at_release_state(self, sim):
        """After plan() transitions to RELEASE, gripper_val should be 1."""
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller, tasks=[("obj_red", "bin_red")])

        steps = 0
        max_steps = 20_000
        while task.state != State.RELEASE and steps < max_steps:
            task.update()
            env.step()
            steps += 1

        assert task.state == State.RELEASE
        assert task.gripper_val == 1.0
        assert task._gripper_open is True


class TestPlanNStepsScalesTimers:
    def test_settle_counter_decrements_by_n_steps(self, sim):
        """plan(n_steps) should decrement settle_counter by n_steps."""
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller, tasks=[("obj_red", "bin_red")])

        # Run until CLOSE_GRIPPER state (which has a settle counter)
        steps = 0
        max_steps = 10_000
        while task.state != State.CLOSE_GRIPPER and steps < max_steps:
            task.update()
            env.step()
            steps += 1

        assert task.state == State.CLOSE_GRIPPER
        counter_before = task.settle_counter

        # Now call plan() with n_steps=5
        task.plan(5)
        assert task.settle_counter == counter_before - 5

    def test_large_n_steps_completes_settle(self, sim):
        """plan() with large n_steps should transition past settle states."""
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller, tasks=[("obj_red", "bin_red")])

        # Advance to CLOSE_GRIPPER
        steps = 0
        while task.state != State.CLOSE_GRIPPER and steps < 10_000:
            task.update()
            env.step()
            steps += 1

        assert task.state == State.CLOSE_GRIPPER

        # A single plan() with enough steps should clear the counter
        task.plan(GRIPPER_SETTLE_STEPS + 10)
        assert task.state == State.LIFT


class TestPlanCompletesEpisode:
    def test_plan_with_action_repeat_completes(self, sim):
        """Run plan(ACTION_REPEAT) + _actuate() in a loop to complete episode."""
        env, robot, controller = sim
        task = PickAndPlaceTask(env, robot, controller, tasks=[("obj_red", "bin_red")])

        gym_steps = 0
        max_gym_steps = 2000
        while not task.is_done and gym_steps < max_gym_steps:
            task.plan(ACTION_REPEAT)
            task._actuate()
            for _ in range(ACTION_REPEAT):
                env.step()
            gym_steps += 1

        assert task.is_done, f"Episode did not finish in {max_gym_steps} gym steps"
