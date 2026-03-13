"""Tests for the bottle packing FSM."""

import numpy as np
import pytest

from mujoco_manip.controller import IKController
from mujoco_manip.robot import PandaRobot
from mujoco_manip.tasks.bottle_packing.constants import ACTION_REPEAT
from mujoco_manip.tasks.bottle_packing.env import BottlePackingEnv
from mujoco_manip.tasks.bottle_packing.fsm import (
    GRIPPER_CLOSE_STEPS,
    Phase,
    State,
    BottlePackingTask,
    _STATE_TO_PHASE,
)


@pytest.fixture
def sim():
    env = BottlePackingEnv(add_wrist_camera=False)
    env.reset_to_keyframe("scene_start")
    env.setup_scene(num_prepacked=0)
    env.spawn_bottle_at_pickup(0)
    robot = PandaRobot(env.model, env.data)
    ctrl = IKController(env.model, env.data, robot)
    return env, robot, ctrl


@pytest.fixture
def task(sim):
    env, robot, ctrl = sim
    return BottlePackingTask(env, robot, ctrl, well_index=0)


# ---------------------------------------------------------------------------
# State ↔ Phase mapping
# ---------------------------------------------------------------------------


class TestStateToPhase:
    def test_all_states_mapped(self):
        for s in State:
            assert s in _STATE_TO_PHASE, f"State {s} not mapped to a Phase"

    def test_phase_property(self, task):
        assert task.phase == Phase.IDLE

    def test_idle_then_approaching(self, task):
        task.plan()
        assert task.phase == Phase.APPROACHING


# ---------------------------------------------------------------------------
# Phase descriptions
# ---------------------------------------------------------------------------


class TestPhaseDescription:
    def test_idle_description(self, task):
        assert task.phase_description == "idle"

    @pytest.mark.parametrize(
        "phase,expected_substring",
        [
            (Phase.APPROACHING, "approaching"),
            (Phase.GRASPING, "grasping"),
            (Phase.LIFTING, "lifting"),
            (Phase.TRANSPORTING, "transporting"),
            (Phase.PLACING, "placing"),
            (Phase.RETREATING, "retreating"),
        ],
    )
    def test_phase_descriptions_contain_keyword(self, task, phase, expected_substring):
        # Force the state to one that maps to this phase
        for s, p in _STATE_TO_PHASE.items():
            if p == phase:
                task.state = s
                break
        desc = task.phase_description
        assert expected_substring in desc.lower()


# ---------------------------------------------------------------------------
# Plan does not actuate
# ---------------------------------------------------------------------------


class TestPlanDoesNotActuate:
    def test_plan_only_updates_state(self, task, sim):
        _, robot, _ = sim
        qpos_before = robot.arm_qpos.copy()
        task.plan()
        qpos_after = robot.arm_qpos.copy()
        np.testing.assert_array_equal(qpos_before, qpos_after)


# ---------------------------------------------------------------------------
# Gripper val
# ---------------------------------------------------------------------------


class TestGripperVal:
    def test_starts_open(self, task):
        assert task.gripper_val == 1.0

    def test_closes_on_close_gripper_state(self, task, sim):
        env, robot, ctrl = sim
        # Drive to CLOSE_GRIPPER state
        task.plan()  # IDLE → PRE_GRASP
        task.state = State.GRASP
        task.controller.pos_tolerance = 999.0  # force reached
        task.plan()  # GRASP → CLOSE_GRIPPER
        assert task.gripper_val == 0.0

    def test_opens_on_release_state(self, task, sim):
        task.state = State.RELEASE
        task._target_pos = np.array([0.0, 0.0, 0.42])
        task.settle_counter = 1
        task.plan()  # ramp finishes → RELEASE_WAIT, gripper fully open
        assert task.gripper_val == 1.0


# ---------------------------------------------------------------------------
# plan(n_steps) scales timers
# ---------------------------------------------------------------------------


class TestPlanNSteps:
    def test_settle_counter_decrements(self, task):
        task.state = State.CLOSE_GRIPPER
        task.settle_counter = GRIPPER_CLOSE_STEPS
        task._target_pos = np.array([0, 0, 0.5])
        task.plan(n_steps=ACTION_REPEAT)
        assert task.settle_counter == GRIPPER_CLOSE_STEPS - ACTION_REPEAT


# ---------------------------------------------------------------------------
# FSM completes an episode
# ---------------------------------------------------------------------------


class TestFSMCompletion:
    def test_plan_actuate_completes(self, task, sim):
        env, robot, ctrl = sim
        for _ in range(30000):
            task.plan(ACTION_REPEAT)
            task._actuate()
            for _ in range(ACTION_REPEAT):
                env.step()
            if task.is_done:
                break
        assert task.is_done, f"FSM stuck in state {task.state}"

    def test_phase_progresses(self, task, sim):
        env, robot, ctrl = sim
        seen_phases = set()
        for _ in range(30000):
            task.plan(ACTION_REPEAT)
            task._actuate()
            for _ in range(ACTION_REPEAT):
                env.step()
            seen_phases.add(task.phase)
            if task.is_done:
                break
        # Should have visited multiple phases
        assert len(seen_phases) >= 4


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_well_index(self, task):
        assert task.well_index == 0

    def test_is_done_false_initially(self, task):
        assert not task.is_done

    def test_target_pos_none_initially(self, task):
        assert task.target_pos is None

    def test_target_pos_set_after_plan(self, task):
        task.plan()
        assert task.target_pos is not None
        assert task.target_pos.shape == (3,)
