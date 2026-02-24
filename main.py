"""Entry point for the Franka Panda pick-and-place simulation."""

import os
import sys
import time

from src.env import PickPlaceEnv
from src.robot import PandaRobot
from src.controller import IKController
from src.pick_and_place import PickAndPlaceTask


SCENE_XML = os.path.join(os.path.dirname(__file__), "pick_and_place_scene.xml")
MENAGERIE_DIR = os.path.join(os.path.dirname(__file__), "third_party", "mujoco_menagerie")


def main():
    # Check menagerie is cloned
    if not os.path.isdir(MENAGERIE_DIR):
        print(f"Error: mujoco_menagerie not found at {MENAGERIE_DIR}")
        print("Run: bash setup_menagerie.sh")
        sys.exit(1)

    # Load environment
    print("Loading scene...")
    env = PickPlaceEnv(SCENE_XML)
    env.reset_to_keyframe("scene_start")

    # Create robot interface and controller
    robot = PandaRobot(env.model, env.data)
    controller = IKController(env.model, env.data, robot)

    # Create task state machine
    task = PickAndPlaceTask(env, robot, controller)

    # Launch viewer
    print("Launching viewer...")
    env.launch_viewer()

    # Main simulation loop
    print("Starting pick-and-place task...")
    last_status = ""
    step_count = 0

    while env.is_running():
        status = task.update()
        if status != last_status and "Gripping" not in status and "Releasing (" not in status:
            print(f"[{step_count:>6d}] {status}")
            last_status = status
        elif status != last_status:
            last_status = status

        env.step()
        env.sync()
        step_count += 1

        if task.is_done:
            print(f"\nTask complete after {step_count} steps!")
            print("Keeping viewer open — close the window to exit.")
            break

    # Keep viewer alive after task completion
    while env.is_running():
        env.step()
        env.sync()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
