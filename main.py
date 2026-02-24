"""Entry point for the Franka Panda pick-and-place simulation."""

import os
import sys
import time

from src.env import PickPlaceEnv
from src.robot import PandaRobot
from src.controller import IKController
from src.pick_and_place import PickAndPlaceTask


SCENE_XML = os.path.join(os.path.dirname(__file__), "pick_and_place_scene.xml")
MENAGERIE_DIR = os.path.join(
    os.path.dirname(__file__), "third_party", "mujoco_menagerie"
)


def main() -> None:
    """Run the interactive pick-and-place simulation with a passive viewer."""
    if not os.path.isdir(MENAGERIE_DIR):
        print(f"Error: mujoco_menagerie not found at {MENAGERIE_DIR}")
        print("Run: bash setup_menagerie.sh")
        sys.exit(1)

    print("Loading scene...")
    env = PickPlaceEnv(SCENE_XML)
    env.reset_to_keyframe("scene_start")

    robot = PandaRobot(env.model, env.data)
    controller = IKController(env.model, env.data, robot)

    task = PickAndPlaceTask(env, robot, controller)

    print("Launching viewer...")
    env.launch_viewer()

    print("Starting pick-and-place task...")
    last_status = ""
    step_count = 0

    while env.is_running():
        status = task.update()
        if (
            status != last_status
            and "Gripping" not in status
            and "Releasing (" not in status
        ):
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

    while env.is_running():
        env.step()
        env.sync()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
