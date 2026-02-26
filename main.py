"""Entry point for the Franka Panda pick-and-place simulation."""

import argparse
import os
import sys
import time

import numpy as np

from mujoco_manip.env import PickPlaceEnv
from mujoco_manip.robot import PandaRobot
from mujoco_manip.controller import IKController
from mujoco_manip.pick_and_place import PickAndPlaceTask


SCENE_XML = os.path.join(os.path.dirname(__file__), "pick_and_place_scene.xml")
MENAGERIE_DIR = os.path.join(
    os.path.dirname(__file__), "third_party", "mujoco_menagerie"
)


def main() -> None:
    """Run the interactive pick-and-place simulation with a passive viewer."""
    parser = argparse.ArgumentParser(description="Run pick-and-place simulation")
    parser.add_argument(
        "--slow", type=float, default=1.0, help="Slowdown factor (e.g. 2 = half speed)"
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize object positions each episode",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for object randomization"
    )
    args = parser.parse_args()

    if not os.path.isdir(MENAGERIE_DIR):
        print(f"Error: mujoco_menagerie not found at {MENAGERIE_DIR}")
        print("Run: bash setup_menagerie.sh")
        sys.exit(1)

    rng = np.random.default_rng(args.seed) if args.randomize else None

    print("Loading scene...")
    env = PickPlaceEnv(SCENE_XML, add_wrist_camera=True)
    env.reset_to_keyframe("scene_start")
    if rng is not None:
        env.randomize_objects(rng)
        print("Randomized object positions (seed={})".format(args.seed))

    robot = PandaRobot(env.model, env.data)
    controller = IKController(env.model, env.data, robot)

    task = PickAndPlaceTask(env, robot, controller)

    print("Launching viewer...")
    env.launch_viewer()

    print("Starting pick-and-place task...")
    last_status = ""
    step_count = 0
    step_time = env.model.opt.timestep * args.slow

    while env.is_running():
        t_start = time.monotonic()

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

        elapsed = time.monotonic() - t_start
        sleep = step_time - elapsed
        if sleep > 0:
            time.sleep(sleep)

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
