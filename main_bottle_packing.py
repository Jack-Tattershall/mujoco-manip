"""Entry point for the Franka Panda bottle packing simulation."""

import argparse
import random
import time

from mujoco_manip.controller import IKController
from mujoco_manip.robot import PandaRobot
from mujoco_manip.tasks.bottle_packing.constants import NUM_WELLS, well_row_col
from mujoco_manip.tasks.bottle_packing.env import BottlePackingEnv
from mujoco_manip.tasks.bottle_packing.fsm import BottlePackingTask, State


def main() -> None:
    """Run the interactive bottle packing simulation with a passive viewer."""
    parser = argparse.ArgumentParser(description="Run bottle packing simulation")
    parser.add_argument(
        "--slow", type=float, default=1.0, help="Slowdown factor (e.g. 2 = half speed)"
    )
    parser.add_argument(
        "--task",
        choices=["sequential", "random"],
        default="sequential",
        help="'sequential' packs wells 0,1,2,...; 'random' picks a random empty well each time",
    )
    parser.add_argument(
        "--well", type=int, default=None, help="Pack a single well index (0-19)"
    )
    parser.add_argument(
        "--num-bottles",
        type=int,
        default=NUM_WELLS,
        help="Number of bottles to pack (default: all 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (for --task random)"
    )
    args = parser.parse_args()

    print("Loading scene...")
    env = BottlePackingEnv(add_wrist_camera=True)
    env.reset_to_keyframe("scene_start")
    env.colorize_bottles()

    robot = PandaRobot(env.model, env.data)
    controller = IKController(env.model, env.data, robot)

    num_bottles = min(args.num_bottles, NUM_WELLS)

    if args.well is not None:
        # Single-well mode
        bottle_indices = [args.well]
        well_schedule = [args.well]
        num_prepacked = args.well
    elif args.task == "random":
        rng = random.Random(args.seed)
        available = list(range(NUM_WELLS))
        rng.shuffle(available)
        well_schedule = available[:num_bottles]
        bottle_indices = list(range(num_bottles))
        num_prepacked = 0
    else:
        # Sequential: bottle i -> well i
        bottle_indices = list(range(num_bottles))
        well_schedule = list(range(num_bottles))
        num_prepacked = 0

    print(f"Task: {args.task}, {len(well_schedule)} bottles")
    if args.task == "random":
        print(f"Well order: {well_schedule}")

    print("Launching viewer...")
    env.launch_viewer()

    env.setup_scene(num_prepacked=num_prepacked)
    env.load_conveyor(bottle_indices)

    step_time = env.model.opt.timestep * args.slow
    total_steps = 0
    bottles_packed = 0
    well_iter = iter(well_schedule)

    task: BottlePackingTask | None = None
    current_well: int | None = None
    current_bottle: int | None = None
    conveyor_resumed = True
    last_status = ""

    while env.is_running():
        t_start = time.monotonic()

        # --- FSM tick ---
        if task is not None:
            status = task.update()

            if (
                status != last_status
                and "Gripping" not in status
                and "Releasing (" not in status
                and "Settling above well (" not in status
            ):
                print(f"  [{total_steps:>6d}] {status}")
                last_status = status
            elif status != last_status:
                last_status = status

            # Resume conveyor once bottle is lifted clear of belt
            if not conveyor_resumed and task.state == State.LIFT:
                env.resume_conveyor()
                conveyor_resumed = True

            if task.is_done:
                env.mark_bottle_packed(current_bottle)
                bottles_packed += 1
                row, col = well_row_col(current_well)
                print(f"  Bottle packed into well ({row},{col}).")
                task = None
                current_well = None

        # --- Physics ---
        env.step()

        # --- Conveyor tick (kinematic, after physics) ---
        env.tick_conveyor()

        # --- Start new task when robot is free and a bottle is waiting ---
        if task is None:
            bottle_idx = env.start_pickup()
            if bottle_idx is not None:
                current_well = next(well_iter, None)
                if current_well is not None:
                    row, col = well_row_col(current_well)
                    print(f"\n=== Bottle {bottle_idx} -> well ({row},{col}) ===")
                    current_bottle = bottle_idx
                    task = BottlePackingTask(
                        env, robot, controller, well_index=current_well
                    )
                    conveyor_resumed = False
                    last_status = ""

        env.sync()
        total_steps += 1

        elapsed = time.monotonic() - t_start
        sleep = step_time - elapsed
        if sleep > 0:
            time.sleep(sleep)

        if bottles_packed >= len(well_schedule) and task is None:
            break

    print(f"\nAll done! {bottles_packed} bottles packed in {total_steps} total steps.")
    print("Keeping viewer open — close the window to exit.")

    while env.is_running():
        env.step()
        env.sync()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
