# mujoco-manip

Franka Panda pick-and-place simulation in MuJoCo with a Gymnasium env and LeRobot dataset generation.

## Setup

```bash
# Install dependencies
uv sync

# Clone the Panda robot model
bash setup_menagerie.sh
```

## Run the interactive demo

Opens a MuJoCo viewer and runs the FSM to pick and place all 3 coloured cubes into their bins.

```bash
uv run python main.py
```

## Gymnasium environment

```python
from src.gym_env import PickPlaceGymEnv

# Default: ee_8dof action mode, random task from all 9 object-bin combos
env = PickPlaceGymEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# 10DOF actions (6D rotation representation)
env = PickPlaceGymEnv(action_mode="ee_10dof")

# Absolute position actions (legacy 4D)
env = PickPlaceGymEnv(action_mode="abs_pos")

# Fix a specific task
env = PickPlaceGymEnv(task=("obj_red", "bin_blue"))

# Use a task set: "all" (9), "match" (3 colour-matched), "cross" (6 cross-colour)
env = PickPlaceGymEnv(tasks="match")
```

## Generate a LeRobot dataset

```bash
# All 9 object-bin combinations, 10 episodes
uv run python scripts/generate_dataset.py --repo-id user/pick-place --num-episodes 10 --tasks all --root ./datasets

# Only colour-matched tasks, 100 episodes
uv run python scripts/generate_dataset.py --repo-id user/pick-place-match --num-episodes 100 --tasks match --root ./datasets

# Only cross-colour tasks
uv run python scripts/generate_dataset.py --repo-id user/pick-place-cross --num-episodes 60 --tasks cross --root ./datasets
```

## Visualise a dataset

Custom visualizer that logs **all** dataset features to [Rerun](https://rerun.io/) — images, EE states (8dof/10dof), actions, keypoints, and target bin one-hot — with named scalar dimensions.

```bash
# Open Rerun viewer for episode 0
uv run python scripts/visualize_dataset.py --repo-id user/pick-place --root ./datasets --episode-index 0

# Save a .rrd file for later viewing
uv run python scripts/visualize_dataset.py --repo-id user/pick-place --root ./datasets --episode-index 0 --save ./viz/ep0.rrd
rerun ./viz/ep0.rrd
```

The built-in `lerobot-dataset-viz` CLI is also available but only shows images and basic action/state scalars:

```bash
uv run lerobot-dataset-viz --repo-id user/pick-place --root ./datasets --episode-index 0 --display-compressed-images 0
```

## Run tests

```bash
uv run pytest tests/ -v
```
