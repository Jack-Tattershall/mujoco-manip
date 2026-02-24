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

Uses [Hydra](https://hydra.cc/) for configuration (see `configs/generate.yaml` for defaults).

```bash
# All 9 object-bin combinations, 10 episodes
uv run python scripts/generate_dataset.py repo_id=user/pick-place num_episodes=10

# Only colour-matched tasks, 100 episodes
uv run python scripts/generate_dataset.py repo_id=user/pick-place-match num_episodes=100 tasks=match

# Only cross-colour tasks, custom root
uv run python scripts/generate_dataset.py repo_id=user/pick-place-cross num_episodes=60 tasks=cross root=./my-datasets
```

## Visualise a dataset

Logs all dataset features to [Rerun](https://rerun.io/) — images, states, actions, keypoints, onehots, and 3D trails.

```bash
uv run python scripts/visualize_dataset.py --repo-id user/pick-place --root ./datasets --episode-index 0
```

## Replay actions in MuJoCo

Feeds recorded actions back through IK in the MuJoCo viewer to verify trajectories.

```bash
uv run python scripts/replay_actions.py \
    --repo-id user/pick-place --root ./datasets/user/pick-place \
    --episode-index 0 --action-key action.ee.8dof_rel
```

## Push a dataset to Hugging Face

```bash
# Login (one-time)
uv run huggingface-cli login

# Upload to the kinisi org
uv run huggingface-cli upload kinisi/pick-place ./datasets/user/pick-place --repo-type dataset --private
```

## Run tests

```bash
uv run pytest tests/ -v
```
