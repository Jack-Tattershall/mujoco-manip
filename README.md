# mujoco-manip

Franka Panda pick-and-place simulation in MuJoCo with a Gymnasium env and LeRobot dataset generation.

## Setup

```bash
uv sync
```

## Gymnasium environment

```python
from mujoco_manip.gym_env import PickPlaceGymEnv

# Default: ee_pos_quat_g_rel action mode (relative to initial EE pose),
# random task from all 9 object-bin combos
env = PickPlaceGymEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# Relative actions with 6D rotation (relative to initial EE pose)
env = PickPlaceGymEnv(action_mode="ee_pos_rot6d_g_rel")

# Absolute SE(3) actions in world frame (quaternion)
env = PickPlaceGymEnv(action_mode="ee_pos_quat_g")

# Absolute SE(3) actions in world frame (6D rotation)
env = PickPlaceGymEnv(action_mode="ee_pos_rot6d_g")

# Absolute position actions (4D)
env = PickPlaceGymEnv(action_mode="abs_pos")

# Fix a specific task
env = PickPlaceGymEnv(task=("obj_red", "bin_blue"))

# Use a task set:
#   "match" (3) — each object to its same-colour bin (red→red, green→green, blue→blue)
#   "cross"  (6) — each object to a different-colour bin (red→green, red→blue, etc.)
#   "all"    (9) — every object-bin combination (match + cross)
env = PickPlaceGymEnv(tasks="match")

# Randomize object positions on each reset
env = PickPlaceGymEnv(randomize_objects=True)
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

# Randomize object positions each episode (seeded for reproducibility)
uv run python scripts/generate_dataset.py repo_id=user/pick-place-rand num_episodes=100 randomize_objects=true seed=42

# Single fixed task (e.g. red cube → red bin) with randomization
uv run python scripts/generate_dataset.py \
    repo_id=user/pick-place-simple-rand num_episodes=1000 \
    tasks='[["obj_red","bin_red"]]' randomize_objects=true seed=42
```

## Visualise a dataset

Logs all dataset features to [Rerun](https://rerun.io/) — images, EE states (pos+quat / pos+rot6d), actions, keypoints, and target one-hots — with named scalar dimensions and 3D EE trails.

```bash
uv run python scripts/visualize_dataset.py --repo-id user/pick-place --episode-index 0

# Save a .rrd file for later viewing
uv run python scripts/visualize_dataset.py --repo-id user/pick-place --episode-index 0 --save ./viz/ep0.rrd
```

## Replay actions in MuJoCo

Feeds recorded dataset actions back through IK in the MuJoCo viewer to verify they reproduce the original trajectory.

```bash
# Replay absolute pos+quat actions
uv run python scripts/replay_actions.py --repo-id user/pick-place --episode-index 0

# Replay relative actions (reconstructed via T_initial)
uv run python scripts/replay_actions.py \
    --repo-id user/pick-place --episode-index 0 --action-key action.ee.pos_quat_g_rel

# Slow motion (2x slower)
uv run python scripts/replay_actions.py --repo-id user/pick-place --episode-index 0 --slow 2
```

## Run the interactive demo

Opens a MuJoCo viewer and runs the FSM to pick and place all 3 coloured cubes into their bins. Requires `bash setup_menagerie.sh` to clone the Panda model.

```bash
uv run python main.py

# Randomize object positions
uv run python main.py --randomize --seed 42
```

## Push a dataset to Hugging Face

```bash
uv run hf login
uv run hf upload user/pick-place ./datasets/user/pick-place --repo-type dataset --private
```

## Run tests

```bash
uv run pytest tests/ -v
```
