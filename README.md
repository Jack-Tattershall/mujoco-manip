# mujoco-manip

Franka Panda manipulation tasks in MuJoCo with Gymnasium environments and LeRobot dataset generation.

## Tasks

- **Pick-and-place** — 3 coloured cubes into 3 bins (9 task combos), with optional position randomization
- **Bottle packing** — pick bottles from a conveyor belt and place into a 5x4 crate (20 wells)

## Setup

```bash
uv sync
```

## Gymnasium environments

```python
from mujoco_manip.tasks.pick_and_place import PickPlaceGymEnv
from mujoco_manip.tasks.bottle_packing import BottlePackingGymEnv

# Pick-and-place (default: relative pose actions, random task from all 9 combos)
env = PickPlaceGymEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# Bottle packing (default: relative pose actions, random well)
env = BottlePackingGymEnv()
obs, info = env.reset()

# Fix a specific target
env = PickPlaceGymEnv(task=("obj_red", "bin_blue"))
env = BottlePackingGymEnv(well_index=5)

# Task sets
env = PickPlaceGymEnv(tasks="match")   # 3 same-colour combos
env = PickPlaceGymEnv(tasks="cross")   # 6 cross-colour combos
env = PickPlaceGymEnv(tasks="all")     # all 9

# Action modes (both tasks): abs_pos, ee_pos_quat_g, ee_pos_rot6d_g, ee_pos_quat_g_rel, ee_pos_rot6d_g_rel
env = PickPlaceGymEnv(action_mode="ee_pos_rot6d_g")
env = BottlePackingGymEnv(action_mode="ee_pos_rot6d_g_rel")

# Object randomization (pick-and-place only)
env = PickPlaceGymEnv(randomize_objects=True)
```

## Generate datasets

Uses [Hydra](https://hydra.cc/) for configuration.

```bash
# Pick-and-place
uv run python scripts/generate_dataset.py repo_id=user/pick-place num_episodes=100
uv run python scripts/generate_dataset.py repo_id=user/pick-place-rand num_episodes=100 randomize_objects=true seed=42

# Bottle packing
uv run python scripts/generate_bottle_packing_dataset.py repo_id=user/bottle-packing num_episodes=100
```

## Visualise a dataset

Logs features to [Rerun](https://rerun.io/) — images with keypoint overlays, EE state/action scalars, 3D trails, rewards, and target one-hots. Auto-detects task type.

```bash
uv run python scripts/visualize_dataset.py --repo-id user/pick-place --episode-index 0
uv run python scripts/visualize_dataset.py --repo-id user/bottle-packing --episode-index 0 --save ./viz/ep0.rrd
```

## Replay actions

Feeds recorded actions back through IK in the MuJoCo viewer.

```bash
uv run python scripts/replay_actions.py --repo-id user/pick-place --episode-index 0
uv run python scripts/replay_bottle_packing.py --repo-id user/bottle-packing --episode-index 0
```

## Interactive demos

```bash
uv run python main.py                          # pick-and-place
uv run python main.py --randomize --seed 42    # with randomization
uv run python main_bottle_packing.py --well 0  # bottle packing, specific well
```

## Tests

```bash
uv run pytest tests/ -v
```
