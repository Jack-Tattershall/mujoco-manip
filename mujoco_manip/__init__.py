from gymnasium.envs.registration import register

register(
    id="mujoco_manip/PickPlace-v0",
    entry_point="mujoco_manip.tasks.pick_and_place.gym_env:PickPlaceGymEnv",
)

register(
    id="mujoco_manip/BottlePacking-v0",
    entry_point="mujoco_manip.tasks.bottle_packing.gym_env:BottlePackingGymEnv",
)
