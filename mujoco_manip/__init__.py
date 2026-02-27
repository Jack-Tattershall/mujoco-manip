from gymnasium.envs.registration import register

register(
    id="mujoco_manip/PickPlace-v0",
    entry_point="mujoco_manip.gym_env:PickPlaceGymEnv",
)
