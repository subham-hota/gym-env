from gym.envs.registration import register

register(
    id='gym_env/UncertainForaging-v0',
    entry_point='gym_env.envs:UncertainForageEnv',
    max_episode_steps=1500,
)
