from gym.envs.registration import register

register(
    id='gym-env/UncertainForaging-v0',
    entry_point='gym-env.envs:UncertainForageEnv',
    max_episode_steps=1500,
)
