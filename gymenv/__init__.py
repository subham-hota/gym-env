from gym.envs.registration import register

register(
    id='gymenv/UncertainForaging-v0',
    entry_point='gymenv.envs:UncertainForageEnv',
    max_episode_steps=1500,
)
