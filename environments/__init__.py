from gym.envs.registration import register

register(
    id='LidarBatEnv-v0',
    entry_point='environments.bat_flying_env:BatFlyingEnv'
)
