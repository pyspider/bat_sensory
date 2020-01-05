from gym.envs.registration import register

register(
    id='LidarBatEnv-v0',
    entry_point='lidar_bat_env.bat_flying_env:BatFlyingEnv'
)
