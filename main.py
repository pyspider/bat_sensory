import math
import gym
from lidar_bat_env.bat_flying_env import BatFlyingEnv
from lidar_bat_env.lidar_bat import LidarBat

TIME_STEP = 0.005

def main():
    bat = LidarBat(0, 0.3, 0.75, 3, dt=TIME_STEP)
    env = BatFlyingEnv(
        world_width=1.5, world_height=1.5, discrete_length=0.01, dt=TIME_STEP, bat=bat)
    env.reset()
    for i_episode in range(10):
        bat = LidarBat(0, 0.3, 0.75, 1, dt=TIME_STEP)
        observation = env.reset(bat)
        for t in range(400):
            env.render(screen_width=500)
            print(f'bat angle: {bat.angle *180 / math.pi}')
            print(f'observation:{observation}')
            action = env.action_space.sample()
            # action[0] = 50
            # action[1] = math.pi/2
            action[2] = 0.9
            action[3] = 0
            print(f'action: {action}')
            observation, reward, done, info = env.step(action)
            print(f'reward: {reward}')
            print(f'done: {done}')
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break
        env.close()
    env.close()

if __name__ == "__main__":
    main()