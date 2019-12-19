import gym
from lidar_bat_env.bat_flying_env import BatFlyingEnv

def main():
    env = BatFlyingEnv(
        world_width=6.5, world_height=1.5, discrete_length=0.01)
    env.reset()
    for i_episode in range(2):
        observation = env.reset()
        for t in range(400):
            env.render()
            print(observation)
            action = env.action_space.sample()
            print(action)
            observation, reward, done, info = env.step(action)
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break
    env.close()

if __name__ == "__main__":
    main()