import math

import numpy as np
import gym
from gym import wrappers

import environments
from agents.dqn.dqn import *

# for debug
from gym import logger
# logger.set_level(10)


TIME_STEP = 0.005

def main():
    env = gym.make('LidarBat-v0')
    # env = gym.make('CartPole-v0')
    # env = gym.make('Pendulum-v0')
    # env = gym.make('BipedalWalker-v2')
    env = wrappers.Monitor(env, 'data/env_test_videos', force=True)
    for i_episode in range(5):
        observation = env.reset()
        for t in range(1000):
            print(f'----step {t}----')
            # print(f'bat angle: {env.bat.angle *180 / math.pi:2f} [degree]')
            print('observation:')
            print(observation)
            action = env.action_space.sample()
            # action[0] = 50
            # action[1] = math.pi/2
            # action[2] = 0.9
            # action[3] = 0
            print(f'action: {action}')
            observation, reward, done, info = env.step(action)
            print(f'reward: {reward}')
            print(f'done: {done}')
            # print(f'time: {env.t:2f} [sec]')
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break
        env.reset()
    env.close()


if __name__ == "__main__":
    main()