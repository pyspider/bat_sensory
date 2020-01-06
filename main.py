import math
import gym
from gym import wrappers

import cv2

import environments
from agents.dqn import *

# for debug
from gym import logger
logger.set_level(10)


TIME_STEP = 0.005

ENV = 'LidarBatEnv-v0'
MAX_STEPS = 1000
NUM_EPISODES = 500

def main():
    env = gym.make('LidarBatEnv-v0')
    env = wrappers.Monitor(env, './videos/test', force=True)
    for i_episode in range(5):
        observation = env.reset()
        for t in range(1000):
            print(f'----step {t}----')
            print(f'bat angle: {env.bat.angle *180 / math.pi:2f} [degree]')
            print('observation:')
            print(observation)
            action = env.action_space.sample()
            # action[0] = 50
            # action[1] = math.pi/2
            action[2] = 0.9
            action[3] = 0
            print(f'action: {action}')
            observation, reward, done, info = env.step(action)
            print(f'reward: {reward}')
            print(f'done: {done}')
            print(f'time: {env.t:2f} [sec]')
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break
        env.reset()
    env.close()

def train_dqn():
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device name: {device}')

    # env = gym.make('LidarBatEnv-v0')
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, './videos/dqn')

    observation = env.reset()
    num_states = observation.n
    agent = Agent()

    for i_episode in range(500):
        observation = env.reset()
        for t in range(200):
            print(f'----step {t}----')
            print(f'bat angle: {env.bat.angle *180 / math.pi:2f} [degree]')
            print('observation:')
            print(observation)
            action = env.action_space.sample()


            print(f'action: {action}')
            observation, reward, done, info = env.step(action)
            print(f'reward: {reward}')
            print(f'done: {done}')
            print(f'time: {env.t:2f} [sec]')
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break
        env.reset()
    env.close()



if __name__ == "__main__":
    # train_dqn()
    main()