import math

import numpy as np
import gym
from gym import wrappers
import cv2

import environments
from agents.dqn.dqn import *

# for debug
from gym import logger
# logger.set_level(10)


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
    env = wrappers.Monitor(env, './videos/dqn', force=True)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = Agent(num_states, num_actions)

    episode_10_list = np.zeros(10)
    complete_episodes = 0
    episode_final = False

    for i_episode in range(500):
        observation = env.reset()
        state = observation
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0)
        for step in range(200):
            # print(f'----step {step}----')
            # print(f'bat angle: {env.bat.angle *180 / math.pi:2f} [degree]')
            # print('observation:')
            # print(observation)

            action = agent.get_action(state, i_episode)
            # print(f'action: {action}')

            observation_next, _, done, _ = env.step(action.item())
            # print(f'reward: {reward}')
            # print(f'done: {done}')
            # print(f'time: {env.t:2f} [sec]')
            if done:
                print(f"Episode finished after {step+1} timesteps")
                state_next = None
                episode_10_list = np.hstack((episode_10_list[1:], step+1))

                if step < 195:
                    reward = torch.FloatTensor([-1.0])
                    complete_episodes = 0
                else:
                    reward = torch.FloatTensor([1.0])
                    complete_episodes = complete_episodes + 1
            else:
                reward = torch.FloatTensor([0.0])
                state_next = observation_next
                state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                state_next = torch.unsqueeze(state_next, 0)

            agent.memorize(state, action, state_next, reward)
            agent.update_q_function()
            state = state_next

            if done:
                print(f'{i_episode} Episode: Finished after {step+1} steps: average 10 steps = {episode_10_list.mean()}')

                if (i_episode % 2 == 0):
                    agent.update_target_q_function()
                break

        if complete_episodes >= 10:
            print('Clear 10 times in a row')
            break

        env.reset()
    env.close()



if __name__ == "__main__":
    train_dqn()
    # main()