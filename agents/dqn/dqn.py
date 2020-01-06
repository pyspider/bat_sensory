import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from ..replay_memory import *

BATCH_SIZE = 32
CAPACITY = 10000
GAMMA = 0.999


class Brain(object):
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)

        self.model = nn.Sequential(
            nn.Linear(num_states, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions))

        print('Q-Network')
        print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        # do nothing when memory is small
        if len(self.memory) < BATCH_SIZE:
            return
        
        # sample mini-batch from memory
        transitions = self.memory.sample(BATCH_SIZE)

        # reshape transition
        # (s_t, a, s_t+1, r) x N -> (s_t x N, a x N, s_t+1 x N, r x N)
        batch = Transition(*zip(*transitions))

        # unpack each value
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([
            s for s in batch.next_state if s is not None])
        
        # switch network mode to evaluate
        self.model.eval()

        # calculate Q(s_t, a_t)
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # index mask for checking whether there is next state 
        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state)))
        
        # initilize to zero
        next_state_values = torch.zeros(BATCH_SIZE) 

        # calculate max Q
        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()

        # calculate Q(s_t, a_t)
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # switch network mode to train
        self.model.train()

        # calculate loss
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1))
        
        # update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])
        
        return action


class Agent(object):
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
    
    def update_q_function(self):
        self.brain.replay()
    
    def get_action(self):
        action = self.brain.dicide_action(state, episode)
        return action
    
    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)