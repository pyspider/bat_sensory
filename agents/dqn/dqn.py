import random
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from ..replay_memory import *

BATCH_SIZE = 32
CAPACITY = 10000
GAMMA = 0.99


class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)
    
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output

class Brain(object):
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)

        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)
        self.target_q_network = Net(n_in, n_mid, n_out)
        print('Q-Network')
        print(self.main_q_network)

        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.0001)

    def replay(self):
        # do nothing when memory is small
        if len(self.memory) < BATCH_SIZE:
            return
        
        (self.batch,
         self.state_batch,
         self.action_batch,
         self.reward_batch,
         self.non_final_next_states
        ) = self.make_minibatch()
         
        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main_q_network()

   
    def make_minibatch(self):
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
        
        return batch, state_batch, action_batch, reward_batch, non_final_next_states
    
    def get_expected_state_action_values(self):
        # switch network mode to evaluate
        self.main_q_network.eval()
        self.target_q_network.eval()

        # calculate Q(s_t, a_t)
        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        # index mask for checking whether there is next state 
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, self.batch.next_state)), 
            dtype=torch.bool)
        
        # initilize to zero
        next_state_values = torch.zeros(BATCH_SIZE) 

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # calculate max Q
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).max(1)[1].detach()
        
        # choose states have next_state
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # calculate target Q
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states
            ).detach().squeeze()

        # calculate Q(s_t, a_t)
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values
    
    def update_main_q_network(self):
        # switch network to train
        self.main_q_network.train()

        # calculate loss
        loss = F.smooth_l1_loss(
            self.state_action_values,
            self.expected_state_action_values.unsqueeze(1))
        # print('loss:', loss.item())

        # update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])
        
        return action


class Agent(object):
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
    
    def update_q_function(self):
        self.brain.replay()
    
    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action
    
    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)
    
    def update_target_q_function(self):
        self.brain.update_target_q_network()
