'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-05-30 14:20:04
LastEditTime: 2022-12-28 12:35:02
@Description: 
'''

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
import os

from .buffer import ReplayBuffer

sys.path.append('../')
from utils import CUDA, CPU, kaiming_init


class Actor_Continuous(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, min_val):
        super(Actor_Continuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.min_val = min_val
        self.action_scale = 1
        self.action_bias = 0
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = self.softplus(self.fc_std(x)) + self.min_val
        return mu, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.min_val)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def select_action(self, state, deterministic=False):
        action, _, mean = self.sample(state)
        if deterministic:
            return action
        else:
            return mean


class Actor_Discrete(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, min_val):
        super(Actor_Discrete, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.min_val = min_val
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

    def sample(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        z = action_probs == 0.0  # deal with situation of 0.0 probabilities because we can't do log 0
        z = z.float() * self.min_val
        action_log_probs = torch.log(action_probs + z)
        return action, action_probs, action_log_probs      

    def select_action(self, state, deterministic=False):
        action, action_probs, _ = self.sample(state)
        if deterministic:
            return torch.argmax(action_probs)
        else:
            return action


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_type):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_type = action_type

        if self.action_type == 'continuous':
            # for continuous Critic, input is state_dim+action_dim and output is 1
            self.fc1 = nn.Linear(state_dim+action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
            self.fc4 = nn.Linear(state_dim+action_dim, hidden_dim)
            self.fc5 = nn.Linear(hidden_dim, hidden_dim)
            self.fc6 = nn.Linear(hidden_dim, 1)
        else:
            # for discrete Critic, input is state_dim and output is action_dim
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
            self.fc4 = nn.Linear(state_dim, hidden_dim)
            self.fc5 = nn.Linear(hidden_dim, hidden_dim)
            self.fc6 = nn.Linear(hidden_dim, action_dim)
        
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, s, a):
        if self.action_type == 'continuous':
            x = torch.cat([s, a], dim=1) # combination s and a
        else:
            x = s
        
        # Q1
        q1 = self.relu(self.fc1(x))
        q1 = self.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        # Q2
        q2 = self.relu(self.fc4(x))
        q2 = self.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2


class SAC():
    ''' SAC model with continuous and discrete action space '''
    name = 'SAC'

    def __init__(self, args):
        super(SAC, self).__init__()
        self.max_buffer_size = args['max_buffer_size']
        self.pretrain_buffer_size = args['pretrain_buffer_size']
        self.lr = args['lr']
        self.min_val = torch.tensor(args['min_Val']).float()
        self.batch_size = args['batch_size']
        self.update_iteration = args['update_iteration']
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        hidden_dim = args['hidden_dim']

        self.model_path = args['model_path']
        self.model_id = args['model_id']

        self.state_dim = args['env_params']['state_dim']
        self.action_dim = args['env_params']['action_dim']
        self.env_name = args['env_params']['env_name']

        if self.env_name == 'chemistry':
            self.action_type = 'discrete'
            self.num_objects = args['env_params']['num_objects']
            self.num_colors = args['env_params']['num_colors']
            self.action_dim = 1
            action_num = self.num_objects * self.num_colors
            self.policy = CUDA(Actor_Discrete(self.state_dim, action_num, hidden_dim, self.min_val))
            self.critic = CUDA(QNetwork(self.state_dim, action_num, hidden_dim, self.action_type))
            self.critic_target = CUDA(QNetwork(self.state_dim, action_num, hidden_dim, self.action_type))
            self._action_postprocess = self._chemistry_action_postprocess

            # build the mapping bettwen index to action
            self.action_map = []
            self.action_map_str = []
            for i in range(action_num):
                onehot = np.zeros((action_num,))
                onehot[i] = 1.0
                self.action_map.append(onehot)
                self.action_map_str.append(str(onehot))
        else:
            raise ValueError('Unknown env name')

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        # buffer for saving data
        self.replay_buffer = ReplayBuffer(self.max_buffer_size, self.state_dim*2+self.action_dim+2)

    def _chemistry_action_postprocess(self, action_idx):
        return self.action_map[action_idx]

    def select_action(self, env, state, deterministic=False):
        state = CUDA(torch.from_numpy(state.astype(np.float32)))
        action = self.policy.select_action(state, deterministic)
        action = self._action_postprocess(CPU(action))
        return action

    def store_transition(self, data):
        # [state, action, reward, next_state, done]
        # for discrete action, we need to store the index
        if self.action_type == 'discrete':
            action = data[1]
            action_idx = self.action_map_str.index(str(action))
            data[1] = np.array([action_idx])

        data = np.concatenate([data[0], data[1], [data[2]], data[3], [np.float32(data[4])]])
        self.replay_buffer.push(data)

    def update_loss_discrete(self, state, action, reward, next_state, mask):
        # update critic
        with torch.no_grad():
            _, action_probs, action_log_probs = self.policy.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, None)
            next_q_value = action_probs * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * action_log_probs)
            next_q_value = reward + (mask * self.gamma * next_q_value.sum(dim=1).unsqueeze(-1)) 

        # Compute critic loss
        qf1, qf2 = self.critic(state, None)
        qf1 = qf1.gather(dim=1, index=action.long())
        qf2 = qf2.gather(dim=1, index=action.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Compute actor loss
        _, action_probs, action_log_probs = self.policy.sample(state)
        qf1, qf2 = self.critic(state, None)
        min_Q = torch.min(qf1, qf2)
        policy_loss = (action_probs * (self.alpha * action_log_probs - min_Q)).sum(1).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
    
    def update_loss_continuous(self, state, action, reward, next_state, mask):
        # update critic
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + mask * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # update policy network
        pi, log_pi, _ = self.policy.sample(state)
        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

    def train(self):
        if self.replay_buffer.memory_len < self.pretrain_buffer_size:
            #print('Skip training, buffer size: [{}/{}]'.format(self.replay_buffer.memory_len, self.max_buffer_size))
            return

        for _ in range(self.update_iteration):
            # Sample replay buffer
            batch_memory = self.replay_buffer.sample(self.batch_size)
            state_batch = CUDA(torch.from_numpy(batch_memory[:, 0:self.state_dim].astype(np.float32)))
            action_batch = CUDA(torch.from_numpy(batch_memory[:, self.state_dim:self.state_dim+self.action_dim].astype(np.float32)))
            reward_batch = CUDA(torch.from_numpy(batch_memory[:, self.state_dim+self.action_dim:self.state_dim+self.action_dim+1].astype(np.float32)))
            next_state_batch = CUDA(torch.from_numpy(batch_memory[:, self.state_dim+self.action_dim+1:2*self.state_dim+self.action_dim+1].astype(np.float32)))
            mask_batch = CUDA(torch.from_numpy(1-batch_memory[:, 2*self.state_dim+self.action_dim+1:2*self.state_dim+self.action_dim+2].astype(np.float32)))

            if self.action_type == 'continuous':
                self.update_loss_continuous(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
            else:
                self.update_loss_discrete(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

    def save_model(self):
        states = {'policy': self.policy.state_dict(), 'critic': self.critic.state_dict(), 'critic_target': self.critic_target.state_dict()}
        filepath = os.path.join(self.model_path, 'model.sac.'+str(self.model_id)+'.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self):
        filepath = os.path.join(self.model_path, 'model.sac.'+str(self.model_id)+'.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy.load_state_dict(checkpoint['policy'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
        else:
            raise Exception('No SAC model found!')
