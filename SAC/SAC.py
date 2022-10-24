import os
import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ReplayBuffer:
    def __init__(self, buffer_size, state_dims, action_dims):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.is_full = False

        # Init buffer
        self.states = np.zeros((self.buffer_size, *state_dims), dtype=np.float32)
        self.states_ = np.zeros((self.buffer_size, *state_dims), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, *action_dims), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, ), dtype=np.float32)
        self.done = np.zeros((self.buffer_size,), dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.states_[self.ptr] = state_
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size

        if self.ptr == 0 and not self.is_full:
            self.is_full = True
            print('... Replay Buffer is full ...')

    def load_batch(self, batch_size):
        if self.is_full:
            samples = np.random.choice(np.arange(self.buffer_size), batch_size, replace=False)
        else:
            samples = np.random.choice(np.arange(self.ptr), batch_size, replace=False)
        states = self.states[samples]
        actions = self.actions[samples]
        rewards = self.rewards[samples]
        states_ = self.states_[samples]
        done = self.done[samples]

        return states, actions, rewards, states_, done



class Critic(nn.Module):
    def __init__(self, beta, state_dims, action_dims, fc1_dims, fc2_dims, name='Critic', ckpt_dir='tmp'):
        super(Critic, self).__init__()
        self.beta = beta
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.ckpt_path = os.path.join(ckpt_dir, name)

        self.fc1 = nn.Linear(*(np.array(self.state_dims) + np.array(self.action_dims)), self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = ('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_action = T.cat([state, action], dim=1).to(self.device)
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.ckpt_path)

    def load_checkpoint(self, gpu_to_cpu=False):
        print('... loading checkpoint ...')
        if gpu_to_cpu:
            self.load_state_dict(T.load(self.ckpt_path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(T.load(self.ckpt_path))


class Actor(nn.Module):
    def __init__(self, alpha, state_dims, action_dims, fc1_dims, fc2_dims, max_action, reparam_noise,
                 name='Actor', ckpt_dir='tmp'):
        super(Actor, self).__init__()
        self.alpha = alpha
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.max_action = max_action
        self.name = name
        self.ckpt_path = os.path.join(ckpt_dir, name)
        self.reparam_noise = reparam_noise

        self.fc1 = nn.Linear(*state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, *self.action_dims)
        self.sigma = nn.Linear(self.fc2_dims, *self.action_dims)
        self.layer_norm = nn.LayerNorm(self.fc1_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = ('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):

        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))

        mu = self.mu(x)
        sigma = self.sigma(x)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1.)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)

        probs = Normal(mu, sigma)
        if reparameterize:
            actions = probs.rsample()
        else:
            actions = probs.sample()
        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probs.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        return action, log_probs

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.ckpt_path)

    def load_checkpoint(self, gpu_to_cpu=False):
        print('... loading checkpoint ...')
        if gpu_to_cpu:
            self.load_state_dict(T.load(self.ckpt_path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(T.load(self.ckpt_path))


class Value(nn.Module):
    def __init__(self, beta, state_dims, fc1_dims, fc2_dims, name='Value', ckpt_dir='tmp'):
        super(Value, self).__init__()
        self.beta = beta
        self.state_dims = state_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.ckpt_path = os.path.join(ckpt_dir, self.name)

        self.fc1 = nn.Linear(*self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = ('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        v = self.v(x)
        return v

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.ckpt_path)

    def load_checkpoint(self, gpu_to_cpu=False):
        print('... loading checkpoint ...')
        if gpu_to_cpu:
            self.load_state_dict(T.load(self.ckpt_path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(T.load(self.ckpt_path))