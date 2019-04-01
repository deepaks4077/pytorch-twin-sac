import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-8

# From https://github.com/openai/spinningup/blob/master/spinup/algos/sac/core.py


def gaussian_likelihood(x, mu, log_std):
    std = log_std.exp()

    pre_sum = -0.5 * (((x - mu) /
                       (std + EPS)).pow(2) + 2 * log_std + np.log(2 * np.pi))
    return pre_sum.sum(-1, keepdim=True)


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = (x > u).float().to(x.device)
    clip_low = (x < l).float().to(x.device)
    return x + ((u - x) * clip_up + (l - x) * clip_low).detach()


def apply_squashing_func(mu, pi, logp_pi):
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= torch.log(
        clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6).sum(
            -1, keepdim=True)
    return mu, pi, logp_pi


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mu_fc = nn.Linear(256, action_dim)
        self.log_std_fc = nn.Linear(256, action_dim)

        self.max_action = max_action

        self.apply(weight_init)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mu = self.mu_fc(x)

        log_std = torch.tanh(self.log_std_fc(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1)

        std = log_std.exp()

        pi = mu + torch.randn_like(mu) * std

        log_pi = gaussian_likelihood(pi, mu, log_std)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, log_pi)

        return mu, pi, log_pi


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.apply(weight_init)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, x):
        x1 = F.relu(self.l1(x))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        return x1


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters())

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mu, pi, _ = self.actor(state)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mu, pi, _ = self.actor(state)
        return pi.cpu().data.numpy().flatten()

    def train(self,
              replay_buffer,
              total_timesteps,
              batch_size=100,
              discount=0.99,
              tau=0.005,
              policy_freq=2,
              temperature=0.2):

        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(x).to(device)
        action = torch.FloatTensor(u).to(device)
        next_state = torch.FloatTensor(y).to(device)
        done = torch.FloatTensor(1 - d).to(device)
        reward = torch.FloatTensor(r).to(device)

        def fit_critic():
            with torch.no_grad():
                _, policy_action, log_pi = self.actor(next_state)
                target_Q1, target_Q2 = self.critic_target(
                    next_state, policy_action)
                target_V = torch.min(target_Q1,
                                     target_Q2) - temperature * log_pi
                target_Q = reward + (done * discount * target_V)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        fit_critic()

        def fit_actor():
            # Compute actor loss
            _, pi, log_pi = self.actor(state)
            actor_Q1, actor_Q2 = self.critic(state, pi)

            actor_Q = torch.min(actor_Q1, actor_Q2)

            actor_loss = (temperature * log_pi - actor_Q).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        if total_timesteps % policy_freq == 0:
            fit_actor()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data +
                                        (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory,
                                                                 filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory,
                                                                   filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(
            torch.load('%s/%s_critic.pth' % (directory, filename)))
