import numpy as np
import os
import random
import time
import json

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py


# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self,
                 window=1e4,
                 norm_ret=False,
                 warm_up=0,
                 delay=0,
                 discount=0.99,
                 alpha=0.001):

        self.storage = []
        #self.max_size = max_size
        self.ptr = 0

        self.episode_lens = []
        self.l_margin = 0
        self.u_margin = 0

        self.window = window
        self.warm_up = warm_up
        self.delay = delay

        self.norm_ret = norm_ret
        if norm_ret:
            self.discount = discount
            self.alpha = alpha
            self.returns = 0.0
            self.returns_count = 0
            self.returns_mean = None
            self.returns_m2 = None

    def _update_stats(self, reward, mask):
        # From https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation
        self.returns = self.returns * self.discount * mask + reward

        if self.returns_mean is None:
            self.returns_count = 1
            self.returns_mean = self.returns
            self.returns_m2 = 0
        else:
            self.returns_count += 1
            delta = self.returns - self.returns_mean
            self.returns_mean += delta / self.returns_count
            delta2 = self.returns - self.returns_mean
            self.returns_m2 += delta * delta2

#    def add(self, data):
#        if self.norm_ret:
#            reward = data[-2]
#            mask = 1 - data[-1]
#            self._update_stats(reward, mask)
#
#        if len(self.storage) == self.max_size:
#            self.storage[int(self.ptr)] = data
#            self.ptr = (self.ptr + 1) % self.max_size
#        else:
#            self.storage.append(data)

    def add_sample(self, data):
        if self.norm_ret:
            reward = data[-2]
            mask = 1 - data[-1]
            self._update_stats(reward, mask)
        
        self.storage.append(data)

    def add_episode_len(self, data):
        self.episode_lens.append(data)

    def set_margins(self):
        u = min(len(self.episode_lens), self.warm_up) + max(0, len(self.episode_lens) - self.warm_up - self.delay)
        l = max(0, u - self.window)

        self.l_margin = np.sum(self.episode_lens[:l])
        self.u_margin = np.sum(self.episode_lens[:u])
            
    def sample(self, batch_size):
        ind = np.random.randint(self.l_margin, self.u_margin, size=batch_size)
        
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        rewards = np.array(r).reshape(-1, 1)

        if self.norm_ret:
            rewards /= np.sqrt(self.returns_m2 / self.returns_count + 1e-8)

        return np.array(x), np.array(y), np.array(u), rewards, np.array(d).reshape(-1, 1)

def create_folder(f): return [os.makedirs(f) if not os.path.exists(f) else False]

class Logger(object):
      def __init__(self, args, experiment_name='', environment_name='', folder='./results'):
            """
            Saves experimental metrics for use later.
            :param experiment_name: name of the experiment
            :param folder: location to save data
            : param environment_name: name of the environment
            """
            self.rewards = []
            self.save_folder = os.path.join(folder, experiment_name, environment_name, time.strftime('%y-%m-%d') + '_' + str(args.window) + '_' + str(args.delay))
            create_folder(self.save_folder)
            with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
                  json.dump(dict(args._get_kwargs()), f)

      def record_reward(self, reward_return):
            self.returns_eval = np.array(reward_return)

      def training_record_reward(self, reward_return):
            self.returns_train = np.array(reward_return)

      def record_losses(self, critic_loss_avg, actor_loss_avg, critic_loss, actor_loss):
            self.critic_loss_avg = np.array(critic_loss_avg)
            self.actor_loss_avg = np.array(actor_loss_avg)
            self.critic_loss = np.array(critic_loss)
            self.actor_loss = np.array(actor_loss)

      def save(self):
            np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.returns_eval)
            np.save(os.path.join(self.save_folder, "returns_train.npy"), self.returns_train)
            np.save(os.path.join(self.save_folder, "critic_loss_avg.npy"), self.critic_loss_avg)
            np.save(os.path.join(self.save_folder, "actor_loss_avg.npy"), self.actor_loss_avg)
            np.save(os.path.join(self.save_folder, "critic_loss.npy"), self.critic_loss)
            np.save(os.path.join(self.save_folder, "actor_loss.npy"), self.actor_loss)
