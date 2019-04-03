import numpy as np

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py


# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self,
                 max_size=1e6,
                 norm_ret=False,
                 discount=0.99,
                 alpha=0.001):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

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

    def add(self, data):
        if self.norm_ret:
            reward = data[-2]
            mask = 1 - data[-1]
            self._update_stats(reward, mask)

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
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

        return np.array(x), np.array(y), np.array(u), rewards, np.array(
            d).reshape(-1, 1)
