import numpy as np


def random_argmax(vector):
    """选择vector中的最大值的下标，如果有多个最大，则随机选择其中一个"""
    index = np.random.choice(np.where(vector == vector.max())[0])
    return index


class BernoulliBanditEpsilonGreedy():
    """Bernoulli Bandit问题中使用ε-greedy算法"""

    def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
        self.n_arm = n_arm
        self.epsilon = epsilon
        self.prior_success = np.array([a0 for arm in range(n_arm)])
        self.prior_failure = np.array([b0 for arm in range(n_arm)])

    def set_prior(self, prior_success, prior_failure):
        # 修改默认先验值
        self.prior_success = np.array(prior_success)
        self.prior_failure = np.array(prior_failure)

    def get_posterior_mean(self):
        # 由beta分布的α和β可知beta分布的期望为 α /(α+β)
        return self.prior_success / (self.prior_success + self.prior_failure)

    def get_posterior_sample(self):
        # 从后验抽样
        return np.random.beta(self.prior_success, self.prior_failure)

    def update_observation(self, observation, action, reward):
        # 简单检查与环境的兼容性
        assert observation == self.n_arm

        if np.isclose(reward, 1):
            self.prior_success[action] += 1
        elif np.isclose(reward, 0):
            self.prior_failure[action] += 1
        else:
            raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

    def pick_action(self, observation):
        if np.random.rand() < self.epsilon:
            # 随机探索
            action = np.random.randint(self.n_arm)
        else:
            # 从后验分布中选取一个均值最大的臂
            posterior_means = self.get_posterior_mean()
            action = random_argmax(posterior_means)

        return action


class BernoulliBanditTS(BernoulliBanditEpsilonGreedy):
    def pick_action(self, observation):
        """汤普森抽样"""
        sampled_means = self.get_posterior_sample()
        action = random_argmax(sampled_means)
        return action
