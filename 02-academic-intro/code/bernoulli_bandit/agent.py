import numpy as np


def random_argmax(vector):
    """随机选择argmax"""
    index = np.random.choice(np.where(vector == vector.max())[0])
    return index


class BernoulliBanditEpsilonGreedy():
    """Using greedy method to solve bernoulli bandit problem"""

    def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
        self.n_arm = n_arm
        self.epsilon = epsilon
        self.prior_success = np.array([a0 for arm in range(n_arm)])
        self.prior_failure = np.array([b0 for arm in range(n_arm)])

    def set_prior(self, prior_success, prior_failure):
        # Overwrite the default prior
        self.prior_success = np.array(prior_success)
        self.prior_failure = np.array(prior_failure)

    def get_posterior_mean(self):
        return self.prior_success / (self.prior_success + self.prior_failure)

    def get_posterior_sample(self):
        return np.random.beta(self.prior_success, self.prior_failure)

    def update_observation(self, observation, action, reward):
        # Naive error checking for compatibility with environment
        assert observation == self.n_arm

        # numpy的isclose方法: 比较两个array是不是每一元素都相等，默认在1e-05的误差范围内
        if np.isclose(reward, 1):
            self.prior_success[action] += 1
        elif np.isclose(reward, 0):
            self.prior_failure[action] += 1
        else:
            raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

    def pick_action(self, observation):
        """Take random action prob epsilon, else be greedy."""
        if np.random.rand() < self.epsilon:
            # 随机选择一个臂
            action = np.random.randint(self.n_arm)
        else:
            # 从后验的平均值中选择最大的臂
            posterior_means = self.get_posterior_mean()
            action = random_argmax(posterior_means)

        return action


class BernoulliBanditTS(BernoulliBanditEpsilonGreedy):
    def pick_action(self, observation):
        """使用beta后验进行动作选择，汤普森抽样"""
        sampled_means = self.get_posterior_sample()
        action = random_argmax(sampled_means)
        return action
