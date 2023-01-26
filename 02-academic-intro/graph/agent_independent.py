import copy
import numpy as np
import random

from graph.env_graph_bandit import IndependentBinomialBridge


class IndependentBBEpsilonGreedy():
    """Independent Binomial Bridge Epsilon Greedy"""

    def __init__(self, n_stages, mu0, sigma0, sigma_tilde, epsilon=0.0):
        """An agent for graph bandits.

        Args:
          n_stages: binomial bridge的阶段数 (必须是偶数)
          mu0: 先验的平均值
          sigma0: 先验的标准差
          sigma_tilde: 观察值的噪声
          epsilon: 用于选择的参数
        """
        assert (n_stages % 2 == 0)
        self.n_stages = n_stages
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.sigma_tilde = sigma_tilde
        self.epsilon = epsilon

        # 使用任意初始值设置内部环境
        self.internal_env = IndependentBinomialBridge(n_stages, mu0, sigma0)

        # 将边的后验保存为后验belief的元组(mean, std)
        self.posterior = copy.deepcopy(self.internal_env.graph)
        for start_node in self.posterior:
            for end_node in self.posterior[start_node]:
                # 用mu0, sigma0进行初始化
                self.posterior[start_node][end_node] = (mu0, sigma0)

    def get_posterior_mean(self):
        """获得每条边的后验均值

        Returns:
          edge_length: edge_length[start_node][end_node] = distance
        """
        edge_length = copy.deepcopy(self.posterior)

        for start_node in self.posterior:
            for end_node in self.posterior[start_node]:
                mean, std = self.posterior[start_node][end_node]
                edge_length[start_node][end_node] = np.exp(mean + 0.5 * std**2)

        return edge_length

    def get_posterior_sample(self):
        """获得每条边的后验抽样

        Return:
          edge_length: edge_length[start_node][end_node] = distance
        """
        edge_length = copy.deepcopy(self.posterior)

        for start_node in self.posterior:
            for end_node in self.posterior[start_node]:
                mean, std = self.posterior[start_node][end_node]
                edge_length[start_node][end_node] = np.exp(mean +
                                                           std * np.random.randn())

        return edge_length

    def update_observation(self, observation, action, reward):
        """为binomial bridge更新观察值.

        Args:
          observation: 阶段数
          action: 智能体选择的路(未使用)
          reward: reward[start_node][end_node] = stochastic_time
        """
        assert observation == self.n_stages

        for start_node in reward:
            for end_node in reward[start_node]:
                y = reward[start_node][end_node]
                old_mean, old_std = self.posterior[start_node][end_node]

                # 利用公式更新后验值
                old_precision = 1. / (old_std**2)
                noise_precision = 1. / (self.sigma_tilde**2)
                new_precision = old_precision + noise_precision

                new_mean = (noise_precision * (np.log(y) + 0.5 /
                            noise_precision) + old_precision * old_mean) / new_precision
                new_std = np.sqrt(1. / new_precision)

                # 更新后验值
                self.posterior[start_node][end_node] = (new_mean, new_std)

    def _pick_random_path(self):
        """在bridge中完全随机地选择一条路径"""
        path = []
        start_node = (0, 0)
        while True:
            path += [start_node]
            if start_node == (self.n_stages, 0):
                break
            start_node = random.choice(list(self.posterior[start_node].keys()))
        return path

    def pick_action(self, observation):
        """贪心路径是后验均值的最短路径"""
        if np.random.rand() < self.epsilon:
            path = self._pick_random_path()

        else:
            posterior_means = self.get_posterior_mean()
            self.internal_env.overwrite_edge_length(posterior_means)
            path = self.internal_env.get_shortest_path()

        return path


class IndependentBBTS(IndependentBBEpsilonGreedy):
    """Independent Binomial Bridge Thompson Sampling"""

    def pick_action(self, observation):
        """从后验中抽样"""
        posterior_sample = self.get_posterior_sample()
        # 根据后验抽样更新内部环境
        self.internal_env.overwrite_edge_length(posterior_sample)
        path = self.internal_env.get_shortest_path()

        return path
