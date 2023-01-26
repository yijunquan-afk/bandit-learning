import numpy as np
from base.environment import Environment
from collections import defaultdict
from graph.dijkstra import Dijkstra


class IndependentBinomialBridge(Environment):
    """Graph shortest path on a binomial bridge.

    The agent proceeds up/down for n_stages, but must end with equal ups/downs.
      e.g. (0, 0) - (1, 0) - (2, 0) for n_stages = 2
                  \        /
                    (1, 1)
    We label nodes (x, y) for x=0, 1, .., n_stages and y=0, .., y_lim
    y_lim = x + 1 if x < n_stages / 2 and then decreases again appropriately.
    """

    def __init__(self, n_stages, mu0, sigma0, sigma_tilde=1.):
        """
           graph[node1][node2] 表示node1 和 node2之间的边距

        Args:
          n_stages: 阶段数必须为偶数
          mu0: 独立分布的边的先验均值
          sigma0: 独立分布的边的先验标准差
          sigma_tilde: 标准差的观察噪声
        """
        assert (n_stages % 2 == 0)
        self.n_stages = n_stages
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.sigma_tilde = sigma_tilde

        self.nodes = set()
        self.graph = defaultdict(dict)

        self.optimal_reward = None  # 当我们计算最短路径时填充

        self._create_graph()
        _ = self.get_shortest_path()

    def get_observation(self):
        """这里的观察值是阶段数"""
        return self.n_stages

    def _get_width_bridge(self, x):
        """在阶段x时计算bridge的宽度
        Args:
          x: 阶段数
        """
        depth = x - 2 * np.maximum(x - self.n_stages / 2, 0) + 1
        return int(depth)

    def _create_graph(self):
        """随机初始化图"""

        # 初始化结点
        for x in range(self.n_stages + 1):
            for y in range(self._get_width_bridge(x)):
                node = (x, y)
                self.nodes.add(node)

        # 添加边
        for x in range(self.n_stages + 1):
            for y in range(self._get_width_bridge(x)):
                node = (x, y)
                # 右上的结点
                right_up = (x + 1, y - 1)
                # 正右的结点
                right_equal = (x + 1, y)
                # 右下的结点
                right_down = (x + 1, y + 1)

                if right_down in self.nodes:
                    distance = np.exp(
                      # np.random.randn: 返回一个或一组服从标准正态分布的随机样本值。
                        self.mu0 + self.sigma0 * np.random.randn())
                    self.graph[node][right_down] = distance

                if right_equal in self.nodes:
                    distance = np.exp(
                        self.mu0 + self.sigma0 * np.random.randn())
                    self.graph[node][right_equal] = distance

                if right_up in self.nodes and right_equal not in self.nodes:
                    # 向上走
                    distance = np.exp(
                        self.mu0 + self.sigma0 * np.random.randn())
                    self.graph[node][right_up] = distance

    def overwrite_edge_length(self, edge_length):
        """用确切的值覆盖原先的边长

        Args:
          edge_length: edge_length[start_node][end_node] = distance
        """
        for start_node in edge_length:
            for end_node in edge_length[start_node]:
                self.graph[start_node][end_node] = edge_length[start_node][end_node]

    def get_shortest_path(self):
        """找到最短路

        Returns:
          path: 遍历的结点列表
        """
        start = (0, 0)
        end = (self.n_stages, 0)
        # 使用Dijkstra方法找到最短路
        final_distance, predecessor = Dijkstra(self.graph, start, end)

        path = []
        iter_node = end
        while True:
            path.append(iter_node)
            if iter_node == start:
                break
            iter_node = predecessor[iter_node]

        path.reverse()

        # Updating the optimal reward
        self.optimal_reward = -final_distance[end]

        return path

    def get_optimal_reward(self):
        return self.optimal_reward

    def get_expected_reward(self, path):
        """给定一个路径，获取奖励

        Args:
          path: 结点列表

        Returns:
          expected_reward: -路长
        """
        expected_distance = 0
        # zip()是Python的一个内建函数，它接受一系列可迭代的对象作为参数，
        # 将对象中对应的元素打包成一个个tuple（元组），
        # 然后返回由这些tuples组成的list（列表）。
        for start_node, end_node in zip(path, path[1:]):
            expected_distance += self.graph[start_node][end_node]

        return -expected_distance

    def get_stochastic_reward(self, path):
        # 随机产生奖励
        time_elapsed = defaultdict(dict)
        for start_node, end_node in zip(path, path[1:]):
            mean_time = self.graph[start_node][end_node]
            lognormal_mean = np.log(mean_time) - 0.5 * self.sigma_tilde**2
            stoch_time = np.exp(
                lognormal_mean + self.sigma_tilde * np.random.randn())
            time_elapsed[start_node][end_node] = stoch_time

        return time_elapsed
