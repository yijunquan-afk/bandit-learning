import copy
import numpy as np
import numpy.linalg as npla

from collections import defaultdict
from base.agent import Agent
from graph.env_graph_bandit import CorrelatedBinomialBridge

_SMALL_NUMBER = 1e-10
###############################################################################
# Helper functions for correlated agents

def _prepare_posterior_update_elements(observation, action, reward, num_edges, \
                                       edge2index, sigma_tilde, internal_env):
  """生成用于相关BB问题后验更新的浓度矩阵
      Inputs:
          observation - 观察数 (= n_stages)
          action - 选择的动作，即一条路
          reward - 观察到的每条边的奖励, 字典的字典
          num_edges - 总的边数
          edge2index - 将每条边映射到一个唯一的下标
          sigma_tilde - 噪声
          internal_env - 内部环境

      Return:
          更新平均向量时使用的向量，更新协方差矩阵和平均向量时使用的浓度矩阵
  """
  # 为每条边生成局部浓度矩阵和对数奖励
  log_rewards = np.zeros(num_edges)
  local_concentration = np.zeros((observation, observation))
  first_edge_counter = 0
  for start_node in reward:
    for end_node in reward[start_node]:
      log_rewards[edge2index[start_node][end_node]] = \
        np.log(reward[start_node][end_node])
      secod_edge_counter = 0
      for another_start_node in reward:
        for another_end_node in reward[another_start_node]:
          if first_edge_counter == secod_edge_counter:
            local_concentration[first_edge_counter,secod_edge_counter] \
              = sigma_tilde ** 2
          elif internal_env.is_in_lower_half(start_node, end_node) \
            == internal_env.is_in_lower_half(another_start_node, another_end_node):
            local_concentration[first_edge_counter, secod_edge_counter] \
              = 2 * (sigma_tilde ** 2) / 3
          else:
            local_concentration[first_edge_counter, secod_edge_counter] \
                      = (sigma_tilde ** 2) / 3
          secod_edge_counter += 1
      first_edge_counter += 1

  # 求局部浓度矩阵的逆
  local_concentration_inv = npla.inv(local_concentration)

  # 生成浓度矩阵
  concentration = np.zeros((num_edges, num_edges))
  first_edge_counter = 0
  for start_node in reward:
    for end_node in reward[start_node]:
      secod_edge_counter = 0
      for another_start_node in reward:
        for another_end_node in reward[another_start_node]:
          concentration[edge2index[start_node][end_node] \
                        ,edge2index[another_start_node][another_end_node]] \
          = local_concentration_inv[first_edge_counter,secod_edge_counter]
          secod_edge_counter += 1
      first_edge_counter += 1

  return log_rewards, concentration


def _update_posterior(posterior, log_rewards, concentration):
  """更新后验参数

      Input:
          posterior - 后验参数的形式为(Mu, Sigma, Sigmainv)
          log_rewards - 对每条遍历边观察到的延迟的日志
          concentration - 根据新的观测计算出的浓度矩阵

      Return:
          updated parameters: Mu, Sigma, Sigmainv
  """

  new_Sigma_inv = posterior[2] + concentration
  new_Sigma = npla.inv(new_Sigma_inv)
  new_Mu = new_Sigma.dot(posterior[2].dot(posterior[0]) +
                         concentration.dot(log_rewards))

  return new_Mu, new_Sigma, new_Sigma_inv


def _find_conditional_parameters(dim, S):
  """给定一个维协方差矩阵S，
  返回一个包含用于计算每个组件的条件分布的元素的列表。"""
  Sig12Sig22inv = []
  cond_var = []

  for e in range(dim):
    S11 = copy.copy(S[e][e])
    S12 = S[e][:]
    S12 = np.delete(S12, e)
    S21 = S[e][:]
    S21 = np.delete(S21, e)
    S22 = S[:][:]
    S22 = np.delete(S22, e, 0)
    S22 = np.delete(S22, e, 1)
    S22inv = npla.inv(S22)
    S12S22inv = S12.dot(S22inv)
    Sig12Sig22inv.append(S12S22inv)
    cond_var.append(S11 - S12S22inv.dot(S21))

  return cond_var, Sig12Sig22inv


##############################################################################


class CorrelatedBBTS(Agent):
  """Correlated Binomial Bridge Thompson Sampling"""

  def __init__(self, n_stages, mu0, sigma0, sigma_tilde, n_sweeps=10):
    """An agent for graph bandits.

    Args:
      n_stages - number of stages of the binomial bridge (must be even)
      mu0 - prior mean
      sigma0 - prior stddev
      sigma_tilde - noise on observation
      n_sweeps - number of sweeps, used only in Gibbs sampling
    """
    assert (n_stages % 2 == 0)
    self.n_stages = n_stages
    self.n_sweeps = n_sweeps

    # 使用任意初始值设置内部环境
    self.internal_env = CorrelatedBinomialBridge(n_stages, mu0, sigma0)

    # 保存一个映射(start_node,end_node)——>R以简化计算
    self.edge2index = defaultdict(dict)
    self.index2edge = defaultdict(dict)
    edge_counter = 0
    for start_node in self.internal_env.graph:
      for end_node in self.internal_env.graph[start_node]:
        self.edge2index[start_node][end_node] = edge_counter
        self.index2edge[edge_counter] = (start_node, end_node)
        edge_counter += 1

    # 保存所有边的数量
    self.num_edges = edge_counter

    # 先验参数
    self.Mu0 = np.array([mu0] * self.num_edges)
    self.Sigma0 = np.diag([sigma0**2] * self.num_edges)
    self.Sigma0inv = np.diag([(1 / sigma0)**2] * self.num_edges)
    self.sigma_tilde = sigma_tilde

    # p后验分布保存为包含平均向量、协方差矩阵及其逆的三重分布
    self.posterior = (self.Mu0, self.Sigma0, self.Sigma0inv)

    # boostrap版本中使用的附加参数
    self.concentration_history = []
    self.log_reward_history = []
    self.history_size = 0

  def get_posterior_mean(self):
    """获得每条边的后验均值

    Return:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    edge_length = copy.deepcopy(self.internal_env.graph)

    for start_node in edge_length:
      for end_node in edge_length[start_node]:
        edge_index = self.edge2index[start_node][end_node]
        mean = self.posterior[0][edge_index]
        var = self.posterior[0][edge_index, edge_index]
        edge_length[start_node][end_node] = np.exp(mean + 0.5 * var)

    return edge_length

  def get_posterior_sample(self):
    """获得每条边的后验抽样

    Return:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    # flattened sample
    flattened_sample = np.random.multivariate_normal(self.posterior[0],
                                                     self.posterior[1])

    edge_length = copy.deepcopy(self.internal_env.graph)

    for start_node in edge_length:
      for end_node in edge_length[start_node]:
        edge_length[start_node][end_node] = \
            np.exp(flattened_sample[self.edge2index[start_node][end_node]])

    return edge_length

  def update_observation(self, observation, action, reward):
    """更新观察值
    Args:
      observation - number of stages
      action - path chosen by the agent (not used)
      reward - dict of dict reward[start_node][end_node] = stochastic_time
    """
    assert (observation == self.n_stages)

    log_rewards, concentration = _prepare_posterior_update_elements(observation,\
            action, reward, self.num_edges, self.edge2index, self.sigma_tilde, \
            self.internal_env)

    # 更新联合分布的均值和方差矩阵
    new_Mu, new_Sigma, new_Sigma_inv = _update_posterior(self.posterior, \
                                                log_rewards, concentration)
    self.posterior = (new_Mu, new_Sigma, new_Sigma_inv)

  def pick_action(self, observation):
    """Greedy shortest path wrt posterior sample."""
    posterior_sample = self.get_posterior_sample()
    self.internal_env.overwrite_edge_length(posterior_sample)
    path = self.internal_env.get_shortest_path()

    return path