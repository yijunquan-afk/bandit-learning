import numpy as np
from base.environment import Environment


class BernoulliBandit(Environment):
  """Bernoulli Bandits"""

  def __init__(self, probs):
    """   
    probs: 每个臂的奖励概率
    optimal_reward: 最大奖励值
    n_arm: 臂的数目 
    """
    self.probs = np.array(probs)
    self.optimal_reward = np.max(self.probs)
    self.n_arm = len(self.probs)

  def get_observation(self):
    return self.n_arm

  def get_optimal_reward(self):
    return self.optimal_reward

  def get_expected_reward(self, action):
    return self.probs[action]

  def get_stochastic_reward(self, action):
    # 重复伯努利试验，产生0/1的奖励
    return np.random.binomial(1, self.probs[action])
  
