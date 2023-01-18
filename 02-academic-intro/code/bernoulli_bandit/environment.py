import numpy as np

class BernoulliBandit():
  """Simple N-armed bandit."""

  def __init__(self, probs):
    self.probs = np.array(probs)
    assert np.all(self.probs >= 0)
    assert np.all(self.probs <= 1)

    self.optimal_reward = np.max(self.probs)
    self.n_arm = len(self.probs)

  def get_observation(self):
    return self.n_arm

  def get_optimal_reward(self):
    return self.optimal_reward

  def get_expected_reward(self, action):
    return self.probs[action]

  def get_stochastic_reward(self, action):
    # 重复伯努利试验
    return np.random.binomial(1, self.probs[action])