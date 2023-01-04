import random
import math

def ind_max(x):
    """  
    获取向量x的最大值的索引
    """
    m = max(x)
    return x.index(m)

class AnnealingEpsilonGreedy():
  def __init__(self, counts, values):
    self.counts = counts
    self.values = values
    return

  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return

  def select_arm(self):
    t = sum(self.counts) + 1
    # 模拟退火
    epsilon = 1 / math.log(t + 0.0000001)
    
    if random.random() > epsilon:
      # 利用已知最佳的臂
      return ind_max(self.values)
    else:
      # 随机探索
      return random.randrange(len(self.values))
  
  def update(self, chosen_arm, reward):
        """更新算法

        Args:
            chosen_arm: 最近选择的arm的索引
            reward: 选择该arm获得的奖励
        """        
        # 选择次数增加
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        # 加权平均
        new_value = ((n-1)/float(n))*value + (1/float(n)) * reward
        self.values[chosen_arm] = new_value
        return
