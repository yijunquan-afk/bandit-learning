
import random


def index_max(x):
    """  
    获取向量x的最大值的索引
    """
    m = max(x)
    return x.index(m)


class EpsilonGreedy():
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return
    
    def select_arm(self):
        if random.random() > self.epsilon:
            # 利用已知最佳的臂
            return index_max(self.values)
        else:
            # 随机探索
            return int(random.uniform(0,len(self.values)))

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

