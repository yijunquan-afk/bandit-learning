import math
import random


def categorical_draw(probs):
    """  
    根据probs按比例以一定概率选择臂
    """
    z = random.random()
    cum_prob = 0.0
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        if cum_prob > z:
            return i

    return len(probs) - 1


class AnnealingSoftmax:
    def __init__(self, counts, values):

        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_arm(self):
        t = sum(self.counts) + 1
        # 模拟退火
        temperature = 1 / math.log(t + 0.0000001)
        # temperature = 1 / t
        # if t < 100:
        #     temperature = 0.5
        # else:
        #     temperature = 0.1
        total = sum([math.exp(v / temperature) for v in self.values])
        probs = [math.exp(v / temperature) / total for v in self.values]
        return categorical_draw(probs)

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
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return
