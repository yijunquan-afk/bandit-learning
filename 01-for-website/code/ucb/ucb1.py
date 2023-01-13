import math


def index_max(x):
    """  
    获取向量x的最大值的索引
    """
    m = max(x)
    return x.index(m)


class UCB1():
    def __init__(self, counts, values):
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_arm(self):
        n_arms = len(self.counts)
        # 确保所有的臂都至少玩了一次
        # 从而可以对所有可用的臂有一个初始化的了解
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            # 使用置信区间上界
            # 置信度为1-2/total_counts
            bonus = math.sqrt((2 * math.log(total_counts)) /
                              float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus

        return index_max(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return
