import numpy as np
from mab import BasicMAB

class EpsilonGreedy(BasicMAB):
    def __init__(self, narms, epsilon=0):
        self.narms = narms
        self.epsilon = epsilon
        # 总的执行次数
        self.step = 0
        # 每只臂被选择过的次数
        self.step_arm = np.zeros(self.narms)
        # 每只臂的平均奖励
        self.mean_reward = np.zeros(self.narms)
        return

    def select_arm(self,context = None):
        # 每个臂至少选择一次
        if len(np.where(self.step_arm==0)[0]) > 0:
            action = np.random.choice(np.where(self.step_arm==0)[0])
            return action + 1
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.narms)
        else:
            # 选择平均奖励最大的臂
            action = np.random.choice(np.where(self.mean_reward==np.max(self.mean_reward))[0])
        # 由于臂的下标为1-10，所以要+1
        return action + 1


    def update(self, arm, reward, context=None):
        self.arm = arm
        self.step += 1
        self.step_arm[self.arm] += 1
        self.mean_reward[self.arm] = (
            self.mean_reward[self.arm] * (self.step_arm[self.arm]-1) + reward)/ float(self.step_arm[self.arm])
