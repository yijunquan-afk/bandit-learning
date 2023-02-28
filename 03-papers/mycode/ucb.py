import numpy as np
from mab import BasicMAB

class UCB(BasicMAB):
    def __init__(self, narms, alpha=1):
        self.narms = narms
        # 超参，>0
        self.alpha = alpha
        # 总的执行次数
        self.step = 0
        # 每只臂被选择过的次数
        self.step_arm = np.zeros(self.narms)
        # 每只臂的平均奖励
        self.mean_reward = np.zeros(self.narms)
        return

    def select_arm(self,  context = None):
        # 每个臂至少选择一次
        if len(np.where(self.step_arm==0)[0]) > 0:
            action = np.random.choice(np.where(self.step_arm==0)[0])
            return action + 1
        # 计算ucb
        ucb_values = np.zeros(self.narms)
        for arm in range(self.narms):
            # 置信区间
            ucb_values[arm] =np.sqrt(self.alpha *(np.log(self.step)) / self.step_arm[arm])
        temp = self.mean_reward + ucb_values
        action = np.random.choice(np.where(temp == np.max(temp))[0])
        return action + 1


    def update(self, arm, reward, context=None):
        self.arm = arm
        self.step += 1
        self.step_arm[self.arm] += 1
        self.mean_reward[self.arm] = (
            self.mean_reward[self.arm] * (self.step_arm[self.arm]-1) + reward)/ float(self.step_arm[self.arm])