import numpy as np
from mab import BasicMAB

class BetaThompsonSampling(BasicMAB):
    def __init__(self, narms, alpha0=1.0, beta0=1.0):
        self.narms = narms
        # 总的执行次数
        self.step = 0
        # 每只臂被选择过的次数
        self.step_arm = np.zeros(self.narms)
        self.alpha0 = np.ones(narms) * alpha0
        self.beta0 = np.ones(narms) * beta0

    def select_arm(self,  context = None):
        # 每个臂至少选择一次
        if len(np.where(self.step_arm==0)[0]) > 0:
            action = np.random.choice(np.where(self.step_arm==0)[0])
            return action + 1
        means = np.random.beta(self.alpha0, self.beta0)
        action = np.random.choice(np.where(means==np.max(means))[0])
        return action + 1

    def update(self, arm, reward, context = None):
        self.arm = arm
        self.step += 1
        self.step_arm[self.arm] += 1
        if reward == 1:
            self.alpha0[arm] += 1
        else:
            self.beta0[arm] += 1