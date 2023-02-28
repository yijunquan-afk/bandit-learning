import numpy as np
from numpy.linalg import inv
from mab import BasicMAB

class LinUCB(BasicMAB):
    def __init__(self, narms, ndims, alpha):
        self.narms = narms
        # 上下文特征的数量
        self.ndims = ndims
        # 超参，>0
        self.alpha = alpha
        # 每个臂的A矩阵为（ndims，ndims）的单位矩阵
        self.A = np.array([np.identity(self.ndims)] * self.narms)
        # 每个臂的b矩阵为（ndims，1）的矩阵
        self.b = np.zeros((self.narms, self.ndims, 1))
        return

    def select_arm(self, context=None):
        p_t = np.zeros((self.ndims,1))
        for i in range(self.narms):
            self.theta = inv(self.A[i]).dot(self.b[i])
            # 获得每个臂的特征
            x = np.array(context[i*10:(i+1)*10]).reshape(self.ndims, 1)
            # 获得每个臂的奖励
            p_t[i] = self.theta.T.dot(x) + \
                     self.alpha * np.sqrt(x.T.dot(inv(self.A[i]).dot(x)))
        action = np.random.choice(np.where(p_t == max(p_t))[0])
        return action+1

    def update(self, arm, reward, context=None):
        self.arm = arm
        x = np.array(context[arm*10:(arm+1)*10]).reshape(self.ndims, 1)
        self.A[arm] = self.A[arm] + x.dot(x.T)
        self.b[arm] = self.b[arm] + reward*x
        return
