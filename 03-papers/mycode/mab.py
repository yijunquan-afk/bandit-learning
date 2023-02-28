import numpy as np
    
class BasicMAB():
    def __init__(self, narms):
        """初始化

        Args:
            narms (int): 臂的数量
        """        
        pass

    def select_arm(self, context = None):
        """选择一个臂

        Args:
            tround (int): 当前的轮次数
            context (1D float array): 给定arm的上下文信息
        """        
        pass

    def update(self, arm, reward, context = None):
        """更新算法

        Args:
            arm (int): 当前选择的臂，1,...,self.narms
            reward (float): 从一个臂中获得的奖励
            context (1D float array): 给定arm的上下文信息
        """        
        pass