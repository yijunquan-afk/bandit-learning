import numpy as np
from base.agent import Agent

class GreedyNewsRecommendation(Agent):
    """ 新闻推荐贪心算法 """
    def __init__(self,num_articles,dim,theta_mean=0,theta_std=1,epsilon=0.0,
               alpha=0.2,beta=0.5,tol=0.0001):
        """初始化
        Args:
            num_articles (int): 新闻文章的数量  
            dim (int): 问题的维度数
            theta_mean (int, optional): θ分量的均值. Defaults to 0.
            theta_std (int, optional): θ分量的方差. Defaults to 1.
            epsilon (float, optional): 在epsilon-greedy算法中使用的参数. Defaults to 0.0.
            alpha (float, optional): 回溯直线搜索使用的参数. Defaults to 0.2.
            beta (float, optional): 回溯直线搜索使用的参数. Defaults to 0.5.
            tol (float, optional): 牛顿方法的停止标准. Defaults to 0.0001.
        """    
        self.num_articles = num_articles
        self.dim = dim
        self.theta_mean = theta_mean
        self.theta_std = theta_std
        self.back_track_alpha = alpha
        self.back_track_beta = beta
        self.tol = tol
        self.epsilon = epsilon            