import numpy as np


from base.environment import Environment

class NewsRecommendation(Environment):
  """新闻推荐环境。环境提供了任何时期的特征向量，并决定了所选动作的奖励。."""

  def __init__(self,num_articles,dim,theta_mean=0,theta_std=1):
    """环境初始化

    Args:
        num_articles (int): 文章数量
        dim (int): 问题维度数
        theta_mean (int, optional): θ分量的均值. Defaults to 0.
        theta_std (int, optional): θ分量的方差. Defaults to 1.
    """      

    self.num_articles = num_articles
    self.dim = dim
    self.theta_mean = theta_mean
    self.theta_std = theta_std
    
    # 生成实际参数
    # 每一个参数都是 ～ N(theta_mean,theta_std)，初始化为N(0,1)
    self.thetas = [self.theta_mean + self.theta_std*np.random.randn(self.dim) 
                                            for _ in range(self.num_articles)]
    
    # 维护当前奖励[0,0,...,0]
    self.current_rewards = [0]*self.num_articles
    
  def get_observation(self):
    '''生成上下文向量并计算每个文章的真实奖励.'''
    
    context = []
    # 上下文向量，重复伯努利试验，做dim次实验，概率 p= max(0,1/(self.dim-1))
    # 示例 [0 1 0 0 0 0 0]
    context_vector = np.random.binomial(1,max(0,1/(self.dim-1)),self.dim)
    context_vector[0] = 1        
    for i in range(self.num_articles):
      context.append(context_vector)
      self.current_rewards[i] = 1/(1+np.exp(-self.thetas[i].dot(context_vector)))
        
    return context
    
  def get_optimal_reward(self):
    return np.max(self.current_rewards)