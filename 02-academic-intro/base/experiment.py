import numpy as np
import pandas as pd

class BaseExperiment(object):
  """基础实验：记录悔值(regret和采取的动作(action)
  """

  def __init__(self, agent, environment, n_steps,
               seed=0,  unique_id='NULL'):
    """
    agent: 智能体
    environment: 环境
    n_steps: 迭代次数
    seed: 随机数种子
    unique_id: 标识实验

    results: 实验结果

    """
    self.agent = agent
    self.environment = environment
    self.n_steps = n_steps
    self.seed = seed
    self.unique_id = unique_id

    self.results = []
    self.data_dict = {}


  def run_step_maybe_log(self, t):
    # 观察环境，选择臂
    observation = self.environment.get_observation()
    action = self.agent.pick_action(observation)

    # 计算有用的值
    optimal_reward = self.environment.get_optimal_reward()
    expected_reward = self.environment.get_expected_reward(action)
    reward = self.environment.get_stochastic_reward(action)

    # 使用获得的奖励和选择的臂更新智能体
    self.agent.update_observation(observation, action, reward)

    # 记录悔值
    instant_regret = optimal_reward - expected_reward
    self.cum_regret += instant_regret

    # 环境进化（非平稳实验中才会用到）
    self.environment.advance(action, reward)

    # 记录产生的数据
    self.data_dict = {'t': (t + 1),
                        'instant_regret': instant_regret,
                        'cum_regret': self.cum_regret,
                        'action': action,
                        'unique_id': self.unique_id}
    self.results.append(self.data_dict)


  def run_experiment(self):
    """运行实验，收集数据"""
    np.random.seed(self.seed)
    self.cum_regret = 0
    self.cum_optimal = 0

    for t in range(self.n_steps):
      self.run_step_maybe_log(t)

    # 使用pandas存储数据
    self.results = pd.DataFrame(self.results)

class ExperimentNoAction(BaseExperiment):

  def run_step_maybe_log(self, t):
    # 观察环境，选择臂
    observation = self.environment.get_observation()
    action = self.agent.pick_action(observation)

    # 计算有用的值
    optimal_reward = self.environment.get_optimal_reward()
    expected_reward = self.environment.get_expected_reward(action)
    reward = self.environment.get_stochastic_reward(action)

    # 使用获得的奖励和选择的臂更新智能体
    self.agent.update_observation(observation, action, reward)

    # 记录需要的值
    instant_regret = optimal_reward - expected_reward
    self.cum_optimal += optimal_reward
    self.cum_regret += instant_regret

    # 环境进化（非平稳实验中才会用到）
    self.environment.advance(action, reward)


    self.data_dict = {'t': (t + 1),
                      'instant_regret': instant_regret,
                      'cum_regret': self.cum_regret,
                      'cum_optimal': self.cum_optimal,
                      'unique_id': self.unique_id}
    self.results.append(self.data_dict)