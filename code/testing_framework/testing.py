import numpy as np
def test_algorithm(algo, arms, num_sims, horizon):
    """测试算法框架

    Args:
        algo: 试图测试的bandit算法框架
        arms: 模拟绘制的臂的数组
        num_sims: 模拟的次数
        horizon: 每个算法在每次模拟期间被允许拉动臂的次数

    Returns:
        会返回一个数据集，告诉我们每次模拟选择了哪个臂，以及算法在每个时间点的表现如何。
    """

    # 初始化算法设置，此时未知哪只臂是最好的
    num = num_sims * horizon
    chosen_arms = [0.0 for i in range(num)]
    rewards = [0.0 for i in range(num)]
    cumulative_rewards = [0.0 for i in range(num)]
    # 模拟下标数组：表示第i次模拟
    sim_nums = [0.0 for i in range(num)]
    # 拉动次数数组
    times = [0.0 for i in range(num)]

    for sim in range(num_sims):
        sim = sim + 1
        algo.initialize(len(arms))

        for t in range(horizon):
            t = t + 1
            index = (sim - 1) * horizon + t - 1
            sim_nums[index] = sim
            times[index] = t

            # 选择臂
            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm

            # 模拟拉动臂的结果
            reward = arms[chosen_arms[index]].draw()
            rewards[index] = reward

            # 记录算法收到的奖励金，然后调用更新
            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

            algo.update(chosen_arm, reward)

    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


def get_cumulative_reward(algo, arms, num_sims, horizon):
    """测试算法框架

    Args:
        algo: 试图测试的bandit算法框架
        arms: 模拟绘制的臂的数组
        num_sims: 模拟的次数
        horizon: 每个算法在每次模拟期间被允许拉动臂的次数

    Returns:
        在每次试验中获得的平均累计奖励
    """
    result = []

    for sim in range(num_sims):
        cumulative_rewards = [0.0 for i in range(horizon)]
        algo.initialize(len(arms))
        for t in range(horizon):
            t = t + 1
            index =  t - 1

            # 选择臂
            chosen_arm = algo.select_arm()

            # 模拟拉动臂的结果
            reward = arms[chosen_arm].draw()

            # 记录算法收到的奖励金，然后调用更新
            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

            algo.update(chosen_arm, reward)
        result.append(cumulative_rewards)
    
    # 多次模拟求平均
    average = np.sum(np.array(result),axis=0) / num_sims
    
    return list(average)


