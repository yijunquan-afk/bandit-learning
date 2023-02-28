import numpy as np


def offlineEvaluate(mab, data_num, arms, rewards, contexts, rounds=None):
    """模拟在线环境的离线测试

    Args:
        mab (BasicMAB): 多臂老虎机算法
        data_num (int): 数据量
        arms (1D int array): 数据集中选择的臂的数量
        rewards (1D float array): 数据集中的奖励
        contexts (2D float array): 数据集中的上下文信息
        rounds (int, optional): 选择限制的轮次. Defaults to None.

    Returns:
        reward_arms (1D float array): 获得的奖励
        chosen_arms (1D int array): 选择的臂
        cumulative_reward (1D float array): 累积的奖励
    """
    chosen_arms = np.zeros(rounds)
    reward_arms = np.zeros(rounds)
    cumulative_reward = np.zeros(rounds)
    # 当前轮次
    T = 0
    # 当前累计奖励
    G = 0
    # 离线历史
    history = []
    # 最初选择一个动作
    action = mab.select_arm(contexts[0, :])

    for i in range(data_num):
        action = mab.select_arm(contexts[i, :])
        if T < rounds:
            # 当算法选择的臂跟实际数据匹配时，更新参数
            if action == arms[i]:
                # 将选择的臂的上下文信息添加到历史信息中
                history.append(contexts[i, :])
                reward_arms[T] = rewards[i]
                # action要-1，便于映射到下标
                mab.update(action-1, rewards[i], contexts[i, :])
                G += rewards[i]
                cumulative_reward[T] = G
                chosen_arms[T] = action
                T += 1
        else:
            break
        
    return reward_arms, chosen_arms, cumulative_reward
