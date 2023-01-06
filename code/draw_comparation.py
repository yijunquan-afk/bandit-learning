import random
from arms.bernoulli import *
from epsilon_greedy.annealing import *
from softmax.annealing import *
from ucb.ucb1 import *
from testing_framework.testing import *
import matplotlib.pyplot as plt

# seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，
# 则每次生成的随即数都相同
random.seed(1)
means = [0.3, 0.1, 0.9, 0.1, 0.9]
n = len(means)
random.shuffle(means)
arms = list(map(lambda mu: BernoulliArm(mu), means))

algo1 = AnnealingEpsilonGreedy([], [])
algo2 = AnnealingSoftmax([], [])
algo3 = UCB1([], [])
algo1.initialize(n)
algo2.initialize(n)
algo3.initialize(n)

cumulative_rewards1 = get_cumulative_reward(algo1, arms, 5000, 250)
cumulative_rewards2 = get_cumulative_reward(algo2, arms, 5000, 250)
cumulative_rewards3 = get_cumulative_reward(algo3, arms, 5000, 250)

average1 = list(
    map(lambda x: x / (cumulative_rewards1.index(x) + 1),
        cumulative_rewards1))
average2 = list(
    map(lambda x: x / (cumulative_rewards2.index(x) + 1),
        cumulative_rewards2))
average3 = list(
    map(lambda x: x / (cumulative_rewards3.index(x) + 1),
        cumulative_rewards3))


plt.plot(average1, label='Annealing Epsilon-Greedy')
plt.plot(average2, label='Annealing Softmax')
plt.plot(average3, label='UCB1')


plt.legend()
plt.xlabel('Time')
plt.ylabel('Average Reward')
plt.title('Performance of Different Algorithms')
plt.show()

# plt.plot(cumulative_rewards1, label='Annealing Epsilon-Greedy')
# plt.plot(cumulative_rewards2, label='Annealing Softmax')
# plt.plot(cumulative_rewards3, label='UCB1')


# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Cumulative Reward')
# plt.title('Performance of Different Algorithms')
# plt.show()


