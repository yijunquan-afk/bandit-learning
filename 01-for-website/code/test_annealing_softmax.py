import random
from arms.bernoulli import *
from softmax.annealing import *
from testing_framework.testing import *
import matplotlib.pyplot as plt


random.seed(1)
means = [0.1, 0.1, 0.1, 0.1, 0.9]
n = len(means)
random.shuffle(means)
arms = list(map(lambda mu:BernoulliArm(mu), means))

algo = AnnealingSoftmax([], [])
algo.initialize(n)
cumulative_rewards = get_cumulative_reward(algo, arms, 5000, 250)

plt.plot(cumulative_rewards, label='annealing')


plt.legend()
plt.xlabel('Time')
plt.ylabel('Average Reward')
plt.title('Performance of the Epsilon Greedy Algorithm')
plt.show()