import random
from arms.bernoulli import *
from epsilon_greedy.standard import *
from testing_framework.testing import *
import matplotlib.pyplot as plt


def draw_average_reward(arms, epsilons, num_sims=5000, times=250):
    result = []

    for epsilon in epsilons:
        algo = EpsilonGreedy(epsilon, [], [])
        algo.initialize(n)
        cumulative_rewards = get_cumulative_reward(algo, arms, num_sims, times)

        average = list(
            map(lambda x: x / (cumulative_rewards.index(x) + 1),
                cumulative_rewards))
        result.append(average)

    i = 0
    for res in result:
        plt.plot(res, label='epsilon = {0}'.format(epsilons[i]))
        i += 1

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.title('Performance of the Epsilon Greedy Algorithm')
    plt.show()


def draw_cumulative_reward(arms, epsilons, num_sims=5000, times=250):
    result = []

    for epsilon in epsilons:
        algo = EpsilonGreedy(epsilon, [], [])
        algo.initialize(n)
        cumulative_rewards = get_cumulative_reward(algo, arms, num_sims, times)

        result.append(cumulative_rewards)

    i = 0
    for res in result:
        plt.plot(res, label='epsilon = {0}'.format(epsilons[i]))
        i += 1

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Cumulative Reward')
    plt.title('Performance of the Epsilon Greedy Algorithm')
    plt.show()

means = [0.1, 0.1, 0.1, 0.1, 0.9]
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
n = len(means)
random.shuffle(means)
arms = list(map(lambda mu: BernoulliArm(mu), means))

draw_average_reward(arms, epsilons)
draw_cumulative_reward(arms, epsilons)
