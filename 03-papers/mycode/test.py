from epsilon_greedy import EpsilonGreedy
from ucb import UCB
from ts import BetaThompsonSampling
from LinUCB import LinUCB
from evaluator import offlineEvaluate
import numpy as np
import matplotlib.pyplot as plt


JOBS = 1
test_rounds = 800
# np.random.seed(7)
data = np.loadtxt('03-papers/dataset.txt')
datalen = np.shape(data)[0]
arms = data[:, 0]
rewards = data[:, 1]
contexts = data[:, 2:102]

epsilon = 0.05
alpha = 0.1
alpha2 = 0.2

mab1 = EpsilonGreedy(10,epsilon)
mab2 = UCB(10, alpha)
mab3 = BetaThompsonSampling(10, 1.0, 1.0)
mab4 = LinUCB(10, 10 ,alpha2)

cul_TS = 0
cul_Eps = 0
cul_UCB = 0
cul_LinUCB = 0

cul2_TS = np.zeros(test_rounds)
cul2_Eps = np.zeros(test_rounds)
cul2_UCB = np.zeros(test_rounds)
cul2_LinUCB = np.zeros(test_rounds)

for j in range(JOBS):
    results_Eps, _ , cumulative_reward_Eps = offlineEvaluate(
        mab1, datalen, arms, rewards, contexts, test_rounds)
    results_UCB, _ , cumulative_reward_UCB = offlineEvaluate(
        mab2, datalen, arms, rewards, contexts, test_rounds)
    results_TS, _ , cumulative_reward_TS = offlineEvaluate(
        mab3, datalen, arms, rewards, contexts, test_rounds)
    results_LinUCB, _ , cumulative_reward_LinUCB = offlineEvaluate(
        mab4, datalen, arms, rewards, contexts, test_rounds)
    
    cul_TS += np.mean(results_TS)
    cul_Eps += np.mean(results_Eps)
    cul_UCB += np.mean(results_UCB)
    cul_LinUCB += np.mean(results_LinUCB)

    cul2_TS += cumulative_reward_TS
    cul2_Eps += cumulative_reward_Eps
    cul2_UCB += cumulative_reward_UCB
    cul2_LinUCB += cumulative_reward_LinUCB

print('Epsilon-greedy average reward', cul_Eps/JOBS)    
print('UCB average reward', cul_UCB/JOBS)
print('Thompson Sampling average reward', cul_TS/JOBS)
print('LinUCB-greedy average reward', cul_LinUCB/JOBS)   


plt.figure(figsize=(12, 8))
plt.plot((cul2_Eps/JOBS)/(np.linspace(1, test_rounds, test_rounds)),label=r"$\epsilon=$ {:.2f}(greedy)".format(epsilon))
plt.plot((cul2_UCB/JOBS)/(np.linspace(1, test_rounds, test_rounds)),label=r"$\alpha=$ {:.2f}(UCB)".format(alpha))
plt.plot((cul2_LinUCB/JOBS)/(np.linspace(1, test_rounds, test_rounds)),label=r"$\alpha=$ {:.2f}(LinUCB)".format(alpha2))
plt.plot((cul2_TS/JOBS)/(np.linspace(1, test_rounds, test_rounds)),label=r"Thompson Sampling")

plt.legend()
plt.xlabel("Rounds")
plt.ylabel(r"regret")
plt.title("Per-round Cumulative Rewards after single simulation")
plt.show()
