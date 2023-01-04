import random
from arms.bernoulli import *
from epsilon_greedy.annealing import *
from testing_framework.testing import *

means = [0.1, 0.1, 0.1, 0.1, 0.9]
n = len(means)
random.shuffle(means)
arms = list(map(lambda mu:BernoulliArm(mu), means))
f = open("annealing_results.tsv", "w")

for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
  algo = AnnealingEpsilonGreedy([], [])
  algo.initialize(n)
  results = test_algorithm(algo, arms, 1000, 250)
  for i in range(len(results[0])):
      f.write(str(epsilon) + "\t")
      f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")

f.close()