import numpy as np
from bernoulli_bandit.agent import BernoulliBanditEpsilonGreedy
from bernoulli_bandit.environment import BernoulliBandit
from bernoulli_bandit.experiment import BaseExperiment

probs = [0.7, 0.8, 0.9]
n_steps = 1000
seed = 0

agent = BernoulliBanditEpsilonGreedy(n_arm=len(probs))
env = BernoulliBandit(probs)
experiment = BaseExperiment(
    agent, env, n_steps=n_steps, seed=seed, unique_id='example')

experiment.run_experiment()

print(experiment.results)