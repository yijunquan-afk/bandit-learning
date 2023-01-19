import numpy as np
import pandas as pd
import plotnine as gg
from bernoulli_bandit.agent import BernoulliBanditEpsilonGreedy
from bernoulli_bandit.agent import BernoulliBanditTS
from bernoulli_bandit.environment import BernoulliBandit
from bernoulli_bandit.experiment import BaseExperiment

probs = [0.7, 0.8, 0.9]
n_steps = 1000
seed = 10000

# agent = BernoulliBanditEpsilonGreedy(n_arm=len(probs))
agent = BernoulliBanditTS(n_arm=len(probs))
env = BernoulliBandit(probs)
experiment = BaseExperiment(
    agent, env, n_steps=n_steps, seed=seed, unique_id='example')

experiment.run_experiment()


df_agent = experiment.results
print(df_agent)
n_action = np.max(df_agent.action) + 1
plt_data = []
for i in range(n_action):
    probs2 = (df_agent.groupby('t')
            .agg({'action': lambda x: np.mean(x == i)})
            .rename(columns={'action': 'action_' + str(i)}))
    plt_data.append(probs2)
plt_df = pd.concat(plt_data, axis=1).reset_index()
print(plt_df)
p = (gg.ggplot(pd.melt(plt_df, id_vars='t'))
    + gg.aes('t', 'value', colour='variable', group='variable')
    + gg.geom_line(size=1.25, alpha=0.75)
    + gg.xlab('time period (t)')
    + gg.ylab('Action probability')
    + gg.ylim(0, 1)
    + gg.scale_colour_brewer(name='Variable', type='qual', palette='Set1'))
print(p)
