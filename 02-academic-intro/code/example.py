import numpy as np
import pandas as pd
import plotnine as gg
from bernoulli_bandit.agent import BernoulliBanditEpsilonGreedy
from bernoulli_bandit.agent import BernoulliBanditTS
from bernoulli_bandit.environment import BernoulliBandit
from bernoulli_bandit.experiment import BaseExperiment

def plotCompara1(algorithm = 'TS'):
    probs = [0.9,0.8,0.7]
    n_steps = 1000
    N_JOBS = 200

    results = []
    for job_id in range(N_JOBS):
        if(algorithm == 'TS'):
            agent = BernoulliBanditTS(n_arm=len(probs))
        else:
            agent = BernoulliBanditEpsilonGreedy(n_arm=len(probs))
        env = BernoulliBandit(probs)
        experiment = BaseExperiment(
            agent, env, n_steps=n_steps, seed=job_id, unique_id=str(job_id))
        experiment.run_experiment()
        results.append(experiment.results)


    df_agent = pd.concat(results)


    n_action = np.max(df_agent.action) + 1
    plt_data = []
    for i in range(n_action):
        probs2 = (df_agent.groupby('t') # 按照t分组
                .agg({'action': lambda x: np.mean(x == i)}) # 按照action求平均
                .rename(columns={'action': 'action_' + str(i + 1)}))  # 重命名列
        plt_data.append(probs2)
    # 重置索引
    plt_df = pd.concat(plt_data, axis=1).reset_index()

    p = (gg.ggplot(pd.melt(plt_df, id_vars='t'))
        + gg.aes('t', 'value', colour='variable', group='variable')
        + gg.geom_line(size=1.25, alpha=0.75)
        + gg.xlab('time period (t)')
        + gg.ylab('Action probability')
        + gg.ylim(0, 1)
        + gg.scale_colour_brewer(name='Variable', type='qual', palette='Set1'))
    print(p)


def plotCompara2():
    probs = [0.9,0.8,0.7]
    n_steps = 1000
    N_JOBS = 200

    results1 = []
    for job_id in range(N_JOBS):
        agent = BernoulliBanditTS(n_arm=len(probs))
        env = BernoulliBandit(probs)
        experiment = BaseExperiment(
            agent, env, n_steps=n_steps, seed=job_id, unique_id=str(job_id))
        experiment.run_experiment()
        results1.append(experiment.results)

    df_agent1 = (pd.concat(results1)).assign(agent='TS')

    results2 = []
    for job_id in range(N_JOBS):
        agent = BernoulliBanditEpsilonGreedy(n_arm=len(probs))
        env = BernoulliBandit(probs)
        experiment = BaseExperiment(
            agent, env, n_steps=n_steps, seed=job_id, unique_id=str(job_id))
        experiment.run_experiment()
        results2.append(experiment.results)
    df_agent2 = (pd.concat(results2)).assign(agent='greedy')


    df_agents = pd.concat([df_agent1,df_agent2])

    plt_df = (df_agents.groupby(['t','agent'])
            .agg({'instant_regret': np.mean})
            .reset_index())
    p = (gg.ggplot(plt_df)
       + gg.aes('t', 'instant_regret', colour='agent')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('per-period regret')
       + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1'))
    print(p)


# plotCompara1('Greedy')
# plotCompara1('TS')

plotCompara2()


    
