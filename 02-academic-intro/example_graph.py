import pandas as pd
import numpy as np
import plotnine as gg
from graph.agent_independent import IndependentBBEpsilonGreedy
from graph.agent_independent import IndependentBBTS
from graph.env_graph_bandit import IndependentBinomialBridge
from base.experiment import ExperimentNoAction


def generateTSAgent(n_steps, n_stages, mu0, sigma0, sigma_tilde, jobs):
    results = []
    for job_id in range(jobs):
        agent = IndependentBBTS(n_stages, mu0, sigma0, sigma_tilde)
        # 初始化环境，产生图
        env = IndependentBinomialBridge(n_stages, mu0, sigma0, sigma_tilde)
        experiment = ExperimentNoAction(agent, env, n_steps=n_steps, seed=job_id, unique_id=str(job_id))
        experiment.run_experiment()
        results.append(experiment.results)

    df_agent = (pd.concat(results)).assign(agent='TS')
    return df_agent


def generateEpsilonAgent(n_steps, n_stages, mu0, sigma0, sigma_tilde, jobs, epsilon=0):
    results = []
    for job_id in range(jobs):
        agent = IndependentBBEpsilonGreedy(
            n_stages, mu0, sigma0, sigma_tilde, epsilon)
        env = IndependentBinomialBridge(n_stages, mu0, sigma0, sigma_tilde)
        experiment = ExperimentNoAction(
            agent, env, n_steps=n_steps, seed=job_id, unique_id=str(job_id))
        experiment.run_experiment()
        results.append(experiment.results)
    df_agent = (pd.concat(results)).assign(agent='greedy-'+str(epsilon))
    return df_agent

def generateAgents():
    n_stages = 20
    n_steps = 500
    mu0 = -0.5
    sigma0 = 1
    sigma_tilde = 1
    N_JOBS = 200

    agents = []

    agents.append(generateEpsilonAgent(
        n_steps, n_stages, mu0, sigma0, sigma_tilde, N_JOBS))
    agents.append(generateEpsilonAgent(n_steps, n_stages, mu0,
                  sigma0, sigma_tilde, N_JOBS, epsilon=0.01))
    agents.append(generateEpsilonAgent(n_steps, n_stages, mu0,
                  sigma0, sigma_tilde, N_JOBS, epsilon=0.05))
    agents.append(generateEpsilonAgent(n_steps, n_stages, mu0,
                  sigma0, sigma_tilde, N_JOBS, epsilon=0.1))
    agents.append(generateTSAgent(n_steps, n_stages,
                  mu0, sigma0, sigma_tilde, N_JOBS))
    df_agents = pd.concat(agents)
    return df_agents   

def plotCompare1():
    df_agents = generateAgents()

    plt_df = (df_agents.groupby(['t', 'agent'])
              .agg({'instant_regret': np.mean})
              .reset_index())
    p = (gg.ggplot(plt_df)
         + gg.aes('t', 'instant_regret', colour='agent')
         + gg.geom_line(size=1.25, alpha=0.75)
         + gg.xlab('time period (t)')
         + gg.ylab('per-period regret')
         + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1'))
    print(p)

def plotCompare2():
    df_agents = generateAgents()

    df_agents['cum_ratio'] = (df_agents.cum_optimal - df_agents.cum_regret) / df_agents.cum_optimal
    plt_df = (df_agents.groupby(['t', 'agent'])
            .agg({'cum_ratio': np.mean})
            .reset_index())
    p = (gg.ggplot(plt_df)
       + gg.aes('t', 'cum_ratio', colour='agent')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('Total distance / optimal')
       + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1')
       + gg.aes(ymin=1)
       + gg.geom_hline(yintercept=1, linetype='dashed', size=2, alpha=0.5))
  
    print(p)

plotCompare2()