import pandas as pd
import plotnine as gg
import numpy as np
from news.agent_news import GreedyNewsRecommendation
from news.env_news import NewsRecommendation
from base.experiment import ExperimentNoAction



def generateGreedyAgent(n_steps,  jobs):
    num_articles = 3
    dim = 7
    theta_mean = 0
    theta_std = 1
    epsilon1 = 0.01
    alpha=0.2
    beta=0.5
    tol=0.0001
    results = []
    for job_id in range(jobs):
        agent = GreedyNewsRecommendation(num_articles, dim, theta_mean, theta_std, epsilon1,alpha,beta,tol)
        env = NewsRecommendation(num_articles, dim, theta_mean, theta_std)
        experiment = ExperimentNoAction(agent, env, n_steps=n_steps, seed=job_id, unique_id=str(job_id))
        experiment.run_experiment()
        results.append(experiment.results)

    df_agent = (pd.concat(results)).assign(agent='TS')
    return df_agent


def generateAgents():
    n_steps = 500
    N_JOBS = 20

    agents = []

    agents.append(generateGreedyAgent(n_steps,N_JOBS))
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

plotCompare1()
