import os
import sys

import numpy as np
import pandas as pd
import plotnine as gg

_DEFAULT_DATA_PATH = '/path/to/your/data'
_DATA_CACHE = {}

def _name_cleaner(agent_name):
  """Renames agent_name to prettier string for plots."""
  rename_dict = {'correct_ts': 'Correct TS',
                 'kl_ucb': 'KL UCB',
                 'misspecified_ts': 'Misspecified TS',
                 'ucb1': 'UCB1',
                 'ucb-best': 'UCB-best',
                 'nonstationary_ts': 'Nonstationary TS',
                 'stationary_ts': 'Stationary TS',
                 'greedy': 'greedy',
                 'ts': 'TS',
                 'action_0': 'Action 0',
                 'action_1': 'Action 1',
                 'action_2': 'Action 2',
                 'bootstrap': 'bootstrap TS',
                 'laplace': 'Laplace TS',
                 'thoughtful': 'Thoughtful TS',
                 'gibbs': 'Gibbs TS'}
  if agent_name in rename_dict:
    return rename_dict[agent_name]
  else:
    return agent_name

def load_data(experiment_name, data_path=_DEFAULT_DATA_PATH):
  """Function to load in the data relevant to a specific experiment.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    df: dataframe of experiment data (uses cache for faster reloading).
  """
  if experiment_name in _DATA_CACHE:
    return _DATA_CACHE[experiment_name]
  else:
    all_files = os.listdir(data_path)
    good_files = []
    for file_name in all_files:
      if '.csv' not in file_name:
        continue
      else:
        file_experiment = file_name.split('exp=')[1].split('|')[0]
        if file_experiment == experiment_name:
          good_files.append(file_name)

    data = []
    for file_name in good_files:
      file_path = os.path.join(data_path, file_name)
      if 'id=' in file_name:
        if os.path.getsize(file_path) < 1024:
          continue
        else:
          data.append(pd.read_csv(file_path))
      elif 'params' in file_name:
        params_df = pd.read_csv(file_path)
        params_df['agent'] = params_df['agent'].apply(_name_cleaner)
      else:
        raise ValueError('Something is wrong with file names.')

    df = pd.concat(data)
    df = pd.merge(df, params_df, on='unique_id')
    _DATA_CACHE[experiment_name] = df
    return _DATA_CACHE[experiment_name]

def plot_action_proportion(df_agent):
  """Plot the action proportion for the sub-dataframe for a single agent."""
  n_action = np.max(df_agent.action) + 1
  plt_data = []
  for i in range(n_action):
    probs = (df_agent.groupby('t')
             .agg({'action': lambda x: np.mean(x == i)})
             .rename(columns={'action': 'action_' + str(i)}))
    plt_data.append(probs)
  plt_df = pd.concat(plt_data, axis=1).reset_index()
  p = (gg.ggplot(pd.melt(plt_df, id_vars='t'))
       + gg.aes('t', 'value', colour='variable', group='variable')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('Action probability')
       + gg.ylim(0, 1)
       + gg.scale_colour_brewer(name='Variable', type='qual', palette='Set1'))
  return p

def compare_action_selection_plot(experiment_name='finite_simple',
                                  data_path=_DEFAULT_DATA_PATH):
  """Specialized plotting script for TS tutorial paper action proportion."""
  df = load_data(experiment_name, data_path)
  plot_dict = {}
  for agent, df_agent in df.groupby(['agent']):
    key_name = experiment_name + '_' + agent + '_action'
    plot_dict[key_name] = plot_action_proportion(df_agent)
  return plot_dict