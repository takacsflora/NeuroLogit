
#%%

from src.ephys.encoding_avg.encoding_avg import fit_dataset, get_winning_model, plot_prediction,filt_trials,get_time_params, get_predictors,get_tested_models
from src.ephys.dat_utils import load_trial_data

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from util_vis import get_component_matrix


fit_type = 'active_choice'

subject,date = 'AV008','2022-03-10'
df_choice,clusters,_ = load_trial_data(subject,date,
                            load_clusters=True,load_raster='choice',
                            avg_kwargs={'pre_time':0.15,'post_time':0.0}).values()

df_stim,clusters,_ = load_trial_data(subject,date,
                            load_clusters=True,load_raster='stim',
                            avg_kwargs={'pre_time':0.15,'post_time':0}).values()


#%%

from src.ephys.cccp.stat import run_combined_condition_U
results = run_combined_condition_U(df_choice[df_choice.rt>0.1],to_discriminate='choice')


# %%
choice_neurons = results[(clusters.BerylAcronym=='SCm') & (clusters.is_good)].neuronID.values
# %%

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=df_choice, x='neuron_0', hue='choice',palette='coolwarm',multiple='layer',element='step',bins=100)

# %%

colors = ['red','blue','blue']
# Use kdeplot for 2D KDE by 'choice'
# Scatterplot with color by choice

nrn = choice_neurons[25]
print(clusters[clusters.neuronID==nrn].BerylAcronym.values[0])
palette = {choice: color for choice, color in zip(df_choice['choice'].unique(), colors)}
for choice_value, group in df_choice[(df_choice.choice>=0)].groupby('choice'):
    sns.scatterplot(
        data=group,
        x='rt', 
        y=nrn,
        hue='choice',
        alpha=1,
        palette=palette)
    
    sns.kdeplot(
        data=group,
        x='rt',
        y=nrn,
        fill=True,
        alpha=0.4,
        label=f'choice={choice_value}',
        levels=30,
        color=palette[choice_value])

plt.legend()
# %%
