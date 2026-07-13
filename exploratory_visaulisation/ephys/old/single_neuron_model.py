#%%
# load a session

import numpy as np
from utils.av_dat_utils import load_trial_data,prepare_for_fit
from src.batch.encoding import fit_nrn,plot_prediction
import seaborn as sns

import matplotlib.pyplot as plt 

# 
# test things with example neuron 


fit_type = 'passive'
df,clusters,rasters = load_trial_data('AV008','2022-03-09',
                            load_clusters=True,load_raster='stim',
                            avg_kwargs={'pre_time':0.0,'post_time':0.2}).values()


#%%
added_conditions = ['choice','visDiff','audDiff','choice_categorical']
rasters = rasters.merge(df[added_conditions], left_on='Trial', right_index=True, how='left')

# plot the behaviour so the fraction of choices per visDiff and audDiff
average_choice = df[df.choice >= 0].groupby(['visDiff', 'audDiff'])['choice'].mean().reset_index()
sns.scatterplot(x='visDiff', y='choice', hue='audDiff', data=average_choice,palette='coolwarm')

#%%
# Create a 2D plot where x axis is visDiff, y axis is audDiff, and hue depends on the fraction of choices that were 1
choice_fraction = df[df.choice >= 0].groupby(['visDiff', 'audDiff']).agg(
    choice_mean=('choice', 'mean'),
    trial_count=('choice', 'size')
).reset_index()
# Adjust the text to be in the middle of each square
# Create a pivot table to reshape the data for matshow
pivot_table = choice_fraction.pivot('audDiff', 'visDiff', 'choice_mean')

# Plot the matrix using matshow
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(pivot_table, cmap='coolwarm')
# Adjust the colorbar height
fig.colorbar(cax, ax=ax, fraction=0.01, pad=0.04, label='Fraction of right choices')

# Annotate each cell with the fraction of choices and trial count
# Reverse the y-axis to make audDiff decrease from bottom to top
for (i, j), val in np.ndenumerate(pivot_table):
    trial_count = choice_fraction[(choice_fraction['audDiff'] == pivot_table.index[i]) & 
                                  (choice_fraction['visDiff'] == pivot_table.columns[j])]['trial_count'].values[0]
    ax.text(j, i, f'{val:.2f}\n({trial_count})', ha='center', va='center', color='black')

ax.set_xlabel('Visual Difference')
ax.set_ylabel('Auditory Difference')
ax.set_xticks(range(len(pivot_table.columns)))
ax.set_xticklabels(pivot_table.columns)
ax.set_yticks(range(len(pivot_table.index)))
ax.set_yticklabels(pivot_table.index)
ax.invert_yaxis()

plt.show()

#%%
df = prepare_for_fit(df,fit_type=fit_type)

# load the results and pick what was selected as the best model 

#%%
# start with example neuron
neuron = 'neuron_1140'


print(clusters[clusters.neuronID==neuron][['BerylAcronym','bombcell_class']])

#%
# plot the simple passive models w\o modulations / single example
predictors =  ['visL','visR', 'audL','audR','baseline','choice']
weights,m = fit_nrn(df,neuron,predictors,model_name = 'audiovisual',fitter='scipy',return_model=True)
fig,ax = plt.subplots(1,1,figsize=(3,3))
plot_prediction(m,df.copy(),predictors,neuron,ax=ax,extra_predictors = {'baseline': 1,'choice':-2})
plt.ylim([-1,150])
plt.xlim([-.41,.41])
ax.set_ylabel('Firing rate (spks/s)')
ax.set_title('')
plt.legend([])



# %%
# choice models/plots


predictors =  ['visL','visR', 'audL','audR','baseline','is_active','choice']
weights,m = fit_nrn(df,neuron,predictors,model_name = 'audiovisual_engagement_gain',fitter='scipy',return_model=True)

print(weights)
unique_columns = ['choice','is_active']

unique_plots = np.sort(np.unique(df[unique_columns],axis=0))


n_plots = unique_plots.shape[0]

fig,ax = plt.subplots(n_plots,1,figsize = (3,3*n_plots),sharex=True,sharey=True)

for i,unique_cond in enumerate(unique_plots):
    added_pred_dict = {k:v for k,v in zip(unique_columns,unique_cond)}
    added_pred_dict.update({'baseline':1})

    plot_prediction(m,df.copy(),predictors,neuron,ax=ax[i],extra_predictors = added_pred_dict)
    #ax[i].set_title(f'{unique_columns}:{unique_cond}')
    ax[i].set_title('')
    ax[i].set_ylabel('Firing rate (spks/s)')

    ax[i].legend().set_visible(False)

    if i == n_plots - 1:
        fig.legend(ax[i].get_legend_handles_labels()[0], ax[i].get_legend_handles_labels()[1], loc='upper right')
    
plt.xlim([-.41,.41])    
plt.ylim([-1,140])

# %%

feature_to_plot = neuron


on_x = 'visDiff'
on_y = 'choice'
on_hue  = 'audDiff'

# on_x = 'choice'
# on_y = 'audDiff'
# on_hue  = 'visDiff'

# Average the responses across trials for each combination of conditions
average_responses = rasters[rasters.Feature == feature_to_plot].groupby(
    [on_y, on_x, on_hue, 'Time']
)['Response'].mean().reset_index()

# Use the averaged responses for plotting
g = sns.FacetGrid(average_responses[average_responses.choice!=-1], 
                  row=on_y, col=on_x, hue=on_hue, margin_titles=True, palette='coolwarm',
                  despine=False, height=2, aspect=1)

g.map(sns.lineplot, 'Time', 'Response')

g.add_legend(title=on_hue)
plt.show()

# %%
