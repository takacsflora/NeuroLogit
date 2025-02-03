#%%
# load a session

import numpy as np
from utils.av_dat_utils import load_trial_data,prepare_for_fit
from src.batch.encoding import fit_nrn,plot_prediction
import seaborn as sns

import matplotlib.pyplot as plt 

# 
# test things with example neuron 


fit_type = 'choice_engagement'
df,clusters,rasters = load_trial_data('AV005','2022-05-18',
                            load_clusters=True,load_raster='choice',
                            avg_kwargs={'pre_time':0.1,'post_time':0}).values()

added_conditions = ['choice','visDiff','audDiff','choice_categorical']
rasters = rasters.merge(df[added_conditions], left_on='Trial', right_index=True, how='left')


#

# plot the behaviour so the fraction of choices per visDiff and audDiff
average_choice = df[df.choice >= 0].groupby(['visDiff', 'audDiff'])['choice'].mean().reset_index()
sns.scatterplot(x='visDiff', y='choice', hue='audDiff', data=average_choice,palette='coolwarm')


#
df = prepare_for_fit(df,fit_type=fit_type)

# load the results and pick what was selected as the best model 

#%%
# start with example neuron
neuron = 'neuron_122'


print(clusters[clusters.neuronID==neuron][['BerylAcronym','bombcell_class']])

#%
# plot the simple passive models w\o modulations / single example
predictors =  ['visL','visR', 'audL','audR','baseline','choice']
weights,m = fit_nrn(df,neuron,predictors,model_name = 'audiovisual',fitter='scipy',return_model=True)
plot_prediction(m,df.copy(),predictors,neuron,ax=None,extra_predictors = {'baseline': 1,'choice':-2})

    


# %%
# choice models/plots


predictors =  ['visL','visR', 'audL','audR','baseline','is_active','choice']
weights,m = fit_nrn(df,neuron,predictors,model_name = 'audiovisual_engagement_gain',fitter='scipy',return_model=True)

print(weights)
unique_columns = ['choice','is_active']

unique_plots = np.sort(np.unique(df[unique_columns],axis=0))


n_plots = unique_plots.shape[0]

fig,ax = plt.subplots(1,n_plots,figsize = (5*n_plots,5),sharex=True,sharey=True)

for i,unique_cond in enumerate(unique_plots):
    added_pred_dict = {k:v for k,v in zip(unique_columns,unique_cond)}
    added_pred_dict.update({'baseline':1})

    plot_prediction(m,df.copy(),predictors,neuron,ax=ax[i],extra_predictors = added_pred_dict)
    ax[i].set_title(f'{unique_columns}:{unique_cond}')
    

    ax[i].legend().set_visible(False)

    if i == n_plots - 1:
        fig.legend(ax[i].get_legend_handles_labels()[0], ax[i].get_legend_handles_labels()[1], loc='upper right')
    
    

# %%

feature_to_plot = neuron


on_x = 'visDiff'
on_y = 'choice'
on_hue  = 'audDiff'

on_x = 'choice'
on_y = 'audDiff'
on_hue  = 'visDiff'

# Average the responses across trials for each combination of conditions
average_responses = rasters[rasters.Feature == feature_to_plot].groupby(
    [on_y, on_x, on_hue, 'Time']
)['Response'].mean().reset_index()

# Use the averaged responses for plotting
g = sns.FacetGrid(average_responses, 
                  row=on_y, col=on_x, hue=on_hue, margin_titles=True, palette='coolwarm',
                  despine=False, height=2, aspect=1)

g.map(sns.lineplot, 'Time', 'Response')

g.add_legend(title=on_hue)
plt.show()

# %%
