# %%

import numpy as np
from utils.av_dat_utils import load_trial_data,prepare_for_fit

import matplotlib.pyplot as plt 
import seaborn as sns
# 
# test things with example neuron 


fit_type = 'choice_engagement'


ev,clusters,rasters = load_trial_data('AV008','2022-03-09',
                            load_clusters=True,load_raster='stim',
                            avg_kwargs=None).values()


added_conditions = ['choice','visDiff','audDiff']
rasters = rasters.merge(ev[added_conditions], left_on='Trial', right_index=True, how='left')

#%%


#%%

# Optimize the plotting by reducing the number of seaborn calls and using vectorized operations

# Prepare data for plotting
feature_to_plot = 'neuron_1177'


on_x = 'visDiff'
on_y = 'choice'
on_hue  = 'audDiff'



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

#%%
# with error

g = sns.FacetGrid(rasters[rasters.Feature==feature_to_plot], 
                  row=on_y, col=on_x, hue=on_hue, margin_titles=True, palette='coolwarm',
                  despine=False, height=1, aspect=1)

g.map(sns.lineplot, 'Time', 'Response')

# Add legend
g.add_legend(title=on_hue)
plt.show()

# %%
