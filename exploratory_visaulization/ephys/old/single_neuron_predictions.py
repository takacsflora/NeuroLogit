
#%%
import numpy as np
from utils.av_dat_utils import load_trial_data,prepare_for_fit
from src.batch.encoding import fit_nrn,plot_prediction,fit_all_neurons
import seaborn as sns

import matplotlib.pyplot as plt 

# 
# test things with example neuron 


fit_type = 'task'
df,clusters,rasters = load_trial_data('AV005','2022-05-12',
                            load_clusters=True,load_raster='stim',
                            avg_kwargs={'pre_time':0.0,'post_time':0.15}).values()



coefs  = fit_all_neurons(df,clusters,fit_type=fit_type)
coefs = coefs.reset_index()
# Keep only the rows with the highest adj_r2 for each neuronID
# Ensure there are no duplicate rows for each neuronID by keeping only the row with the highest adj_r2
coefs = coefs.loc[coefs.groupby('neuronID', group_keys=False)['adj_r2'].idxmax()]

#%%

df = prepare_for_fit(df,fit_type=fit_type)

# load the results and pick what was selected as the best model 

#%
# start with example neuron
neuron = 'neuron_130'
neuron_properties = clusters[clusters.neuronID==neuron].copy()
print(neuron_properties[['BerylAcronym','bombcell_class']])
nrn_hemisphere = neuron_properties['hemi'].values[0]

#%
# plot the simple passive models w\o modulations / single example
predictors =  ['visL','visR', 'audL','audR','baseline','choice']



weights,m = fit_nrn(df,neuron,predictors,model_name = 'audiovisual',return_model=True,hemisphere=nrn_hemisphere)

#
fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=300)


plot_prediction(m,df.copy(),predictors,neuron,ax=ax,extra_predictors = {'baseline': 1,'choice':-2},hemisphere=nrn_hemisphere) # this isn't with vis C now so its shit
# plt.ylim([-1,150])
# plt.xlim([-.41,.41])
ax.set_ylabel('')
ax.set_title('')

# Remove the top and right spines for a cleaner plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# with error
# %%


# Plot a histogram of counts of the column 'model_type' in the coefs DataFrame
plt.figure(figsize=(6, 4), dpi=300)
sns.histplot(coefs['model_type'], bins=10, kde=False, color='blue')
plt.xlabel('Model Type')
plt.ylabel('Count')
plt.title('Histogram of Model Type Counts')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# %%
