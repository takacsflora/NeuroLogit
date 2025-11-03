

#%%

from src.ephys.encoding_avg.encoding_avg import fit_dataset, get_winning_model, plot_prediction,filt_trials,get_time_params, get_predictors,get_tested_models
from src.ephys.dat_utils import load_trial_data

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from util_vis import get_component_matrix

timing = {'time_window':'stim_bin','pre_time':0.15,'post_time':0.0}

fit_type = 'active_choice'

# load the models 
subject = 'AV030'
date = '2022-12-07'

coefs = fit_dataset(fit_type = fit_type,
    dataset_kwargs={'set_name':'all', 'subset':f'{subject}_{date}'},
    time_kwargs=timing,
    recompute=False
)

models = get_winning_model(coefs,thr_scorer='adj_r2',thr=0.005) # across all models

# winning model for gamma
best_models = coefs.sort_values('adj_r2', ascending=False).groupby(['model', 'fitID'], as_index=False).first()

goodClus  = models[models.is_good] # the units that passed the quality metric thresholds by bombcell

# reload the data
sess_params ={
    'subject':subject,
    'date':date
}
time_params = get_time_params(**timing)
df,clusters,_ = load_trial_data(**sess_params,**time_params).values()

df = filt_trials(df,fit_type)
  #%%

model_names = goodClus.model.unique()
n_models = len(model_names)

max_n_nrns = 4
fig,axs = plt.subplots(n_models,max_n_nrns*3, figsize=(4.5*max_n_nrns,1*n_models), dpi=150, sharex=True,sharey=False)
fig.subplots_adjust(hspace=0.4,wspace=0.5)

for x,model in enumerate(model_names):    

    nrns = goodClus[goodClus.model==model].neuronID.values

    n_nrns = len(nrns)

    for y,nrn in enumerate(nrns):
        if y< max_n_nrns:
            ax =axs[x,3*y:3*y+3]
            nrn_fitID = f"{sess_params['subject']}_{sess_params['date']}_{nrn}"
            # find neuron in the best_mode_df
            nrn_model = models[models.fitID==nrn_fitID].copy()

            print(nrn_model.BerylAcronym.values[0],nrn_model.model.values[0],nrn_model.bombcell_class.values[0])
            #nrn_model = coefs[(coefs.fitID==nrn_fitID) & (coefs.model=='av') & (coefs.gamma==2)].copy()

            plot_prediction(df,nrn_model,plot_gamma_transformed_v=False,axes=ax)
            ax[0].set_title(nrn.split('_')[-1], fontsize=8)
            ax[1].set_title('')
            ax[2].set_title('')

            ymax = max(a.get_ylim()[1] for a in ax)
            ymin = min(a.get_ylim()[0] for a in ax)
            for a in ax:
                a.set_ylim(ymin, ymax)
    
            for a in ax:
                a.set_xlabel('')
                a.set_ylabel('')

        if y== 0:
            axs[x,0].set_ylabel(model, fontsize=10)
    
    # Remove unused axes if fewer than max_n_nrns neurons for this model
    if n_nrns < max_n_nrns:
        for y in range(n_nrns, max_n_nrns):
            for a in axs[x, 3*y:3*y+3]:
                a.remove()



#  #%%

# components = get_component_matrix(fit_type = fit_type)

# fig,ax = plt.subplots(figsize=(2, n_models), dpi=150)
# ax.matshow(components[model_names].values.astype('int').T,cmap='Greys',aspect='auto')
# ax.set_xticklabels([''] + list(components.index), rotation=90)

# # for i, label in enumerate(model_names):
# #     curr_comp = components[label]
#     ax.scatter(
#         np.arange(len(curr_comp)),np.ones_like(curr_comp)*i,c=curr_comp, cmap='Greys', s=50)        

# %%

# okay. Instead of get total winning model, for eah model type we just get the one with the best gamma as measured by adj_r2

#nrns = goodClus[goodClus.model=='choice'].neuronID.values


nrns = ['neuron_210']

for nrn in nrns:

    nrn_fitID = f"{sess_params['subject']}_{sess_params['date']}_{nrn}"
    # find neuron in the best_mode_df
    nrn_model = models[models.fitID==nrn_fitID].copy()

    print(nrn_model.BerylAcronym.values[0],
          nrn_model.model.values[0],
          nrn_model.bombcell_class.values[0],
          nrn_model.choice.values[0])
    #nrn_model = coefs[(coefs.fitID==nrn_fitID) & (coefs.model=='av') & (coefs.gamma==2)].copy()

    plot_prediction(df,nrn_model,plot_gamma_transformed_v=False)

    plt.suptitle(nrn)




    nrn_bests = best_models[best_models.neuronID==nrn]



    fig,axs = plt.subplots(2,1,figsize=(12, 6),dpi=150,sharex=False)
    sns.lineplot(data=nrn_bests, x='model', y='adj_r2', marker='o',ax=axs[0])

    xtick_pos = axs[0].get_xticks()
    xtick_labels = [label.get_text() for label in axs[0].get_xticklabels()]



    components = get_component_matrix(fit_type = fit_type)
    desired_order = [
        'baseline', 'visC', 'visI', 'visC_congruent', 'visI_congruent', 'Vpres',
        'audC', 'audI','task','choice'
    ]
    components = components.reindex([idx for idx in desired_order if idx in components.index])
    # rerder the rows of components so that it is ['baseline', 'visC', 'visI', 'visC_congruent',... ]

    for i, label in enumerate(xtick_labels):
        curr_comp = components[label]
        axs[1].scatter(
            np.ones_like(curr_comp)*xtick_pos[i],np.arange(len(curr_comp)),c=curr_comp, cmap='Greys', s=50)
        
    axs[1].set_yticks(np.arange(len(curr_comp)))
    axs[1].set_yticklabels(components.index)

    axs[0].set_xticks([])
    axs[1].set_xticks([])




# %%

## # plot the avearge 


# Count and calculate percentages
model_counts = goodClus['model'].value_counts()
model_percentages = (model_counts / model_counts.sum()) * 100

# Create a DataFrame for plotting
model_stats = model_counts.to_frame(name='count')
model_stats['percentage'] = model_percentages
model_stats = model_stats.sort_values(by='count', ascending=False)

# Plot as a barplot
fig,axs = plt.subplots(2,1,figsize=(5, 6),dpi=150,sharex=False,height_ratios=[2, 1])

sns.barplot(x=model_stats.index, y=model_stats['percentage'], order=model_stats.index,ax=axs[0])
axs[0].set_ylabel('%')
axs[0].axhline(y=5, color='k', linestyle='--', linewidth=0.5)


for i, label in enumerate(model_stats.index):
    curr_comp = components[label]
    axs[1].scatter(
        np.ones_like(curr_comp)*xtick_pos[i],np.arange(len(curr_comp)),c=curr_comp, cmap='Greys', s=50)
    
axs[1].set_yticks(np.arange(len(curr_comp)))
axs[1].set_yticklabels(components.index)

axs[0].set_xticks([''])
axs[1].set_xticks([])
axs[0].set_xlabel('Model components')

#%%

# many units are extremely unreliable ... 



# %%
