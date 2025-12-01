#%%

from NeuroLogit.src.ephys.results_utils import read_files 
from NeuroLogit.src.ephys.encoding_avg.batch_proc import get_winning_model, plot_prediction,filt_trials,get_time_params

coefs = read_files(which_result = 'encoding_avg_results', filestub='passive_all_stim_pre0.0_post0.15', extension='csv', sessions='AV034_2022-12-07')


# load the mean psths -- well the idea is, let me plot the model predictions and I will deal with the data later




models = get_winning_model(coefs,thr_scorer='adj_r2',thr=0.005) # across all models

# winning model for gamma
best_models = coefs.sort_values('adj_r2', ascending=False).groupby(['model', 'fitID'], as_index=False).first()

goodClus  = models[models.is_good] # the units that passed the quality metric thresholds by bombcell



#%%

model_names = goodClus.model.unique()
n_models = len(model_names)

max_n_nrns = 10
fig,axs = plt.subplots(n_models,max_n_nrns, figsize=(1.5*max_n_nrns,1*n_models), dpi=150, sharex=True,sharey=False)
fig.subplots_adjust(hspace=0.4,wspace=0.5)

for x,model in enumerate(model_names):    

    nrns = goodClus[goodClus.model==model].neuronID.values

    n_nrns = len(nrns)

    for y,nrn in enumerate(nrns):
        if y< max_n_nrns:
            ax =axs[x,y]
            nrn_fitID = f"{sess_params['subject']}_{sess_params['date']}_{nrn}"
            # find neuron in the best_mode_df
            nrn_model = models[models.fitID==nrn_fitID].copy()

            print(nrn_model.BerylAcronym.values[0],nrn_model.model.values[0],nrn_model.bombcell_class.values[0])
            #nrn_model = coefs[(coefs.fitID==nrn_fitID) & (coefs.model=='av') & (coefs.gamma==2)].copy()

            plot_prediction(filt_trials(df,fit_type),nrn_model,plot_gamma_transformed_v=False,axes=ax)
            ax.set_title(nrn.split('_')[-1], fontsize=8)
            ax.set_ylabel('')
            ax.set_xlabel('')

        if y== 0:
            axs[x,0].set_ylabel(model, fontsize=10)
    
    if y < max_n_nrns-1:
        for y2 in range(y+1,max_n_nrns):
            axs[x,y2].axis('off') 




# %%
