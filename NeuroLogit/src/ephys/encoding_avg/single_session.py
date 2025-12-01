
#%% fit and look at results from a single session. 


from NeuroLogit.src.ephys.encoding_avg.batch_proc import fit_dataset, get_winning_model, plot_prediction,filt_trials,get_time_params, get_predictors,get_tested_models
from NeuroLogit.src.ephys.dat_utils import load_trial_data

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from NeuroLogit.src.ephys.encoding_avg.visualisation_helpers import get_component_matrix

timing = {'time_window':'stim','pre_time':0.0,'post_time':0.15}

fit_type = 'passive'

# load the models 
subject = 'AV034'
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



# %%
