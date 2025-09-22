
# utility functions for large scale visualisation of the ephys population
# e.g. reading in the results/adding the behaviural params etc.

import pandas as pd
from src.ephys.encoding_avg import fit_dataset, get_winning_model


def add_behavioural_params(coefs):
    """
    Add behavioural parameters to the neural coefficients dataframe by perging on sessionID
    
    """
    #  adding the behavioural parameters 
    behav_params = pd.read_csv(r'C:\Users\Flora\Documents\Github\NeuroLogit\data\active_behaviour_params.csv')
    # Rename all columns in behav_params except 'sessionID' to add '_behav' suffix
    cols_to_rename = {col: f"{col}_behav" for col in behav_params.columns if col != 'sessionID'}
    behav_params = behav_params.rename(columns=cols_to_rename)
    coefs = coefs.merge(behav_params, on='sessionID', how='left')
    return coefs


def read_in_all_coefs(fit_type='active', subset='', 
                      recompute_summary=False,recompute_parts = False,
                      add_behav_params=True,time_window = 'stim_bin',
                      get_best = True,pre_time=0.0,post_time=0.15):
    """
    Read in or compute the coefficients for a given fit type and subset."""


    timing = {'time_window': time_window,'pre_time':pre_time,'post_time':post_time}

    path = rf'C:\Users\Flora\Documents\Github\NeuroLogit\data\{fit_type}_{time_window}_{subset}_pre{pre_time}_post{post_time}_coefs.csv'


    if recompute_summary:

        coefs = fit_dataset(fit_type = fit_type,
            dataset_kwargs={'set_name':'all', 'subset':subset},
            time_kwargs=timing,recompute=recompute_parts,
        )

        
        coefs.to_csv(path,index=False)


    else:
        coefs = pd.read_csv(path,low_memory=False)


    coefs['sessionID'] = coefs.subject + '_' + coefs.date

    if add_behav_params:
        coefs = add_behavioural_params(coefs)

    if get_best:
        # selecting the best/winning models
        best = get_winning_model(coefs,thr_scorer='adj_r2',thr=0.005)
    else:
        best = None


    return coefs, best


def get_common_cols():
    common_cols = ['neuronID',
                'hemi',
                'BerylAcronym',
                'bombcell_class',
                'is_good',
                'ml',
                'ap',
                'dv',
                'presenceRatio',
                'percentageSpikesMissing_gaussian',
                'fractionRPVs_estimatedTauR',
                'signalToNoiseRatio',
                'subject',
                'date',
                'sessionID',
                'audR_behav',
                'audL_behav',
                'visR_behav',
                'visL_behav',
                'gamma_behav',
                'bias_behav',
                'subject_behav',
                'date_behav',
                'SPL_behav',
                'log_loss_behav']
    
    return common_cols
