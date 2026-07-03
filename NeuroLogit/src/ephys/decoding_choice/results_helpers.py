
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from floras_helpers.plotting import off_axes


def format_scores(metrics):
    
    """"
    Format the metrics dataframe by calculating delta logLoss relative to stim model"""
    # calculate the delta logLoss on  the test set by subtracting the stim model logLoss

    delta_metrics_to_compute = ['logLik_test','auc_roc_R_vs_L_test','auc_roc_NoGo_vs_rest_test',
                                'auc_roc_R_vs_Nogo_test','auc_roc_L_vs_Nogo_test',
                                'logLik_detect_test_avg_per_trial','logLik_discrim_test_avg_per_trial',
                                'brier_detect_test', 'brier_discrim_test']
    
    identifiers = ['sessionID','roi_fitted','stub']

    stim_metrics = metrics[metrics.model=='stim'][identifiers + delta_metrics_to_compute]
    stim_metrics = stim_metrics.rename(columns={col:f'stim_{col}' for col in delta_metrics_to_compute})

    bias_only_metrics = metrics[metrics.model=='bias_only'][identifiers + delta_metrics_to_compute]
    bias_only_metrics = bias_only_metrics.rename(columns={col:f'bias_only_{col}' for col in delta_metrics_to_compute})

    metrics = metrics.merge(stim_metrics, on=['sessionID','roi_fitted','stub'], how='left')
    metrics = metrics.merge(bias_only_metrics, on=['sessionID','roi_fitted','stub'], how='left')  

    

    for col in delta_metrics_to_compute:
        metrics[f'stimdelta_{col}'] = metrics[col] - metrics[f'stim_{col}']
        metrics[f'biasdelta_{col}'] = metrics[col] - metrics[f'bias_only_{col}']


    metrics['mcFadden'] = 1 - (metrics['logLik_test'] / metrics['bias_only_logLik_test'])
    metrics['mcFadden_detect'] = 1 - (metrics['logLik_detect_test_avg_per_trial'] / metrics['bias_only_logLik_detect_test_avg_per_trial'])
    metrics['mcFadden_discrim'] = 1 - (metrics['logLik_discrim_test_avg_per_trial'] / metrics['bias_only_logLik_discrim_test_avg_per_trial'])

    
    
    # Ensure that the time_bin_mid column is treated as a categorical variable with a specific order

    # metrics['time_bin_mid'] = pd.Categorical(metrics['time_bin_mid'], 
    #                                          categories=sorted(metrics['time_bin_mid'].unique(), key=float), 
    #                                          ordered=True)
    metrics['is_at_choice_time'] = metrics['stub'].str.contains('choice') 
    metrics['is_at_prestim_time'] = metrics['stub'].str.contains('prestim')

    for col in delta_metrics_to_compute:

        metrics[f'delta_{col}'] = np.where(metrics['is_at_prestim_time'], metrics[f'biasdelta_{col}'], metrics[f'stimdelta_{col}'])
    
    return metrics
    
