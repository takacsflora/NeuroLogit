
# %%
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from floras_helpers.io import get_latest_file
from utils.av_dat_utils import get_paths
from utils.plot_utils import get_region_colors

_,_,results_path  = get_paths(set_name= r'opto\Rinberg')
results = pd.read_csv(get_latest_file(results_path,'summary'))

colors = get_region_colors()

def get_param_predictions(results):
    results['optoL_coeff'] = -results.optoL/(1+results.optoL)
    results['optoR_coeff'] = -results.optoR/(1+results.optoR)

    # predictions of the model
    results['bias_predicted'] = results['biasR'] + results['biasL']
    results['audR_opto_predicted'] = results.audR * results.optoL_coeff
    results['audL_opto_predicted'] = results.audL * results.optoR_coeff
    results['visR_opto_predicted'] = results.visR * results.optoL_coeff
    results['visL_opto_predicted'] = results.visL * results.optoR_coeff
    results['biasR_opto_predicted'] = results.biasR * results.optoL_coeff * (-1)
    results['biasL_opto_predicted'] = results.biasL * results.optoR_coeff

    predicted_param_list = [
        'bias',
        'audR_opto',
        'audL_opto',
        'visR_opto',
        'visL_opto',
        'biasR_opto',
        'biasL_opto'
    ]

    for p in predicted_param_list:
        pred_name = p + '_predicted'
        results[pred_name] = results.apply(lambda row: row[p] if 'additive' in row['model'] else row[pred_name], axis=1)

    return results


results = get_param_predictions(results)
# %%

metrics = [
    'log_loss', 
    'auc',
    'visL',
    'visR',
    'audR',
    'audL',
    'gamma', 
    'bias_predicted',
    'audR_opto_predicted',
    'audL_opto_predicted',
    'visR_opto_predicted',
    'visL_opto_predicted',
    'biasR_opto_predicted',
    'biasL_opto_predicted'
    ]

rois = ['Frontal','SC','Vis','Lateral']

for metric in metrics:
    fig, ax = plt.subplots(1,1,figsize = (4,4))

    for roi in rois:
        results_roi =  results[results.region == roi]
        pivot_df = results_roi.pivot_table(index=['mouseID'], columns='model', values = metric)

        sns.scatterplot(pivot_df,x = 'av_opto_hemispheric_additive',
                            y='av_opto_hemispheric_divisive',
                            color = colors[roi],edgecolor='k',s=50,linewidth=1,alpha=.9, 
                            ax=ax)

    ax.axline([0.9,0.9],[1,1],color='k',linestyle='--')
    ax.axhline(0,color ='k',linestyle='--')
    ax.axvline(0,color ='k',linestyle='--')

    ax.set_xlabel('additive')
    ax.set_ylabel('divisive')
    ax.set_title(metric)


# %%
# now we will compare the parameters for each model
fig,ax = plt.subplots(1,1)
divisive = results[results.model == 'av_opto_hemispheric_divisive']
divisive['biasL_pos'] = divisive['biasL'] * (-1)
sns.swarmplot(divisive,x='region',y = 'biasL_pos',ax = ax)
ax.set_yscale('log')
# %%
results.to_csv(results_path / 'edited.csv')
# %%
