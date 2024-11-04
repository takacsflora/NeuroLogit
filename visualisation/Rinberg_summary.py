
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
    'log_loss_opto', 
    'auc_opto',
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

from scipy.stats import wilcoxon,ttest_rel

#rois = ['Frontal','SC','Vis','Lateral']
metrics = ['log_loss_opto']
rois  = ['Lateral','Vis']
for metric in metrics:
    fig, ax = plt.subplots(1,1,figsize = (4,4))

    tot = []
    for roi in rois:
        results_roi =  results[results.region == roi]
        pivot_df = results_roi.pivot_table(index=['mouseID'], columns='model', values = metric)

        sns.scatterplot(pivot_df,x = 'av_opto_hemispheric_additive',
                            y='av_opto_hemispheric_divisive',
                            color = colors[roi],edgecolor='k',s=50,linewidth=1,alpha=.9, 
                            ax=ax)
        
        tot.append(pivot_df)

    tot = pd.concat(tot,ignore_index=True)
    _, p = ttest_rel(tot.av_opto_hemispheric_additive.values,
        tot.av_opto_hemispheric_divisive.values)

    ax.axline([0.9,0.9],[1,1],color='k',linestyle='--')
    # ax.axhline(0,color ='k',linestyle='--')
    # ax.axvline(0,color ='k',linestyle='--')

    ax.set_xlabel('additive')
    ax.set_ylabel('divisive')
    ax.set_title(f'{metric}, {rois}, p = {p.round(3)}')
    ax.set_xlim([.2,.7])
    ax.set_ylim([.2,.7])

#%%


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
# check how equivalences hold in the additive moddle
results['audR_ratio'] = results['audR_opto'] / results['audR']
results['visR_ratio'] = results['visR_opto'] / results['visR']
results['audL_ratio'] = results['audL_opto'] / results['audL']
results['visL_ratio'] = results['visL_opto'] / results['visL']

results['bias_from_vis'] = (results['biasR_opto'] / results['visR_ratio']) - (results['biasL_opto'] / results['visL_ratio']) 

results['bias_from_aud'] = results['biasR_opto'] / results['audR_ratio'] - results['biasL_opto'] / results['audL_ratio'] 


additive = results[results.model == 'av_opto_hemispheric_additive']
# %%
fig,ax = plt.subplots(1,4,figsize = (12,3))

common_dict = { 
    'hue': 'region',
    'palette': colors, 
    'legend': False
}
sns.scatterplot(additive, x = 'visR_ratio', y = 'audR_ratio',ax= ax[0], **common_dict)
sns.scatterplot(additive, x = 'visL_ratio', y = 'audL_ratio',ax= ax[1], **common_dict)
sns.scatterplot(additive, x = 'bias_from_vis', y = 'bias',ax= ax[2], **common_dict)
sns.scatterplot(additive, x = 'bias_from_aud', y = 'bias',ax= ax[3], **common_dict)

for i in range(ax.size):
    ax[i].axline([0.9,0.9],[1,1],color='k',linestyle='--')
    ax[i].axhline(0,color ='k',linestyle='--')
    ax[i].axvline(0,color ='k',linestyle='--')


ax[2].set_xlim([-25,25])


plt.tight_layout()
# %%
