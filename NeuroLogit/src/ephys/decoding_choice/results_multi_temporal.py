#%%

from pathlib import Path
import pandas as pd 
import numpy as np 

from NeuroLogit.src.ephys.results_utils import read_files 
import NeuroLogit.src.ephys.results_utils as hP

import matplotlib.pyplot as plt 
import seaborn as sns
from results_helpers import format_scores
from floras_helpers.plotting import off_axes
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind,ttest_rel
# %


SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Choice')

plt.rcParams.update({'font.size': 7,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})

# SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\conferences\Cosyne2026\Poster figures')

# plt.rcParams.update({'font.size': 24,'font.family':'Calibri','axes.linewidth':1,'axes.spines.top':False,'axes.spines.right':False,
#                      'axes.spines.left':True,'axes.spines.bottom':True,
#                      'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})

times = [f'choice{i}' for i in range(9)] # finer bins around choice
post_times = np.linspace(0.1,-0.3,9)
pre_times = post_times - 0.05

times += [f'prestim{i}' for i in range(4)] # finer bins around prestim
post_times = np.concatenate([post_times, np.linspace(0.05,-0.15,4)])
pre_times = np.concatenate([pre_times, post_times - 0.05])

model_type = 'scipy_taskorchoice' # specify the way we select the neurons
which_sesh = None  # session to load

#%% results and metrics
# in ev there is a column called sessionID and another one called roi_fitted
all_ev,all_weights,all_metrics = [],[],[]
for i, time_stub in enumerate(times):
    ev = read_files(which_result = 'decoding_choice_results', filestub=f'task_ev_{model_type}_{time_stub}', extension='csv', sessions=which_sesh)
    weights = read_files(which_result = 'decoding_choice_results', filestub=f'weights_{model_type}_{time_stub}', extension='csv', sessions=which_sesh)
    metrics = read_files(which_result = 'decoding_choice_results', filestub=f'metrics_{model_type}_{time_stub}', extension='csv', sessions=which_sesh)
    
    ev['stub'] = time_stub
    ev['pre_time'] = pre_times[i]
    ev['post_time'] = post_times[i]
    all_ev.append(ev)

    metrics = metrics.merge(ev[['sessionID','roi_fitted']].drop_duplicates(),on=['sessionID'],how='left')

    metrics['stub'] = time_stub
    metrics['pre_time'] = pre_times[i]
    metrics['post_time'] = post_times[i]
    
    all_metrics.append(metrics)

    weights = weights.merge(ev[['sessionID','roi_fitted']].drop_duplicates(),on=['sessionID'],how='left')

    weights['stub'] = time_stub
    weights['pre_time'] = pre_times[i]
    weights['post_time'] = post_times[i]

    all_weights.append(weights)

metrics = pd.concat(all_metrics)
ev = pd.concat(all_ev)
weights = pd.concat(all_weights)

metrics['time_bin_mid'] = np.round(metrics['post_time']-0.025, 4).astype(float)

metrics = format_scores(metrics)
# %%
fig,ax = plt.subplots(1,2,figsize=(2.5,2),dpi=150,sharex=False,sharey=True,width_ratios=[4,9])
#fig,ax = plt.subplots(1,2,figsize=(6,4),dpi=150,sharex=False,sharey=True,width_ratios=[4,9])
avg_subject_roi_metrics = metrics.groupby(['subject','roi_fitted','model','stub','pre_time','post_time','time_bin_mid']).mean().reset_index()


region_colors = hP.get_brain_region_colors()
model_type= 'full'
plotted_metric =  'delta_auc_test'
rois = ['MOs','SCs','SCm']
for time_flag ,ax_i in zip(['is_at_prestim_time','is_at_choice_time'],ax):

    if time_flag == 'is_at_prestim_time':
        stim_metrics = avg_subject_roi_metrics[(avg_subject_roi_metrics.model=='bias_only') &
                                            (avg_subject_roi_metrics[time_flag]) &
                                            (avg_subject_roi_metrics.roi_fitted.isin(rois))]
        

        curr_metrics = avg_subject_roi_metrics[(avg_subject_roi_metrics.model=='neural_bias_only') & 
                                        (avg_subject_roi_metrics[time_flag]) & 
                                        (avg_subject_roi_metrics.roi_fitted.isin(rois))]
        
    else: 
        stim_metrics  = avg_subject_roi_metrics[(avg_subject_roi_metrics.model=='stim') &
                                                (avg_subject_roi_metrics[time_flag]) & 
                                                (avg_subject_roi_metrics.roi_fitted.isin(rois))]

        curr_metrics = avg_subject_roi_metrics[(avg_subject_roi_metrics.model==model_type) & 
                                (avg_subject_roi_metrics[time_flag]) & 
                                (avg_subject_roi_metrics.roi_fitted.isin(rois))]


    sns.lineplot(data=curr_metrics, x='time_bin_mid', y=plotted_metric,
            hue='roi_fitted',estimator=np.mean,errorbar='se',ax=ax_i,legend=False,
            palette=region_colors,linewidth=2)
    
    sns.lineplot(data=stim_metrics, x='time_bin_mid', y=plotted_metric,
                 estimator=np.mean,errorbar='se',
                 ax=ax_i,legend=False,color='black',linestyle='-')
    


    ax_i.axhline(0,color='gray',ls=':',linewidth=1.5)
    ax_i.axvline(0,color='gray',ls=':',linewidth=1.5)


#ax[0].set_ylim(0.7,1)
#ax[1].get_yaxis().set_visible(False)
ax[0].set_ylabel(plotted_metric)
#ax[1].spines['left'].set_visible(False)
#ax[0].set_xlim([-0.15,0])

# for ax_i in ax:
#     ax_i.set_ylim([-0.15, 0.18])
#     ax_i.set_yticks([-0.05, 0, 0.05, 0.1, 0.15])

ax[0].set_ylim([-0.1,0.3])
#ax[0].set_ylim()
ax[0].set_ylabel(plotted_metric)
#ax[0].set_ylabel('Likelihood ratio \n (rel. to stim. only model)')
# ax[0].set_yticks([0,0.1,0.2])
# ax[0].set_yticklabels(['1','1.3','1.6'])
# ax[0].set_xticks([-0.1,0])
# ax[0].set_xticklabels(['-0.1','0'])
# ax[1].set_xticks([-0.15,0])
# ax[1].set_xticklabels(['-0.15','0'])

ax[0].set_xlabel('Time from stim. (s)')
ax[1].set_xlabel('Time from choice (s)')
#fig.suptitle(rois)

fig.tight_layout()
plt.savefig(SAVE_PATH / f'delta_logLoss_vs_time_{model_type}_{plotted_metric}.png', dpi=300)
# %%

test = avg_subject_roi_metrics[(avg_subject_roi_metrics.stub=='choice2') & 
                        (avg_subject_roi_metrics.model=='full')]

sns.stripplot(data=test, x='roi_fitted', y=plotted_metric,jitter=True)

# %%

metric_SCm =  avg_subject_roi_metrics[(avg_subject_roi_metrics.model==model_type) & 
                        (avg_subject_roi_metrics.roi_fitted=='SCm') & 
                        (avg_subject_roi_metrics.post_time==0.0)]

metric_MOs = avg_subject_roi_metrics[(avg_subject_roi_metrics.model==model_type) & 
                        (avg_subject_roi_metrics.roi_fitted=='MOs') & 
                        (avg_subject_roi_metrics.post_time==0.0)]

metric_SCs = avg_subject_roi_metrics[(avg_subject_roi_metrics.model==model_type) & 
                        (avg_subject_roi_metrics.roi_fitted=='SCs') &   
                        (avg_subject_roi_metrics.post_time==0.0)]
# %%
metric = plotted_metric
myx  = metric_SCm[metric]
myy = metric_SCs[metric]

if len(myx) > 1 and len(myy) > 1:
    t_stat, p_val = ttest_ind(myx, myy, equal_var=False)
else:
    raise ValueError("The test requires at least 2 samples for both groups.")
print(f"T-statistic: {t_stat}, P-value: {p_val}")
# %%
