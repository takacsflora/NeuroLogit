#%%



import pandas as pd 
import numpy as np 

from src.ephys.results_utils import read_files,get_brain_region_colors
import src.ephys.decoding_choice.results_helpers as hP

import matplotlib.pyplot as plt
import seaborn as sns
from floras_helpers.plotting import off_axes
# %
from sklearn.metrics import log_loss, roc_auc_score

unique_only = True

all_ev = []
n_steps = 4
for i in range(n_steps):
    ev = read_files(which_result = 'decoding_choice_results',
                     filestub=f'task_ev_prestim{i}_', 
                     extension='csv', sessions=None)
    ev['time_bin'] = i
    all_ev.append(ev)

task_ev = pd.concat(all_ev)


# %%

models_to_evaluate = ['l1_0.05','pca_5','stim']
tot_col_to_groupby = ['is_test_set','sessionID','subject','date','roi_fitted','time_bin']
scores = []
for model_name in models_to_evaluate:

    probabilities = f'proba_right_{model_name}'
    pred_choices = f'predicted_choice_{model_name}'

    if task_ev[probabilities].isna().any():
        print(f"Skipping model {model_name} for session {task_ev.sessionID.unique()} due to all NaN probabilities. Model probably didn't fit.")
        continue

    grouped_scores = task_ev.groupby(tot_col_to_groupby).apply(
            lambda group: pd.Series({
                'log_loss': log_loss(group['choice'], group[probabilities],normalize=True) if len(group['choice'].unique()) > 1 else np.nan,
                'auc': roc_auc_score(group['choice'], group[pred_choices]) if len(group['choice'].unique()) > 1 else np.nan
            })).unstack('is_test_set')


    grouped_scores.columns = [f"{col}_{'train' if idx == 0 else 'test'}" for col, idx in grouped_scores.columns]
    grouped_scores = grouped_scores.reset_index()
    grouped_scores['model'] = model_name
    scores.append(grouped_scores)

scores = pd.concat(scores)

#%% calculate delta scores between model and stim model
stim_scores = scores[scores.model=='stim'][['sessionID','time_bin','log_loss_test','auc_test']]
stim_scores = stim_scores.rename(columns={'log_loss_test':'log_loss_test_stim','auc_test':'auc_test_stim'})
scores = scores.merge(stim_scores,on=['sessionID','time_bin'])
scores['delta_log_loss_test'] = scores['log_loss_test_stim'] - scores['log_loss_test']
scores['delta_auc_test'] = scores['auc_test'] - scores['auc_test_stim']
# %%
scores['subject_roi'] = scores['subject'] + '_' + scores['roi_fitted']

# average across sessions for each subject_roi
avg_scores = scores.groupby(['subject_roi','subject','roi_fitted','model','time_bin']).mean().reset_index()
# %%

region_colors = get_brain_region_colors()


fig,ax = plt.subplots(1,1,figsize=(1.5,2),dpi=150)
sns.lineplot(data=avg_scores[avg_scores.model=='pca_5'], 
             x='time_bin', y='delta_auc_test',
              hue='roi_fitted',ax=ax,errorbar=('ci',75),palette=region_colors,legend=False)

# ax.set_xticks([1,4])
# ax.set_xticklabels(np.round([0,-0.15],2))

ax.set_xticks([1,3])
ax.set_xticklabels(np.round([0,-0.1],2))

ax.axvline(1,color='gray',ls=':',alpha=0.5)
ax.invert_xaxis()
#ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
off_axes(ax,which='top')
ax.set_xlabel('Rel. time (s)')
ax.set_ylabel(r'$\Delta$AUC')
ax.set_ylim([-0.05,0.15])
ax.set_yticks([0,0.1])


savepath = r'C:\Users\Flora\OneDrive - University College London\Cortexlab\conferences\Cosyne2026\Raw_figures'
fig.savefig(f'{savepath}/Decoding_choice_SCm_deltaAUC_prestim.svg',dpi=300,bbox_inches='tight',transparent=True)

#ax.axhline(0,color='k',ls=':',alpha=0.5)
# %%
# %%
