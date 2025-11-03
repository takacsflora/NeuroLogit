#%%

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

from src.ephys.cccp.results_helpers import add_ipsi_contra
from src.ephys.results_utils import read_files 
from floras_helpers.plotting import off_axes
# %

unique_only = True
clusters = read_files(which_result = 'ccCP_results', filestub='clusters', extension='csv', sessions='unique')
clusters = add_ipsi_contra(clusters)


# %%

sel_nrns = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCm','SCs']))# &
] 

# get the same neurons across subjects
min_neurons = sel_nrns.groupby('subject').size().min()
balanced_subjects = sel_nrns.groupby('subject').filter(lambda x: len(x) >= min_neurons)
balanced_subjects = balanced_subjects.groupby('subject').apply(lambda x: x.sample(n=min_neurons, random_state=42)).reset_index(drop=True)
sel_nrns = balanced_subjects
# %%

time_points  = [f'choice_choice{i}_Go' for i in range(6,-1,-1)]

is_significant_time = [f'is_{f}' for f in time_points]
cp_values_time = [f'cp_{f}' for f in time_points]

real_time_bins = np.linspace(-0.25,0.1,7)
# fraction of signifiant neurons at each time point
fraction_significant_ipsi = {tp: sel_nrns[(sel_nrns[f'is_{tp}']==1) & (sel_nrns[f'cp_{tp}']<0.5)].shape[0]/sel_nrns.shape[0] for tp in time_points}
fraction_significant_contra = {tp: sel_nrns[(sel_nrns[f'is_{tp}']==1) & (sel_nrns[f'cp_{tp}']>0.5)].shape[0]/sel_nrns.shape[0] for tp in time_points}

#%%
# can we canculate the same thing per subject and then average across subjects?
per_subject_ipsi = sel_nrns.groupby(['subject']).apply(lambda x: pd.Series({
    tp: x[(x[f'is_{tp}']==1) & (x[f'cp_{tp}']<0.5)].shape[0]/x.shape[0] for tp in time_points
}))
per_subject_ipsi = per_subject_ipsi.reset_index()

per_subject_contra = sel_nrns.groupby(['subject']).apply(lambda x: pd.Series({
    tp: x[(x[f'is_{tp}']==1) & (x[f'cp_{tp}']>0.5)].shape[0]/x.shape[0] for tp in time_points
}))
per_subject_contra = per_subject_contra.reset_index()
# make new df for plotting, keep the subject column
scores_ipsi = per_subject_ipsi.melt(var_name='time_bin',value_name='fraction_significant',id_vars=['subject'])
scores_contra = per_subject_contra.melt(var_name='time_bin',value_name='fraction_significant',id_vars=['subject'])
scores_ipsi['type'] = 'ipsi'
scores_contra['type'] = 'contra'

scores = pd.concat([scores_ipsi,scores_contra])

#%%

fig,ax = plt.subplots(1,1,figsize=(1.2,1.6),dpi=150)
sns.lineplot(data=scores, x='time_bin', y='fraction_significant', hue='type',errorbar=('ci',68),palette=['blue','red'],legend=False,ax=ax)

ax.set_xticks([1,4],labels=['-0.15','0'])

ax.axhline(0.05,color='grey',linestyle=':',linewidth=0.8)
ax.set_xlabel('Time from \n choice (s)')
ax.set_ylabel('% neurons')
ax.set_yticks([0,0.1,0.2],labels=['0','10','20'])
off_axes(ax,which='top')

ax.axvline(4,color='gray',ls=':',alpha=0.5)

savepath = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\conferences\Cosyne2026\Raw_figures')
fig.savefig(f'{savepath}/Fraction_ipsi_contra_neurons_SCm_timecourse.svg',dpi=300,bbox_inches='tight',transparent=True)
# %%
time_range = np.linspace(-0.25,0.1,7)
fig,ax = plt.subplots(1,1,figsize=(3,2),dpi=150)
ax.plot(range(len(time_points)),[fraction_significant_ipsi[tp] for tp in time_points],label='ipsi',color='blue')
ax.plot(range(len(time_points)),[fraction_significant_contra[tp] for tp in time_points],label='contra',color='red')

ax.set_xticks(range(len(time_points)),labels=[f'{t:.2f}' for t in time_range],rotation=45)



# %%
