#%%

from util_dat import read_in_all_coefs,get_common_cols
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

coefs_pre,models_pre = read_in_all_coefs(fit_type='active', subset='', recompute=False, add_behav_params=True, get_best=True,pre_time=0.15,post_time=0)
coefs_post,models_post = read_in_all_coefs(fit_type='active', subset='', recompute=False, add_behav_params=True, get_best=True,pre_time=0,post_time=0.15)


common_cols = get_common_cols()
models_post_renamed = models_post.drop(columns=common_cols)

models = models_pre.merge(models_post_renamed, on='fitID',suffixes=('_pre', '_post'))

goodClus = models[(models.is_good) &
                   ((models.BerylAcronym=='SCm')|(models.BerylAcronym=='SCs'))
                   ].copy()

weights_comparison = goodClus.copy()
weights_comparison = weights_comparison.fillna(0)
# %%


param = 'audI'

g = sns.relplot(data=weights_comparison, x=f'{param}_pre', y=f'{param}_post', row='SPL_behav',col='BerylAcronym',height=4, aspect=1)
#
for ax in g.axes.flatten():
    ax.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1)

# %%

weights_comparison['stim_onset_resp'] = weights_comparison['baseline_post'] -  weights_comparison['baseline_pre']

weights_comparison['action'] = weights_comparison['task_post'] - weights_comparison['task_pre']




# %%
sns.kdeplot(data=weights_comparison, x='stim_onset_resp',
             y='audC_post')
# %%
#


weights_comparison['choice_post'].hist(bins=150)
# %%
