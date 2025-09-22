
#%%
from util_dat import read_in_all_coefs,get_common_cols
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

coefs_pre ,models_pre = read_in_all_coefs(fit_type='active_choice', subset='',
                                  recompute_summary = False, recompute_parts=False,
                                  add_behav_params=True, get_best=True,
                                  time_window ='stim_bin',pre_time=0.15,post_time=0)

coefs_choice ,models_choice = read_in_all_coefs(fit_type='active_choice', subset='',
                                  recompute_summary = False, recompute_parts=False,
                                  add_behav_params=True, get_best=True,
                                  time_window ='choice_bin',pre_time=0.15,post_time=0)






common_cols = get_common_cols()
models_post_renamed = models_choice.drop(columns=common_cols)

models = models_pre.merge(models_post_renamed, on='fitID',suffixes=('_pre', '_choice'))

goodClus = models[(models.is_good) &
                   ((models.BerylAcronym=='SCm')|(models.BerylAcronym=='SCs'))
                   ].copy()

weights_comparison = goodClus.copy()
weights_comparison = weights_comparison.fillna(0)

# %%


param = 'choice'

g = sns.relplot(data=weights_comparison, x=f'{param}_pre', y=f'{param}_choice', row='SPL_behav',col='BerylAcronym',height=4, aspect=1)
#
for ax in g.axes.flatten():
    ax.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1)


weights_comparison['action'] = weights_comparison['baseline_choice'] - weights_comparison['baseline_pre']

weights_comparison['choice'] = weights_comparison['choice_choice'] - weights_comparison['choice_pre']


#%%

# how does it compare to the sensory variables?
sns.relplot(data=weights_comparison, x='action', y='choice', col='BerylAcronym', height=4, aspect=1)


# %%
