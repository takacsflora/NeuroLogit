
# this script addresses from several angles -- whether there is choice related activity in the prestimulus period

#%%
from NeuroLogit.src.ephys.encoding_avg.results_helpers import read_in_all_coefs,get_winning_model
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

coefs,models = read_in_all_coefs(fit_type='active_choice', subset='',
                                  recompute_summary = False, recompute_parts=False,
                                  add_behav_params=True, get_best=True,
                                  time_window ='stim_bin',pre_time=0.15,post_time=0)


# %%

goodClus = models[(models.is_good) &
                   ((models.BerylAcronym=='SCm') |(models.BerylAcronym=='SCs')) & 
                     (np.abs(models.bias_behav)<1)
                   ].copy()

# goodClus = models[(models.is_good) &
#                    ((models.BerylAcronym=='MOs')) & 
#                    (models.subject.isin(['AV023'])) 
#                    ].copy()


#goodClus = models[(models.is_good) & (models.BerylAcronym=='SCs')].copy()



model_counts = goodClus.groupby('sessionID')['model'].value_counts(normalize=True) * 100
model_counts = model_counts.rename('percentage').reset_index()


# Order the models by descending model counts
model_order = model_counts.groupby('model')['percentage'].sum().sort_values(ascending=False).index
model_counts['model'] = pd.Categorical(model_counts['model'], categories=model_order, ordered=True)
# Barplot showing the percentage of each model
plt.figure(figsize=(10, 6))
sns.barplot(data=model_counts, x='model', y='percentage', ci=None)
plt.title('Percentage of Models Across All Sessions')
plt.ylabel('Percentage')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


model_counts = model_counts.merge(goodClus[['sessionID','SPL_behav','bias_behav']].drop_duplicates(), on='sessionID', how='left')


#%%


sns.scatterplot(data=model_counts[model_counts.model=='choice'], x='bias_behav', y='percentage',hue='SPL_behav')

# Lineplot showing the percentage of each model per session
#%%
#model_counts['subject'] = model_counts['sessionID'].str.split('_').str[0]
plt.figure(figsize=(12, 8))

from src.ephys.encoding_avg.encoding_avg import get_predictors

fig,ax = plt.subplots(1, 1, figsize=(12, 8))

sns.lineplot(data=model_counts, x='model', y='percentage', hue='SPL_behav', marker='o',ax=ax)
ax.set_title('Percentage of neurons per sesssion, on avg')
ax.set_ylabel('Percentage')
ax.set_xlabel('Model')
ax.legend(title='Session ID', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.axhline(y=5, color='k', linestyle='--', linewidth=0.5)


xticklabels = [t.get_text() for t in ax.get_xticklabels()]
xticklabels = [get_predictors(model) for model in xticklabels]
ax.set_xticklabels(xticklabels, rotation=45, ha='right')

plt.tight_layout()
plt.show()
# %%

# %%


sns.scatterplot(data=goodClus, x='bias_behav', y='choice')


# %%


# 
choice_model = coefs[coefs.model=='choice'].copy()
baseline_model = coefs[coefs.model=='baseline'].copy()

scorer = 'r2_score'

choice_adj_r2 = choice_model[['fitID', scorer]].rename(columns={scorer: f'choice_{scorer}'})
baseline_adj_r2 = baseline_model[['fitID', scorer]].rename(columns={scorer: f'baseline_{scorer}'})


r2_comps = pd.merge(choice_adj_r2, baseline_adj_r2, on='fitID', how='inner')


sns.scatterplot(data=r2_comps, x=f'baseline_{scorer}', y=f'choice_{scorer}')
plt.axline((0, 0), slope=1, color='k', linestyle='--', linewidth=1)



# other question
# %%


