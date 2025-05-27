#%%
from src.ephys.encoding_avg import fit_dataset, get_winning_model
import pandas as pd

timing = {'time_window':'stim','pre_time':0.0,'post_time':0.15}

fit_type = 'active'
subset = ''

recompute = False
path = rf'C:\Users\Flora\Documents\Github\NeuroLogit\data\{fit_type}_{subset}_coefs.csv'


if recompute:

    coefs = fit_dataset(fit_type = fit_type,
        dataset_kwargs={'set_name':'all', 'subset':subset},
        time_kwargs=timing
    )

    
    coefs.to_csv(path,index=False)
else:
    coefs = pd.read_csv(path,low_memory=False)

coefs['sessionID'] = coefs.subject + '_' + coefs.date
models = get_winning_model(coefs,thr_scorer='adj_r2',thr=0)



#%%


# goodClus = models[(models.is_good) &
#                    ((models.BerylAcronym=='SCm')|(models.BerylAcronym=='SCs')) & 
#                    (models.subject.isin(['AV025','AV030','AV034'])) 
#                    ].copy()

goodClus = models[(models.is_good) &
                   ((models.BerylAcronym=='MOs')) & 
                   (models.subject.isin(['AV023'])) 
                   ].copy()


#goodClus = models[(models.is_good) & (models.BerylAcronym=='SCs')].copy()

import seaborn as sns
import matplotlib.pyplot as plt


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

# Lineplot showing the percentage of each model per session
#%%
model_counts['subject'] = model_counts['sessionID'].str.split('_').str[0]
plt.figure(figsize=(12, 8))

from src.ephys.encoding_avg import get_predictors

fig,ax = plt.subplots(1, 1, figsize=(12, 8))

sns.lineplot(data=model_counts, x='model', y='percentage', hue='subject', marker='o',ax=ax)
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
fullm = coefs[(coefs.model == 'av_aud_bilateral_choice')].copy()
fullm = get_winning_model(fullm,thr_scorer='explained_variance_score',thr=-40)

evthr = 0.05
fullm['is_audC'] = fullm['explained_variance_score_audC'] > evthr
fullm['is_visC'] = fullm['explained_variance_score_visC'] > evthr
fullm['is_audI'] = fullm['explained_variance_score_audI'] > evthr
fullm['is_task'] = fullm['explained_variance_score_task'] > evthr
fullm['is_choice'] = fullm['explained_variance_score_choice'] > evthr

fullm = fullm[fullm.is_good].copy()


#%%
ps = ['audC','visC','audI','task','choice']

explained_vars = [f'explained_variance_score_{p}' for p in ps]

sns.scatterplot(data=fullm,x='audI',y='choice',hue='date',palette='Set2',legend=False)

#plt.ylim([-.25,2.6])
#plt.xlim([-.25,1])

# %%
