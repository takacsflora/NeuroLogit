
#%%

from src.ephys.encoding_avg import fit_dataset, get_winning_model
import pandas as pd

timing = {'time_window':'stim','pre_time':0.0,'post_time':0.15}

fit_type = 'passive'
subset = ''

recompute = True
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


# %%


goodClus = models[(models.is_good) &
                   ((models.BerylAcronym=='SCm')|(models.BerylAcronym=='SCs')) & 
                   (models.subject.isin(['AV005','AV008','AV014','FT030','FT032','AV025','AV030','AV034'])) 
                   ].copy()

# goodClus = models[(models.is_good) &
#                    ((models.BerylAcronym=='MOs')) & 
#                    (models.subject.isin(['AV007','AV013'])) 
#                    ].copy()

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
# select individual neurons --- 
coefs_full = coefs[
    (coefs.is_good) &
    ((coefs.BerylAcronym == 'SCs')|(coefs.BerylAcronym=='SCm')) & 
    (coefs.model == 'av_bilateral') 
      ].copy().reset_index(drop=True)


# get the winnder gamma 

#
#%%
average_dv_per_session = coefs.groupby(['subject', 'date'])['dv'].mean().reset_index()

# Select sessions within each subject that are at least 600 dv apart
selected_sessions = []
for subject, group in average_dv_per_session.groupby('subject'):
    group = group.sort_values('dv')
    selected = []
    for _, row in group.iterrows():
        if not selected or all(abs(row['dv'] - s['dv']) >= 600 for s in selected):
            selected.append(row)
    selected_sessions.extend(selected)

selected_sessions = pd.DataFrame(selected_sessions)

# Filter selected_coefs to keep only neurons from selected_sessions
selected_coefs = coefs.merge(
    selected_sessions[['subject', 'date']],
    on=['subject', 'date'],
    how='inner'
)





# maybe let's do the anatomy plot for each mouse or group of cells? 
from floras_helpers.anat_plots import anatomy_plotter


anat = anatomy_plotter()
fig,ax = plt.subplots(1,1,figsize=(4, 5),dpi=300)
anat.plot_anat_canvas(ax=ax,coord = 3800, axis='ap')

anat.plot_points(selected_coefs['ml'],selected_coefs['dv'],unilateral=True,c = 'grey',alpha=0.2,marker = '.',s=100,edgecolor=None)

# param = 'aud_ipsi'
# cc = selected_coefs[(selected_coefs['model_type'] != 'baseline') & (selected_coefs['model_type'] != 'vis')].copy()
# #cc = selected_coefs[(selected_coefs['model_type'] != 'baseline') & (selected_coefs['model_type'] != 'aud')].copy()

# anat.plot_points(cc['ml'],cc['dv'],unilateral=True,c = cc[param],alpha=1,marker = '.',s=200,edgecolor='k',cmap='coolwarm',vmin=-20,vmax=20)

ax.set_xlim([-2200, -200])
ax.set_ylim([-3000, -500])
ax.invert_xaxis()
# %%
