
#%%

from NeuroLogit.src.ephys.encoding_avg.results_helpers import read_in_all_coefs
import pandas as pd

coefs,models = read_in_all_coefs(fit_type='passive', subset='', 
                                recompute=False, 
                                add_behav_params=True,
                                get_best=True)

# %%
# 
# 'AV005','AV008','AV014','FT030','FT032'

goodClus = models[(models.is_good) &
                  ((models.BerylAcronym=='SCm')|(models.BerylAcronym=='SCs'))  
                  # (models.subject.isin(['AV005','AV008','AV014','FT030','FT032','AV025','AV030','AV034'])) 
                   ].copy()


# Calculate the fraction of SCm and SCs neurons per sessionID
sc_counts = goodClus.groupby(['sessionID', 'BerylAcronym']).size().unstack(fill_value=0)
sc_counts['total'] = sc_counts.sum(axis=1)
sc_counts['SCs_frac'] = sc_counts.get('SCs', 0) / sc_counts['total']

# Map the winner back to goodClus
goodClus = goodClus.merge(sc_counts['SCs_frac'], left_on='sessionID', right_index=True, how='left')

#%%

import seaborn as sns
import matplotlib.pyplot as plt


model_counts = goodClus.groupby('sessionID')['model'].value_counts(normalize=True) * 100
model_counts = model_counts.rename('percentage').reset_index()
model_counts[['subject', 'date']] = model_counts['sessionID'].str.split('_', expand=True)
model_counts = model_counts.merge(goodClus[['sessionID', 'SCs_frac','SPL_behav']].drop_duplicates(), on='sessionID', how='left')

#%%
import numpy as np



desired_model_order = ['baseline', 'vis', 'vis_bilateral',
                    'aud','aud_ipsi','aud_bilateral',
                    'av','av_aud_bilateral','av_bilateral',
                    'av_multiplicative','av_bilateral_multiplicative','a_vPres'] 


model_counts_per_sess = model_counts.pivot('model', 'sessionID', 'percentage').reindex(desired_model_order)

# Order sessionIDs by SCs_frac (fraction of SCs neurons per session)
session_order = sc_counts['SCs_frac'].sort_values(ascending=False).index
model_counts_per_sess = model_counts_per_sess[session_order]

# plot per sessionID 
fig,ax = plt.subplots(1, 1, figsize=(20, 5), dpi=150)
sns.heatmap(data=model_counts_per_sess,
            annot=True, fmt=".1f", cmap='Oranges', cbar_kws={'label': 'Percentage'},vmin=5,vmax=20,ax=ax)
        # Impose a specific row order on the heatmap

#%%






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

from src.ephys.encoding_avg.encoding_avg import get_predictors

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



#
#%%
average_dv_per_session = goodClus.groupby(['subject', 'date'])['dv'].mean().reset_index()

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
selected_coefs = goodClus.merge(
    selected_sessions[['subject', 'date']],
    on=['subject', 'date'],
    how='inner'
)




# %%
# maybe let's do the anatomy plot for each mouse or group of cells? 
from floras_helpers.anat_plots import anatomy_plotter


anat = anatomy_plotter()
fig,ax = plt.subplots(1,1,figsize=(4, 5),dpi=300)
anat.plot_anat_canvas(ax=ax,coord = 3800, axis='ap')

anat.plot_points(selected_coefs['ml'],selected_coefs['dv'],unilateral=True,c = 'grey',alpha=0.2,marker = '.',s=100,edgecolor=None)

param = 'Vpres'
cc = selected_coefs[selected_coefs[param].notna()].copy()

import numpy as np
# cc['is_extrame_gamma'] = np.abs((cc['gamma']-1))
# anat.plot_points(cc['ml'],cc['dv'],unilateral=True,c = cc['gamma'],alpha=1,marker = '.',s=100,edgecolor='k',cmap='RdYlBu',vmin=0,vmax=2)


anat.plot_points(cc['ml'],cc['dv'],unilateral=True,c = cc[param],alpha=1,marker = '.',s=200,edgecolor='k',cmap='coolwarm',vmin=-20,vmax=20)

ax.set_xlim([-2200, -200])
ax.set_ylim([-3000, -500])
ax.invert_xaxis()
ax.set_title(f'{param} weight')
# %%

weights_comparison = selected_coefs.copy()
weights_comparison = weights_comparison.fillna(0)


            # Make all pairplots have the same x and y axes
g = sns.pairplot(weights_comparison,
                    vars=['baseline', 'visC', 'visI', 'audC', 'audI', 'Vpres'],
                    kind='scatter',
                    diag_kind='kde',
                    markers='o')

# # Find global min/max for all variables
# cols = ['baseline', 'visC', 'visI', 'audC', 'audI', 'Vpres']
# min_val = weights_comparison[cols].min().min()
# max_val = weights_comparison[cols].max().max()

# # Set the same limits for all axes
# for i, row in enumerate(g.axes):
#     for j, ax in enumerate(row):
#         if ax is not None:
#             ax.set_xlim(min_val, max_val)
#             ax.set_ylim(min_val, max_val)

# %%
