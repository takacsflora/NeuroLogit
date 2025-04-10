
#%%
import pandas as pd 
import numpy as np
from utils.neural_results_utils import load_results,select_best_models


regions = ['SCs','SCm']
#coefs = pd.concat([load_results(dataset='all',region = r,fit_type = 'passive',time_bin = 'stim_bin_pre_0.40_post_0.60') for r in regions])
coefs = pd.concat([load_results(dataset=None,region = r,fit_type = 'passive',time_bin = 'poststim') for r in regions])

selected_coefs = select_best_models(coefs,thr=-0.01)
selected_coefs = selected_coefs.fillna(0)


# plot the average dv in selected coefs.session 

# Compute the average dv per session
#%%
average_dv_per_session = selected_coefs.groupby(['subject', 'session'])['dv'].mean().reset_index()

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
selected_coefs = selected_coefs.merge(
    selected_sessions[['subject', 'session']],
    on=['subject', 'session'],
    how='inner'
)



#%%






selected_coefs['-dv'] = selected_coefs.dv *-1





# %%
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(1,4,figsize=(10, 3),sharex=True,sharey=True)

kws = {
    # 'stat':'probability',
    # 'binwidth':(5,60),
}


sns.scatterplot(x='vis_contra',y='-dv',data=selected_coefs,ax=ax[0],**kws)

sns.scatterplot(x='aud_contra',y='-dv',data=selected_coefs,ax=ax[1],**kws)

sns.scatterplot(x='aud_ipsi',y='-dv',data=selected_coefs,ax=ax[2],**kws)


sns.scatterplot(x='baseline',y='-dv',data=selected_coefs,ax=ax[3],**kws)

for a in ax:
    a.axhline(-1750, color='k', linestyle='--')
#sns.scatterplot(x='vis_ipsi',y='-dv',data=selected_coefs,ax=ax)


#sns.scatterplot(x='aud_ipsi',y='-dv',data=selected_coefs,ax=ax)

# %%
selected_coefs['dv_bin'] = pd.cut(selected_coefs['-dv'], bins=10)
selected_coefs['dv_bin_mean'] = selected_coefs['dv_bin'].apply(lambda x: x.mid if pd.notnull(x) else None)

model_counts = selected_coefs.groupby(['dv_bin_mean', 'model_type']).size().unstack(fill_value=0)
model_counts = model_counts.div(model_counts.sum(axis=1), axis=0) * 100
print(model_counts)
fig, axes = plt.subplots(1, len(model_counts.columns), figsize=(8, 5), sharey=True,sharex=True)

color_map = {'baseline': 'grey', 'vis': 'black', 'aud': 'red', 'audiovisual': 'brown'}


for i, model in enumerate(['baseline','vis','aud','audiovisual']):
    ax = axes[0]
    color = color_map[model]
    #sns.lineplot(x=model_counts[model], y=model_counts.index, ax=ax)
    ax.plot(model_counts[model], model_counts.index,lw=5,color=color)
    ax.set_title(model)
    ax.set_xlabel('%')
    ax.set_ylabel('DV Bin' if i == 0 else '')



for ax in axes:
    ax.axvline(5, color='k', linestyle='--')
    ax.axhline(-1750, color='k', linestyle='--')

plt.tight_layout()
plt.show()

#%%
#similarly 
sns.scatterplot(x='ml',y='-dv',hue = 'vis_contra',data=selected_coefs)
sns.scatterplot(x='ml',y='-dv',hue = 'aud_contra',data=selected_coefs)


# %%
non_baseline = selected_coefs[selected_coefs['model_type'] != 'baseline'].copy()

# get the averge vsi_contra weight per dv bin
weight_mean_per_bin = non_baseline.groupby('dv_bin_mean')['aud_ipsi'].mean()

plt.plot()
plt.plot(weight_mean_per_bin.values, weight_mean_per_bin.index,lw=5,color=color)

#%%

#%%
plt.plot(selected_coefs['aud_ipsi'],selected_coefs['aud_contra'],'.')
#%%
y = (selected_coefs['aud_contra'])#+np.abs(selected_coefs['aud_ipsi'])

fig,ax = plt.subplots(1,1,figsize=(5, 3),dpi=300)
ax.scatter(selected_coefs['vis_contra'],y,
            marker = '.',s=200,edgecolor=None,c='grey',alpha=0.5)


ax.axvline(0, color='k', linestyle='--')
ax.axhline(0, color='k', linestyle='--')
ax.axline((0, 0), slope=1, color='k', linestyle='--')

ax.set_xlim([-20,80])
ax.set_ylim([-20,40])

#%%
from floras_helpers.anat_plots import anatomy_plotter

anat = anatomy_plotter()
fig,ax = plt.subplots(1,1,figsize=(4, 5),dpi=300)
anat.plot_anat_canvas(ax=ax,coord = 3800, axis='ap')

anat.plot_points(selected_coefs['ml'],selected_coefs['dv'],unilateral=True,c = 'grey',alpha=0.2,marker = '.',s=100,edgecolor=None)

param = 'aud_ipsi'
cc = selected_coefs[(selected_coefs['model_type'] != 'baseline') & (selected_coefs['model_type'] != 'vis')].copy()
#cc = selected_coefs[(selected_coefs['model_type'] != 'baseline') & (selected_coefs['model_type'] != 'aud')].copy()

anat.plot_points(cc['ml'],cc['dv'],unilateral=True,c = cc[param],alpha=1,marker = '.',s=200,edgecolor='k',cmap='coolwarm',vmin=-20,vmax=20)

ax.set_xlim([-2200, -200])
ax.set_ylim([-3000, -500])
ax.invert_xaxis()
# %%

from floras_helpers.anat_plots import anatomy_plotter

anat = anatomy_plotter()
fig,ax = plt.subplots(1,1,figsize=(9, 5),dpi=300)
anat.plot_anat_canvas(ax=ax,coord = 3800, axis='ap')

#anat.plot_points(selected_coefs['ml'],selected_coefs['dv'],unilateral=True,c = 'grey',alpha=0.2,marker = '.',s=100,edgecolor=None)


ax.set_xlim([-2250, 2250])
ax.set_ylim([-3000, -500])
# %%
