#%%

# another attempt to visualise the rseponses


import numpy as np
import matplotlib.pyplot as plt

import NeuroLogit.src.ephys.psth_visualise.psth_helpers as ph
from NeuroLogit.src.ephys.cccp.results_helpers import add_ipsi_contra

from floras_helpers.plotting import off_axes


#%%
# get all the combination of the above conditions 

trig_kws = {'vis_conds':  [-1,-0.5,-0.25,0,0.25,0.5,1], 
            'aud_conds': ['no_sound',-1,0,1], 
            'choice_conds': ['passive'],
            'sessions': 'unique',
            'dat_type':'zscored',
            'swap_to_ipsi_contra': True,
            'odds_conds': [-1,-0.5,0,0.5,1]
            }

clusters = ph.hP.read_files(filestub='clusters',extension='csv',sessions=trig_kws['sessions'])
clusters = add_ipsi_contra(clusters)

clusters['sessionID'] = clusters.subject.astype(str) + '_' + clusters.date.astype(str)


#%%
mean,sem,tscale = ph.read_all_psths(triggered_at='stim_stimCategory', **trig_kws)


#%% get the power content for each neuron in each stim condition 

time_period = (-0.1,.7) # time period to look at
t_idx = (tscale >= time_period[0]) & (tscale < time_period[1])
tscale_t = tscale[t_idx]

audstim = trig_kws['aud_conds']
visstim = trig_kws['vis_conds']
passive_keys = [(v,a,'passive') for v in visstim for a in audstim]

# %%

#%%


key = 'vis_stim_passive'

sel_nrn = clusters[
    (clusters.BerylAcronym.isin(['MOs'])) & 
    (clusters.bombcell_class=='good')  & 
    #(clusters['is_choice_choice_Go']==True)
     ((clusters[f'is_vis_stim_passive']==True) &
      (clusters[f'is_aud_stim_passive']==False ))
].copy()

nrn_idxs = sel_nrn.index.values # all indices


#%%
time_period = (-0.1,.7) # time period to average over
t_idx = (tscale >= time_period[0]) & (tscale < time_period[1])
tscale_t = tscale[t_idx]
psth_idx = np.ix_(nrn_idxs,t_idx)

# baseline subtract
blank_psth = mean[(0,0,'passive')][psth_idx]

passive_mean_resps = np.concatenate([mean[(v,a,'passive')][psth_idx] for v in visstim for a in audstim],axis=1)

n_neurons = passive_mean_resps.shape[0]


from sklearn.decomposition import PCA 
pca = PCA(n_components=int(n_neurons/3))
pca.fit(passive_mean_resps)
pc1 = pca.components_[0,:]


#%%
neuron_loadings = pca.transform(passive_mean_resps)  # neurons x components



#%%
# look at explained variance across components
explained_var = pca.explained_variance_ratio_
fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150) 
ax.plot(np.arange(len(explained_var))+1,explained_var*100,'-')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')

#%%
# reshape pc1 to time x viscond x audcond
# atm I think basically it is 

pc1_reshaped = pc1.reshape(len(visstim), len(audstim), -1)
aud_colors = ['k','blue','grey','red']
fig,axs = plt.subplots(len(audstim),len(visstim),figsize=(4,2),sharex=True,sharey=True)
fig.subplots_adjust(wspace=0.1,hspace=0.1)
for i,(aud,color) in enumerate(zip(audstim,aud_colors)):
    for j,vis in enumerate(visstim):
        ax=axs[len(audstim)-i-1,j]
        ax.plot(tscale_t,pc1_reshaped[j,i,:],color=color)
        ax.axvline(0,color='k',ls=':',alpha=0.2)
        off_axes(ax)

  #%%

from rastermap import Rastermap


# SCm visual neurons 10PC 10 clusters 1.0 locality 5 time lag
power_pcs = np.where((pca.explained_variance_ratio_>.02))[0][-1] +10  # number of PCs to keep
clus_scler = int(n_neurons/5)
model = Rastermap(n_PCs=power_pcs, n_clusters=clus_scler, 
                  locality=1.0, time_lag_window=5).fit(passive_mean_resps)
y = model.embedding # neurons x 1
isort = model.isort

# visualize binning over neurons
X_embedding = model.X_embedding



# rehape the embedding as pc1
# fig = plt.figure(figsize=(12,5))
# ax = fig.add_subplot(111)
ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
# %%

time_period = (-0.1,.7) # time period to average over
t_idx = (tscale >= time_period[0]) & (tscale < time_period[1])
tscale_t = tscale[t_idx]
psth_idx = np.ix_(nrn_idxs,t_idx)

n_aud = len(audstim)
n_vis = len(visstim)

fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(10,5))
fig.subplots_adjust(wspace=0.1,hspace=0.1)

blank_psth = mean[(0,0,'passive')][psth_idx]

for i,aud in enumerate(audstim):
    for j,vis in enumerate(visstim):
        ax=axs[n_aud-i-1,j]
        resp = mean[(vis,aud,'passive')][psth_idx]#-blank_psth
        ax.matshow(resp[isort,:],aspect='auto',cmap='coolwarm',vmin=-1.5,vmax=1.5)
        
        off_axes(ax)

# %%
