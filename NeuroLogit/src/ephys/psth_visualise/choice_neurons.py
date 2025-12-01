#%%


import numpy as np
import matplotlib.pyplot as plt

import NeuroLogit.src.ephys.psth_visualise.psth_helpers as ph
from NeuroLogit.src.ephys.cccp.results_helpers import add_ipsi_contra

from floras_helpers.plotting import off_axes

trig_kws = {'vis_conds':  [-1, -0.5,-0.25,0,0.25,0.5,1], 
            'aud_conds': [-1,0,1], 
            'choice_conds': ['left','right'],
            'sessions': 'unique',
            'dat_type':'zscored',
            'swap_to_ipsi_contra': True,
            'odds_conds': [-1,-0.5,0,0.5,1]
            }



mean_stim,sem_stim,tscale_stim = ph.read_all_psths(triggered_at='stim_stimCategory', **trig_kws)
mean_move,sem_move,tscale_move = ph.read_all_psths(triggered_at='move_stimCategory', **trig_kws)


clusters = ph.hP.read_files(filestub='clusters',extension='csv',sessions=trig_kws['sessions'])
clusters = add_ipsi_contra(clusters)


#%%
# and after I will show basically the

sel_nrn = clusters[
    (clusters.BerylAcronym.isin(['SCm'])) & 
    (clusters.bombcell_class=='good')  & 
    (clusters['is_choice_choice_Go']==True)
].copy()

nrn_idxs = sel_nrn.index.values 



#%%

def get_time_indices(tscale, time_period):
    return (tscale >= time_period[0]) & (tscale < time_period[1])

# select the fist 0.45 secords after stim onset 

t_idx_stim = get_time_indices(tscale_stim, (-0.05,0.15))
tscale_t = tscale_stim[t_idx_stim]
psth_idx_stim = np.ix_(nrn_idxs,t_idx_stim)

t_idx_move = get_time_indices(tscale_move, (-0.2,0.05))
tscale_t_move = tscale_move[t_idx_move]
psth_idx_move = np.ix_(nrn_idxs,t_idx_move)

#%%

visstim = trig_kws['vis_conds']
audstim = [-1,0,1]

stim_ipsi = np.concatenate([mean_stim[(v,a,'ipsi')][psth_idx_stim] for v in visstim for a in audstim],axis=1)
stim_contra = np.concatenate([mean_stim[(v,a,'contra')][psth_idx_stim] for v in visstim for a in audstim],axis=1)

move_ipsi = np.concatenate([mean_move[(v,a,'ipsi')][psth_idx_move] for v in visstim for a in audstim],axis=1)
move_contra = np.concatenate([mean_move[(v,a,'contra')][psth_idx_move] for v in visstim for a in audstim],axis=1)

all_resps = np.concatenate([stim_ipsi, stim_contra, move_ipsi, move_contra], axis=1)

#%%

# replace nans with 0s ...
all_resps = np.nan_to_num(all_resps)

from rastermap import Rastermap

model = Rastermap(n_PCs=10, n_clusters=10, 
                  locality=1.0, time_lag_window=5).fit(all_resps)
y = model.embedding # neurons x 1
isort = model.isort

# visualize binning over neurons
X_embedding = model.X_embedding
fig,ax = plt.subplots(1,1,figsize=(10,2),dpi=150)
ax.imshow(X_embedding, vmin=-2, vmax=2, cmap="coolwarm", aspect="auto")

## or pca

#%%
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
pca = PCA(n_components=10)
y = pca.fit_transform(all_resps)


# plot the first component across time
# look at explained variance across components
explained_var = pca.explained_variance_ratio_
fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150) 
ax.plot(np.arange(len(explained_var))+1,explained_var*100,'-')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')


#%%
pc1 = pca.components_[0,:]
n = 0
pc1_stim_ipsi = pc1[:stim_ipsi.shape[-1]].reshape(len(visstim), len(audstim), -1)
n= stim_ipsi.shape[-1]
pc1_stim_contra = pc1[n:n+stim_contra.shape[-1]].reshape(len(visstim), len(audstim), -1)
n += stim_contra.shape[-1]
pc1_move_ipsi = pc1[n:n+move_ipsi.shape[-1]].reshape(len(visstim), len(audstim), -1)
n += move_ipsi.shape[-1]
pc1_move_contra = pc1[n:n+move_contra.shape[-1]].reshape(len(visstim), len(audstim), -1)


#%%

n_vis = len(visstim)
n_aud = len(audstim)

fig,axs = plt.subplots(n_aud, n_vis*2, figsize=(n_vis*1, n_aud*1),dpi=150,sharex=False,sharey=True,
                       gridspec_kw={'width_ratios':[t_idx_stim.sum(),t_idx_move.sum()]*n_vis})

# Adjust the spacing between subplots
fig.subplots_adjust(wspace=0.01, hspace=0.4)

choice_colors = ['green','magenta']

for i,aud in enumerate(audstim):
    for j,vis in enumerate(visstim):
        stim_ax=axs[n_aud-i-1,j*2]
        move_ax=axs[n_aud-i-1,j*2+1]

        stim_ax.plot(tscale_t,pc1_stim_ipsi[j,i,:],color='green')
        stim_ax.plot(tscale_t,pc1_stim_contra[j,i,:],color='magenta')
        
        move_ax.plot(tscale_t_move,pc1_move_ipsi[j,i,:],color='green')
        move_ax.plot(tscale_t_move,pc1_move_contra[j,i,:],color='magenta')
        stim_ax.axvline(0,color='k',linestyle=':',linewidth=0.5)
        move_ax.axvline(0,color='k',linestyle=':',linewidth=0.5)

        off_axes(stim_ax)
        off_axes(move_ax)


#%%
n_aud = len(audstim)
n_vis = len(visstim)

fig,axs = plt.subplots(n_aud, n_vis*2, figsize=(n_vis*2, n_aud*2),dpi=150)

choice = 'ipsi'
move_extent = [tscale_move[t_idx_move][0], tscale_move[t_idx_move][-1], 0, len(nrn_idxs)]
stim_extent = [tscale_stim[t_idx_stim][0], tscale_stim[t_idx_stim][-1], 0, len(nrn_idxs)]
for i,aud in enumerate(audstim):
    for j,vis in enumerate(visstim):
        
        stim_ax=axs[n_aud-i-1,j*2]
        resp = mean_stim[(vis,aud,'contra')][psth_idx_stim] - mean_stim[(vis,aud,'ipsi')][psth_idx_stim] #-blank_psth
        stim_ax.matshow(resp[isort,:],aspect='auto',cmap='coolwarm',vmin=-4,vmax=4,extent=stim_extent)
        stim_ax.axvline(0,color='k',linestyle=':',linewidth=0.5)        
        off_axes(stim_ax)

        move_ax=axs[n_aud-i-1,j*2+1]
        resp = mean_move[(vis,aud,'contra')][psth_idx_move] - mean_move[(vis,aud,'ipsi')][psth_idx_move]
        move_ax.matshow(resp[isort,:],aspect='auto',cmap='coolwarm',vmin=-4,vmax=4,extent=move_extent)
        move_ax.axvline(0,color='k',linestyle=':',linewidth=0.5)
        off_axes(move_ax)
#%%

