#%%

import numpy as np
import matplotlib.pyplot as plt

import NeuroLogit.src.ephys.psth_visualise.psth_helpers as ph
from NeuroLogit.src.ephys.cccp.results_helpers import add_ipsi_contra

from floras_helpers.plotting import off_axes

trig_kws = {'vis_conds':  [-1, -0.5,-0.25,0,0.25,0.5,1], 
            'aud_conds': ['no_sound',-1,0,1], 
            'choice_conds': ['left','right','nogo','passive'],
            'sessions': 'unique',
            'dat_type':'zscored',
            'swap_to_ipsi_contra': True,
            'odds_conds': [-1,-0.5,0,0.5,1]
            }



mean,sem,tscale = ph.read_all_psths(triggered_at='stim_stimCategory', **trig_kws)
clusters = ph.hP.read_files(filestub='clusters',extension='csv',sessions=trig_kws['sessions'])
clusters = add_ipsi_contra(clusters)

#%% calculate the responses on correct trials only
audstim = [-1,0,1]
visstim = trig_kws['vis_conds']
# get the means on correct trials 
for i,aud in enumerate(audstim):
    for j,vis in enumerate(visstim):

        if (np.sign(vis)==-1) & (np.sign(aud)<=0):
            
            which_choice_cond = 'ipsi'
        elif (np.sign(vis)==1) & (np.sign(aud)>=0):
            which_choice_cond = 'contra'
        elif (vis==0) & (aud==1):
            which_choice_cond = 'contra'
        elif (vis==0) & (aud==-1):
            which_choice_cond = 'ipsi'
        else:
            which_choice_cond = 'ipsi'   

        print(f'vis: {vis}, aud: {aud}, choice: {which_choice_cond}') 

        resp = mean[(vis,aud,which_choice_cond)]

        mean[(vis,aud,'correct')] = resp


#%%
all_active_resps = np.concatenate([mean[(v,a,'correct')] 
                                     for v in visstim for a in audstim],axis=1)



sel_nrn = clusters[
    (clusters.BerylAcronym.isin(['SCm'])) & 
    (clusters.bombcell_class=='good') & 
     ((clusters[f'is_vis_stim_Go']==True) &
      (clusters[f'is_aud_stim_Go']==False ))
].copy()

nrn_idxs = sel_nrn.index.values # all indices

time_period = (-0.01,.35) # time period to average over
t_idx = (tscale >= time_period[0]) & (tscale < time_period[1])
tscale_t = tscale[t_idx]
psth_idx = np.ix_(nrn_idxs,t_idx)

#%%
from rastermap import Rastermap

correct_mean_resps = np.concatenate([mean[(v,a,'correct')][psth_idx] 
                                     for v in visstim for a in audstim],axis=1)

passive_mean_resps = np.concatenate([mean[(v,a,'passive')][psth_idx] 
                                     for v in visstim for a in audstim],axis=1)


correct_and_passive_resps = np.concatenate([correct_mean_resps,passive_mean_resps],axis=1)

model = Rastermap(n_PCs=10, n_clusters=2, 
                  locality=1.0, time_lag_window=5).fit(correct_and_passive_resps)
y = model.embedding # neurons x 1
isort = model.isort

# visualize binning over neurons
X_embedding = model.X_embedding
fig,ax = plt.subplots(1,1,figsize=(10,2),dpi=150)
ax.imshow(X_embedding, vmin=-2, vmax=2, cmap="coolwarm", aspect="auto")

# %%


audstim = [-1,0,1]
visstim = trig_kws['vis_conds']


n_aud = len(audstim)
n_vis = len(visstim)

fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(10,5))
fig.subplots_adjust(wspace=0.1,hspace=0.1)


choice_cond = 'correct'
blank_psth = mean[(0,0,choice_cond)][psth_idx]


for i,aud in enumerate(audstim):
    for j,vis in enumerate(visstim):
        ax=axs[n_aud-i-1,j]

        resp = mean[(vis,aud,choice_cond)][psth_idx]-blank_psth
        ax.matshow(resp[:,:],aspect='auto',cmap='coolwarm',vmin=-4,vmax=4)
        
        off_axes(ax)


# %%
