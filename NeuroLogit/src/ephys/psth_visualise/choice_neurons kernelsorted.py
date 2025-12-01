#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

from NeuroLogit.src.ephys.results_utils import read_files 
import NeuroLogit.src.ephys.psth_visualise.psth_helpers as ph

from NeuroLogit.src.ephys.encoding_kernel.results_helpers import read_all_results  
from floras_helpers.plotting import off_axes


from floras_helpers.plotting import off_axes



SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Passive')

plt.rcParams.update({'font.size': 6,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})
#%% load the data
trig_kws = {'vis_conds':  [-1, -0.5,-0.25,0,0.25,0.5,1], 
            'aud_conds': [-1,0,1], 
            'choice_conds': ['left','right'],
            'sessions': 'unique',
            'dat_type':'zscored',
            'swap_to_ipsi_contra': True,
            'odds_conds': [-1,-0.5,0,0.5,1]
            }

mean_move,sem_move,tscale_move = ph.read_all_psths(triggered_at='move_stimCategory', **trig_kws)


results = read_all_results(model_type='passive_active_Ridge10', sessions=trig_kws['sessions'],
                            trig_kws=None,VE_threshold = 0.005)




#%%
kernel_fitted_clusters  = results['clusters']
clusters = ph.hP.read_files(filestub='clusters_ccCP',extension='csv',sessions=trig_kws['sessions'])


# shared columns
shared_cols = np.intersect1d(clusters.columns, kernel_fitted_clusters.columns)
merger_colums = ['subject','date','_av_IDs']

kernels_fitted_unique = list(np.setdiff1d(kernel_fitted_clusters.columns, clusters.columns))

clusters = pd.merge(clusters,
                           kernel_fitted_clusters[merger_colums + kernels_fitted_unique],
                           on=merger_colums,
                           how='left',
                           suffixes=('_psth', '_kernel'))


#%% select neurons of interest and get their indices both in the kernel matrix and in the psth matrix

def filter_clusters(cluster_df,subject='AV030'):
    # filter clusters to good neurons with model fit and return indices
    sel_nrns = cluster_df[
        (cluster_df.bombcell_class!='noise') &
        (cluster_df.BerylAcronym.isin(['MOs'])) & 
       # (cluster_df.r2_tot>0.00001) #& # any neuron where model has predicitve power
        (cluster_df.subject==subject) &
        (cluster_df['VE_choice']>0.005)

        ]
    return sel_nrns.index.values

subject  = 'AV007'

kernel_fitted_sidx = filter_clusters(kernel_fitted_clusters, subject=subject)
clusters_sidx = filter_clusters(clusters, subject=subject)


assert kernel_fitted_sidx.size == clusters_sidx.size, "Mismatch in number of selected neurons between kernel fitted clusters and psth clusters"

#%%  sort the kernels with rastermap
from rastermap import Rastermap

feature_column_dict = results['feature_column_dict']
kernels = results['kernels']
# SCm visual neurons 10PC 10 clusters 1.0 locality 5 time lag
temp_features = [k for k in feature_column_dict.keys() if ((k!='b'))]

all_columns_except_b = np.concatenate([v for k,v in feature_column_dict.items() if ((k!='b'))])

model = Rastermap(n_PCs=20, n_clusters=30, 
                  locality=1.0, time_lag_window=5).fit(kernels[np.ix_(kernel_fitted_sidx,all_columns_except_b)])
y = model.embedding # neurons x 1
isort = model.isort

# visualize binning over neurons
X_embedding = model.X_embedding


fig,ax = plt.subplots(1,1,figsize=(3,3),dpi=150)
ax.matshow(X_embedding, vmin=-4, vmax=4, cmap="coolwarm",aspect='auto')

for a in temp_features:
    last_ind = feature_column_dict[a][-1]
    ax.axvline(last_ind, color='k', lw=0.5)

    midpoint = (feature_column_dict[a][0] + feature_column_dict[a][-1]) / 2
    ax.text(midpoint, -5, a, ha='center', va='bottom', fontsize=6, rotation=90)

    off_axes(ax)

#%%
kernel_dict = results['kernel_dict']

#%%

fig,ax = plt.subplots(1,1,figsize=(1,1),dpi=150)
contra = kernel_dict['choice_contra']['coefficients'][kernel_fitted_sidx,:]
ipsi = kernel_dict['choice_ipsi']['coefficients'][kernel_fitted_sidx,:]



diff_kernel = contra - ipsi

mean_diff = np.sum(diff_kernel[:,10:16],axis=1)
isort = np.argsort(mean_diff)[::-1]

tscale_kernel = kernel_dict['choice_contra']['tscale']

extent = [tscale_kernel[0], tscale_kernel[-1], 0, len(clusters_sidx)]
ax.imshow(ipsi[isort,:], aspect='auto', cmap='bwr', vmin=-2, vmax=2, extent=extent)
ax.axvline(0,color='k',linestyle=':',linewidth=0.5)
ax.set_xlim(-0.15,0.1)

#%%
n_neurons = 15

# n_rows = int(np.ceil(np.sqrt(n_neurons)))
# n_cols = int(np.ceil(n_neurons / n_rows))
# fig,axs = plt.subplots(n_rows,n_cols,figsize=(n_cols*.5,n_rows *0.5),dpi=150,sharex=True,sharey=True)

fig,axs = plt.subplots(1,n_neurons,figsize=(n_neurons*.5,0.5),dpi=150,sharex=True,sharey=True)
fig.subplots_adjust(wspace=0.0,hspace=0.0)
for i in range(n_neurons):
    ax = axs.flatten()[i]
    myiloc = -i-1
    ax.plot(tscale_kernel,contra[isort[myiloc],:], color='magenta', alpha=0.7, linewidth=1)
    ax.plot(tscale_kernel,ipsi[isort[myiloc],:], color='green', alpha=0.7, linewidth=1)
    off_axes(ax)
    ax.set_xlim(-0.12,0.1)
    ax.axvline(0,color='k',linestyle=':',linewidth=0.5)

    ax.set_ylim([-2.5,2.5])
    ax.axhline(0,color='k',linestyle=':',linewidth=0.5)
    # Remove unused axes


#%%
from sklearn.decomposition import PCA

fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150)

tscale_kernel = kernel_dict['choice_contra']['tscale']

unique_subjects = clusters['subject'].unique()
for subject in unique_subjects:
    kernel_fitted_sidx = filter_clusters(kernel_fitted_clusters, subject=subject)
    clusters_sidx = filter_clusters(clusters, subject=subject)

    if kernel_fitted_sidx.size ==0:
        continue

    contra = kernel_dict['choice_contra']['coefficients'][kernel_fitted_sidx,:]
    ipsi = kernel_dict['choice_ipsi']['coefficients'][kernel_fitted_sidx,:]
    diff_kernel = contra - ipsi

    pca = PCA(n_components=2)
    pca.fit(np.concatenate([contra, ipsi], axis=1))
    pc1 = pca.components_[0]
    pc1_contra = pc1[:contra.shape[1]]
    pc1_ipsi = pc1[contra.shape[1]:]


    # contra_pref_neurons = np.sum(diff_kernel,axis=1)>0
    # ipsi_pref_neurons = np.sum(diff_kernel,axis=1)<=0
    # # do a pca on the choice kernels
    # n_contra = np.sum(contra_pref_neurons)
    # n_ipsi = np.sum(ipsi_pref_neurons)
    # for i in range(n_contra):
    #     ax.plot(ipsi[contra_pref_neurons][i],contra[contra_pref_neurons][i],'-',color="#F1CADD",alpha=0.7,
    #             markeredgecolor='k',linewidth = 0.5,markeredgewidth=0.5)


    # for i in range(n_ipsi):
    #     ax.plot(ipsi[ipsi_pref_neurons][i],contra[ipsi_pref_neurons][i],'-',color="#BEC6C5",alpha=0.7,
    #             markeredgecolor='k',linewidth = 0.5,markeredgewidth=0.5)


    # mean_ipsi = np.mean(ipsi[contra_pref_neurons],axis=0)
    # mean_contra = np.mean(contra[contra_pref_neurons],axis=0)

    mean_ipsi = pc1_ipsi
    mean_contra = pc1_contra
    before_choice_idx = tscale_kernel<=0.05
    before_choice_ipsi = mean_ipsi[before_choice_idx]
    before_choice_contra = mean_contra[before_choice_idx] 
    
    ax.plot(before_choice_ipsi, before_choice_contra, '->', color="#F47BB7", alpha=0.7, linewidth=1,markersize=1)

    after_choice_idx = tscale_kernel>=0
    after_choice_ipsi = mean_ipsi[after_choice_idx]
    after_choice_contra = mean_contra[after_choice_idx] 

    ax.plot(after_choice_ipsi, after_choice_contra, '->', color="#D70173", alpha=0.7, linewidth=1,markersize=1)
    #ax.scatter(mean_ipsi,mean_contra,c=(tscale_kernel<0),cmap='copper',s=5,edgecolor='k',linewidth=0.2)     
    # mean_ipsi = np.mean(ipsi[ipsi_pref_neurons],axis=0)
    # mean_contra = np.mean(contra[ipsi_pref_neurons],axis=0)
    # before_choice_ipsi = mean_ipsi[before_choice_idx]
    # before_choice_contra = mean_contra[before_choice_idx]
    # ax.plot(before_choice_ipsi, before_choice_contra, '-', color="#70DECD", alpha=0.7, linewidth=1)
    # after_choice_ipsi = mean_ipsi[after_choice_idx]
    # after_choice_contra = mean_contra[after_choice_idx]
    # ax.plot(after_choice_ipsi, after_choice_contra, '-', color="#17C0BB", alpha=0.7, linewidth=1)
    #ax.plot(mean_ipsi,mean_contra,color='green',alpha=0.7,linewidth=1)
    # # # or just the mean



ax.axhline(0,color='k',linestyle=':',linewidth=0.5)

ax.axvline(0,color='k',linestyle=':',linewidth=0.5)
ax.set_aspect('equal')
ax.axline((0,0),slope=1,color='k',linestyle=':',linewidth=0.5)

# ax.set_xlim([-1,5])
# ax.set_ylim([-1,5])


#%%


fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150)
n_neurons_to_plot = contra.shape[0]
for i in range(n_neurons_to_plot):
    plt.plot(ipsi[i],contra[i],'-',color='grey',alpha=0.7,markeredgecolor='k',linewidth = 0.5,markeredgewidth=0.5)

ax.axline((0,0),slope=1,color='k',linestyle=':',linewidth=0.5)
ax.set_aspect('equal')
ax.axhline(0,color='k',linestyle=':',linewidth=0.5)
ax.axvline(0,color='k',linestyle=':',linewidth=0.5)


ax.set_xlim([-2,8])
ax.set_ylim([-2,8])

#%%

def get_time_indices(tscale, time_period):
    return (tscale >= time_period[0]) & (tscale < time_period[1])

t_idx_move = get_time_indices(tscale_move, (-0.2,0.1))
tscale_t_move = tscale_move[t_idx_move]
psth_idx_move = np.ix_(clusters_sidx,t_idx_move)
#%%

visstim = trig_kws['vis_conds']
audstim = [1,0,-1]


#%%
n_aud = len(audstim)
n_vis = len(visstim)

fig,axs = plt.subplots(n_aud, n_vis, figsize=(n_vis*2, n_aud*2),dpi=150)

choice = 'ipsi'
move_extent = [tscale_move[t_idx_move][0], tscale_move[t_idx_move][-1], 0, len(clusters_sidx)]
for i,aud in enumerate(audstim):
    for j,vis in enumerate(visstim):
        
        # stim_ax=axs[n_aud-i-1,j]
        # resp = mean_stim[(vis,aud,'contra')][psth_idx_stim] - mean_stim[(vis,aud,'ipsi')][psth_idx_stim]
        # stim_ax.imshow(resp[isort,:],aspect='auto',cmap='coolwarm',vmin=-2,vmax=2,extent=stim_extent)
        # stim_ax.axvline(0,color='k',linestyle=':',linewidth=0.5)        
        # off_axes(stim_ax)

        move_ax=axs[n_aud-i-1,j]
        resp = mean_move[(vis,aud,'contra')][psth_idx_move] 
        move_ax.imshow(resp[isort,:],aspect='auto',cmap='bwr',vmin=-5,vmax=5,extent=move_extent)
        move_ax.axvline(0,color='k',linestyle=':',linewidth=0.5)
        off_axes(move_ax)

#%%


#



#%%
# individual neuron traces
example_neurons = np.arange(0,108,1)
for respid_to_plot in example_neurons:

    fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(2.25,1.25))
    fig.subplots_adjust(wspace=0.05,hspace=0.0)
    fig.patch.set_alpha(0.0)

    aud_colors = ['k','blue','grey','red']
    choice_colors = ['green','magenta']
    for i,aud in enumerate(audstim):
        for j,vis in enumerate(visstim):
            ax=axs[n_aud-i-1,j]
            
            for ch_i, ch_dir in enumerate(['ipsi','contra']):
                resp = mean_move[(vis,aud,ch_dir)][psth_idx_move]
                resp_sem = sem_move[(vis,aud,ch_dir)][psth_idx_move]

                trace = resp[respid_to_plot]
                trace_sem = resp_sem[respid_to_plot]

                
                
                ax.fill_between(tscale_t_move, trace - trace_sem, trace + trace_sem, 
                                color=choice_colors[ch_i], alpha=0.3)
                ax.plot(tscale_t_move,trace, color=choice_colors[ch_i],linewidth = .5)

            ax.axvline(0,color='k',linestyle=':',linewidth=0.5)

            off_axes(ax)


# %%
