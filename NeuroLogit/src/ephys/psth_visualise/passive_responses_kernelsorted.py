#%%

# another attempt to visualise the rseponses
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from NeuroLogit.src.ephys.results_utils import read_files 
import NeuroLogit.src.ephys.psth_visualise.psth_helpers as ph
from floras_helpers.plotting import off_axes

SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Passive')

plt.rcParams.update({'font.size': 10,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})

#%% load the data 

trig_kws = {'vis_conds':  [-1,-0.5,-0.25,0,0.25,0.5,1], 
            'aud_conds': ['no_sound',-1,0,1], 
            'choice_conds': ['passive'],
            'sessions': 'unique',
            'dat_type':'zscored',
            'swap_to_ipsi_contra': True,
            'odds_conds': [-1,-0.5,0,0.5,1]
            }


# load kernel analysis and sort neurons
model_type = 'passive_active_Ridge100'

# note -- we don't fit all the neurons with the kernels so this will not be the same length as the psth clusters
kernels = read_files(which_result = 'encoding_kernel_results', filestub=f'model_{model_type}', extension='npz', npz_dat_type='coefficients', sessions=trig_kws['sessions'])
kernel_fitted_clusters = read_files(which_result = 'encoding_kernel_results', filestub=f'clusters_{model_type}', extension='csv', sessions=trig_kws['sessions'])
feature_column_dict = read_files(which_result = 'encoding_kernel_results', filestub = f'model_{model_type}',extension='npz', npz_dat_type='feature_column_dict', sessions=trig_kws['sessions']).item()
feature_tscale_dict = read_files(which_result = 'encoding_kernel_results', filestub = f'model_{model_type}',extension='npz', npz_dat_type='feature_tscale_dict', sessions=trig_kws['sessions']).item()

# get raw data psths

mean,sem,tscale = ph.read_all_psths(triggered_at='stim_stimCategory', **trig_kws)
#  raw daata corresponding clusterIDs


# making sure we can cross-reference between raw data clusterIDs and kernel-fitted clusterIDs

clusters = ph.hP.read_files(filestub='clusters',extension='csv',sessions=trig_kws['sessions'])

# shared columns
shared_cols = np.intersect1d(clusters.columns, kernel_fitted_clusters.columns)
merger_colums = ['subject','date','_av_IDs']

kernels_fitted_unique = list(np.setdiff1d(kernel_fitted_clusters.columns, clusters.columns))

clusters = pd.merge(clusters,
                           kernel_fitted_clusters[merger_colums + kernels_fitted_unique],
                           on=merger_colums,
                           how='left',
                           suffixes=('_psth', '_kernel'))
# sort neurons according to their kernels 

#%%
import seaborn as sns
fig,ax = plt.subplots(1,1,figsize=(3,3),dpi=150)
clusters['VE_choice'] = clusters[['VE_choice_ipsi','VE_choice_contra']].sum(axis=1)
clusters['VE_vis'] = clusters[['VE_vis_contra_0.5','VE_vis_contra_1.0']].sum(axis=1)
clusters['p_val_choice'] = 1- (clusters['p_choice_choice_Go']-0.5).abs()*2


sns.scatterplot(data=clusters, x='p_val_choice', y='VE_choice', alpha=0.7, s=10,hue='is_choice_choice_Go')
ax.legend(title='significant by ccCP',bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, title_fontsize=6)
ax.set_xscale('symlog', linthresh=0.01)
ax.set_yscale('symlog', linthresh=0.01)

#ax.axvline(0.5,color='k',linestyle='--',linewidth=0.5)
#%% select neurons of interest and get their indices both in the kernel matrix and in the psth matrix

def filter_clusters(cluster_df):
    # filter clusters to good neurons with model fit and return indices
    sel_nrns = cluster_df[
        (cluster_df.bombcell_class=='good') &
        (cluster_df.BerylAcronym.isin(['SCs','SCm'])) & 
        (cluster_df.r2_tot>0) #& # any neuron where model has predicitve power
       # (cluster_df['VE_vis_contra_1.0']>0.01)

        ]
    return sel_nrns.index.values


kernel_fitted_sidx = filter_clusters(kernel_fitted_clusters)
clusters_sidx = filter_clusters(clusters)


assert kernel_fitted_sidx.size == clusters_sidx.size, "Mismatch in number of selected neurons between kernel fitted clusters and psth clusters"




#%%  sort the kernels with rastermap
from rastermap import Rastermap
# SCm visual neurons 10PC 10 clusters 1.0 locality 5 time lag
temp_features = [k for k in feature_column_dict.keys() if ((k!='b'))]

all_columns_except_b = np.concatenate([v for k,v in feature_column_dict.items() if ((k!='b'))])

model = Rastermap(n_PCs=10, n_clusters=5, 
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


# cluster_labels = model.embedding_clust[isort]
# # pick one unit per cluster label
# unique_clusters = np.unique(cluster_labels)
# cluster_borders = []
# example_neurons = []
# for c in unique_clusters:
#     cluster_indices = np.where(cluster_labels == c)[0]
#     if len(cluster_indices) > 0:
#         cluster_borders.append(cluster_indices[0])
#         example_neurons.append(cluster_indices[1])

#%% get the power content for each neuron in each stim condition 

example_neurons = [48,211,323,367]


audstim = trig_kws['aud_conds']
visstim = trig_kws['vis_conds']

#

time_period = (-0.1,.7) # time period to average over
t_idx = (tscale >= time_period[0]) & (tscale < time_period[1])
tscale_t = tscale[t_idx]
psth_idx = np.ix_(clusters_sidx,t_idx)

n_aud = len(audstim)
n_vis = len(visstim)

fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(5,2))
fig.subplots_adjust(wspace=0.0,hspace=0.0)
# Set transparent background for the figure
fig.patch.set_alpha(0.0)
blank_psth = mean[(0,0,'passive')][psth_idx]


extent = [tscale_t[0], tscale_t[-1], 0, len(clusters_sidx)]
for i,aud in enumerate(audstim):
    for j,vis in enumerate(visstim):
        ax=axs[n_aud-i-1,j]
        resp = mean[(vis,aud,'passive')][psth_idx]#-blank_psth
        ax.matshow(resp,aspect='auto',cmap='coolwarm',vmin=-3,vmax=3,extent=extent)
        if j==n_vis-1: 
            for respid_to_plot in example_neurons:
                ax.annotate('←', xy=(0.75,respid_to_plot), 
                            xycoords='data', color='black', fontsize=6,
                              ha='center', va='center')
            ax.set_xlim([0,0.8])
            ax.set_ylim([-35,len(clusters_sidx)])
        # for border in cluster_borders:
        #     ax.axhline(border, color='k', lw=0.5)
        off_axes(ax)
    
        if (i==0) & (j==n_vis-1):
            ax.plot([0.4,0.6],[-25,-25],color='k',linewidth=2)

# Add a colorbar to the figure
cbar_ax = fig.add_axes([0.95, 0.4, 0.01, 0.2])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=-2, vmax=2)
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Response (z-score)', fontsize=8)


fig.savefig(SAVE_PATH / 'passive_responses_all_psths.svg', dpi=300)


#%%
for respid_to_plot in example_neurons:

    fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(2.25,2))
    fig.subplots_adjust(wspace=0.05,hspace=0.0)
    fig.patch.set_alpha(0.0)

    aud_colors = ['k','blue','grey','red']
    for i,aud in enumerate(audstim):
        for j,vis in enumerate(visstim):
            ax=axs[n_aud-i-1,j]
            resp = mean[(vis,aud,'passive')][psth_idx]
            resp_sem = sem[(vis,aud,'passive')][psth_idx]

            trace = resp[respid_to_plot]
            trace_sem = resp_sem[respid_to_plot]       
            ax.fill_between(tscale_t, trace - trace_sem, trace + trace_sem, 
                            color=aud_colors[i], alpha=0.3)
            ax.plot(tscale_t,trace, color=aud_colors[i],linewidth = .5)
            
            off_axes(ax)
    #fig.suptitle(f'Neuron {respid_to_plot}')
    fig.savefig(SAVE_PATH / f'passive_responses_nrn{respid_to_plot}.svg', dpi=300)


# %%
