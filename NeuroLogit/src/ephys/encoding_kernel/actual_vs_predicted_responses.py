#%%



import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


from NeuroLogit.src.ephys.results_utils import read_files 
from floras_helpers.plotting import off_axes

SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Passive')

plt.rcParams.update({'font.size': 6,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})
#%% load all the data




#%%



# %% slect all the good neurons and sort them with rastermap

sel_nrns = clusters[
    (clusters.bombcell_class=='good') &
   # (clusters.BerylAcronym.isin(['SCs','SCm'])) & 
   (clusters.SC_pos.isin(['SCig'])) &
    (clusters.r2_tot>0) #& # any neuron where model has predicitve power
    # (clusters['VE_vis_contra_1.0']>0.01)
    ]
goodclus_idx  = sel_nrns.index.values

#   sort the kernels with rastermap
from rastermap import Rastermap
# SCm visual neurons 10PC 10 clusters 1.0 locality 5 time lag
temp_features = [k for k in feature_column_dict.keys() if ((k!='b'))]

all_columns_except_b = np.concatenate([v for k,v in feature_column_dict.items() if ((k!='b'))])

model = Rastermap(n_PCs=15, n_clusters=7, 
                  locality=1.0, time_lag_window=5).fit(kernels[np.ix_(goodclus_idx,all_columns_except_b)])
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
# sort by non-kernel related things  


#isort = np.argsort(sel_nrns['ap'].values)



#%%

# show all the kernels
plotted_kernels = ['vis_contra_0.25','vis_contra_0.5','vis_contra_1.0','aud_onset','aud_ipsi','aud_contra','task','choice_ipsi','choice_contra']

#plotted_kernels = ['vis_contra_0.5','vis_contra_1.0','aud_onset','aud_ipsi','aud_contra']
n_kernels = len(plotted_kernels)
fig,axs = plt.subplots(1,n_kernels,figsize=(n_kernels,2),dpi=150)
fig.subplots_adjust(wspace=0.1, hspace=0.3)

for i,k in enumerate(plotted_kernels):
    ax = axs[i]
    
    if k=='task':        
        engagement = kernels[np.ix_(goodclus_idx[isort], feature_column_dict['engagement'])]
        action_onset = kernels[np.ix_(goodclus_idx[isort], feature_column_dict['action_onset'])]

        tscale_engagement = kernel_tscale['engagement']
        tscale_action_onset = kernel_tscale['action_onset']


        tscle_engagement_only = np.setdiff1d(tscale_engagement, tscale_action_onset)
        coef_engaement = np.tile(engagement, (1, len(tscle_engagement_only)))

        k_tscale = np.sort(np.concatenate([tscle_engagement_only,tscale_action_onset]), axis=0)
        coeffs = np.concatenate([coef_engaement,action_onset+engagement], axis=1)
    else:
        kernel_inds = feature_column_dict[k]
        coeffs = kernels[np.ix_(goodclus_idx[isort],kernel_inds)]
        k_tscale = kernel_tscale[k]

    extent = [k_tscale[0], k_tscale[-1], 0, coeffs.shape[1]]


    ax.imshow(coeffs, aspect='auto', cmap='bwr', vmin=-1.5, vmax=1.5,extent=extent)
    #ax.set_xlim([0,0.6])
    
    if ('task' in k) or ('choice' in k):
        ax.axvline(0, color='k', lw=0.5)
    ax.set_title(k)
    off_axes(ax)




# %%

#  plot the acutal and the predicted means and then the indivisual neurons

example_neurons = [218,259,369,327]


audstim = trig_kws['aud_conds']
visstim = trig_kws['vis_conds']

#

time_period = (-0.1,.7) # time period to average over
t_idx = (tscale >= time_period[0]) & (tscale < time_period[1])
tscale_t = tscale[t_idx]
psth_idx = np.ix_(goodclus_idx[isort],t_idx)

n_aud = len(audstim)
n_vis = len(visstim)

fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(4,1.75))
fig.subplots_adjust(wspace=0.0,hspace=0.0)
# Set transparent background for the figure
fig.patch.set_alpha(0.0)

extent = [tscale_t[0], tscale_t[-1], 0, len(goodclus_idx)]
for i,aud in enumerate(audstim):
    for j,vis in enumerate(visstim):
        ax=axs[n_aud-i-1,j]
        resp = actual_mean[(vis,aud,'contra')][psth_idx]#-blank_psth
        ax.imshow(resp,aspect='auto',cmap='bwr',vmin=-1.5,vmax=1.5,extent=extent)
        if j==n_vis-1: 
            for respid_to_plot in example_neurons:
                ax.annotate('←', xy=(0.75,respid_to_plot), 
                            xycoords='data', color='black', fontsize=6,
                              ha='center', va='center')
            ax.set_xlim([0,0.8])
            ax.set_ylim([-30,len(goodclus_idx)])
        # for border in cluster_borders:
        #     ax.axhline(border, color='k', lw=0.5)
        off_axes(ax)
    
        if (i==0) & (j==n_vis-1):
            ax.plot([0.4,0.6],[-25,-25],color='k',linewidth=2)

# Add a colorbar to the figure
cbar_ax = fig.add_axes([0.95, 0.4, 0.01, 0.2])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Response (z-score)', fontsize=8)

fig.savefig(SAVE_PATH / 'passive_responses_prediction_all_psths.svg', dpi=300)
# %%

resp_type = 'contra'
example_neurons = np.arange(0,sel_nrns.shape[0],1)

for respid_to_plot in example_neurons:

    fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(5,3))
    fig.subplots_adjust(wspace=0.05,hspace=0.0)
    fig.patch.set_alpha(0.0)

    aud_colors = ['k','blue','grey','red']
    for i,aud in enumerate(audstim):
        for j,vis in enumerate(visstim):
            ax=axs[n_aud-i-1,j]
            resp = actual_mean[(vis,aud,resp_type)][psth_idx]
            resp_sem = actual_sem[(vis,aud,resp_type)][psth_idx]

            resp_pred = predicted_mean[(vis,aud,resp_type)][psth_idx]
            pred_trace = resp_pred[respid_to_plot]


            resp_passive = actual_mean[(vis,aud,'ipsi')][psth_idx]
            passive_trace = resp_passive[respid_to_plot]


            trace = resp[respid_to_plot]
            trace_sem = resp_sem[respid_to_plot]       
            ax.fill_between(tscale_t, trace - trace_sem, trace + trace_sem, 
                            color=aud_colors[i], alpha=0.3)
            ax.plot(tscale_t,pred_trace, color=aud_colors[i],linewidth = 1)

            ax.plot(tscale_t,passive_trace, color=aud_colors[i],linewidth = 1,linestyle='--')

            if (i==0) and (j==n_vis-1):
                ax.plot([0.4,0.6],[-0.5,-0.5],color='k',linewidth=2)
                
            if (i==1) and (j==n_vis-1):
                ax.plot([.61,.61],[1,2],color='k',linestyle='-',linewidth=2)
            
            off_axes(ax)
    
    fig.suptitle(f'{respid_to_plot}', fontsize=8)

# %%
