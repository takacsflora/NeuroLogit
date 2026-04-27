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



SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Active')

plt.rcParams.update({'font.size': 6,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})
#%% load the data
trig_kws = {'vis_conds':  [-1, -0.5,-0.25,0,0.25,0.5,1], 
            'aud_conds': [-1,0,1], 
            'choice_conds': ['left','right','passive','NoGo'],
            'sessions': 'unique',
            'dat_type':'zscored',
            'swap_to_ipsi_contra': True,
            'odds_conds': [-1,-0.5,0,0.5,1]
            }

mean,sem,tscale = ph.read_all_psths(triggered_at='stim_stimCategory', **trig_kws)


results = read_all_results(model_type='passive_active_Ridge10', sessions=trig_kws['sessions'],
                            trig_kws=None,VE_threshold = 0.005)
# %%
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

brain_region = 'SCs'
def filter_clusters(cluster_df):
    # filter clusters to good neurons with model fit and return indices
    sel_nrns = cluster_df[
        (cluster_df.bombcell_class=='good') &
        (cluster_df.BerylAcronym.isin([brain_region])) & #& 
       (cluster_df.r2_tot>10e-5)# & # any neuron where model has predicitve power
       # (cluster_df.subject.isin([subject])) #
        #(cluster_df['VE_choice']>0.005)

        ]
    return sel_nrns.index.values


kernel_fitted_sidx = filter_clusters(kernel_fitted_clusters)
clusters_sidx = filter_clusters(clusters)


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
kernels = results['kernel_dict']
plotted_kernels = ['vis_contra_0.25','vis_contra_0.5','vis_contra_1.0',
                   'aud_onset','aud_ipsi','aud_contra',
                   'choice_ipsi','choice_contra','task']

kernel_names = ['Visual \n low','Visual \n medium','Visual \n high',
                'Auditory \n onset','Auditory \n ipsi','Auditory \n contra',
                'Action \n ipsi','Action \n contra','Task']
# get the sizes of each kernels tscale


tscale_sizes = {k: len(kernels[k]['tscale']) for k in plotted_kernels}
#plotted_kernels = ['vis_contra_0.5','vis_contra_1.0','aud_onset','aud_ipsi','aud_contra']
n_kernels = len(plotted_kernels)
n_neurons = len(kernel_fitted_sidx)
fig,axs = plt.subplots(1,n_kernels,figsize=(n_kernels/2,1),dpi=150,sharex=False,sharey=True,gridspec_kw={'width_ratios':list(tscale_sizes.values())})
#fig.subplots_adjust(wspace=-20, hspace=-20.5)

for i,k in enumerate(plotted_kernels):
    ax = axs[i]
    coeffs = kernels[k]['coefficients'][kernel_fitted_sidx[isort],:]
    k_tscale =kernels[k]['tscale']

        
    extent = [k_tscale[0], k_tscale[-1], n_neurons, 0]
    ax.imshow(coeffs, aspect='auto', cmap='bwr', vmin=-2, vmax=2,extent=extent)
    #ax.set_xlim([0,0.6])
    
    #ax.axvline(0, color='k', lw=0.5)
    ax.set_title(kernel_names[i], fontsize=6)
    ax.set_ylim([0,n_neurons])
    off_axes(ax)

    if (k=='task') or ('choice' in k):
        ax.axvline(0, color='k', lw=0.5)
    
    ax.invert_yaxis()
    
plt.tight_layout()
fig.savefig(SAVE_PATH / f'all_kernels_rastermap_sorted_{brain_region}.svg', dpi=300)


#%%




audstim = [1,0,-1]
visstim = [-0.5,0,0.5]

time_period = (-0.1,.45) # time period to average over
t_idx = (tscale >= time_period[0]) & (tscale < time_period[1])
tscale_t = tscale[t_idx]
psth_idx = np.ix_(clusters_sidx[isort],t_idx)

n_aud = len(audstim)
n_vis = len(visstim)
n_neurons = len(clusters_sidx)

from scipy.ndimage import gaussian_filter1d

fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(3,3))
fig.subplots_adjust(wspace=0.1,hspace=0.2)
# Set transparent background for the figure
fig.patch.set_alpha(0.0)
resptype = 'contra'
extent = [tscale_t[0], tscale_t[-1],  n_neurons, 0]
for i,aud in enumerate(audstim):
    for j,vis in enumerate(visstim):
        ax=axs[n_aud-i-1,j]
        key = (vis,aud,resptype)

        resp = mean[key][psth_idx]#-blank_psth
        
        # otherwise the imshow interpolation is a bit annoying 
        #resp = np.nan_to_num(resp, nan=0)
        resp = gaussian_filter1d(resp, sigma=2, axis=1)
        resp = gaussian_filter1d(resp, sigma=1, axis=0)
        ax.matshow(resp,aspect='auto',cmap='bwr',vmin=-2,vmax=2,
                  extent=extent,interpolation='none')

        ax.set_ylim([0,n_neurons+10])
        
        # for border in cluster_borders:
        #     ax.axhline(border, color='k', lw=0.5)
        off_axes(ax)
        ax.invert_yaxis()
    
        if (i==0) & (j==n_vis-1):
            ax.plot([0.3,0.4],[n_neurons+5,n_neurons+5],color='k',linewidth=3)

# Add a colorbar to the figure
cbar_ax = fig.add_axes([0.93, 0.4, 0.01, 0.2])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=-3, vmax=3)
sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Response (z-score)', fontsize=8)
# Save the figure as an SVG file
fig.savefig(SAVE_PATH / f'responses_{brain_region}_{resptype}.svg', format='svg', bbox_inches='tight')

# %%
