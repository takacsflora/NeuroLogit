# supplementary Figure to visualize the feature matrix used for kernel fitting
#%%
from pathlib import Path
from turtle import rt
import numpy as np
import matplotlib.pyplot as plt
from NeuroLogit.src.ephys.dat_utils import load_trial_data,smooth_raster
from NeuroLogit.src.ephys.encoding_kernel.feature_matrix import construct_feature_matrix
from floras_helpers.plotting import off_axes

import NeuroLogit.src.ephys.encoding_kernel.batch_proc as bp
# code to fit a single session atm
SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Active')

plt.rcParams.update({'font.size': 7,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})

#%% load data and get the feature matrx
subject,date = 'AV030','2022-12-07'
ev,clusters,rasters_stim = load_trial_data(subject,date,
                            load_clusters=True,load_raster='prestim',include_no_sound_trials=True).values()

# smooth data
tscale = rasters_stim['tscale']

# # some time could be saved here by smoothing only valid trials and clusters. especauuly that we don't do bl subrraction here..
# # maybe later.... or save it...
r = smooth_raster(rasters_stim['data_binned'],tscale,smoothing=0.025,
                kernel_dir='forward',baseline_subtract=None,zscore=False) 
# smooth the raster

# Replace hemisphere values in clusters
clusters['hemi'] = clusters['hemi'].replace({-1: 'right', 1: 'left'})
unique_hemispheres = clusters['hemi'].dropna().unique() #

# remove nan from unique hemisphere
model_ID = 'passive_active_Ridge10'

valid_trial_ID,kernel_kws,preproc_kws,fitting_kws = bp.get_model_param_combos(model_type=model_ID)
is_valid_trial = bp.get_valid_trials(ev, valid_trial_ID = valid_trial_ID)

ev_p,clusters_p,r_p,tscale_p = bp.filter_data(ev,clusters,r,tscale,is_valid_trial,
                                        hemisphere='right',**preproc_kws)
feature_matrix,feature_column_dict,feature_tscale_dict = construct_feature_matrix(ev_p,tscale_p,**kernel_kws)

#m,clusters_fitted = bp.fit_evaluate(ev_p,clusters_p,r_p,feature_matrix,feature_column_dict,**fitting_kws)
# %% plot a random selection of trianls wit the feature matrix

np.random.seed(20)
trial_inds = np.random.choice(ev_p.index.values,size=10,replace=False)

feature_borders =[f[-1] for f in feature_column_dict.values()]
feature_border_mids = [f[-1] - (f[-1]-f[0])/2 for f in feature_column_dict.values()]

n_plots = len(trial_inds)
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*.5,1.5),dpi=150)
fig.subplots_adjust(wspace=0.05, hspace=0.001)

rts = ev_p.iloc[trial_inds].rt.values
rt_inds = [np.argmin(np.abs(tscale_p - rt)) for rt in rts]
stim_onset_ind = np.argmin(np.abs(tscale_p - 0))


rename_of_features = {
    'b':'baseline',
    'engagement':'engagement',
    'choice_ipsi':'action ipsi',
    'choice_contra':'action contra',
    'action_onset':'task',
    'vis_contra_1.0':'Visual high',
    'vis_contra_0.5':'Visual medium',
    'vis_contra_0.25':'Visual low',
    'aud_onset':'Auditory onset',
    'aud_contra':'Auditory Contra',
    'aud_ipsi':'Auditory Ipsi',
}

for i,trial_idx in enumerate(trial_inds):
    ax = axs[i]
    im = ax.imshow(feature_matrix[trial_idx,:,:],aspect='auto',cmap='Greys',vmin=0.01,vmax=0.02)

    for b in feature_borders:
        ax.axhline(b,color='k',linestyle=':',linewidth=0.5)

    if rt_inds[i]>0:
        ax.axvline(rt_inds[i],color='r',linestyle='--',linewidth=0.5)
    ax.axvline(stim_onset_ind,color='b',linestyle='--',linewidth=0.5)

    if i==0:
        for j, feature_name in enumerate(feature_column_dict.keys()):
            feature_plotted_name = rename_of_features.get(feature_name, feature_name)
            if feature_plotted_name=='task':
                ax.text(-1, feature_borders[j]+2, 
                    feature_plotted_name, va='center', ha='right', fontsize=7, rotation=0)
            elif feature_plotted_name=='engagement':
                ax.text(-1, feature_border_mids[j]+5, 
                        feature_plotted_name, va='center', ha='right', fontsize=7, rotation=0)
            else: 
                ax.text(-1, feature_border_mids[j], 
                        feature_plotted_name, va='center', ha='right', fontsize=7, rotation=0)
                
    ax.set_title(f'Trial {trial_idx}',fontsize=6)
    off_axes(ax)


plt.savefig(SAVE_PATH / f'feature_matrix_example_{subject}_{date}.svg',dpi=300,bbox_inches='tight')
# %%
