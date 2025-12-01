#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


from NeuroLogit.src.ephys.encoding_kernel.results_helpers import read_all_results  
from floras_helpers.plotting import off_axes

SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Passive')

plt.rcParams.update({'font.size': 6,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})

#%% load results

trig_kws = {
    'vis_conds': [-.25,0,.25],
    'aud_conds': ['no_sound',-1,0,1],
    'choice_conds': ['passive','contra','ipsi','NoGo']
} 

results = read_all_results(model_type='passive_active_Ridge100', sessions='unique',
                            trig_kws=trig_kws,VE_threshold = 0.005)

#
# at this point this takes like 5 mins to load

#%% select the SC neurons
clusters = results['clusters']

sel_clusters = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCs']))# &
].copy()

SC_neuron_inds = sel_clusters.index.values

n_neurons = len(SC_neuron_inds)

 #%% sort by rastermap
kernels = results['kernels']
feature_column_dict = results['feature_column_dict']
from rastermap import Rastermap
# SCm visual neurons 10PC 10 clusters 1.0 locality 5 time lag
temp_features = [k for k in feature_column_dict.keys() if ((k!='b'))]

all_columns_except_b = np.concatenate([v for k,v in feature_column_dict.items() if ((k!='b'))])

model = Rastermap(n_PCs=10, n_clusters=10, 
                  locality=1.0, time_lag_window=5).fit(kernels[np.ix_(SC_neuron_inds,all_columns_except_b)])
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

#%% plot all kernels rastermap sorted

kernels = results['kernel_dict']
plotted_kernels = ['vis_contra_0.25','vis_contra_0.5','vis_contra_1.0',
                   'aud_onset','aud_ipsi','aud_contra',
                   'choice_ipsi','choice_contra','task']

kernel_names = ['Visual \n 0.25','Visual \n 0.5','Visual \n 1.0',
                'Auditory \n onset','Auditory \n ipsi','Auditory \n contra',
                'Action \n ipsi','Action \n contra','Task']
# get the sizes of each kernels tscale


tscale_sizes = {k: len(kernels[k]['tscale']) for k in plotted_kernels}
#plotted_kernels = ['vis_contra_0.5','vis_contra_1.0','aud_onset','aud_ipsi','aud_contra']
n_kernels = len(plotted_kernels)
fig,axs = plt.subplots(1,n_kernels,figsize=(n_kernels/2,1),dpi=150,sharex=False,sharey=True,gridspec_kw={'width_ratios':list(tscale_sizes.values())})
#fig.subplots_adjust(wspace=-20, hspace=-20.5)

for i,k in enumerate(plotted_kernels):
    ax = axs[i]
    coeffs = kernels[k]['coefficients'][SC_neuron_inds[isort],:]
    k_tscale =kernels[k]['tscale']

        
    extent = [k_tscale[0], k_tscale[-1], n_neurons, 0]
    ax.imshow(coeffs, aspect='auto', cmap='bwr', vmin=-1, vmax=1,extent=extent)
    #ax.set_xlim([0,0.6])
    
    #ax.axvline(0, color='k', lw=0.5)
    ax.set_title(kernel_names[i], fontsize=6)
    ax.set_ylim([0,n_neurons])
    off_axes(ax)

    if (k=='task') or ('choice' in k):
        ax.axvline(0, color='k', lw=0.5)
    
    ax.invert_yaxis()
    
plt.tight_layout()
fig.savefig(SAVE_PATH / 'all_kernels_rastermap_sorted.svg', dpi=300)



 #%% plot all responses, rastermap sorted on passive and correct


#  plot the acutal and the predicted means and then the indivisual neurons

example_neurons = [218,259,369,327]


audstim = [1,0,-1]
visstim = trig_kws['vis_conds']

time_period = (-0.1,.45) # time period to average over
t_idx = (results['response_tscale'] >= time_period[0]) & (results['response_tscale'] < time_period[1])
tscale_t = results['response_tscale'][t_idx]
psth_idx = np.ix_(SC_neuron_inds[isort],t_idx)

n_aud = len(audstim)
n_vis = len(visstim)
n_neurons = len(SC_neuron_inds)

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
        if key not in results['predicted_mean'].keys():
            ax.axis('off')
        else:
            resp = results['actual_mean'][key][psth_idx]#-blank_psth
            predicted = results['predicted_mean'][key][psth_idx]
            residual = resp - predicted
            print(key,resp.shape)
            ax.imshow(resp,aspect='auto',cmap='bwr',vmin=-2,vmax=2,extent=extent)

            ax.set_ylim([-10,n_neurons])
        
        # for border in cluster_borders:
        #     ax.axhline(border, color='k', lw=0.5)
        off_axes(ax)
        ax.invert_yaxis()
    
        if (i==0) & (j==n_vis-1):
            ax.plot([0.3,0.4],[-5,-5],color='k',linewidth=2)

# Add a colorbar to the figure
cbar_ax = fig.add_axes([0.93, 0.4, 0.01, 0.2])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=-2, vmax=2)
sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Response (z-score)', fontsize=8)



fig.savefig(SAVE_PATH / f'active_responses_rastermap_sorted_{resptype}.svg', dpi=300)

# %% location of units  and spatial distribution across layers
region_counts = sel_clusters.groupby('SC_pos').size()

d_strings = ['choice','task']

n_plots = len(d_strings)
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*.5, 1.2),sharey=True,sharex=True,dpi=150)
fig.subplots_adjust(wspace=0.7)
for ax,d_string in zip(axs.flatten(),d_strings):    
    #counts = sel_clusters.groupby('SC_pos').apply(lambda x: (x[f'VE_{d_string}']).sum())
    counts = sel_clusters[sel_clusters[f'significant_{d_string}']].groupby('SC_pos').size()
    frac = (counts / region_counts).fillna(0)*100

    # Plot the fractions
    frac.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_xlabel('% neurons')
    ax.axvline(5, color='k', linestyle=':',alpha=0.3,linewidth=0.5)
    off_axes(ax, which='top')
    ax.set_title(d_string)
plt.tight_layout()
fig.savefig(SAVE_PATH / 'active_fraction_significant_neurons_per_SCpos.svg', dpi=300)


#%% anatomy plot 

from floras_helpers.anat_plots import anatomy_plotter

d_strings = ['choice','task']
n_plots = len(d_strings)

# anatomical plot of all neurons from a brain region and significant neurons
coord = 850

anat = anatomy_plotter()
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*1, 1.3),dpi=300,sharex=True,sharey=True)
fig.subplots_adjust(wspace=0.1, hspace=0.1)
sel_nrns = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCm','SCs']))# &
]

for i,type in enumerate(d_strings):

    ax =axs[i]
    anat.plot_anat_canvas(ax=ax,coord = coord, axis='ml')

    anat.plot_points(sel_nrns['ap_gauss'],sel_nrns['dv'],unilateral=True,c = 'gray',
                        alpha=0.1,marker = '.',s=5,edgecolor=None)

    significants = sel_nrns[sel_nrns[f'significant_{type}']].copy()

    anat.plot_points(significants['ap_gauss'],significants['dv'],unilateral=True, 
                     c=significants[f'amp_{type}'],
                        alpha=0.85,marker = '.',
                        s=np.sqrt(significants[f'VE_{type}'])*100,edgecolor='k',
                        linewidth=0.25,cmap = 'coolwarm',vmin=-1,vmax=1)
    ax.set_title(type)

    

# ax.set_xlim([-2200, 0])
ax.set_xlim([-2700, -4900])
ax.set_ylim([-3000, -500])
#off_axes(ax)

ax.set_xticks([-3000,-4000])
ax.set_xticklabels(['-3 mm','-4 mm'])

ax.set_yticks([-1000,-2000])
ax.set_yticklabels(['-1 mm','-2 mm'])

plt.tight_layout()
fig.savefig(SAVE_PATH / 'active_significant_neuron_locations.svg', dpi=300)


# %%s catters of the units 
# percentage of neurons with siginificant kernels
# look at variance explained per kernel

region = 'SCm'

nrns_in_regin = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin([region]))# &
]

on_x = ['vis_spatial','aud_spatial','aud_onset']
y = 'choice'

metric = 'VE'
n_plots = len(on_x)
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*1,1),dpi=150,sharex=True,sharey=True)
fig.subplots_adjust(wspace=0.2, hspace=0.1)
for i,x in enumerate(on_x):
    if n_plots==1:
        ax = axs
    else:
        ax = axs[i]
    sns.scatterplot(data=nrns_in_regin, x=f'{metric}_{x}', y=f'{metric}_{y}',s=5,color='skyblue',
                    edgecolor='k',linewidth=0.3,alpha=0.5, ax=ax)

    # sns.histplot(data=sel_nrns, x='VE_vis_spatial', y='VE_aud_spatial',bins=np.arange(0,1,0.05),cmap='viridis',ax=ax)

    #sns.kdeplot(data=sel_nrns, x='VE_vis_spatial', y='VE_aud_spatial', fill=True, thresh=.1, levels=10, cmap="Blues",ax=ax)
# 45 deg line


    # Pearson correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(nrns_in_regin[f'{metric}_{x}'], nrns_in_regin[f'{metric}_{y}'])
    ax.text(0.6, 0.95, f'r={r:.2f}', transform=ax.transAxes, fontsize=6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))

    ax.set_ylim([-0.01,1])
    ax.set_xlim([-0.01,1])

    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_xticks([0,0.01,0.1,1])
    ax.set_yticks([0,0.01,0.1,1])

    ax.set_xticklabels(['0','.01','0.1','1'])
    ax.set_yticklabels(['0','.01','0.1','1'])

    if i==0:
        ax.set_ylabel(y)
    else:
        ax.set_ylabel('')
    
    ax.set_xlabel(x)



off_axes(ax,which='top')

plt.tight_layout()
fig.savefig(SAVE_PATH / f'active_VE_scatter_{region}_{y}_vs_{on_x[0]}.svg', dpi=300)
# %% example neurons

example_neurons = np.arange(3,n_neurons,1)


for respid_to_plot in example_neurons:

    fig, axs = plt.subplots(n_aud,n_vis*3,sharex=True,sharey=True,figsize=(10,1.7))
    fig.subplots_adjust(wspace=0.05,hspace=0.0)
    fig.patch.set_alpha(0.0)

    choice_colors = {'contra':"#c760ac",'ipsi':"#3d7d54",'passive':"#1c1c1c"}
    for i,aud in enumerate(audstim):
        for j,vis in enumerate(visstim):
            for k,resptype in enumerate(['passive','ipsi','contra']):
                ax=axs[n_aud-i-1,j+k*n_vis]
                resp = results['actual_mean'][(vis,aud,resptype)][psth_idx]
                resp_sem = results['actual_sem'][(vis,aud,resptype)][psth_idx]
                resp_pred = results['predicted_mean'][(vis,aud,resptype)][psth_idx]
                pred_trace = resp_pred[respid_to_plot]

                trace = resp[respid_to_plot]
                trace_sem = resp_sem[respid_to_plot]       
                ax.fill_between(tscale_t, trace - trace_sem, trace + trace_sem, 
                                color=choice_colors[resptype], alpha=0.4)
                ax.plot(tscale_t,pred_trace, color=choice_colors[resptype],linewidth = 1)
                
                off_axes(ax)
                    
                if (i==0) and (j==n_vis-1):
                    ax.plot([0.3,0.4],[-0.5,-0.5],color='k',linewidth=1)
                    
                if (i==1) and (j==n_vis-1):
                    ax.plot([.4,.4],[1,2],color='k',linestyle='-',linewidth=1)


   
    #fig.suptitle(f'{respid_to_plot}', fontsize=8)

# %%


