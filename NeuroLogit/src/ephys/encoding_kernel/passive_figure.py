#%% imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


from NeuroLogit.src.ephys.encoding_kernel.results_helpers import read_all_results  
from floras_helpers.plotting import off_axes
from floras_helpers.anat_plots import anatomy_plotter
from scipy.stats import pearsonr,spearmanr


SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Passive')

plt.rcParams.update({'font.size': 8,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})


# SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\conferences\Cosyne2026\Poster figures')

# plt.rcParams.update({'font.size': 24,'font.family':'Calibri','axes.linewidth':1,'axes.spines.top':False,'axes.spines.right':False,
#                      'axes.spines.left':True,'axes.spines.bottom':True,
#                      'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3})


#%% load results

trig_kws = {
    'vis_conds': [-1,-0.5,-0.25,0,0.25,0.5,1],
    'aud_conds': ['no_sound',-1,0,1],
    'choice_conds': ['passive','ipsi','contra','NoGo'],
} 

results = read_all_results(model_type='passive_active_Ridge10', 
                           sessions='unique',
                            trig_kws=trig_kws,VE_threshold = 0.005)

#%% select SC neurons


clusters = results['clusters']
clusters['session'] = clusters['subject'] + '_' + clusters['date'].astype(str)

sel_clusters = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.SC_pos.isin(['SCs','SCiw','SCig','SCd'])) 
].copy()

SC_neuron_inds = sel_clusters.index.values


#%% sort neurons by SC pos and then within each SC pos sort by depth
sel_clusters['clus_index'] = np.arange(len(sel_clusters))
sel_clusters['SC_pos'] = pd.Categorical(sel_clusters['SC_pos'], categories=['SCd','SCiw','SCig','SCs'], ordered=True)
sel_clusters_sorted = sel_clusters.groupby('SC_pos').apply(lambda x: x.sort_values('dv',ascending=False)).reset_index(drop=True)

isort = sel_clusters_sorted['clus_index'].values

#%% rasters of neurons responses sorted by location

#  plot the acutal and the predicted means and then the indivisual neurons

example_neurons = []


audstim = [1,0,-1,'no_sound']
visstim = [-1, -0.5,-0.25, 0, 0.25, 0.5, 1]# trig_kws['vis_conds']

time_period = (-0.1,.46) # time period to average over
t_idx = (results['response_tscale'] >= time_period[0]) & (results['response_tscale'] < time_period[1])
tscale_t = results['response_tscale'][t_idx]
psth_idx = np.ix_(SC_neuron_inds[isort],t_idx)

n_aud = len(audstim)
n_vis = len(visstim)
n_neurons = len(SC_neuron_inds)

# main figure figsize 
#fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(4,4))

# supplement al figure figsize
#fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(8,8))
#fig.subplots_adjust(wspace=0.0,hspace=0.2)

fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(2,2))
fig.subplots_adjust(wspace=0.05,hspace=0.1)

# Set transparent background for the figure#
#fig.patch.set_alpha(0.0)

resp_type = 'passive'
extent = [tscale_t[0], tscale_t[-1],  n_neurons, 0]
for i,aud in enumerate(audstim):
    for j,vis in enumerate(visstim):
        ax=axs[n_aud-i-1,j]
        resp = results['actual_mean'][(vis,aud,resp_type)][psth_idx]#-blank_psth
        ax.imshow(resp,aspect='auto',cmap='bwr',vmin=-0.75,vmax=0.75,extent=extent)


        if j==n_vis-1: 
            for respid_to_plot in example_neurons:
                ax.annotate('←', xy=(0.65,respid_to_plot), 
                            xycoords='data', color='black', fontsize=6,
                              ha='center', va='center')
            ax.set_xlim([0,0.7])
            ax.set_ylim([-30,n_neurons])
        
        
        # add a hline at the border each SC pos 
        for border in sel_clusters_sorted.groupby('SC_pos').size().cumsum():
            if border < n_neurons:
                ax.axhline(border, color='k',linestyle=':', lw=1)

        # Add arrows at the rows of example neurons
        if j==n_vis-1:
            for respid_to_plot in example_neurons:
                ax.annotate('←', xy=(0.5, respid_to_plot), 
                            xycoords='data', color='black', fontsize=6,
                            ha='center', va='center')
        
        # for border in cluster_borders:
        #     ax.axhline(border, color='k', lw=0.5)
        off_axes(ax)
        #ax.invert_yaxis()
    
        if (i==0) & (j==n_vis-1):
            ax.plot([0.3,0.4],[-25,-25],color='k',linewidth=2)

# Add a colorbar to the figure
cbar_ax = fig.add_axes([0.93, 0.4, 0.01, 0.2])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=-0.75, vmax=0.75)
sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params()
cbar.set_label('Response (z-score)')


# Adjust the x-axis limits of the first column to accommodate the boxes
axs[0, 0].set_xlim([time_period[0], time_period[1]+0.05])


fig.savefig(SAVE_PATH / f'{resp_type}_responses_depth_sorted_reduced_conds.svg', dpi=300)
# fig.savefig(SAVE_PATH / f'{resp_type}_responses_depth_sorted_with_scales.png', dpi=400)

# %% plot the kernels
kernels = results['kernel_dict']
sensory_kernels  = ['vis_contra_0.25','vis_contra_0.5','vis_contra_1.0',
                   'aud_onset','aud_ipsi','aud_contra']
sensory_kernels = [f'{k}_baseline_padded' for k in sensory_kernels]
task_kernels = ['choice_ipsi','choice_contra','task']

plotted_kernels = sensory_kernels + task_kernels

kernel_names = ['Visual \n low','Visual \n mid','Visual \n high',
                'Auditory \n onset','Auditory \n ipsi','Auditory \n contra','Action \n ipsi','Action \n contra','Task']
# get the sizes of each kernels tscale


tscale_sizes = {k: len(kernels[k]['tscale']) for k in plotted_kernels}
#plotted_kernels = ['vis_contra_0.5','vis_contra_1.0','aud_onset','aud_ipsi','aud_contra']
n_kernels = len(plotted_kernels)
fig,axs = plt.subplots(1,n_kernels,figsize=(4,1),dpi=150,
                       sharex=False,sharey=True,gridspec_kw={'width_ratios':list(tscale_sizes.values())})
# fig,axs = plt.subplots(1,n_kernels,figsize=(n_kernels,2),dpi=150,
#                        sharex=False,sharey=True,gridspec_kw={'width_ratios':list(tscale_sizes.values())})
#fig.subplots_adjust(wspace=-20, hspace=-20.5)

for i,k in enumerate(plotted_kernels):
    ax = axs[i]
    coeffs = kernels[k]['coefficients'][SC_neuron_inds[isort],:]
    k_tscale =kernels[k]['tscale']

        
    extent = [k_tscale[0], k_tscale[-1], n_neurons, 0]
    ax.imshow(coeffs, aspect='auto', cmap='bwr', vmin=-1, vmax=1,extent=extent)

    # Add horizontal lines at the borders of each SC position
    for border in sel_clusters_sorted.groupby('SC_pos').size().cumsum():
        if border < n_neurons:
            ax.axhline(border, color='k', linestyle=':', lw=0.5)

    ax.set_title(kernel_names[i], fontsize=6)
    ax.set_ylim([0,n_neurons])
    off_axes(ax)

    if (k=='task') or ('choice' in k):
        ax.axvline(0, color='k', lw=0.5)
    
plt.tight_layout()
fig.savefig(SAVE_PATH / 'passive_kernels_depth_sorted.svg', dpi=300)






# %% plot the fraction of significant neurons per SC pos

region_counts = sel_clusters.groupby('SC_pos').size()

#d_strings = ['vis_spatial','aud_spatial','aud_onset']

d_strings = ['task','choice']

#d_strings = ['visual','auditory','AV']
n_plots = len(d_strings)
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*1, 1.2),sharey=True,sharex=True,dpi=150)
#fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*1.3, 3),sharey=True,sharex=True,dpi=150)
fig.subplots_adjust(wspace=0.7)
for ax,d_string in zip(axs.flatten(),d_strings):    
    counts = sel_clusters[sel_clusters[f'is_{d_string}']].groupby('SC_pos').size()
    
    frac = (counts / region_counts).fillna(0)*100

    # Plot the fractions
    frac.plot(kind='barh', color="#A7A5A596", ax=ax)
    
    #ax.set_xlabel('% nrns')
    ax.axvline(5, color='k', linestyle=':',alpha=1,linewidth=0.6)
    off_axes(ax, which='top')
    #ax.set_title(d_string)
plt.tight_layout()
fig.savefig(SAVE_PATH / 'passive_fraction_significant_neurons_per_SCpos.svg', dpi=300)
#fig.savefig(SAVE_PATH / 'passive_fraction_significant_neurons_per_SCpos_with_scales.png', dpi=300)


#%% anatomy plot 

d_strings = ['visual','auditory','task','choice']

n_plots = len(d_strings) + 1

# anatomical plot of all neurons from a brain region and significant neurons
coord = 850

anat = anatomy_plotter()
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*1, 1.3),dpi=300,sharex=True,sharey=True)

#fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*2.7, 3.5),dpi=300,sharex=True,sharey=True)
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

    significants = sel_nrns[sel_nrns[f'is_{type}']].copy()

    anat.plot_points(significants['ap_gauss'],significants['dv'],unilateral=True, 
                     c=significants[f'amp_{type}'],
                        alpha=0.85,marker = '.',
                        s=np.sqrt(significants[f'VE_{type}'])*200,edgecolor='k',
                        linewidth=0.25,cmap = 'coolwarm',vmin=-1,vmax=1)

    # anat.plot_points(significants['ap_gauss'],significants['dv'],unilateral=True, 
    #                  c='red',
    #                  alpha=0.85,marker = '.',
    #                  s=20,edgecolor='k',
    #                  linewidth=0.25)
    ax.set_title(type)


# final plot with the scales of the labels
ax =axs[-1]
sizes = np.array([0.005,0.01,0.1,0.2,0.4])

anat.plot_anat_canvas(ax=ax,coord = coord, axis='ml')
xpos = np.ones_like(sizes)*9300
ypos = np.linspace(2500,1200,len(sizes))
anat.plot_points(xpos,ypos,unilateral=True, 
                 c='grey',
                 alpha=0.85,marker = '.',
                 s=np.sqrt(sizes)*200,edgecolor='k',
                 linewidth=0.25)


# ax.set_xlim([-2200, 0])
ax.set_xlim([-2700, -4900])
ax.set_ylim([-3000, -500])
#off_axes(ax)

ax.set_xticks([-3000,-4000])
ax.set_xticklabels(['-3 mm  ','-4 mm'])

ax.set_yticks([-1000,-2000])
ax.set_yticklabels(['-1 mm','-2 mm'])

plt.tight_layout()
# Add a colorbar to represent the significance levels
cbar_ax = fig.add_axes([0.99, 0.3, 0.006, 0.3])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)


fig.savefig(SAVE_PATH / 'passive_significant_neuron_locations.svg', dpi=150)
#fig.savefig(SAVE_PATH / 'passive_significant_neuron_locations_with_scales.png', dpi=400)

# %% variance explained per kernel 

# percentage of neurons with siginificant kernels
# look at variance explained per kernel

region = 'SCs'

nrns_in_region = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin([region]))# &
]

# plot passive visaud
on_x = ['aud_onset','aud_spatial','aud_spatial']
y = 'vis_spatial'

# plot passve aud onset vs spaital 

# on_x = ['aud_spatial','aud_spatial','aud_spatial']
# y = 'aud_onset'

# # # active task vs sensory
# on_x = ['visual','aud_onset','aud_spatial']
# y = 'task'

# on_x = ['task', 'task','task']
# y = 'choice'

# on_x = ['choice_ipsi']
# y = 'choice_contra'

n_plots = len(on_x)
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*1,1.3),dpi=150,sharex=True,sharey=True)
#fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*2.8,3.3),dpi=150,sharex=True,sharey=True)
fig.subplots_adjust(wspace=0.05, hspace=0.1)
for i,x in enumerate(on_x):
    if n_plots==1:
        ax=axs
    else:
        ax = axs[i]

    sns.scatterplot(data=nrns_in_region, x=f'VE_{x}', y=f'VE_{y}',s=8,
                    color='#A7A5A596',
                    edgecolor='k',linewidth=0.3,alpha=0.7, ax=ax)
    
    significant_AV = nrns_in_region[
        (nrns_in_region['is_AV'])]
    sns.scatterplot(data=significant_AV, x=f'VE_{x}', y=f'VE_{y}',s=8,
                    color='red',
                    edgecolor='k',linewidth=0.3,alpha=0.8, ax=ax)
    
    
    # sns.histplot(data=sel_nrns, x='VE_vis_spatial', y='VE_aud_spatial',bins=np.arange(0,1,0.05),cmap='viridis',ax=ax)

    #sns.kdeplot(data=sel_nrns, x='VE_vis_spatial', y='VE_aud_spatial', fill=True, thresh=.1, levels=10, cmap="Blues",ax=ax)

    ax.set_ylim([-0.01,1])
    ax.set_xlim([-0.01,1])

    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_xticks([0,0.01,0.1,1])
    ax.set_yticks([0,0.01,0.1,1])
    ax.axhline(0,color='k',linestyle=':',linewidth=1)
    ax.axvline(0,color='k',linestyle=':',linewidth=1)

    ax.set_xticklabels(['0','.01','0.1','1'])
    ax.set_yticklabels(['0','.01','0.1','1'])

    if i==0:
        ax.set_ylabel(y)
    else:
        ax.set_ylabel('')
    
    ax.set_xlabel(x)

off_axes(ax,which='top')


# axs.set_xlabel('')
# axs.set_ylabel('')

axs[0].set_ylabel('')
axs[0].set_xlabel('')
axs[1].set_xlabel('')
axs[2].set_xlabel('')

# axs[0].set_ylabel('Visual')
# axs[0].set_xlabel('Auditory')
# axs[1].set_xlabel('Task')
# axs[2].set_xlabel('Action')

plt.tight_layout()
fig.savefig(SAVE_PATH / f'passive_variance_explained_scatter_{region}_{y}_vs_{on_x[0]}.svg', dpi=150)
# fig.savefig(SAVE_PATH / f'passive_variance_explained_scatter_{region}_{y}_vs_{on_x[0]}_with_scales.png', dpi=300)

# %%  example neurons PSTHs with model predictions

time_period = (-0.1,.45) # time period to average over
t_idx = (results['response_tscale'] >= time_period[0]) & (results['response_tscale'] < time_period[1])
tscale_t = results['response_tscale'][t_idx]
psth_idx = np.ix_(SC_neuron_inds[isort],t_idx)


example_neurons = [24,175,330,540]
#example_neurons = np.arange(151,404,20)
for respid_to_plot in example_neurons:

    fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(3,1.5))
    fig.subplots_adjust(wspace=0.05,hspace=0.0)
    fig.patch.set_alpha(0.0)

    aud_colors = ['red','grey','blue','k']
    for i,aud in enumerate(audstim):
        for j,vis in enumerate(visstim):
            ax=axs[n_aud-i-1,j]
            resp = results['actual_mean'][(vis,aud,'passive')][psth_idx]
            resp_sem = results['actual_sem'][(vis,aud,'passive')][psth_idx]

            resp_pred = results['predicted_mean'][(vis,aud,'passive')][psth_idx]
            pred_trace = resp_pred[respid_to_plot]


            trace = resp[respid_to_plot]
            trace_sem = resp_sem[respid_to_plot]       
            ax.fill_between(tscale_t, trace - trace_sem, trace + trace_sem, 
                            color='k', alpha=0.3)
            ax.plot(tscale_t,pred_trace, color='k',linewidth = 1)

            if (i==0) and (j==n_vis-1):
                ax.plot([0.2,0.4],[-0.5,-0.5],color='k',linewidth=2)
                
            if (i==1) and (j==n_vis-1):
                ax.plot([.61,.61],[1,2],color='k',linestyle='-',linewidth=2)
            
            off_axes(ax)
    
    #fig.suptitle(f'{respid_to_plot}', fontsize=8)
    fig.savefig(SAVE_PATH / f'passive_responses_kernelsorted_nrn{respid_to_plot}.svg', dpi=300)


# %%


region = 'SCm'

nrns_in_region = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin([region]))# &
]


# %%

which_amp = ['visual','aud_total_contra','choice_ipsi','choice_contra','engagement']
which_sig = ['vis_spatial','aud_total','choice','choice','task']
n_plots = len(which_sig)
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots,2),dpi=150,sharey=True)

for i,(feature_sig, feature_amp) in enumerate(zip(which_sig, which_amp)):
    ax = axs[i]
    sns.stripplot(data=nrns_in_region[nrns_in_region['significant_'+feature_sig]],
                   y=f'amp_{feature_amp}', color='#A7A5A596', ax=ax, size=3, edgecolor='k', linewidth=0.3, alpha=0.5)
    ax.set_ylabel('')
    ax.set_title(feature_amp)
    ax.axhline(0,color='k',linestyle=':',linewidth=0.5)
# %%
fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150)
on_x = nrns_in_region['amp_choice_contra']
on_y = nrns_in_region['amp_aud_total_contra'] 
sns.scatterplot(x=on_x, y=on_y,s=10,
                color='#A7A5A596', alpha=0.5,ax=ax,edgecolor='k',linewidth=0.3)

                # Calculate Pearson correlation
corr, p_value = pearsonr(on_x, on_y)
ax.text(2.5, 2, f'r={corr:.2f}\n', fontsize=6, ha='center', va='center', transform=ax.transData)
ax.set_aspect('equal')
ax.axhline(0,color='k',linestyle=':',linewidth=0.5)
ax.axvline(0,color='k',linestyle=':',linewidth=0.5)
ax.set_xlim([-1,3])
ax.set_ylim([-1,3])
ax.axline((0,0),slope=1,color='k',linestyle='--',linewidth=0.5)


ax.set_xscale('symlog', linthresh=1)
ax.set_yscale('symlog', linthresh=1)
# %%
