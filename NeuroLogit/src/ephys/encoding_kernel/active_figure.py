#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

from scipy.ndimage import gaussian_filter1d

from NeuroLogit.src.ephys.encoding_kernel.results_helpers import read_all_results  
from floras_helpers.plotting import off_axes

SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Active')

plt.rcParams.update({'font.size': 7,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})


# SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\conferences\Cosyne2026\Poster figures')

# plt.rcParams.update({'font.size': 24,'font.family':'Calibri','axes.linewidth':1,'axes.spines.top':False,'axes.spines.right':False,
#                      'axes.spines.left':True,'axes.spines.bottom':True,
#                      'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})


#%% load results

trig_kws = {
    'vis_conds': [-1,-.5,0,.5,1],
    'aud_conds': ['no_sound',-1,0,1],
    'choice_conds': ['passive','contra','ipsi','NoGo']
} 

results = read_all_results(model_type='passive_active_Ridge10', sessions='unique',
                            trig_kws=trig_kws,VE_threshold = 0.005)

clusters = results['clusters']

# at this point this takes like 5 mins to load


clusters['is_sensory'] = clusters[[f'is_visual',f'is_auditory',f'is_AV']].any(axis=1)

#%%
brain_region = 'SCm'
sel_clusters = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin([brain_region]))# &
].copy()

SC_neuron_inds = sel_clusters.index.values

n_neurons = len(SC_neuron_inds)

print(n_neurons, 'neurons in', brain_region)

# also print the percentage of significant neurons for choice and task

significant_choice = sel_clusters['is_choice'].mean()
significant_task = sel_clusters['is_task'].mean()


print(f"Percentage of significant neurons in {brain_region}: choice {significant_choice*100:.1f}%, task {significant_task*100:.1f}%")


# also need the count of neurons that were sensory (any) and that were  sensory and task or choice etc.

significant_sensory = sel_clusters['is_sensory'].mean()
sigificant_sensory_and_choice = sel_clusters[sel_clusters['is_sensory'] & sel_clusters['is_choice']].shape[0]
sigificant_sensory_and_task = sel_clusters[sel_clusters['is_sensory'] & sel_clusters['is_task']].shape[0]
sensory_not_choice = sel_clusters[sel_clusters['is_sensory'] & ~sel_clusters['is_choice']].shape[0]


print(f"Percentage of sensory neurons in {brain_region}: {significant_sensory*100:.1f}%")
print(f"Total percentage choice: {sel_clusters['is_choice'].sum()/n_neurons*100:.1f}%")
print (f"Total percentage task: {sel_clusters['is_task'].sum()/n_neurons*100:.1f}%")

print(f"sensory and choice: {sigificant_sensory_and_choice/n_neurons*100:.1f}%")
print(f"sensory and task: {sigificant_sensory_and_task/n_neurons*100:.1f}%")
print(f"sensory but not choice: {sensory_not_choice/n_neurons*100:.1f}%")

signficant_task_and_choice = sel_clusters[sel_clusters['is_task'] & sel_clusters['is_choice']].shape[0]
print(f"Task and choice: {signficant_task_and_choice/n_neurons*100:.1f}%")




# for significant choice perfrom test of ipsi vs contra amp asking whether contra is bigger than ipsi on average across neurons

mean_amp_choice_ipsi_per_subject = (
    sel_clusters[sel_clusters['is_choice']]
    .groupby('subject')['amp_choice_ipsi']
    .mean()
    .sort_index()
)
print('Average amp_choice_ipsi per subject (choice neurons):')
print(mean_amp_choice_ipsi_per_subject)


# contra 
mean_amp_choice_contra_per_subject = (
    sel_clusters[sel_clusters['is_choice']]
    .groupby('subject')['amp_choice_contra']
    .mean()
    .sort_index()
)



from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(mean_amp_choice_ipsi_per_subject, mean_amp_choice_contra_per_subject,alternative='less')
print(f"Paired t-test between ipsi and contra choice amplitudes: t-statistic = {t_stat:.2f}, p-value = {p_value:.4f}")




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
    ax.text(midpoint, -5, a, ha='center', va='bottom', fontsize=8, rotation=90)

off_axes(ax)




#%% plot all kernels rastermap sorted

kernels = results['kernel_dict']
plotted_kernels = ['vis_contra_0.25','vis_contra_0.5','vis_contra_1.0',
                   'aud_onset','aud_ipsi','aud_contra','task',
                   'choice_ipsi','choice_contra']

kernel_names = ['Visual \n low','Visual \n medium','Visual \n high',
                'Auditory \n common','Auditory \n ipsi','Auditory \n contra',
                'Engagement',
                'Action \n ipsi','Action \n contra']
# get the sizes of each kernels tscale


tscale_sizes = {k: len(kernels[k]['tscale']) for k in plotted_kernels}
#plotted_kernels = ['vis_contra_0.5','vis_contra_1.0','aud_onset','aud_ipsi','aud_contra']
n_kernels = len(plotted_kernels)
#
fig,axs = plt.subplots(1,n_kernels,figsize=(n_kernels/1.3,1.5),dpi=150,sharex=False,sharey=True,gridspec_kw={'width_ratios':list(tscale_sizes.values())})
# fig,axs = plt.subplots(1,n_kernels,figsize=(n_kernels*1.6,3.4),dpi=150,sharex=False,sharey=True,
#                        gridspec_kw={'width_ratios':list(tscale_sizes.values())})
#
fig.subplots_adjust(wspace=-20, hspace=-20.5)

for i,k in enumerate(plotted_kernels):
    ax = axs[i]
    coeffs = kernels[k]['coefficients'][SC_neuron_inds[isort],:]
    k_tscale =kernels[k]['tscale']

        
    extent = [k_tscale[0], k_tscale[-1], n_neurons, 0]
    ax.imshow(coeffs, aspect='auto', cmap='bwr', vmin=-2, vmax=2,extent=extent)
    #ax.set_xlim([0,0.6])
    
    #ax.axvline(0, color='k', lw=0.5)
    ax.set_title(kernel_names[i])
    ax.set_ylim([0,n_neurons])
    off_axes(ax)

    if (k=='task') or ('choice' in k):
        ax.axvline(0, color='k', lw=0.5)
    
    ax.invert_yaxis()

    # need to put a timebar on these ones
    if i==n_kernels-1:
        ax.plot([0.0,0.1],[n_neurons-10,n_neurons-10],color='k',linewidth=2)

    
plt.tight_layout()

# Add a colorbar to the figure
cbar_ax = fig.add_axes([0.99, 0.2, 0.01, 0.4])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=-2, vmax=2)
sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params()
cbar.set_label('Kernel Coefficients')

plt.tight_layout()

#fig.savefig(SAVE_PATH / f'all_kernels_rastermap_sorted_{brain_region}.svg', dpi=300, bbox_inches='tight')
fig.savefig(SAVE_PATH / f'{brain_region}_rastermap_sorted_{brain_region}_with_colorbar.svg', dpi=300, bbox_inches='tight')


 #%% plot all responses, rastermap sorted on passive and correct


#  plot the acutal and the predicted means and then the indivisual neurons

example_neurons = [218,259,369,327]

audstim = [1,0,-1]
visstim = trig_kws['vis_conds']

time_period = (-0.1,.46) # time period to average over
t_idx = (results['response_tscale'] >= time_period[0]) & (results['response_tscale'] < time_period[1])
tscale_t = results['response_tscale'][t_idx]
psth_idx = np.ix_(SC_neuron_inds[isort],t_idx)

n_aud = len(audstim)
n_vis = len(visstim)
n_neurons = len(SC_neuron_inds)

fig, axs = plt.subplots(n_aud,n_vis,sharex=True,sharey=True,figsize=(2,2))
fig.subplots_adjust(wspace=0.05,hspace=0.1)
# Set transparent background for the figure
fig.patch.set_alpha(0.0)
resptype = 'contra' 
clip_limit = 2
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

            resp = gaussian_filter1d(resp, sigma=2, axis=1)
            #resp = gaussian_filter1d(resp, sigma=.1, axis=0)

            ax.imshow(resp,aspect='auto',cmap='bwr',vmin=-clip_limit,vmax=clip_limit,extent=extent,interpolation='none')

            ax.set_ylim([-10,n_neurons])
        
        # for border in cluster_borders:
        #     ax.axhline(border, color='k', lw=0.5)
        off_axes(ax)
        ax.invert_yaxis()
    
        if (i==0) & (j==n_vis-1):
            ax.plot([0.3,0.4],[n_neurons-10,n_neurons-10],color='k',linewidth=2)

# Add a colorbar to the figure
cbar_ax = fig.add_axes([0.93, 0.3, 0.01, 0.4])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=-clip_limit, vmax=clip_limit)
sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_ticks([-clip_limit, 0, clip_limit])
cbar.set_ticklabels([f'{-clip_limit}', '0', f'{clip_limit}'])
cbar.ax.tick_params(labelsize=6)
cbar.set_label('Response (z-score)')

print('no of nuerons plotted:', n_neurons)

fig.savefig(SAVE_PATH / f'active_responses_rastermap_sorted_{brain_region}_{resptype}.svg', dpi=300,bbox_inches='tight')


# %% location of units  and spatial distribution across layers

sel_nrns = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCm','SCs']))# &
]


region_counts = sel_nrns.groupby('SC_pos').size()

d_strings = ['choice','task']

n_plots = len(d_strings)
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*.5, 1.2),sharey=True,sharex=True,dpi=150)
fig.subplots_adjust(wspace=0.5)
for ax,d_string in zip(axs.flatten(),d_strings):    
    #counts = sel_clusters.groupby('SC_pos').apply(lambda x: (x[f'VE_{d_string}']).sum())
    counts = sel_nrns[sel_nrns[f'significant_{d_string}']].groupby('SC_pos').size()
    frac = (counts / region_counts).fillna(0)*100

    # Plot the fractions
    frac.plot(kind='barh', color="#A7A5A596", ax=ax)
    ax.set_xlabel('% neurons')
    ax.axvline(5, color='k', linestyle=':',alpha=0.3,linewidth=0.5)
    off_axes(ax, which='top')
    ax.set_title(d_string)
plt.tight_layout()
plt.tight_layout()
fig.savefig(SAVE_PATH / 'active_fraction_significant_neurons_per_SCpos.svg', dpi=300, bbox_inches='tight')

#%% anatomy plot 

from floras_helpers.anat_plots import anatomy_plotter

d_strings = ['task','choice']
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
plt.tight_layout()
fig.savefig(SAVE_PATH / f'active_significant_neuron_locations.svg', dpi=300, bbox_inches='tight')

# %%s catters of the units 
# percentage of neurons with siginificant kernels
# look at variance explained per kernel

region = 'MOs'

nrns_in_regin = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin([region]))# &
]

on_x = ['vis_spatial','aud_spatial','aud_onset']
#on_x  = ['choice']
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
    print(f'Correlation between {x} and {y} in {region}: r={r:.2f}')
    # ax.text(0.6, 0.95, f'r={r:.2f}', transform=ax.transAxes, fontsize=6,
    #         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))

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


fig.savefig(SAVE_PATH / f'active_VE_scatter_{region}_{y}_vs_{on_x[0]}.svg', dpi=300, bbox_inches='tight')



#%% example neurons 
example_neurons = np.arange(3,5,1)


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


#%% the choice kernels -- example neurons (Fig5A)

def filter_clusters(clusters, brain_region=None, is_choice=None, subject=None):
    """
    Filters clusters based on specified criteria.

    Parameters:
    - clusters (pd.DataFrame): DataFrame containing cluster information.
    - brain_region (str, optional): Brain region to filter by.
    - is_choice (bool, optional): Whether to filter by choice neurons.
    - subject (str, optional): Subject to filter by.

    Returns:
    - pd.DataFrame: Filtered clusters.
    """
    criteria = (clusters['bombcell_class'] == 'good')
    
    if brain_region:
        criteria &= clusters['BerylAcronym'].isin([brain_region])
    if is_choice is not None:
        criteria &= clusters['is_choice'] == is_choice
    if subject:
        criteria &= clusters['subject'] == subject

    return clusters[criteria]

# Example usage:

brain_region = 'MOs'
sel_nrns = filter_clusters(clusters, brain_region=brain_region, is_choice=True)

print(f"Number of choice neurons in {brain_region}: {len(sel_nrns)}")
SC_choice_neuron_inds = sel_nrns.index.values



contra = kernels['choice_contra']['coefficients'][SC_choice_neuron_inds]
ipsi = kernels['choice_ipsi']['coefficients'][SC_choice_neuron_inds]
diff_kernel = contra - ipsi
tscale_kernel = kernels['choice_contra']['tscale']

# the ratio of choice neurons etc. 


# calculate the number of neurons where the two kernels are min. 1SD apart at any timepoint
pre_choice_diff = diff_kernel[:, tscale_kernel < 0]
post_choice_diff = diff_kernel[:, tscale_kernel >= 0]


# number of neurons with pre-choice difference > 1 SD
n_action_neurons = len(SC_choice_neuron_inds)
pre_choice_sig = np.abs(pre_choice_diff[:,-1]) > 1 
post_choice_sig = np.abs(post_choice_diff[:,5]) > 1

print(f" 50ms pre choice (% of action neurons) {np.sum(pre_choice_sig)*100/n_action_neurons:.1f}%")
print(f"50ms post choice (% of action neurons) {np.sum(post_choice_sig)*100/n_action_neurons:.1f}%")

print(f"Number of neurons with significant pre-choice kernel: {np.sum(pre_choice_sig)}")
print(f"Number of neurons with significant post-choice kernel: {np.sum(post_choice_sig)}")

isort = np.argsort(np.sum(diff_kernel,axis=1))


#  example ipsi and contra action kernels

n_neurons = 2
which_end_to_plot = 'contra'
fig,axs = plt.subplots(1,n_neurons,figsize=(n_neurons*.5,0.5),dpi=150,sharex=True,sharey=True)
fig.subplots_adjust(wspace=0.0,hspace=0.0)
for i in range(n_neurons):
    ax = axs.flatten()[i]
    # contra pref
    if which_end_to_plot=='contra':
        myiloc = -i-1
    elif which_end_to_plot=='ipsi':
        myiloc = i
    ax.plot(tscale_kernel,contra[isort[myiloc],:], color='magenta', alpha=0.7, linewidth=1)
    ax.plot(tscale_kernel,ipsi[isort[myiloc],:], color='green', alpha=0.7, linewidth=1)
    off_axes(ax)
    ax.set_xlim(-0.12,0.1)
    ax.axvline(0,color='k',linestyle=':',linewidth=0.5)

    ax.set_ylim([-2.5,2.5])
    ax.axhline(0,color='k',linestyle=':',linewidth=0.5)
    # Remove unused axes 

    if i==n_neurons-1:
        ax.plot([0.05,0.1],[-2,-2],color='k',linewidth=1)
        ax.plot([0.09,0.09],[0.5,1.5],color='k',linewidth=1)


#fig.savefig(SAVE_PATH / f'example_choice_kernels_{which_end_to_plot}_{brain_region}.svg', dpi=300, bbox_inches='tight')
#fig.savefig(SAVE_PATH / f'example_choice_kernels_{which_end_to_plot}_{brain_region}.png', dpi=300, bbox_inches='tight')



#%% all the kernels plotted



fig,ax = plt.subplots(1,1,figsize=(1.2,1.2),dpi=150)

plot_type = 'mean_per_subject'  # 'all_neurons' or 'mean_per_subject'
brain_region = 'SCs'
unique_subjects = clusters['subject'].unique()
for subject in unique_subjects:

    sel_nrns = filter_clusters(clusters, brain_region=brain_region, is_choice=True, subject=subject)
    SC_choice_neuron_inds = sel_nrns.index.values
    contra = kernels['choice_contra']['coefficients'][SC_choice_neuron_inds]
    ipsi = kernels['choice_ipsi']['coefficients'][SC_choice_neuron_inds]
    diff_kernel = contra - ipsi

    contra_pref_neurons = np.sum(diff_kernel,axis=1)>0
    ipsi_pref_neurons = np.sum(diff_kernel,axis=1)<0


    if plot_type=='all_neurons':
        n_contra = np.sum(contra_pref_neurons)
        n_ipsi = np.sum(ipsi_pref_neurons)

        for i in range(n_contra):
            ax.plot(ipsi[contra_pref_neurons][i],contra[contra_pref_neurons][i],'-',
                    color="#D70173",alpha=.5,
                    markeredgecolor='k',linewidth = 0.6,markeredgewidth=0.6)


        for i in range(n_ipsi):
            ax.plot(ipsi[ipsi_pref_neurons][i],contra[ipsi_pref_neurons][i],'-',
                    color="#0BACA7",alpha=.7,
                    markeredgecolor='k',linewidth = 0.6,markeredgewidth=0.1)
            
    elif plot_type=='mean_per_subject':
        mean_ipsi = np.mean(ipsi[contra_pref_neurons],axis=0)
        mean_contra = np.mean(contra[contra_pref_neurons],axis=0)

        lw=1.3
        before_choice_idx = tscale_kernel<=0.05
        before_choice_ipsi = mean_ipsi[before_choice_idx]
        before_choice_contra = mean_contra[before_choice_idx] 
        
        ax.plot(before_choice_ipsi, before_choice_contra, '-', color="#F47BB7", alpha=0.7, linewidth=lw,markersize=1)

        after_choice_idx = (tscale_kernel>=0) & (tscale_kernel<0.08)
        after_choice_ipsi = mean_ipsi[after_choice_idx]
        after_choice_contra = mean_contra[after_choice_idx] 

        ax.plot(after_choice_ipsi, after_choice_contra, '-', color="#D70173", alpha=0.7, linewidth=lw,markersize=1)
       # ax.scatter(mean_ipsi,mean_contra,c=(tscale_kernel<0),cmap='copper',s=5,edgecolor='k',linewidth=0.2)     
        mean_ipsi = np.mean(ipsi[ipsi_pref_neurons],axis=0)
        mean_contra = np.mean(contra[ipsi_pref_neurons],axis=0)
        before_choice_ipsi = mean_ipsi[before_choice_idx]
        before_choice_contra = mean_contra[before_choice_idx]
        ax.plot(before_choice_ipsi, before_choice_contra, '-', color="#A8EAE0", alpha=0.7, linewidth=lw)
        after_choice_ipsi = mean_ipsi[after_choice_idx]
        after_choice_contra = mean_contra[after_choice_idx]
        ax.plot(after_choice_ipsi, after_choice_contra, '-', color="#0BACA7", alpha=0.7, linewidth=lw)
        # # or just the mean

        


ax.axhline(0,color='k',linestyle=':',linewidth=0.5)

ax.axvline(0,color='k',linestyle=':',linewidth=0.5)
ax.set_aspect('equal')
ax.axline((0,0),slope=1,color='k',linestyle=':',linewidth=0.5)

ax.set_xlim([-1.2,3.5])
ax.set_ylim([-1.2,3.5])
ax.set_xticks([-1,0,2])
ax.set_yticks([-1,0,2])

fig.savefig(SAVE_PATH / f'choice_kernels_scatter_{plot_type}_{brain_region}.svg', dpi=300, bbox_inches='tight')

# %% average ipsi and contra kernels per subject


plot_type = 'mean_per_subject'  # 'all_neurons' or 'mean_per_subject'
brain_region = 'SCm'
unique_subjects = clusters['subject'].unique()
n_subsjects = len(unique_subjects)
#fig,ax = plt.subplots(1,n_subsjects,figsize=(1*n_subsjects,1.2),dpi=150,sharey=True)

#avg_fig,avg_ax = plt.subplots(1,2,figsize=(2,1),dpi=150,sharey=True)
avg_fig,avg_ax = plt.subplots(1,2,figsize=(5,3),dpi=300,sharey=True)


for p,neuron_type in enumerate(['contra_preferring','ipsi_preferring']):
    ipsi_resp_per_subject = np.zeros((n_subsjects, len(tscale_kernel)))
    contra_resp_per_subject = np.zeros((n_subsjects, len(tscale_kernel)))
    for i,subject in enumerate(unique_subjects):
        sel_nrns = filter_clusters(clusters, brain_region=brain_region, is_choice=True, subject=subject)
        SC_choice_neuron_inds = sel_nrns.index.values
        contra = kernels['choice_contra']['coefficients'][SC_choice_neuron_inds]
        ipsi = kernels['choice_ipsi']['coefficients'][SC_choice_neuron_inds]
        diff_kernel = contra - ipsi

        if neuron_type=='ipsi_preferring':
            pref_neurons = np.sum(diff_kernel,axis=1)<0
        elif neuron_type=='contra_preferring':
            pref_neurons = np.sum(diff_kernel,axis=1)>0


        mean_ipsi = np.mean(ipsi[pref_neurons],axis=0)
        mean_contra = np.mean(contra[pref_neurons],axis=0)

        ipsi_resp_per_subject[i,:] = mean_ipsi
        contra_resp_per_subject[i,:] = mean_contra


        ax[i].plot(tscale_kernel, mean_ipsi)
        ax[i].plot(tscale_kernel, mean_contra)

    mean_ipsi_all = np.nanmean(ipsi_resp_per_subject,axis=0)
    mean_contra_all = np.nanmean(contra_resp_per_subject,axis=0)

    std = np.nanstd(ipsi_resp_per_subject,axis=0) / np.sqrt(n_subsjects)
    avg_ax[p].fill_between(tscale_kernel, mean_ipsi_all - std, mean_ipsi_all + std, color='green', alpha=0.3)
    std = np.nanstd(contra_resp_per_subject,axis=0) / np.sqrt(n_subsjects)
    avg_ax[p].fill_between(tscale_kernel, mean_contra_all - std, mean_contra_all + std, color='magenta', alpha=0.3)
    avg_ax[p].plot(tscale_kernel, mean_ipsi_all, label='Ipsi-tuned', color='green')
    avg_ax[p].plot(tscale_kernel, mean_contra_all, label='Contra-tuned', color='magenta')
    avg_ax[p].axvline(0,color='k',linestyle=':',linewidth=0.5)
    avg_ax[p].axhline(0,color='k',linestyle=':',linewidth=0.5)

    avg_ax[p].set_ylim([-0.5,2.5])

# save avg figure
avg_fig.savefig(SAVE_PATH / f'choice_kernels_avg_per_subject_{brain_region}.png', dpi=300, 
                bbox_inches='tight')
# %%
