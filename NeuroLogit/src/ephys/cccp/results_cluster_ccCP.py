#%%


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

from src.ephys.cccp.results_helpers import add_ipsi_contra
from src.ephys.results_utils import read_files 
from floras_helpers.plotting import off_axes
# %

unique_only = True
clusters = read_files(which_result = 'ccCP_results', filestub='clusters', extension='csv', sessions='unique')
clusters = add_ipsi_contra(clusters)

#%% add a few things I newly calculate in clusters
# Add Gaussian noise to the 'ml' axis
noise_std = 50  # Standard deviation of the noise
clusters['ml_gauss'] = clusters['ml'] + np.random.normal(0, noise_std, size=len(clusters))
clusters['ap_gauss'] = clusters['ap'] + np.random.normal(0, noise_std, size=len(clusters))
d_strings = ['vis_stim_passive','vis_stim_Go','aud_stim_passive','aud_stim_Go','choice_prestim_Go','choice_choice_Go']

columns_to_check = [f'is_{d}' for d in d_strings]

clusters['is_responsive'] = clusters[columns_to_check].any(axis=1)

for d in d_strings:
    clusters[f'fr_diff_{d}'] = clusters[f'fr_contra_{d}'] - clusters[f'fr_ipsi_{d}']


# recalculate the vector angles and magnitudes based on ipsi contra
for type in d_strings:
    d,d_t,d_i = type.split('_')
    baseline_stub_ipsi = f'fr_ipsi_{d}_prestim_{d_i}'
    baseline_stub_contra = f'fr_contra_{d}_prestim_{d_i}'
    response_stub_ipsi = f'fr_ipsi_{d}_{d_t}_{d_i}'
    response_stub_contra = f'fr_contra_{d}_{d_t}_{d_i}'

    baseline_vector = np.array([clusters[baseline_stub_ipsi], clusters[baseline_stub_contra]])
    response_vector = np.array([clusters[response_stub_ipsi], clusters[response_stub_contra]])
    vector_diff = response_vector - baseline_vector

    clusters[f'vector_magnitude_{d}_{d_i}'] = np.sqrt(vector_diff[0]**2 + vector_diff[1]**2)
    clusters[f'vector_angle_{d}_{d_i}'] = np.degrees(np.arctan2(vector_diff[1], vector_diff[0]))

# %% select the neurons to plot/analyze
sel_nrns = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCm','SCs','MOs']))# &
]
# %%  the fraction of significant neurons per brain region
region_counts = sel_nrns.groupby('BerylAcronym').size()

d_strings =['vis_stim_passive','aud_stim_passive','choice_choice3_Go','choice_choice1_Go']
n_plots = len(d_strings)
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*.8, 1.5),sharey=True,sharex=True,dpi=150)
fig.subplots_adjust(wspace=0.7)
for ax,d_string in zip(axs.flatten(),d_strings):    
    counts = sel_nrns[sel_nrns[f'is_{d_string}'] == 1].groupby('BerylAcronym').size()
    frac = (counts / region_counts).fillna(0)

    # Plot the fractions
    frac = frac.reindex(['MOs', 'SCm', 'SCs'])
    print(frac)
    frac.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_xlabel('% choice \n neurons')
    ax.axvline(0.05, color='k', linestyle=':',alpha=0.3)
    off_axes(ax, which='top')
    ax.set_title(d_string)
    # nn = d_string.split('_')
    # ax.set_title(f'{nn[0]}, {nn[-1]}')
savepath = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\conferences\Cosyne2026\Raw_figures')
fig.savefig(savepath/'Fraction_choice_neurons.svg',dpi=300,bbox_inches='tight',transparent=True)
#%% 
# now test vis/aud vs audiovisual separately
passive_keys = ['vis_stim_passive','aud_stim_passive']
permutations = [(True, False), (False, True),(True, True)]

n_permutations = len(permutations)

fig, axs = plt.subplots(1,n_permutations, figsize=(8, .7), dpi=150, sharey=True,sharex=True)
fig.subplots_adjust(wspace=0.4)
for ax, (vis, aud) in zip(axs.flatten(), permutations):
    counts = sel_nrns[(sel_nrns.is_vis_stim_passive == vis) & (sel_nrns.is_aud_stim_passive == aud)].groupby('BerylAcronym').size()
    frac = (counts / region_counts).fillna(0)
    frac = frac.reindex(['MOs', 'SCm', 'SCs'])
    print(frac)
    frac.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_xlabel('% of Neurons')
    ax.axvline(0.05, color='k', linestyle=':',alpha=0.3)
    off_axes(ax, which='top')

savepath = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\conferences\Cosyne2026\Raw_figures')
fig.savefig(savepath/'Fraction_significant_neurons.svg',dpi=300,bbox_inches='tight',transparent=True)

#%% anatomy plots colored by cp value 

from floras_helpers.anat_plots import anatomy_plotter

# anatomical plot of all neurons from a brain region and significant neurons
coord = 850

n_plots = len(d_strings)
anat = anatomy_plotter()
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*2, 2),dpi=300,sharex=True,sharey=True)

sel_nrns = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCm','SCs']))# &
]

for i, type in enumerate(d_strings):
    ax =axs[i]
    anat.plot_anat_canvas(ax=ax,coord = coord, axis='ml')

    anat.plot_points(sel_nrns['ap_gauss'],sel_nrns['dv'],unilateral=True,c = 'grey',
                     alpha=0.2,marker = '.',s=20,edgecolor=None)

    significants = sel_nrns[sel_nrns[f'is_{type}']].copy()

    anat.plot_points(significants['ap_gauss'],significants['dv'],unilateral=True,c = significants[f'cp_{type}'],
                     alpha=0.85,marker = '.',s=70,edgecolor='k',cmap='coolwarm',vmin=.25,vmax=.75)

    # ax.set_xlim([-2200, 0])
    ax.set_xlim([-2700, -4900])
    ax.set_ylim([-3000, -500])
    nn = type.split('_')
    ax.set_title(f'{nn[0]}, {nn[-1]}')
    off_axes(ax)

fig.savefig(savepath/'Anatomy_cp_values.svg',dpi=300,bbox_inches='tight',transparent=True)
#%%

sel_nrns = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCm','SCs']))# &
]



anat_dim = 'ap'
pkws = {
    'x': anat_dim,
    'linewidth':4,
}
sns.kdeplot(sel_nrns, color='grey', fill=True, alpha=0.3, **pkws)

significants = sel_nrns[sel_nrns['is_vis_stim_passive']].copy()
sns.kdeplot(significants, color='blue', **pkws)

significants = sel_nrns[sel_nrns['is_aud_stim_passive']].copy()
sns.kdeplot(significants, color='magenta', **pkws)


significants = sel_nrns[sel_nrns['is_choice_choice3_Go']].copy()
sns.kdeplot(significants, color='black',linestyle='--', **pkws)

significants = sel_nrns[sel_nrns['is_choice_choice1_Go']].copy()
sns.kdeplot(significants, color='black', **pkws)





# %%  histogram of cp values
fig,axs = plt.subplots(1,len(d_strings),figsize=(1.5*len(d_strings),1.5),dpi=150,sharey=True,sharex=True)
regions = ['MOs','SCm','SCs']
region_colors = ["#2F8A67","#EA76C7","#2BE5E5"]
for i, type in enumerate(d_strings):
    ax = axs[i]
    for region,color in zip(regions,region_colors):
        sel = sel_nrns[sel_nrns.BerylAcronym==region]
        #sns.kdeplot(sel[sel[f'is_{type}'] == 1], x=f'cp_{type}', bw_adjust=0.3,color=color,ax=ax)
        sns.histplot(sel[sel[f'is_{type}'] == 1],x = f'cp_{type}',bins=np.arange(0, 1.1, 0.05),color=color,alpha=0.9,ax=ax,stat='density',element='step',fill=False)

    #sns.kdeplot(sel_nrns[(sel_nrns.is_responsive==False)], x=f'cp_{type}', bw_adjust=0.5,color='grey',ax=ax)
    sns.histplot(sel_nrns[(sel_nrns.is_responsive==False)],x=f'cp_{type}',bins=np.arange(0, 1.1, 0.05),color='grey',alpha=0.9,ax=ax,stat='density',element='step',fill=False)
    ax.axvline(0.5,color='k',ls=':')
    ax.set_xlim([0,1])
    nn = type.split('_')
    ax.set_title(f'{nn[0]}, {nn[-1]}')
    off_axes(ax,which='top')
ax.legend(regions + ['non-responsive'], loc='center left', bbox_to_anchor=(1, 0.5))



#%%
# Add a coolwarm colorbar
fig, ax = plt.subplots(figsize=(1, 4), dpi=150)
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0.25, vmax=0.75)), cax=ax)
cbar.set_label('CP Value', rotation=270, labelpad=15)
cbar.ax.tick_params(labelsize=8)

# %% Polar plot of vector angles
fig, axs = plt.subplots(1, len(d_strings), subplot_kw={'projection': 'polar'}, figsize=(2*len(d_strings), 2), dpi=150)
fig.subplots_adjust(wspace=0.3)
# Convert angles to radians
for i, type in enumerate(d_strings):
    ax = axs[i]
    d,d_t,d_i = type.split('_') 

    # Separate responsive and non-responsive neurons
    responsive = sel_nrns[sel_nrns[f'is_{type}'] == 1]
    non_responsive = sel_nrns[sel_nrns[f'is_{type}'] == 0]

    # Plot responsive neurons
    # ax.hist(np.radians(non_responsive[f'vector_angle_{d}_{d_i}']), bins=45, alpha=0.5, label='Non-Responsive', color='gray', density=True)

    # ax.hist(np.radians(responsive[f'vector_angle_{d}_{d_i}']), bins=45, alpha=0.7, label='Responsive', color='blue', density=True)
    # Plot each neuron with vector angle and magnitude
    magnitudes = responsive[f'vector_magnitude_{d}_{d_i}']
    angles = np.radians(responsive[f'vector_angle_{d}_{d_i}'])
    ax.scatter(angles, magnitudes, alpha=0.7, label='Neurons', color='blue', s=1)

    # Plot non-responsive neurons

    # ax.set_theta_zero_location("N")
    # ax.set_theta_direction(-1)
    ax.set_title(f'{d}, {d_i}', va='bottom')
    #ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1))
    # Move the concentric labels to the outside of the plot
    ax.set_rlabel_position(-75)  # Adjust the position of radial labels
    if i > 0:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    ax.set_yscale('symlog',linthresh=1)


# %% comparison of cps

#on_y = d_strings[5]
for on_y in d_strings:
    on_xs  = [d for d in d_strings if d!=on_y] 
    # 
    n_plots  = len(on_xs)+1
    # plot the selectivity indices against each other 
    fig,axs = plt.subplots(2,n_plots,figsize=(2*n_plots,3),dpi=150,sharex=False,sharey=False,
                        gridspec_kw={'height_ratios':[2.5,1],'width_ratios':[1]+[2]*len(on_xs)})

    significants = sel_nrns[sel_nrns[f'is_{on_y}']].copy()
    non_responsives = sel_nrns[sel_nrns['is_responsive']==False].copy().sample(n=len(significants),random_state=42)
    # first plot is the histogram of the y axis
    # sns.histplot(non_responsives,y=f'cp_{on_y}',bins=np.arange(0, .9, 0.05),ax=axs[0,0],color='grey',stat='density',element='step',fill=False)
    # sns.histplot(significants,y=f'cp_{on_y}',bins=np.arange(0, .9, 0.05),ax=axs[0,0],color='cyan',stat='density',element='step',fill=False)

    # instead of histplot try kdeplot
    sns.kdeplot(non_responsives,y=f'cp_{on_y}',ax=axs[0,0],color='grey',fill=False,cumulative=True)
    sns.kdeplot(significants,y=f'cp_{on_y}',ax=axs[0,0],color='cyan',fill=False,cumulative=True)
    axs[0,0].axhline(0.5,color='k',ls=':')

    axs[0, 0].invert_xaxis()
    off_axes(axs[0,0],which='excepty')
    axs[0, 0].set_ylim([0,1])
    scatter_kwargs = {'alpha':0.7,'s':5,'edgecolor':'black','legend':None}
    #
    for i, on_x in enumerate(on_xs):
        ax_scatter = axs[0,i+1]
        sns.scatterplot(data=non_responsives,x=f'cp_{on_x}',y=f'cp_{on_y}',color='grey',ax=ax_scatter,**scatter_kwargs)
        sns.scatterplot(data=significants,x=f'cp_{on_x}',y=f'cp_{on_y}',color='cyan',ax=ax_scatter,**scatter_kwargs)
        ax_scatter.axhline(0.5,color='k',ls=':')
        ax_scatter.axvline(0.5,color='k',ls=':')
        ax_scatter.axline((0.5,0.5),slope=1,color='k',ls=':')
        off_axes(ax_scatter,which='all')
        ax_scatter.set_xlim([0,1])
        ax_scatter.set_ylim([0,1])
        ax_scatter.set_ylabel('')
        ax_scatter.set_title(f'{on_x}')

        ax_hist = axs[1,i+1]
        # sns.histplot(non_responsives,x=f'cp_{on_x}',bins=np.arange(0, .9, 0.05),ax=ax_hist,color='grey',stat='density',element='step',fill=False)
        # sns.histplot(significants,x=f'cp_{on_x}',bins=np.arange(0, .9, 0.05),ax=ax_hist,color='cyan',stat='density',element='step',fill=False)

        # try kdeplot instead of histplot
        sns.kdeplot(non_responsives,x=f'cp_{on_x}',ax=ax_hist,color='grey',fill=False,cumulative=True)
        sns.kdeplot(significants,x=f'cp_{on_x}',ax=ax_hist,color='cyan',fill=False,cumulative=True)

        off_axes(ax_hist,which='exceptx')
        ax_hist.set_xlim([0,1])
        ax_hist.set_ylabel('')
        ax_hist.axvline(0.5,color='k',ls=':')

    off_axes(axs[1,0],which='all')
# %%
