#%%

# new plots related to passive neurons'responses to audiovisual stimuli

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

from NeuroLogit.src.ephys.cccp.results_helpers import add_ipsi_contra
from NeuroLogit.src.ephys.results_utils import read_files 
from floras_helpers.plotting import off_axes

SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Passive')

plt.rcParams.update({'font.size': 6,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})



# %

unique_only = True
clusters = read_files(which_result = 'ccCP_results', filestub='clusters', extension='csv', sessions='unique')
clusters = add_ipsi_contra(clusters)


clusters['is_vis'] = (clusters['is_vis_stim_passive']==True) & (clusters['is_aud_stim_passive']==False)
clusters['is_aud'] = (clusters['is_aud_stim_passive']==True) & (clusters['is_vis_stim_passive']==False)
clusters['is_AV'] = (clusters['is_vis_stim_passive']==True) & (clusters['is_aud_stim_passive']==True)
clusters['is_choice']  = (clusters['is_choice_choice_Go']==True)

noise_std = 50  # Standard deviation of the noise
clusters['ap_gauss'] = clusters['ap'] + np.random.normal(0, noise_std, size=len(clusters))
clusters['ap_bregma'] = -clusters['ap'] + 5400
clusters['dv_bregma'] = -clusters['dv'] + 332

# %%  the fraction of significant neurons per brain region

all_good_nrns = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCm','SCs','MOs']))# &
]
region_counts = all_good_nrns.groupby('BerylAcronym').size()



region_colors =  {"SCs":"#04D9FF",
            "SCm":"#FF04D9",
            "MOs":"#7BD1A3",
            "VISp":"#C0C9CF",
        }


#d_strings =['vis','aud','AV']
d_strings = ['choice']
n_plots = len(d_strings)
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*1, .6),sharey=True,sharex=True,dpi=150)
fig.subplots_adjust(wspace=0.7)



for i,d_string in enumerate(d_strings):    
    if n_plots ==1:
        ax = axs
    else:
        ax = axs[i]
    counts = all_good_nrns[all_good_nrns[f'is_{d_string}'] == 1].groupby('BerylAcronym').size()
    frac = (counts / region_counts).fillna(0)*100


    # Plot the fractions
    frac = frac.reindex(['MOs', 'SCm', 'SCs'])
    print(frac)

    ax.barh(frac.index, frac.values, height=.9, color=[region_colors.get(x, '#333333') for x in frac.index], edgecolor='k', linewidth=0.4, alpha=0.6)
    #frac.plot(kind='barh', color=[region_colors.get(x, '#333333') for x in frac.index], ax=ax,edgecolor='k',linewidth=0.4,alpha=0.6)

    ax.bar_label(ax.containers[0], fmt='%.0f', fontsize=6, padding=1)
    ax.set_xlim(0, 35)
    ax.set_xlabel('% neurons')
    ax.axvline(0.05, color='k', linestyle=':',alpha=0.3)
    off_axes(ax, which='top')
    ax.set_title(d_string)
    ax.axvline(5, color='k', linestyle=':',alpha=0.5,linewidth=0.8)
    ax.set_xticks([0,15,30])




fig.savefig(SAVE_PATH/f'percentage_{d_string}.svg',dpi=300,bbox_inches='tight',transparent=True)

# %%  the location and distribution of each neuron type in SC

sc_neurons = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCm','SCs']))# & 
]

from floras_helpers.anat_plots import anatomy_plotter



type = 'choice'
cp_type = 'cp_choice_choice_Go'

# anatomical plot of all neurons from a brain region and significant neurons
coord = 850
anat = anatomy_plotter()
fig,axs = plt.subplots(2,2,figsize=(1.2, 1.2),dpi=300,sharex='col',sharey='row',gridspec_kw={'height_ratios':[1,4],'width_ratios':[4,1]})

fig.subplots_adjust(wspace=0.1,hspace=0.1)
anat_ax = axs[1,0]
anat.plot_anat_canvas(ax=anat_ax,coord = coord, axis='ml')

anat.plot_points(sc_neurons['ap_gauss'],sc_neurons['dv'],unilateral=True,c = 'grey',
                    alpha=0.2,marker = '.',s=10,edgecolor=None)

significants = sc_neurons[sc_neurons[f'is_{type}']].copy()

anat.plot_points(significants['ap_gauss'],significants['dv'],unilateral=True,c = significants[f'{cp_type}'],
                    alpha=0.85,marker = '.',s=25,edgecolor='k',cmap='coolwarm',vmin=.25,vmax=.75,linewidth=0.2)

# ax.set_xlim([-2200, 0])
anat_ax.set_xlim([-2600, -4800])
anat_ax.set_ylim([-2900, -700])



ap_hist_ax = axs[0,0]
dv_hist_ax = axs[1,1]


ap_kws = {
    'x': 'ap_bregma',
}
sns.kdeplot(sc_neurons, color='grey', fill=True, alpha=0.3,linewidth=.1, **ap_kws, ax=ap_hist_ax)
sns.kdeplot(significants, color='k', **ap_kws,linewidth=1,ax=ap_hist_ax)


dv_kws = {
    'y': 'dv_bregma',
}

sns.kdeplot(sc_neurons, color='grey', fill=True, alpha=0.3,linewidth=.1, **dv_kws, ax=dv_hist_ax)
sns.kdeplot(significants, color='k', **dv_kws,linewidth=1, ax=dv_hist_ax)

fig.delaxes(axs[0,1])
off_axes(ap_hist_ax,which='exceptx')
off_axes(dv_hist_ax,which='excepty')
off_axes(anat_ax,which='top')

ap_hist_ax.set_ylabel('')

anat_ax.set_xlabel('AP distance from Bregma')
anat_ax.set_ylabel('DV')

anat_ax.set_xticks([-3000,-4000])
anat_ax.set_yticks([-1000,-2500])
anat_ax.set_xticklabels(['-3 mm','-4 mm'])
anat_ax.set_yticklabels(['-1 mm','-2.5 mm'])

fig.savefig(SAVE_PATH/f'Anatomy__passive_{type}.svg',dpi=300,bbox_inches='tight',transparent=True)

# %%
fig,ax = plt.subplots(1,1,figsize=(.5,.5),dpi=300)
# plot a coolwarm palette between 0 and 1 
#just the palette
cmap = plt.get_cmap('coolwarm')
norm = plt.Normalize(vmin=0, vmax=1)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=1, pad=0.5)
cbar.set_label('AUC', fontsize=6)
cbar.set_ticks([0, 1])
#cbar.set_ticklabels(['ipsi', 'contra'],rotation=90)
cbar.ax.tick_params(labelsize=6)
ax.axis('off')

fig.savefig(SAVE_PATH/f'colorbar_passive_AUC.svg',dpi=300,bbox_inches='tight',transparent=True)
# %%

# same figure for MOs


mo_neurons = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['MOs']))# &
]

type = 'aud'
cp_type = 'cp_aud_stim_passive'
# anatomical plot of all neurons from a brain region and significant neurons
coord = 500

anat = anatomy_plotter()
fig,axs = plt.subplots(2,2,figsize=(1.8,    1.2),dpi=300,sharex='col',sharey='row',gridspec_kw={'height_ratios':[1,4],'width_ratios':[4,1]})
fig.subplots_adjust(wspace=0.1,hspace=0.1)
anat_ax = axs[1,0]
anat.plot_anat_canvas(ax=anat_ax,coord = coord, axis='ml')
anat.plot_points(mo_neurons['ap_gauss'],mo_neurons['dv'],unilateral=True,c = 'grey',
                    alpha=0.2,marker = '.',s=10,edgecolor=None)

significants = mo_neurons[mo_neurons[f'is_{type}']].copy()
anat.plot_points(significants['ap_gauss'],significants['dv'],unilateral=True,c = significants[f'{cp_type}'],
                    alpha=0.85,marker = '.',s=25,edgecolor='k',cmap='coolwarm',vmin=.25,vmax=.75,linewidth=0.2)
# ax.set_xlim([-2200, 0])
anat_ax.set_xlim([3500, -500])
anat_ax.set_ylim([-2800, -200])
ap_hist_ax = axs[0,0]
dv_hist_ax = axs[1,1]
ap_kws = {
    'x': 'ap_bregma',
}
sns.kdeplot(mo_neurons, color='grey', fill=True, alpha=0.3,linewidth=.1, **ap_kws, ax=ap_hist_ax)
sns.kdeplot(significants, color='k', **ap_kws,linewidth=1,ax=ap_hist_ax)
dv_kws = {
    'y': 'dv_bregma',
}
sns.kdeplot(mo_neurons, color='grey', fill=True, alpha=0.3,linewidth=.1, **dv_kws, ax=dv_hist_ax)
sns.kdeplot(significants, color='k', **dv_kws,linewidth=1, ax=dv_hist_ax)
fig.delaxes(axs[0,1])
off_axes(ap_hist_ax,which='exceptx')
off_axes(dv_hist_ax,which='excepty')
off_axes(anat_ax,which='top')
ap_hist_ax.set_ylabel('')

anat_ax.set_xlabel('AP distance from Bregma')
anat_ax.set_ylabel('DV')
anat_ax.set_xticks([3000,2000,1000,0])
anat_ax.set_xticklabels(['3 mm','2 mm','1 mm','0 mm'])
anat_ax.set_yticks([-1000,-2000])
anat_ax.set_yticklabels(['-1 mm','-2 mm'])
fig.savefig(SAVE_PATH/f'Anatomy__passive_MOs_{type}.svg',dpi=300,bbox_inches='tight',transparent=True)
# %%
