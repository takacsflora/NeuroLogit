#%%
import numpy as np

import src.ephys.results_utils  as hP 


import matplotlib.pyplot as plt
import seaborn as sns
from floras_helpers.plotting import off_axes


def read_all_psths(triggered_at='stim',vis_conds=[-1,0,1],aud_conds=[-1,0,1],odds_conds=None,choice_conds=['left','right'],sessions='unique',swap_to_ipsi_contra=False,dat_type='fr'):
    # get all the combination of the above conditions 
    if 'odds' not in triggered_at:
        conds = [(v,a,c) for v in vis_conds for a in aud_conds for c in choice_conds]
    else: 
        conds = [(o,c) for o in odds_conds for c in choice_conds]
    
    read_file_kws = {
        'which_result':'psth_visualisations',
        'extension': 'npz',
        'sessions': sessions, 
        'filestub':f'{dat_type}_psths_at_{triggered_at}'
        }
    # read in all the psths for the conditions

    means = {c: hP.read_files(psth_dat_type='mean',psth_condition=c,**read_file_kws) for c in conds}
    sems = {c: hP.read_files(psth_dat_type='sem',psth_condition=c,**read_file_kws) for c in conds}
    tscale = hP.read_files(psth_dat_type='tscale',psth_condition=None,**read_file_kws)

    # need to be optionally swapping to ipsi and contra
    # basically to swap psths we would need to loop over the hemispheres and if the hemisphere is right we would need to take the psth from the opposite side 
    # so for example (-1,-1,'left') is left stimulus, left stimulus, left choice, so ipsi for left hemisphere but contra for right hemisphere # so a swap would mean 
    if swap_to_ipsi_contra:
        hemis = hP.read_files(psth_dat_type='hemis',psth_condition=None,**read_file_kws)

        if 'odds' not in triggered_at:
            reverse_conds = [(-v, -a, 'right' if c == 'left' else 'left' if c == 'right' else c) for v, a, c in conds]
            new_keys = [(v, a, 'ipsi' if c == 'left' else 'contra' if c == 'right' else c) for v, a, c in conds] # changing the choice names
        else:
            reverse_conds = [(-o, 'right' if c == 'left' else 'left' if c == 'right' else c) for o, c in conds]
            new_keys = [(o, 'ipsi' if c == 'left' else 'contra' if c == 'right' else c) for o, c in conds] # changing the choice names

        # need to make copies because otherwise we will be reswapping back and forth (and btw .copy() is a shallow copy so we need to use deepcopy)
        import copy
        means_swapped = copy.deepcopy(means)
        sems_swapped = copy.deepcopy(sems)
        for i, h in enumerate(hemis):
            if h == -1:
                for c, rc in zip(conds, reverse_conds):
                    means_swapped[c][i] = means[rc][i]
                    sems_swapped[c][i] = sems[rc][i]
        # Now we need to apply the new keys to the means and sems
        means = {new_keys[i]: v for i, (k, v) in enumerate(means_swapped.items())}
        sems = {new_keys[i]: v for i, (k, v) in enumerate(sems_swapped.items())}

    return means,sems,tscale

def plot_psth_for_key(means,tscale,key,nrn_idx,t_idx,color,ax,sems=None):
    """
    heper to plot a single psth for a given key and neuron index and time index
    Args:
        means (dict): dictionary of mean rasters
        sems (dict): dictionary of sem rasters
        tscale (np.ndarray): time scale
        key (tuple): key to select the mean and sem rasters
        nrn_idx (list): list of neuron indices to plot
        t_idx (np.ndarray): boolean array to select time points
        color (str): color to plot
        ax (matplotlib.axes.Axes): axis to plot on
    """

    mean_raster = means[key][np.ix_(nrn_idx,t_idx)][0]
    ax.plot(tscale[t_idx], mean_raster, color=color,linewidth=1)
    if sems is not None:
        sem_raster = sems[key][np.ix_(nrn_idx,t_idx)][0]
        ax.fill_between(tscale[t_idx], mean_raster - sem_raster, mean_raster + sem_raster, color=color, alpha=0.3)


def plot_psth_per_condition(means,sems,tscale,
                            on_vis=[-1,0,1],
                            on_aud=[-1,0,1],
                            on_choice=['left','right'],
                            nrn_idx=[0],
                            pre_time=0.1,
                            post_time=0.5,
                            axs=None):
    if axs is None:
        _,axs = plt.subplots(len(on_aud),len(on_vis), figsize=(1*len(on_vis),1*len(on_aud)),dpi=150,sharex=True,sharey=True)

    color_map = {'left':'blue','right':'red','ipsi':'blue','contra':'red','NoGo':'grey','passive':'grey'}
    colors = [color_map[c] for c in on_choice]

    t_idx = (tscale>-pre_time) & (tscale<post_time)

    on_vis = np.sort(np.array(on_vis))
    on_aud = np.sort(np.array(on_aud))[::-1]

    for i, x in enumerate(on_vis):
        for j, y in enumerate(on_aud):
            ax = axs[j,i] if len(on_vis)>1 else axs[j]
            for h, color in zip(on_choice, colors):
                    key = (x,y,h)
                    if key in means:
                        plot_psth_for_key(means,tscale,key,nrn_idx,t_idx,color,ax,sems=sems)

            if i==0:                
                off_axes(ax,which='top')
            else: 
                off_axes(ax,which='exceptx')
            ax.axvline(0,color='k',ls=':',alpha=0.3)


def plot_psth_per_choice_stimsep(means,sems,tscale,which_stim ='aud',stim_conds=[-1,0,1],choices = ['left','right'],pre_time=0.2,post_time=0.1,ax=None,nrn_idx=[0]):
    if ax is None:
        _,ax = plt.subplots(1,1,figsize=(2,2),dpi=150)
    
    stim_conds = np.sort(np.array(stim_conds))


    for choice in choices:
        if choice == choices[1]:
            stim_cond= stim_conds[stim_conds>-1]
            colors = sns.color_palette("Reds", n_colors=len(stim_cond))
        else:
            stim_cond= stim_conds[stim_conds<1]
            colors  = sns.color_palette("Blues_r", n_colors=len(stim_cond))

        for color, hue in zip(colors, stim_cond):
            if which_stim=='aud':
                key = (0,hue,choice)
            elif which_stim=='vis': 
                key = (hue,0,choice)
            elif which_stim=='odds':
                key = (hue,choice)
            if key in means:
                plot_psth_for_key(means,tscale,key,nrn_idx,
                                t_idx=(tscale>-pre_time) & (tscale<post_time),
                                color=color,ax=ax,sems=sems)
    ax.axvline(0,color='k',ls=':',alpha=0.3)
    off_axes(ax,which='top')


def plot_psth_per_stimcond_aggregate(means,sems,tscale,which_stim ='aud',stim_cond=[-1,0,1],choice='left',pre_time=0.2,post_time=0.1,ax=None,nrn_idx=[0]):
    if ax is None:
        _,ax = plt.subplots(1,1,figsize=(2,2),dpi=150)
    
    stim_cond = np.sort(np.array(stim_cond))

    colors = sns.color_palette("coolwarm", n_colors=len(stim_cond))

    for color, hue in zip(colors, stim_cond):
        if which_stim=='aud':
            key = (0,hue,choice)
        elif which_stim=='vis': 
            key = (hue,0,choice)
        if key in means:
            plot_psth_for_key(means,tscale,key,nrn_idx,
                            t_idx=(tscale>-pre_time) & (tscale<post_time),
                            color=color,ax=ax,sems=sems)
    ax.axvline(0,color='k',ls=':',alpha=0.3)
    off_axes(ax,which='top')
