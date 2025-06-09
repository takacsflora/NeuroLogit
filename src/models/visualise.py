
# function to plot a single session
from src.models.av_models_opto import av_pseudoPlotter
import matplotlib.pyplot as plt

from utils.plot_utils import get_colormap
import numpy as np


def get_choice_fractions(ev):
    """from trial data get the choice fractions and the Odds for each choice/stimulus combination
    Args:
        ev (_type_): _description_

    Returns:
        _type_: _description_
    """

    ev = ev.copy()
    response_mapping = {-1: 'NoGo', 0: 'Left', 1: 'Right', 2: 'Left', 3: 'Right'}

    ev['choice_'] = ev['choice'].map(response_mapping)

    choice_fraction = (
        ev.groupby(['visDiff', 'audDiff', 'choice_'])
        .size()
        .unstack(fill_value=0)
        .div(ev.groupby(['visDiff', 'audDiff']).size(), axis=0)
        .reset_index()
    )

    for col in ['Right', 'Left', 'NoGo']:
        if col in choice_fraction:
            choice_fraction[col] = choice_fraction[col].replace(0, np.nan)
    choice_fraction['logOdds_right_vs_left'] = np.log(choice_fraction['Right'] / choice_fraction['Left'])
    

    is_nogo = np.isin(['NoGo'],choice_fraction.columns)[0]
    if is_nogo:
        choice_fraction['logOdds_right_vs_NoGo'] = np.log(choice_fraction['Right'] / choice_fraction['NoGo'])
        choice_fraction['logOdds_left_vs_NoGo'] = np.log(choice_fraction['Left'] / choice_fraction['NoGo'])
        choice_fraction['logOdds_NoGo_vs_Go'] = np.log(choice_fraction['NoGo'] / (choice_fraction['Right'] + choice_fraction['Left']))
    else:
        choice_fraction['NoGo'] = np.nan

    return choice_fraction



def plot_psychometric_multi(ev, m=None, space="exp", ax=None):
    """
    Plot session behaviour in exp or log space.

    Args:
        ev: DataFrame with behavioural data.
        m: Fitted model or None. If None, only data is plotted.
        space: "exp" or "log".
        ax: Optional tuple/list of matplotlib axes (length 2). If None, creates new.
    """


    choice_fraction = get_choice_fractions(ev)  
    
    custom_palette = get_colormap('auditory', type='continuous')
    colors = custom_palette(np.linspace(.2, .8, len(unique_aud)))


    if ax is None:
        fig, ax = plt.subplots(2, 1,figsize=(1.5,2),height_ratios=[2.5,1],dpi=300, sharex=True, sharey=False)
        created_fig = True
    else:
        fig = None
        created_fig = False
    fig_axes = ax if isinstance(ax, (list, tuple, np.ndarray)) else [ax]

    if space == "exp":
        hlines = [0.5, 0]
    else:
        hlines = [0, 0]
    fig_axes[0].axhline(hlines[0], color='black', linestyle=':', linewidth=0.5)
    fig_axes[1].axhline(hlines[1], color='black', linestyle=':', linewidth=0.5) # the basic setup...


    unique_aud = np.sort(choice_fraction.audDiff.unique())    
    markersize = 4

    for i, (a, c) in enumerate(zip(unique_aud, colors)):
        pkws = dict(
            color=c, markersize=markersize, marker='o', linestyle='None',
            markeredgecolor='k', markeredgewidth=0.5
        )
        cur_choices = choice_fraction[choice_fraction.audDiff == a]
        if m is not None:
            ps = av_pseudoPlotter()
            matrix = ps.pseudo[i]
            gamma = m.params['gamma']
            visDiff_gamma = matrix.visR ** gamma - matrix.visL ** gamma
            zL, zR = m.predict_log_proba(matrix)
            Nogo_Go_pred = -np.log(np.exp(zR) + np.exp(zL))
            pNoGo = 1 / (1 + np.exp(zR) + np.exp(zL))
            pR = np.exp(zR) / (1 + np.exp(zR) + np.exp(zL))
            visDiff_gamma_data = np.abs(cur_choices.visDiff) ** gamma * np.sign(cur_choices.visDiff)
        else:
            visDiff_gamma_data = cur_choices.visDiff

        if space == "exp":
            if m is not None:
                fig_axes[0].plot(visDiff_gamma, pR, color=c)
                fig_axes[1].plot(visDiff_gamma, pNoGo, color=c)
            if 'Right' in cur_choices:
                fig_axes[0].plot(visDiff_gamma_data, cur_choices.Right, **pkws)
            if 'NoGo' in cur_choices:
                fig_axes[1].plot(visDiff_gamma_data, cur_choices.NoGo, **pkws)
        else:
            if m is not None:
                fig_axes[0].plot(visDiff_gamma, zR - zL, color=c)
                fig_axes[1].plot(visDiff_gamma, Nogo_Go_pred, color=c)
            if 'logOdds_right_vs_left' in cur_choices:
                fig_axes[0].plot(visDiff_gamma_data, cur_choices.logOdds_right_vs_left, **pkws)
            if 'logOdds_NoGo_vs_Go' in cur_choices:
                fig_axes[1].plot(visDiff_gamma_data, cur_choices.logOdds_NoGo_vs_Go, **pkws)

    if space == "log":
        fig_axes[0].set_yticks([-3, 0, 3])
        fig_axes[0].set_yticklabels([1e-3, 1e0, 1e3])
        fig_axes[1].set_yticks([-2, -3])
        fig_axes[1].set_yticklabels([1e-2, 1e-3])

    for a in fig_axes:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.axvline(0, color='black', linestyle=':', linewidth=0.5)
        a.set_xticklabels('')
        a.spines['left'].set_position(('outward', 3))
        a.spines['bottom'].set_position(('outward', 3))

    if created_fig:
        return fig, ax
    else:
        return ax
