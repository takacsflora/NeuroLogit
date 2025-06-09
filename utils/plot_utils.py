# for computations
import numpy as np
import pandas as pd

# for plot handling
import matplotlib.pyplot as plt

# because taking the log inevitably results in runtime errors
np.seterr(divide="ignore")


def get_region_colors(show_cbar = True):
    """
    HEX codes of the different brain regions I often use
    """
    regions = {
        'SC': '#D392C0',
        'Frontal': '#3CA37B',
        'Vis':'#278F8F',
        'Parietal': '#28A3B7',
        'Lateral': '#00BCD1',
    }

    # Create a figure and axis
    if show_cbar: 
        _, ax = plt.subplots(figsize=(6, 4))

        # Create a color bar with the regions
        for i, (region, color) in enumerate(regions.items()):
            ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
            ax.text(1.05, i + 0.5, region, va='center', fontsize=36)

        # Set limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(regions))
        ax.axis('off')  # Hide axes


    return regions 

from matplotlib.colors import LinearSegmentedColormap


def get_colormap(which = 'choice',type = 'discrete'):
    """
    _function to get a dict of categorical labels and associated colors_
    """
    if type == 'discrete':
        if which == 'choice':
            palette  = {'NoGo': 'orange', 'Left': 'magenta', 'Right': 'green'}


        elif which == 'auditory':
            palette  = {'Center': 'grey', 'Left': 'blue', 'Right': 'red'}

    if type == 'continuous':
        if which == 'choice':
            pass
        elif which == 'auditory':
            colors = [(0, 0, 1), (0.5,0.5, 0.5), (1, 0, 0)]  # Blue -> Grey -> Red
            positions = [0, 0.5, 1]  # Positions for the colors
            palette = LinearSegmentedColormap.from_list("custom_coolwarm", list(zip(positions, colors)))

    return palette
