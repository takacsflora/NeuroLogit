#%%
# this is how baseline + baseline active changes over time for neurons
from utils.neural_results_utils import load_results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import plotly.io as pio
# pd.options.plotting.backend = "plotly"

myROI = 'SCs'

pre = 'prestim'
post = 'poststim'
coefs_pre = load_results(region = myROI,fit_type = 'choice_engagement',time_bin = pre)
coefs_post = load_results(region = myROI,fit_type = 'choice_engagement',time_bin = post)


model_type = 'audiovisual_choice_engaged' 
#model_type= 'audiovisual_engagement_gain'

# Merge the pre and post coefficients on session and neuronID
merged_coefs = pd.merge(coefs_pre[coefs_pre.model_type == model_type], 
                        coefs_post[coefs_post.model_type == model_type], 
                        on=['session', 'neuronID','model_type'], suffixes=(f'_{pre}', f'_{post}'))



#%%
import seaborn as sns
params = ['tot_vis_ipsi', 'tot_vis_contra','tot_aud_ipsi','tot_aud_contra', 'tot_baseline','choice']

fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 5),sharex=True,sharey=True)

for ax, param in zip(axes, params):
    #sns.kdeplot(x=merged_coefs[f'{param}_{pre}'], y=merged_coefs[f'{param}_{post}'], ax=ax, cmap='viridis', shade=True, thresh=0.05)
    #ax.hexbin(merged_coefs[f'{param}_{pre}'], merged_coefs[f'{param}_{post}'], gridsize=50, cmap='viridis', mincnt=1, bins='log')
    merged_coefs.plot.scatter(x=f'{param}_{pre}', y=f'{param}_{post}', c='hemi_poststim',
                              cmap='coolwarm', ax=ax,alpha=0.5, s=30, edgecolors='k') 
    ax.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1)  # Unity line
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)  # Horizontal zero line
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)  # Vertical zero line
    ax.set_title(param)

plt.tight_layout()
plt.show()


# %%
