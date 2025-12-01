#%%

# load in cluster results
# look at 
# 1) compare Ridge vs Reduced_rank
# 2) look at the significant kernels in different brain regions 
# 3) look at VE per kernel within brain regions
# +1 with the psth -- look at the raw responses

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

from NeuroLogit.src.ephys.results_utils import read_files 
# %%


model1 = 'Ridge50'
model2 = 'Ridge100'

clusters_fit1 = read_files(which_result = 'encoding_kernel_results', filestub=f'clusters_passive_active_{model1}', extension='csv', sessions=None)

clusters_fit2 = read_files(which_result = 'encoding_kernel_results', filestub=f'clusters_passive_active_{model2}', extension='csv', sessions=None)

# %%

cols = ['subject','date','_av_IDs','r2_tot','VE_tot','BerylAcronym','bombcell_class']
# obs1 -- plenty of things didn't fit.... 
comparison_table = pd.merge(clusters_fit1[cols], 
                            clusters_fit2[cols],
                            on=['subject','date','_av_IDs'],
                            suffixes=(f'_{model1}', f'_{model2}'))


# %%

sel_nrns = comparison_table[
    (comparison_table[f'bombcell_class_{model1}']=='good') & 
    (comparison_table[f'BerylAcronym_{model1}'].isin(['SCm','SCs','MOs']))
    ]

metric = 'VE_tot'
fig,ax = plt.subplots(figsize=(2,2),dpi=150)
#
#sns.histplot(data=sel_nrns, x=f'{metric}_{model1}', y=f'{metric}_{model2}',bins=(500,500),cmap='viridis',ax=ax)

sns.scatterplot(data=sel_nrns, x=f'{metric}_{model1}', y=f'{metric}_{model2}',s=10,edgecolor='k',linewidth=0.3,alpha=0.1, ax=ax)
ax.plot([-.2,0.5],[-.2,0.5],'k--',linewidth=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xlabel(f'{model1} R2 total')
ax.set_ylabel(f'{model2} R2 total')

# ax.set_xscale('symlog', linthresh=0.1)
# ax.set_yscale('symlog', linthresh=0.1)


# %%
# %%
