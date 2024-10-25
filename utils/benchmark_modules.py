
# %% 

# this code is used to benchmark the torch pipe I am building against the sklearn pipe I have


import pandas as pd

from dat_utils import get_paths
from fit_opto import fit_opto_model

import warnings
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

_,formatted_path,savepath = get_paths(r'opto\region_comparison\uni')
all_files = list(formatted_path.glob('*.csv'))

results = []
for rec in all_files:
    brain_region = rec.stem
    trials = pd.read_csv(rec)

    subjects = trials.subject.unique()
    for subject in subjects:
        trials_of_subject = trials[trials.subject==subject]
        
        # fit the sklearn model
        params_sklearn,loss,auc,t = fit_opto_model(trials_of_subject,nametag=None,gammafit=True)
        params_sklearn['loss'] = loss
        params_sklearn['auc'] = auc
        params_sklearn['method']='sklearn'


        params_sklearn['subject'] = subject
        params_sklearn['brain_region'] = brain_region

        # fit torch model 
        params_scipy,loss,auc,t = fit_opto_model(trials_of_subject,nametag=None,gammafit=True,scipyfit=True)

        params_scipy['loss'] = loss
        params_scipy['auc'] = auc
        params_scipy['method']='scipy'

        params_scipy['subject'] = subject
        params_scipy['brain_region'] = brain_region


        results.append(pd.concat([params_sklearn,params_scipy],ignore_index=True))

results = pd.concat(results,ignore_index=True)
results = results.reset_index()


# %%
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as stats 

results['ID'] = results['subject'].astype(str) + '_' + results['brain_region']
to_compare = 'auc'
fig,ax = plt.subplots(1,1,figsize=(4,4))
for myID in results.ID.unique():
    df = results[results.ID==myID]
    ax.plot(df.method,df[to_compare],color='grey',alpha=0.5)


ax.set_ylabel(to_compare)
scipy_result = results[results.method=='scipy'][to_compare].values
sklearn_result = results[results.method=='sklearn'][to_compare].values
_,p = stats.ttest_rel(scipy_result,sklearn_result)

ax.set_title(f'p={p.round(5)}')
# %%
