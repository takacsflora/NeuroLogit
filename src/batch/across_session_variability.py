#%%
import pandas as pd
from utils.av_dat_utils import get_paths,filt_split_trials
from src.models.logit_sklearn import av_opto_sk

set_name  = r'opto/region_comparison/bi'
_,formatted_path,save_path = get_paths(set_name)


region_paths = list(formatted_path.glob('*.csv'))


#

results = []

for region_path in region_paths:
    trials = pd.read_csv(region_path)

    X_tot,_,_,_,_ = filt_split_trials(trials,test_size = 0.5 ,balance_sensory = False)
    params,log_loss,auc = av_opto_sk(X_tot,gridCV_vis=True)
    avg_gamma = params['gamma'].values[0]

    trials['combined_session_subject'] = trials['subject'].astype(str) + "_" + trials['sessionID'].astype(str)

    unique_sessions = trials.combined_session_subject.unique()
    
    for sess in unique_sessions:
            trials_of_session = trials[trials.combined_session_subject == sess]
            X_tot,X_train,X_test, y_train,y_test = filt_split_trials(trials_of_session,test_size = 0.33 ,balance_sensory = False)

            # nTrials = X_tot.shape[0]
            # if nTrials>20:
            params,log_loss,auc = av_opto_sk(X_tot,power = avg_gamma,gridCV_vis=False)


            # params['log_loss_opto'] = m.score(X_test[test_is_opto],y_test[test_is_opto],scorer = 'log_loss')
            # params['auc_opto'] = m.score(X_test[test_is_opto],y_test[test_is_opto],scorer = 'roc_auc_score')
            params['mouseID'] = trials_of_session.subject.unique()[0]
            params['region'] = region_path.stem
            results.append(params)
        



results = pd.concat(results, ignore_index=True)

results['combinedID'] = results['mouseID'].astype(str) + "_" + results['region']
# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

ax = sns.stripplot(data = results,x='region',y = 'bias_opto',hue='combinedID',palette = 'tab10' )

ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="MouseID + region")
plt.show()

# %%

import numpy as np
delta_b_variance = []
unique_res = results.combinedID.unique()
for i in unique_res:
    results_per_subject = results[results.combinedID==i]
    variance = np.var(results_per_subject.bias_opto.values)

    delta_b_variance.append({
         'bias_opto_variance':variance,
         'combinedID': i,
         'region':results_per_subject.region.values[0]
    })

delta_b_variance = pd.DataFrame(delta_b_variance)


sns.scatterplot(data=delta_b_variance,x='region',y='bias_opto_variance')
# %%
