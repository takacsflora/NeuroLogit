# %% 
import numpy as np
from utils.av_dat_utils import get_benchmark_opto_dataset
from utils.plot_utils import plot_psychometric

trials, X_train, X_test, y_train, y_test = get_benchmark_opto_dataset(subject=1)
# %%
from src.models.av_models import av_opto_hemispheric_additive,av_opto_hemispheric_divisive

m = av_opto_hemispheric_additive()
m.fit(X_train,y_train)

m.score(X_test,y_test,scorer = 'roc_auc_score')


# %%
import matplotlib.pyplot as plt
_, ax  = plt.subplots(1,4,figsize=(12,3),sharex=True,sharey=True)



trials.hemisphere.fillna(100,inplace=True)
dat_hemisphere_name = ['ctrl','left','right','bilateral']
dat_hemisphere_value = [100,-1,1,0]
pred_hemisphere_value  = [np.nan,-1,1,0]

for i,(name,trial_idx,pred_idx) in enumerate(zip(
    dat_hemisphere_name,dat_hemisphere_value,pred_hemisphere_value
)):

    plot_kwargs = {
        'gamma':m.params['gamma'],
        'ax': ax[i], 
        'yscale' : 'log'
    }

    plot_psychometric(trials[trials.hemisphere==trial_idx], **plot_kwargs)    
    m.plot_pseudo_predictions(hemisphere=pred_idx,**plot_kwargs)

    ax[i].set_title(name)


# %%
