
#%%

from src.ephys.encoding_avg import fit_dataset, get_winning_model
import pandas as pd

timing = {'time_window':'stim','pre_time':0.0,'post_time':0.15}

fit_type = 'passive'
subset = ''

recompute = True
path = rf'C:\Users\Flora\Documents\Github\NeuroLogit\data\{fit_type}_{subset}_coefs.csv'


if recompute:

    coefs = fit_dataset(fit_type = fit_type,
        dataset_kwargs={'set_name':'all', 'subset':subset},
        time_kwargs=timing
    )

    
    coefs.to_csv(path,index=False)
else:
    coefs = pd.read_csv(path,low_memory=False)


models = get_winning_model(coefs,thr_scorer='adj_r2',thr=0)


# %%

goodClus = models[(models.is_good) & (models.BerylAcronym=='SCm')].copy()


goodClus.model.hist()

# %%

import seaborn as sns

fullm = coefs[(coefs.model=='av_aud_bilateral') & (coefs.is_good)].copy()



sns.scatterplot(data=fullm,x='visC',y='gamma')



# %%
