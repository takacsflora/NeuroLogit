#%%
from visualisation.ephys.util_dat import read_in_all_coefs
import pandas as pd

coefs,models = read_in_all_coefs(fit_type='passive', subset='', recompute=False, add_behav_params=True, get_best=True)

# %%
