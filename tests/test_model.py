# %% 
from utils.dat_utils import get_benchmark_opto_dataset

trials, X_train, X_test, y_train, y_test = get_benchmark_opto_dataset()
# %%
from src.av_models import av_base

m = av_base()
m.fit(X_train,y_train)

# %%
