
# this bascially take all the subjects and fits them. things to take care of: 
# when some subjects have not been inactivated on both hemispheres
# output is 1) scores 2) parameters

# %%
from utils.av_dat_utils import get_region_data
trials = get_region_data(region='SC')


# %%
