#%%



import numpy as np
subject = 'AV005'
date = '2022-05-12'

from NeuroLogit.src.ephys.dat_utils import load_trial_data,smooth_raster


ev,clusters,rasters_stim = load_trial_data(subject,date,
                            load_clusters=True,load_raster='prestim',include_no_sound_trials=True).values()


# if stimAudAmplitude contains 0 then rewrite its audDiff_categorical value to -100 (no sound)
ev.loc[ev.stim_audAmplitude==0,'audDiff_categorical'] = 'no_sound'
# %%
blank_trials = ev[(ev.choice_categorical=='passive') & (ev.is_blankTrial==True)].index.values

# other way 

blank_trials2 = ev[(ev.audDiff_categorical=='no_sound') & (ev.visDiff_categorical==0.0)].index.values
# %%
r = rasters_stim['data_binned'][blank_trials2,:,:]


# %%
import matplotlib.pyplot as plt
plt.imshow(np.nanmean(r,axis=0).T,aspect='auto',cmap='viridis')

# %%
blank_trialswrid = ev[(ev.audDiff_categorical=='no_sound') & (ev.visDiff_categorical==0.0)]
# %%
