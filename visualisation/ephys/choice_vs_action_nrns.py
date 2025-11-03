#%%

# plot a couple of choice neurons at the time of choice


# what is the size of the action kernel relative to choice
# do the different trial types peak at the same firing rate or not? 

# one can show this for the average activity as well as the individual neurons



from src.ephys.encoding_avg.encoding_avg import fit_dataset, get_winning_model, plot_prediction,filt_trials,get_time_params, get_predictors,get_tested_models
from src.ephys.dat_utils import load_trial_data

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from util_vis import get_component_matrix
from util_dat import read_in_all_coefs

# read in all the coeficients instead

fit_type = 'active_choice'
timing = {'time_window':'choice_bin','pre_time':0.15,'post_time':0}
coefs_all ,models_all = read_in_all_coefs(fit_type=fit_type, subset='',
                                  recompute_summary = False, recompute_parts=False,
                                  add_behav_params=True, get_best=True,
                                 **timing)



# %%

unique_sessions = models_all.sessionID.unique()
subjects = [sess.split('_')[0] for sess in unique_sessions]
dates = [sess.split('_')[1] for sess in unique_sessions]

# Determine unique subjects and dates
unique_subjects = sorted(set(subjects))
subject_dates = {subj: sorted([date for s, date in zip(subjects, dates) if s == subj]) for subj in unique_subjects}

# Prepare figure: rows = subjects, columns = max number of dates * 2 (for left/right hemispheres)
n_rows = len(unique_subjects)
n_cols = max(len(dates) for dates in subject_dates.values()) * 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), sharey=True, squeeze=False,dpi=150)
plt.rcParams.update({'font.size': 6})

for row_idx, subj in enumerate(unique_subjects):
    for col_idx, date in enumerate(subject_dates[subj]):
        sess = f"{subj}_{date}"
        models = models_all[models_all.sessionID == sess].copy()
        sess_params = {'subject': subj, 'date': date}
        time_params = get_time_params(**timing)
        df, clusters, rasters = load_trial_data(**sess_params, **time_params).values()
        df = filt_trials(df, fit_type)
        # further filtering for blank trials onlly 
        df = df[df.is_blankTrial].copy()
        rasters = rasters[np.isin(rasters.Trial, df.index)].copy()
        added_conditions = ['choice', 'visDiff', 'audDiff']
        rasters = rasters.merge(df[added_conditions], left_on='Trial', right_index=True, how='left')
        non_noise = models[((models.bombcell_class == 'good') | (models.bombcell_class == 'mua')) &
                           ((models.BerylAcronym == 'SCm')|(models.BerylAcronym=='MOs'))].copy()
        

        choice_clus = non_noise.copy()
        for i, (hemiID, hemi) in enumerate(zip([1, -1], ['right', 'left'])):
            choice_clus_hemi = choice_clus[choice_clus.hemi == hemiID].copy()
            n_clus = len(choice_clus_hemi)
            selected_activity = rasters[rasters.Feature.isin(choice_clus_hemi.neuronID)].copy()
            avg_activity = selected_activity.groupby(['choice', 'visDiff', 'audDiff', 'Time'])['Response'].mean().reset_index()
            ax = axes[row_idx, col_idx * 2 + i]
            sns.lineplot(data=avg_activity, x='Time', y='Response', hue='choice',
                         markers=True, dashes=False, ax=ax)
            ax.axvline(0, color='gray', linestyle='--', label='Choice Time')
            ax.set_xlim([-.3, 0.1])
            
            if i == 0:
                ax.set_title(f'{date},#:{n_clus}')
            else:
                ax.set_title(f'#:{n_clus}')
            if col_idx == 0 and i == 0:
                ax.set_ylabel(f'{subj},{non_noise.BerylAcronym.iloc[0]}',fontweight='bold')
            else:
                ax.set_ylabel('')
            
            
plt.tight_layout()



# %%
