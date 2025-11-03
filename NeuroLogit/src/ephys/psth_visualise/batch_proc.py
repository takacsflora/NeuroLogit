#%% save out various psths from the data that I can use for visualisation later

# supresss printing warming related to zscoring (the result should be just nan for 0 std anyway)
# i.e. supress all RuntimeWarnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from pathlib import Path

from src.ephys.dat_utils import load_trial_data,smooth_raster,get_ephys_dataset

PSTH_SAVE_PATH = Path(r'D:\AV_Neural_Data_Sept2025\psth_visualisations')

def raster_to_mean_sem(raster,tscale,**smoothing_kws):
    mean = np.nanmean(raster, axis=0)
    sem = np.nanstd(raster, axis=0) / np.sqrt(raster.shape[0])
    mean_smoothed = smooth_raster(mean,tscale,**smoothing_kws)
    sem_smoothed = smooth_raster(sem,tscale,**smoothing_kws)

    return mean_smoothed, sem_smoothed

def get_psths_by_conditions(df,rasters,baseline_raster,groupby=['visDiff', 'audDiff','choice_categorical'],
                            smooth_per_trial=0.025,smooth_avg=0.025,
                            baseline_period = (-0.4,0)
                            ):
    """group rasters by the combinations of visDiff,audDiff and choice. 
    With optional zscoring based on a passed raster and time period. (that is so that we don't assume the baseline is the same raster as the inputted raster)
    which is useful if we want to zscore the raster at choice with the baseline before stim onset.

    Args:
        df (pd.DataFrame): dataframe with the trial info
        rasters (np.ndarray): raster data of shape (n_trials,n_neurons,n_timepoints)
        baseline_raster (dict): dictionary with keys 'data_binned' and 'tscale' containing the raster to use for zscoring. Defaults to None.
        groupby (list, optional): list of columns to group by. Defaults to ['visDiff', 'audDiff','choice_categorical'].
        smooth_per_trial (float, optional): smoothing kernel size for smoothing each trial. Defaults to 0.025.
        smooth_avg (float, optional): smoothing kernel size for smoothing the average psth. Defaults to 0.025.
        baseline_period (tuple, optional): time period to use for zscoring. Defaults to (-0.4,0).
        baseline_correction_type (str, optional): 'subtract' or 'zscore'. Defaults to 'subtract'.

    Returns:
        dict: dictionary with keys as (visDiff,audDiff,choice) and values as the mean raster for that condition
    """
    grouped_indices = df.groupby(groupby).indices
    
    r = rasters['data_binned']
    tscale = rasters['tscale']

    # 
    fr_mean,fr_sem = {},{}
    fr_blsub_mean, fr_blsub_sem = {},{}
    fr_zscored_mean, fr_zscored_sem = {},{}


    # compute the the baseline 
    r_bl = baseline_raster['data_binned']
    t_scale_bl = baseline_raster['tscale']
    baseline_mask = (t_scale_bl >= baseline_period[0]) & (t_scale_bl <= baseline_period[1])
    avg_baseline_per_trial = np.mean(r_bl[:, :, baseline_mask], axis=2)  # (n_trials,n_neurons)

    # now I just zscore with the total average. Could do per group if needed.
    # do nan because we propagate in trials where e.g. we could not identify the trigger time (for consistency with the trial matrix)
    mean_bl = np.nanmean(avg_baseline_per_trial,axis=0) # (n_neurons,)
    std_bl = np.nanstd(avg_baseline_per_trial,axis=0) # (n_neurons,) # or we could do it per condition
    # expand dims so that it can be subtracted from the rasters
    mean_bl = np.expand_dims(mean_bl, axis=(0, 2))  # Shape: (1, n_neurons, 1)
    std_bl = np.expand_dims(std_bl, axis=(0, 2))    # Shape: (1, n_neurons, 1)


    for group, indices in grouped_indices.items():
        group_rasters = smooth_raster(r[indices, :, :],tscale,smoothing=smooth_per_trial,kernel_dir='forward',baseline_subtract=False)
        fr_mean[group], fr_sem[group] = raster_to_mean_sem(group_rasters,tscale,smoothing=smooth_avg,kernel_dir='forward',baseline_subtract=False)
        

        group_rasters_blsub = group_rasters - mean_bl
        group_rasters_zscored = group_rasters_blsub / std_bl
        fr_blsub_mean[group], fr_blsub_sem[group] = raster_to_mean_sem(group_rasters_blsub,tscale,smoothing=smooth_avg,kernel_dir='forward',baseline_subtract=False)
        fr_zscored_mean[group], fr_zscored_sem[group] = raster_to_mean_sem(group_rasters_zscored,tscale,smoothing=smooth_avg,kernel_dir='forward',baseline_subtract=False)

    mean_rasters = {'fr': fr_mean, 'fr_blsub': fr_blsub_mean, 'zscored': fr_zscored_mean}
    sem_rasters = {'fr': fr_sem, 'fr_blsub': fr_blsub_sem, 'zscored': fr_zscored_sem}


    return mean_rasters, sem_rasters, tscale

def add_behav_odds_to_ev(ev,subject,date):
    """add the behavioural odds to the event dataframe

    Args:
        ev (pd.DataFrame): event dataframe
        subject (str): subject ID
        date (str): session date

    Returns:
        pd.DataFrame: event dataframe with added 'stimOdds' column
    """
    behav_params = pd.read_csv(r'C:\Users\Flora\Documents\Github\NeuroLogit\data\behaviour\logit_params_per_session_ephys.csv')
    behav_params = behav_params[(behav_params.subject==subject) & (behav_params.date==date)].iloc[0]

    ev['stimOdds'] = (
        ev.visR**behav_params.gamma * behav_params.visR -
        ev.visL**behav_params.gamma * behav_params.visL +
        ev.audR * behav_params.audR -
        ev.audL * behav_params.audL 
    )
    # bin the stimOdds into 5 bins 

    #ev['stimOdds_binned'] = pd.qcut(ev['stimOdds'], q=5,labels=np.linspace(-1,1,5))
    # or equal sized bins 
    ev['stimOdds_binned'] = pd.cut(ev['stimOdds'], bins=5,labels=np.linspace(-1,1,5))

    return ev

def save_session_psths(subject,date,session_path):
    """save out psths for a given session for visualisation later

    Args:
        subject (str): subject ID
        date (str): session date
        session_path (Path): path to save the psths
    """
        
    ev,clusters,rasters_stim = load_trial_data(subject,date,
                                load_clusters=True,load_raster='prestim').values()

    _,_,rasters_move = load_trial_data(subject,date,
                                load_clusters=False,load_raster='choice').values()


    cluster_hemispheres = clusters.hemi.values
    cluster_ids = clusters.neuronID.values

    ev = add_behav_odds_to_ev(ev,subject,date)

    # psths for each stimulus/choice condition
    kws = {
            'smooth_per_trial': 0.025,
            'smooth_avg': 0.025,
            'baseline_period': (-0.4,0)
            }


    groupings = {
        'stimCategory': ['visDiff_categorical', 'audDiff_categorical','choice_categorical'],
        'oddsCategory': ['stimOdds_binned','choice_categorical']
    }

    for grouping_name,grouping in groupings.items():
        mean_stim,sem_stim,tscale_stim = get_psths_by_conditions(ev,
                                                                rasters=rasters_stim,
                                                                baseline_raster=rasters_stim,
                                                                groupby=grouping,**kws)
        
        mean_move,sem_move,tscale_move = get_psths_by_conditions(ev,
                                                                rasters=rasters_move,
                                                                baseline_raster=rasters_stim,
                                                                groupby=grouping,**kws)

        # pure firing rate psths
        for k in mean_stim.keys():
            np.savez(session_path / f'{k}_psths_at_stim_{grouping_name}.npz',
                    mean=mean_stim[k], sem=sem_stim[k], tscale=tscale_stim,cIDs = cluster_ids, hemis = cluster_hemispheres)

        for k in mean_move.keys():
            np.savez(session_path / f'{k}_psths_at_move_{grouping_name}.npz',
                    mean=mean_move[k], sem=sem_move[k], tscale=tscale_move,cIDs = cluster_ids, hemis = cluster_hemispheres)

# process all sessions
def run_all_sessions(recompute=False):
    """run all sessions to get ccCP results and selected psths

    Args:
        subject (str): subject ID
    """
    # load data for the session

    df = get_ephys_dataset()
    sessions = df[['subject','date']].drop_duplicates().values
    for subject,date in sessions:
        session_path = PSTH_SAVE_PATH / f'{subject}_{date}'

        if (not session_path.exists()) or recompute:
            print(f'processing {subject} {date}')
            try:
                session_path.mkdir(exist_ok=True,parents=True)
                save_session_psths(subject,date,session_path)
            except Exception as e:
                print(f'error processing {subject} {date}: {e}')
                with open(session_path / 'error_log.txt', 'w') as f:
                    f.write(str(e))
        else:
            print(f'skipping {subject} {date} because already processed')
            continue

if __name__ == '__main__':
    run_all_sessions(recompute=False)