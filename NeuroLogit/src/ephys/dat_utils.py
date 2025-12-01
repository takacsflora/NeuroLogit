from pathlib import Path
import numpy as np
import pandas as pd


from scipy.signal.windows import gaussian
from scipy.signal import fftconvolve


def get_source_folder():

    if 'zcbtfta' in str(Path.home()):
        home_rep = '/lustre/home/zcbtfta'
        source_folder = f'{home_rep}/AV_Neural_data/data'


    else:
        home_rep = 'D:'
        source_folder = f'{home_rep}\\AV_Neural_Data_Sept2025\\data'


    return Path(source_folder)

def get_ephys_dataset(set_name='all',subset=''):
    """
    function that finds all sessions within a dataset based on trial data

    """

    source_folder = get_source_folder()


    if set_name=='all':
        session_paths = list((f for f in source_folder.rglob(f'{subset}*') if f.is_dir()))
        sessions = [f.name for f in session_paths]
        
        session_info = [session.split('_') for session in sessions]
        subjects = [info[0] for info in session_info]
        dates = [info[1] for info in session_info]

        # create df of all the sesions with subject,date and folder 
        df = pd.DataFrame({'subject':subjects,'date':dates})


    else:

        source_folder = get_source_folder('meta_info')
        meta_data = pd.read_csv(source_folder / f'{set_name}_sessInfo.csv')
        df = pd.DataFrame({'subject':meta_data.subject.values,'date':meta_data.expDate.values})
    


    # add set_name to the df    
    df['set_name'] = set_name

    return df

def calculate_differences(ev):
        ev=ev.copy()
        ev.visDiff = ev.stim_visContrast * np.sign(ev.stim_visAzimuth)        
       
        ev.loc[np.isnan(ev.visDiff), "visDiff"] = 0

        unique_audAmps = np.unique(ev.stim_audAmplitude)
        assert (unique_audAmps!=0).sum()==1, 'more than 1 SPLs are played so audDiff is complex to calculate. Currently audDiff is just realted to auditory azimuth..'
        ev.audDiff = ev.stim_audAzimuth.copy()
        ev.loc[np.isnan(ev.audDiff), "audDiff"] = 0
        return ev

def normalize_event_values(ev,maxV=None,maxA=None):
        ev = calculate_differences(ev)
        
        if maxV is None:
                maxV = np.max(np.abs(ev.visDiff)) 

        if maxA is None:
                maxA = np.max(np.abs(ev.audDiff))
        ev['visDiff']=ev.visDiff/maxV
        ev['audDiff']=ev.audDiff/maxA
        # also the option to lateralise them
        ev['visR']=np.abs(ev.visDiff)*(ev.visDiff>0)
        ev['visL']=np.abs(ev.visDiff)*(ev.visDiff<0)
        ev['audR']=np.abs(ev.audDiff)*(ev.audDiff>0)
        ev['audL']=np.abs(ev.audDiff)*(ev.audDiff<0)

        if hasattr(ev,'response_direction'): 
                ev['choice'] = ev.response_direction-1
                ev['feedback'] = ev.response_feedback
        return ev

# average firing rates
def add_average_to_ev(ev,raster,pre_time = 0.1,post_time = 0.15):
    """
    function that adds the average response to the event data
    """
    # filter the raster data for the desired timepoints


    t_idxs = (raster['tscale'] >= -pre_time) & (raster['tscale'] <= post_time)
    responses = raster['data_binned'][:,:,t_idxs]
    average_activity = np.mean(responses,axis=-1)
    name_features = [f'neuron_{int(n)}' for n in raster['cscale']]

    # group by neuron and trial, then calculate the average response
    average_activity = pd.DataFrame(average_activity,columns=name_features)

    # merge the average activity with the event data

    assert ev.shape[0]==average_activity.shape[0], 'cannot unite as evetns and rasters have different number of trials'
    ev = ev.merge(average_activity, left_index=True, right_index=True)

    return ev

def preproc_events_data(ev, include_no_sound_trials = False):
    # Create a new column 'session' to indicate active or passive session
    ev['session'] = np.where(ev['choice'].isna(), 'passive', 'active')
    # Replace NaNs in 'is_validTrial' with True (as atms nans are the passive trials basically)
    ev['is_validTrial'].fillna(True, inplace=True)

    # Create a new column 'choice_categorical' to categorize choices
    ev['choice_categorical'] = ev['choice'].map({-1: 'NoGo', 0: 'left', 1: 'right'})
    # Update 'choice_categorical' to 'passive' where 'choice' is NaN
    ev['choice_categorical'] = ev['choice_categorical'].fillna('passive')



    # Create a new column 'is_validTrial' to indicate valid trials    
    # there have been essentailly two experiments that I want to keep
    noGos =  (ev.response_direction==0).astype('int')
    window_size = 3 
    isnogo_block_conv = np.convolve(noGos, np.ones(window_size, dtype=int), mode='valid') >= window_size
    isnogo_block = np.concatenate([isnogo_block_conv, np.zeros(window_size - 1, dtype=bool)])

    # how about valid trials -- in the active ...?

    ev = ev[(ev.is_validTrial) &
                (~isnogo_block) & 
                (ev.stim_audAzimuth.isin([np.nan,-60,0,60])) &   # in some later mice we have 30 degrees but I don't analyse those
                (ev.stim_visAzimuth.isin([np.nan,-60,60]))
                ].copy()
    
    if not include_no_sound_trials:
         ev = ev[~(np.isnan(ev.timeline_audPeriodOn))].copy()

    # basically numerous combinations of spl and contrast have been trialled during the ephys recordings. 

    spl = np.sort(ev.stim_audAmplitude.unique())[-1]
    
    if spl==0.25:
            if include_no_sound_trials:
                 aud_cond = [0,0.25]
            else:
                 aud_cond = [0.25]
            
            ev = ev[ev.stim_visContrast.isin([-0.25,-0.1,-0.05,0,0.05,0.1,0.25]) &
                    (ev.stim_audAmplitude.isin(aud_cond))
                    ].copy()
    elif spl==0.1:
            if include_no_sound_trials:
                 aud_cond = [0,0.1]
            else:
                 aud_cond = [0.1]
            
            ev = ev[ev.stim_visContrast.isin([-0.4,-0.2,-0.1,0,0.1,0.2,0.4]) & 
                    (ev.stim_audAmplitude.isin(aud_cond))
                    ].copy()
    
    else:
            return None
    

    maxV = np.max(np.abs(ev.visDiff))
    maxA = np.max(np.abs(ev.audDiff))

    if (maxV==0) or (maxA==0):
        return None
    

    # recalculate the normalised values (visDiff, audDiff) for the remainder of stimulus combinations
    ev = normalize_event_values(ev,maxV = .4, maxA = None)

    # Overwrite NaNs in 'choice' column with -2
    ev['choice'].fillna(-2, inplace=True)

    # visdiff is normalised to contrast. However sometimes it just needs to be categorised for plotting. 
    # so that the highest contrast maps to 1 and the lowest to -1    
    
    def map_stim_category(value):
        visDiff_categories = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
        idx = np.argmin(np.abs(visDiff_categories - value))
        return visDiff_categories[idx]
    
    ev['visDiff_categorical'] = (ev['visDiff']/ev.visDiff.max()).apply(map_stim_category)
    ev['audDiff_categorical'] = (ev['audDiff']/ev.audDiff.max()).apply(map_stim_category) # we do this to ensire 0 is always 0 not -0 etc.
    
   # all of this could be replaces with pd.cut potentially ...


    return ev

def load_trial_data(subject,date,load_clusters = True,load_raster = None,avg_kwargs=None,include_no_sound_trials=False):
    """function to call the a particular sessions' trials and clusters data

    Args:
        set_name (str): which dataset to use
        subject (str): subject name, Defaults to None.
        date (str), Defaults to None.

    Returns:
        _type_: _description_
    """    

    data_source = get_source_folder()
    session = f'{subject}_{date}'
    session_path = data_source / session

    df = pd.read_csv((session_path / f'trials.csv'),low_memory=False)

    
    if load_clusters:
        # deal with the cluster data
        clusters = pd.read_csv((session_path / f'clusters.csv'),low_memory=False)
        # adding this line to create common column for merging
        clusters['neuronID'] = clusters['_av_IDs'].apply(lambda x: f'neuron_{x}')
    else: 
        clusters = None

    # deal with the spike data

    if load_raster is not None:
        raster_path = session_path / f'raster_{load_raster}_aligned.npz'
        rasters = np.load(raster_path,allow_pickle=True)
    else: 
        rasters = None
        
    if avg_kwargs is not None: 
        df = add_average_to_ev(df,rasters,**avg_kwargs)

    df = preproc_events_data(df, include_no_sound_trials=include_no_sound_trials)

    # ensure that the raster data has he same trial indices left as the event data
    if rasters is not None:
        idx = df.index.values
        rasters = {
            'data_binned': rasters['data_binned'][idx,:,:],
            'tscale': rasters['tscale'],
            'cscale': rasters['cscale']
        }
    
    return { 
        'ev': df.reset_index(drop=True), # reset index so that it can be used in the rasters  data again
        'clusters':clusters,
        'rasters':rasters
    }

def prepare_for_fit(ev,fit_type = None):
    """this function prepares the ev data for fitting by 
    1) filtering out the trials that are not relevant for the fit (e.g. passive or noGo trials)
    2) explicitly calculating predictors that will be needed fo the model

    Args:
        df (pd.df): events with each row as a trial
        fit_type (str): The identifier for the filtering. Defaults to None.
    """
    

    if fit_type == 'passive': 
        ev = ev[ev['session']=='passive'].copy()
        ev['baseline'] = 1

    elif fit_type == 'engagement':
        ev = ev[ev['choice_categorical']!='NoGo'].copy()
        ev['baseline'] = 1
        ev['is_active'] = (ev['session'] == 'active').astype(float)

    elif (fit_type == 'task'):
        ev = ev[ev['choice_categorical']!='NoGo'].copy()
        ev['baseline'] = 1
        ev['is_active'] = (ev['session'] == 'active').astype(float)

        # also if response direction was prior to stimulus onset it will appear as nan in cohiceMoveOn but not in choice, so have to filter out
        ev = ev[(~ev.timeline_choiceMoveOn.isna())|(ev.session=='passive')]

    elif (fit_type == 'choice_logit'):
        ev = ev[ev['choice_categorical']!='NoGo']
        ev = ev[ev['choice_categorical']!='passive']
        ev = ev[(~ev.timeline_choiceMoveOn.isna())]

    else:
        print('fit_type not recognised')

    return ev

## 
# spiking data tools 


# gaussian smoothing along the time axis with causal kernel (so half gaussian

def smooth_raster(r, tscale, smoothing=0.025, kernel_dir='forward',baseline_subtract=None,zscore = False):
    """smooth raster data along the time axis with a gaussian kernel.

    Args:
        r (np.array): raster data (trials x neurons x time)
        tscale (np.array): time scale
        smoothing (float, optional): smoothing in seconds. Defaults to 0.025.
        kernel_dir (str, optional): direction of the kernel. 'forward' for causal, 'backward' for anti-causal, 'both' for non-causal. Defaults to 'forward'.
        baseline_subtract (str or None, optional): whether to subtract the baseline. and if so, how. options:'per_trial','all_trials' or None. 
        zscore (bool, optional): whether to zscore the data after baseline subtraction. Defaults to False. (only if baseline_subtract is 'all_trials')

    Returns:
        np.array: smoothed raster data
    """
    r_smoothed = r.copy()  # No smoothing applied

    if smoothing>0: 
        w = tscale.size  
        tbin = np.diff(tscale).mean()
        window = gaussian(w, std=smoothing / tbin)

        if kernel_dir == 'forward':
            # half (causal) gaussian filter
            window[: int(np.ceil(w / 2))] = 0
        elif kernel_dir == 'backward':
            # applied for preceding events (anti-causal)
            window[int(np.floor(w / 2)) :] = 0
        window /= np.sum(window)  

        # Convolve along the last axis (time axis) using FFT for faster computation
        # Expand the window to match the dimensions of `r` except for the time axis
        window = window[(np.newaxis,) * (r.ndim - 1) + (slice(None),)]
        
        # pad with zeros at the beginning and end to avoid edge effects
        pad_width = [(0, 0)] * (r.ndim - 1) + [(w // 2, w // 2)]
        r_padded = np.pad(r, pad_width, mode="constant", constant_values=0)
        r_smoothed = fftconvolve(r_padded, window, axes=-1, mode="same")
        
        # remove the padding       
        r_smoothed = r_smoothed[(slice(None),) * (r.ndim - 1) + (slice((w // 2), -(w // 2)),)] 

        assert r_smoothed.shape == r.shape, "Smoothed data shape mismatch after padding removal."
        #r_smoothed = fftconvolve(r, window, axes=-1, mode="same")

    if baseline_subtract is not None:
        if baseline_subtract == 'per_trial':
        
            baseline = r[(slice(None),) * (r.ndim - 1) + (tscale < 0,)].mean(axis=-1, keepdims=True)  # Compute baseline modularly
            r_smoothed = r_smoothed - baseline  # Subtract baseline from smoothed data

        elif baseline_subtract == 'all_trials':
            baseline = np.nanmean(r[(slice(None),) * (r.ndim - 1) + (tscale < 0,)],axis=(0, -1), keepdims=True)  # Compute overall baseline
            r_smoothed = r_smoothed - baseline  # Subtract baseline from smoothed data
 
            if zscore:
                baseline_std = np.nanstd(r[(slice(None),) * (r.ndim - 1) + (tscale < 0,)],axis=(0, -1), keepdims=True) 
                r_smoothed = (r_smoothed - baseline) / baseline_std
            
    return r_smoothed
