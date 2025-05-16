from pathlib import Path
import numpy as np
import pandas as pd


def get_source_folder(dat_type='trial_data'):

    if 'zcbtfta' in str(Path.home()):
        home_rep = '/lustre/home/zcbtfta'
        source_folder = f'{home_rep}/AV_Neural_data/{dat_type}'


    else:
        home_rep = 'D:'
        source_folder = f'{home_rep}\\AV_Neural_Data\\{dat_type}'


    return Path(source_folder)

def get_ephys_dataset(set_name,subset=''):
    """
    function that finds all sessions within a dataset based on trial data

    """

    source_folder = get_source_folder('trial_data')


    if set_name=='all':
        sessions = list(source_folder.glob(f'{subset}*.csv'))
    # parse each session's namestring to subject and date
        session_info = [session.stem.split('_') for session in sessions]
        subjects = [info[0] for info in session_info]
        dates = [info[1] for info in session_info]

        # create df of all the sesions with subject,date and folder 
        df = pd.DataFrame({'subject':subjects,'date':dates})


    else:

        source_folder = get_source_folder('meta_data')
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

def add_average_to_ev(ev,raster,pre_time = 0.1,post_time = 0.15):
    """
    function that adds the average response to the event data
    """
    # filter the raster data for the desired timepoints
    raster_filtered = raster[(raster['Time'] >= -pre_time) & (raster['Time'] <= post_time)]
    # group by neuron and trial, then calculate the average response
    average_activity = raster_filtered.groupby(['Feature', 'Trial'])['Response'].mean().unstack()
    # keep only features that have 'neuron' in them from the average activity
    #neuron_average_activity = average_activity.loc[average_activity.index.str.contains('neuron')]

    # merge the average activity with the event data

    assert ev.shape[0]==average_activity.shape[1], 'cannot unite as evetns and rasters have different number of trials'
    ev = ev.merge(average_activity.T, left_index=True, right_index=True)

    return ev

def preproc_events_data(ev):
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
                (ev.stim_audAzimuth.isin([-60,0,60])) &  # in some later mice we have 30 degrees but I don't analyse those
                 ~(np.isnan(ev.timeline_audPeriodOn)) 
                ].copy()
    

    # basically numerous combinations of spl and contrast have been trialled during the ephys recordings. 

    spl = ev.stim_audAmplitude.unique()[-1]
    
    if spl==0.25:
            ev = ev[ev.stim_visContrast.isin([-0.25,-0.1,-0.05,0,0.05,0.1,0.25]) &
                    (ev.stim_audAmplitude==0.25)
                    ].copy()
    elif spl==0.1:
            ev = ev[ev.stim_visContrast.isin([-0.4,-0.2,0.1,0,0.2,0.4]) & 
                    (ev.stim_audAmplitude==0.1)
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

    # maybe make sure to return indices? 
    return ev

def load_trial_data(subject,date,load_clusters = True,load_raster = None,avg_kwargs=None):
    """function to call the a particular sessions' trials and clusters data

    Args:
        set_name (str): which dataset to use
        subject (str): subject name, Defaults to None.
        date (str), Defaults to None.

    Returns:
        _type_: _description_
    """    

    trial_data_source = get_source_folder('trial_data')
    cluster_data_source = get_source_folder('cluster_data')
    raster_data_source = get_source_folder('raster_data')

    session = f'{subject}_{date}'


    df = pd.read_csv((trial_data_source / f'{session}.csv'),low_memory=False)

    
    if load_clusters:
        # deal with the cluster data
        clusters = pd.read_csv((cluster_data_source / f'{session}.csv'),low_memory=False)
        # adding this line to create common column for merging
        clusters['neuronID'] = clusters['_av_IDs'].apply(lambda x: f'neuron_{x}')
    else: 
        clusters = None

    # deal with the spike data

    if load_raster is not None:
        raster_path = raster_data_source / load_raster / f'{session}.csv'
        rasters = pd.read_parquet(raster_path)
    else: 
        rasters = None
        
    if avg_kwargs is not None: 
        df = add_average_to_ev(df,rasters,**avg_kwargs)

    df = preproc_events_data(df)

    # ensure that the raster data has he same trial indices left as the event data
    if rasters is not None:
        rasters = rasters[rasters['Trial'].isin(df.index)]   
    
    return { 
        'ev':df,
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

