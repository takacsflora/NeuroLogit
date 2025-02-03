

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from pinkrigs_tools.utils.ev_utils import normalize_event_values

def get_paths(set_name):
    """
    Creates standard path structure for data_management and can recall Paths
    Parameters: 
        set_name: str
            identifier of the dataset that points to the raw data 

    Returns:
        basepath,formatted_path,savepath 
        pathlib.Paths 
        raw data, processed data for fitting, results of fitting
    """
    basepath = Path(f"D:\LogRegression\{set_name}")
    formatted_path = basepath / 'formatted'
    formatted_path.mkdir(parents=False,exist_ok=True)
    savepath = formatted_path / 'fit_results'
    savepath.mkdir(parents=False,exist_ok=True)

    return basepath,formatted_path,savepath

def preproc_av_opto_data(set_name = r'opto\Rinberg'):
    """
    matlab query module saves out a bunch of data. This function make these datasets uniform and formatted and must be fun prior to all new fitting procedure
    """
    basepath,formatted_path,_ = get_paths(set_name)
    sessions = list(basepath.glob('*.csv'))


    for cpath in sessions:

        ev = pd.read_csv(cpath)
        
        # determine whether data is the SC or the old data because the variable names are different
        # so need to extract visdiff, audDiff and choice differently
        if 'timeline_isMovedAtStim' in ev.columns:
            dataset_type = 'PinkRigs'
        else:
            dataset_type = 'CoenSit' 


        trials = pd.DataFrame()

        if dataset_type=='CoenSit':
            trials['choice'] = ev.response_direction - 1 

            #normalise to max
            maxV = np.max(np.abs(ev.stim_visDiff))
            maxA = np.max(np.abs(ev.stim_audDiff)) 

            trials['visDiff'] = ev.stim_visDiff/maxV
            trials['audDiff'] = ev.stim_audDiff/maxA

            ev.loc[~ev.is_laserTrial.astype(bool), 'hemisphere'] = np.nan
            trials['hemisphere'] = ev.hemisphere 
            # rewrite nan when nothing is inactivated


        if dataset_type=='PinkRigs':
            
            # extracting choice
            
            # filter out the trials when the animal was moving prior to laser onset
            ev = ev[~ev.timeline_isMovedAtLaser.astype('bool')]
            # further convert trials when the animal was already moving at stimulus presentation
            moved_at_stim = ev.timeline_isMovedAtStim.astype('bool')
            choice = ev.response_direction        
            # when the animal already made a choice prior to stimulus presentation
            assert (choice[moved_at_stim]!=0).all(),'there are some nogos that moved..?'
            choice[moved_at_stim] = choice[moved_at_stim] + 2
            trials['choice'] = choice-1

            # extracting stim
            maxV = np.max(np.abs(ev.stim_visDiff))
            maxA = np.max(np.abs(ev.stim_audAzimuth))        
            trials['visDiff']=ev.stim_visDiff/maxV
            trials['audDiff']=ev.stim_audAzimuth/maxA

            trials['hemisphere'] = np.sign(ev.laser_power_signed)
            

        # things tha tare common in all datasets 
        
        trials['feedback'] = ev.response_feedback
        trials['opto'] = ev.is_laserTrial

        # some further processing so that the predictors and visL/visR
        trials['visR']=np.abs(trials.visDiff)*(trials.visDiff>0)
        trials['visL']=np.abs(trials.visDiff)*(trials.visDiff<0)
        trials['audR']=(trials.audDiff>0).astype('int')
        trials['audL']=(trials.audDiff<0).astype('int')

        # opto predictors for each
        trials['visR_opto'] = trials.visR * trials.opto
        trials['visL_opto'] = trials.visL * trials.opto
        trials['audR_opto'] = trials.audR * trials.opto
        trials['audL_opto'] = trials.audL * trials.opto
        trials['trialtype_id'] = trials.copy().groupby(['visDiff','audDiff']).ngroup()

        #IDs about sessions,hemispheres, mice etc.
        trials['subject']=ev.subjectID_    
        trials['sessionID'] = ev.sessionID

        # add the bias predictors if we want to use them.
        trials['bias_opto'] = trials.opto
        trials['bias'] = 1



            # make sure that each class of trials will have min 2 types for splitting
        uniqueIDs,test_counts = np.unique(trials.trialtype_id,return_counts=True)

        if (test_counts<2).any():
            print('In %s I am dropping some trial types...' % cpath)
            rare_trialtypes = uniqueIDs[test_counts<2]
            for rareID in rare_trialtypes:
                trials = trials[trials.trialtype_id!=rareID]

        # for now we don't use the 
        trials.to_csv((formatted_path / cpath.name))

def filt_split_trials(trials, test_size = 0.33, balance_sensory = True,balance_control=False):
    # 
    stim_predictors = ["visR", "visL", "audR", "audL", "bias"]
    
    opto_predictors = [
        "visR_opto",
        "visL_opto",
        "audR_opto",
        "audL_opto",
        "bias_opto",
        "hemisphere",
    ]

    all_predictors = stim_predictors + opto_predictors

    # filter the trial matrix
    trials = trials[
        (trials.choice == 0) | (trials.choice == 1)
    ]  # keep only the post-stim correct trials
    if balance_control: 
        n_trials = trials.bias_opto.value_counts().min()
        trials_ctrl = trials[trials.bias_opto == 0].sample(n_trials*2, random_state=1)
        trials_opto = trials[trials.bias_opto == 1].sample(n_trials, random_state=1)
        trials = pd.concat([trials_ctrl, trials_opto])

    X = trials[all_predictors]
    y = trials["choice"]

    if balance_sensory:
        stratifyIDs = trials.trialtype_id
    else:
        stratifyIDs = None
    # stratifyIDs = stratifyIDs.fillna(100) # nan means control trials

    # balance opto & non-opto trials

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= test_size, random_state=1, shuffle=True, stratify=stratifyIDs
    )

    return trials, X_train, X_test, y_train, y_test
    
def get_benchmark_opto_dataset(region = 'SC',subject=1):
    """
    allows the easy call of an example dataset, i.e. subject 1
    """
    dat_path = rf"D:\LogRegression\opto\Rinberg\formatted\{region}.csv"
    trials = pd.read_csv(dat_path)
    trials_of_subject = trials[trials.subject == subject]

    return filt_split_trials(trials_of_subject)


def get_ephys_dataset(set_name):
    """
    function that finds all sessions within a dataset based on trial data

    """

    # this is the old structure
    #source_folder = f'D:\AVTrialData\{set_name}\\trial_data'
    
    if 'lustre' in str(Path.home()):
        home_rep = '/lustre/home/zcbtfta'
    else:
        home_rep = 'D:'

    if set_name=='all':
        source_folder = f'{home_rep}\AV_Neural_Data\\trial_data'
        sessions = list(Path(source_folder).glob('*.csv'))
    # parse each session's namestring to subject and date
        session_info = [session.stem.split('_') for session in sessions]
        subjects = [info[0] for info in session_info]
        dates = [info[1] for info in session_info]

        # create df of all the sesions with subject,date and folder 
        df = pd.DataFrame({'subject':subjects,'date':dates})


    else:
        meta_data = pd.read_csv(f'{home_rep}\AV_Neural_Data\\meta_data\\{set_name}_sessInfo.csv')
        df = pd.DataFrame({'subject':meta_data.subject.values,'date':meta_data.expDate.values})
    




    # add set_name to the df    
    df['set_name'] = set_name

    return df


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

def preproc_events_data(df):
    # Create a new column 'session' to indicate active or passive session
    df['session'] = np.where(df['choice'].isna(), 'passive', 'active')

    # Create a new column 'choice_categorical' to categorize choices
    df['choice_categorical'] = df['choice'].map({-1: 'NoGo', 0: 'left', 1: 'right'})
    # Update 'choice_categorical' to 'passive' where 'choice' is NaN
    df['choice_categorical'] = df['choice_categorical'].fillna('passive')



    # and  we return only the stimulus combinations that are also present in active
    # add to only < 40% vis contrast too and normalise always to the same number

    # i.e. audAzimuth = [60,0,-60] and visAzimuth = [60,-60]
    df = df[(df['stim_audAzimuth'].isin([60,0,-60])) & 
            (df['stim_visAzimuth'].isin([60,-60]) | df['stim_visAzimuth'].isna()) & 
            (df['stim_visContrast'] <= 0.4)
            ]

    # recalculate the normalised values (visDiff, audDiff) for the remainder of stimulus combinations
    df = normalize_event_values(df,maxV = None, maxA = None)

    # Overwrite NaNs in 'choice' column with -2
    df['choice'].fillna(-2, inplace=True)

    # maybe make sure to return indices? 
    return df


def load_trial_data(subject,date,load_clusters = True,load_raster = None,avg_kwargs=None):
    """function to call the a particular sessions' trials and clusters data

    Args:
        set_name (str): which dataset to use
        subject (str): subject name, Defaults to None.
        date (str), Defaults to None.

    Returns:
        _type_: _description_
    """    

    source_folder = f'D:\AV_Neural_Data'

    session = f'{subject}_{date}'


    # deal with the trial data
    df = pd.read_csv(fr'{source_folder}\trial_data\{session}.csv',low_memory=False)

    
    if load_clusters:
        # deal with the cluster data
        clusters = pd.read_csv(fr'{source_folder}\cluster_data\{session}.csv',low_memory=False)
        # adding this line to create common column for merging
        clusters['neuronID'] = clusters['_av_IDs'].apply(lambda x: f'neuron_{x}')
    else: 
        clusters = None

    # deal with the spike data

    if load_raster is not None:
        rasters = pd.read_parquet(fr'{source_folder}\raster_data\{load_raster}\{session}.csv')
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
        ev = ev[ev['session']=='passive']
        ev['baseline'] = 1

    elif fit_type == 'engagement':
        ev = ev[ev['choice_categorical']!='NoGo']
        ev['baseline'] = 1
        ev['is_active'] = (ev['session'] == 'active').astype(float)

    elif (fit_type == 'choice')|(fit_type == 'choice_engagement'):
        ev = ev[ev['choice_categorical']!='NoGo']
        ev['baseline'] = 1
        ev['is_active'] = (ev['session'] == 'active').astype(float)

        # also if response direction was prior to stimulus onset it will appear as nan in cohiceMoveOn but not in choice, so have to filter out
        ev = ev[(~ev.timeline_choiceMoveOn.isna())|(ev.session=='passive')]

    elif (fit_type == 'choice_logit'):

        ev = ev[(~ev.timeline_choiceMoveOn.isna())]

    else:
        print('fit_type not recognised')

    return ev