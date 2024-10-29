

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

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

def filt_split_trials(trials):
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
    n_trials = trials.bias_opto.value_counts().min()
    trials_ctrl = trials[trials.bias_opto == 0].sample(n_trials*2, random_state=1)
    trials_opto = trials[trials.bias_opto == 1].sample(n_trials, random_state=1)
    trials = pd.concat([trials_ctrl, trials_opto])

    X = trials[all_predictors]
    y = trials["choice"]
    stratifyIDs = trials.trialtype_id
    # stratifyIDs = stratifyIDs.fillna(100) # nan means control trials

    # balance opto & non-opto trials

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1, shuffle=True, stratify=stratifyIDs
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

