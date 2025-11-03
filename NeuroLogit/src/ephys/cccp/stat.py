import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata



def get_mann_whitneyU(x,y,n_shuffles=20):
    """
    function to calculate the mann-Whitney U(x) statistic for each unit
    Parameters .e.g.
    x = Right choice
    y = left choice
    n_shuffles = n of permitation sets to create

    x: np.ndarray where statistic is compared along the 1st axis (can be as many axes as one wants otherwise e.g. choice x nrn x time)
    y: same as x just for ~choice

    (reason why I input the entire parameter array is because I want to make the permutation sets the same for each nrn)

    """    

    x_y = np.concatenate((x,y),axis=0)
    nx = x.shape[0]

    v_ = np.arange(x_y.shape[0])
    shuffle_idxs = [np.random.permutation(v_)[:nx][np.newaxis,:] for s in range(n_shuffles)]
    shuffle_idxs = np.concatenate(shuffle_idxs)
    shuffle_idxs = np.concatenate((v_[:nx][np.newaxis,:],shuffle_idxs)) # add first row as the actual 
    
    # rank for each nrn
    t = rankdata(x_y,axis=0)
    t= t[shuffle_idxs]

    numer = t.sum(axis=1)
    numer = numer - (nx*(nx+1)/2)

    return numer

def combined_condition_U(spike_counts,trialChoice,trialConditions,n_shuffles=2000):
    """
    function to calculate te combined ranksum across conditions (i.e. cccp anaysis established by Steinmetz et al.)

    Parameters:
    -----------
    spike_counts: np.ndarray : trials x (can be variable dim e.g. trials x nrns or trials x nrns x time etc)
        actual values to rank
    trialChoice: bool np ndarray
        (trials) the two possibilities whose discriminability are comparing in the ROC (e.g. choice under the same stimulus condition)
    trialConditions: np.array 
        (int, trials): unique classes that define which condition the trial belongs to 
    n_shuffles: float (for cv)

    Returns: 
    --------
    : np.ndarray
        U-statistic, with regairds to whatever was considered True by trialChoice
    : np.ndarray
        p-value 
    : np.ndarray
        shuffled U-statisitics


    """
    uCond =np.unique(trialConditions)
    
    nTotal = np.zeros(((n_shuffles+1,) + spike_counts.shape[1:]))

    dTotal = 0
    for c in uCond: 
        inclT = trialConditions==c
        chA = trialChoice & inclT
        nA = chA.sum()
        chB = ~trialChoice & inclT
        nB = chB.sum()
        uA = get_mann_whitneyU(spike_counts[chA],spike_counts[chB],n_shuffles=n_shuffles)
        nTotal = nTotal+uA
        dTotal = dTotal+nA*nB

    cp = nTotal/dTotal
    t = rankdata(cp,axis=0)
    p = t[0]/(n_shuffles+1)

    return cp[0],p,cp[1:]


################## av dataset specific functions #######################
# these functions and specific to how to run ccCP on the av dataset
def get_trialtypes(df,to_discriminate='choice'):
    """function to craate disrete trial type classes for cccp analysis based on conditionst that appear on df. 
    df. should be the trial dataframe with the following columns:
    visDiff, audDiff, choice, stim_visContrast, is_visualTrial, is_auditoryTrial, is_blankTrial
    
    choice: right vs left, conditions balanced by visDiff and audDiff
    vis: vis left vs right, balanced by choice, audDiff and visContrast. Excludes auditory trials and blank trials
    aud: aud left vs right, balanced by choice, visDiff and stim_visContrast. Excludes visual trials and blank trials

    Args:
        df (pd.df): trial data 
        to_discriminate (str, optional): what to dicriminate, atm supports vis, aud and choice. Defaults to 'choice'.

    Returns:
        pd.df: fprmatted trial data with the following new columns: 
        to_discriminate (bool): whether the trial is in up down state for the particular thing we are trying to discriminate 
        trialtype (int): the class label (1-x) for the trials of the same condition
    """
    df = df.copy()
    if to_discriminate == 'choice':
        df['to_discriminate'] = df[to_discriminate].astype('bool')
        df['trialtype'] = df.groupby(['visDiff','audDiff']).ngroup()

    elif to_discriminate == 'aud':
        df = df[df.audDiff.isin([1, -1])].copy()
        df['to_discriminate'] = df['audDiff']>0
        df['trialtype'] = df.groupby(['visDiff','choice']).ngroup()

    elif to_discriminate == 'vis':
        df = df[(df.visDiff!=0)].copy()
        df['to_discriminate'] = df['visDiff']>0
        df['trialtype'] = df.groupby(['audDiff','choice','stim_visContrast']).ngroup()
    
    elif to_discriminate == 'passive':
        df['to_discriminate'] = (df['session']=='active').astype('bool')
        
        # this is done basically for active vs passive. To try to account for drift we can create artificial trial types as 
        # the first half of passive 2nd half etc.
        n_groups = 5

        active_trials = df[df['to_discriminate']]
        passive_trials = df[~df['to_discriminate']]
        n_active_trials = active_trials.shape[0]
        n_passive_trials = passive_trials.shape[0]
        # 
        min_trials = min(n_active_trials, n_passive_trials)
        active_trials = active_trials.iloc[:min_trials]
        passive_trials = passive_trials.iloc[:min_trials]

        trialtype_labels = np.repeat(np.arange(n_groups),min_trials//n_groups)
        if len(trialtype_labels)<min_trials:
            trialtype_labels = np.append(np.zeros(min_trials - len(trialtype_labels)),trialtype_labels)

        active_trials['trialtype'] = trialtype_labels.astype('int')
        passive_trials['trialtype'] = trialtype_labels.astype('int')

        df = pd.concat([active_trials, passive_trials])

    # count the number of rows in each trialtype 
    trialtype_counts = df['trialtype'].value_counts()
    # Filter out trialtypes that do not have at least 2 types of choices
    valid_trialtypes = df.groupby('trialtype')['to_discriminate'].nunique()
    valid_trialtypes = valid_trialtypes[valid_trialtypes == 2].index
    df = df[df['trialtype'].isin(valid_trialtypes)]

    return df

def run_combined_condition_U(df, to_discriminate='choice',min_trials = 10, **kws):
    """helper function to run ccCP on all neuron columns in trial data.

    Args:
        df (pd.df): trial data 
        to_discriminate (str, optional): type of ccCp to run (vis/aud/choice). Defaults to 'choice'.

    Returns:
        _type_: _description_
    """
    neuron_columns = [col for col in df.columns if 'neuron' in col]

    ccCP_df = get_trialtypes(df, to_discriminate=to_discriminate)

    if len(ccCP_df)<min_trials:
        print(f"not enough trials to run ccCP for {to_discriminate}")
        return pd.DataFrame()

    cp, p, _ = combined_condition_U(ccCP_df[neuron_columns].values,
                                   ccCP_df['to_discriminate'].values,
                                   ccCP_df['trialtype'].values,
                                   n_shuffles=2000)

    # for each neuron we will compute the variance and the average fr in the tested period
    fr_mean = ccCP_df[neuron_columns].mean()
    fr_std = ccCP_df[neuron_columns].std()

    # get the fr for each condition
    fr_per_condition = ccCP_df.groupby('to_discriminate')[neuron_columns].mean()
    fr_std_per_condition = ccCP_df.groupby('to_discriminate')[neuron_columns].std()
    # subtract True - False (from what is being discriminated)

    results= pd.DataFrame({f'p_{to_discriminate}':p,
                            f'cp_{to_discriminate}':cp,
                            f'fr_mean_{to_discriminate}': fr_mean, 
                            f'fr_std_{to_discriminate}': fr_std,
                            f'fr_Right_{to_discriminate}': fr_per_condition.loc[True], 
                            f'fr_Left_{to_discriminate}': fr_per_condition.loc[False],
                            f'fr_std_Right_{to_discriminate}': fr_std_per_condition.loc[True],
                            f'fr_std_Left_{to_discriminate}': fr_std_per_condition.loc[False],
                             })

    # calculate the average difference between the two options tested
    
    return results

def check_cp_significance(clusters,p_thr= 0.01,std_thr=0.1,to_discriminate = 'aud'):
    clusters[f'is_{to_discriminate}'] = (((clusters[f'p_{to_discriminate}'] < p_thr)| (clusters[f'p_{to_discriminate}'] > 1-p_thr)) & 
                                         (clusters[f'fr_std_Right_{to_discriminate}'] > std_thr) & 
                                         (clusters[f'fr_std_Left_{to_discriminate}'] > std_thr))
    # potentially change the fr_mean thing to fr_std <0.3 or something 
    return clusters