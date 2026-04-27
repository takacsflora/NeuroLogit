import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, log_loss

def raise_to_sigm(logOdds):
    "function to calculate pR from logOdds"
    return np.exp(logOdds) / (1 + np.exp(logOdds))

def add_behav_to_ev(ev,subject,date):
    """add the behavioural model parameters to the event dataframe

    Args:
        ev (pd.DataFrame): event dataframe
        subject (str): subject ID
        date (str): date of the session
    """

    behav_params = pd.read_csv(r'C:\Users\Flora\Documents\Github\NeuroLogit\data\behaviour\logit_params_per_session_ephys.csv')
    behav_params = behav_params[(behav_params.subject==subject) & (behav_params.date==date)].iloc[0]
    ev['visDiff_gamma'] = ev.visR**behav_params.gamma - ev.visL**behav_params.gamma

    ev['visR_gamma'] = ev.visR**behav_params.gamma
    ev['visL_gamma'] = ev.visL**behav_params.gamma

    ev['logOdds_stim'] = (
        ev.visR**behav_params.gamma * behav_params.visR -
        ev.visL**behav_params.gamma * behav_params.visL +
        ev.audR * behav_params.audR -
        ev.audL * behav_params.audL 
    )

    ev['gamma'] = behav_params.gamma

    ev['logOdds_behav'] = ev['logOdds_stim'] + (behav_params.biasR - behav_params.biasL)
    ev['proba_right_behav'] = raise_to_sigm(ev['logOdds_behav'])
    ev['predicted_choice_behav'] = ev['proba_right_behav'].apply(lambda x: 1 if x > 0.5 else 0)
    ev['logOdds_stim_category'] = pd.cut(ev.logOdds_stim, bins=[-np.inf, -1, -0.5, 0.5, 1,np.inf], labels=np.linspace(-1,1,5))
    return ev

def select_clusters_from_main_region(clusters):
    """
    function filters out noise and non-somatic spikes (keeps mua and good)
    and select the region with the most neurons (to avoid mixing regions) for the session
    """
    clusters_sel = clusters[
        (clusters.bombcell_class.isin(['mua','good'])) & 
        (clusters.BerylAcronym.isin(['SCm','SCs','MOs'])) 
    ].copy()


    # Determine which BerylAcronym has the highest counts
    top_acronym = clusters_sel['BerylAcronym'].value_counts().idxmax()

    # Filter clusters to only include those with the top BerylAcronym
    clusters_sel = clusters_sel[clusters_sel['BerylAcronym'] == top_acronym]
    
    return clusters_sel,top_acronym

def pairwise_auc_roc(y_true,z,pair=[0,1]): 
    y_true_pair = y_true[y_true.isin(pair)]
    z_pair = z[y_true.isin(pair)]
    y_true_binary = (y_true_pair == pair[0]).astype(int)

    return roc_auc_score(y_true_binary, z_pair)