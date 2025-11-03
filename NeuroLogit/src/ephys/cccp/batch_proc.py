#%%



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from pathlib import Path

from src.ephys.dat_utils import load_trial_data,smooth_raster,get_ephys_dataset
import src.ephys.cccp.stat as cp


def get_discrimination_period(which):
    if which=='prestim':
        return {'load_raster':'stim','avg_kwargs':{'pre_time':0.15,'post_time':0.0}}
    elif which=='stim':
        return {'load_raster':'stim','avg_kwargs':{'pre_time':0.0,'post_time':0.15}}
    elif which=='choice':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.15,'post_time':0.0}}
    elif which == 'choice0':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':-0.05,'post_time':0.1}}
    elif which == 'choice1':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.0,'post_time':0.05}}
    elif which == 'choice2':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.05,'post_time':0.0}}
    elif which == 'choice3':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.1,'post_time':-0.05}}
    elif which == 'choice4':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.15,'post_time':-0.1}}
    elif which == 'choice5':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.2,'post_time':-0.15}}
    elif which == 'choice6':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.25,'post_time':-0.2}}
    else:
        raise ValueError(f"unknown discrimination period {which}")

def get_included_trials(df,which='passive'):
    if which=='passive':
        df = df[df.session=='passive'].copy()
    elif which=='Go':
        df = df[df.choice.isin([0,1]) & df.rt>0.15].copy()
    return df

def get_combinations():
    to_discriminate = ['vis','aud']
    discrimination_periods = ['prestim','stim']
    included_trials = ['Go','passive']

    permutations = list(product(to_discriminate, discrimination_periods, included_trials))
    permutations.extend([('choice','prestim','Go'),('choice','choice','Go')])

    temporal_permutations = [('choice',f'choice{i}','Go') for i in range(7)]
    permutations.extend(temporal_permutations)

    return permutations

def get_cps_for_session(subject,date):
    """run ccCP for all combinations of to_discriminate, discrimination period and included trials

    Args:
        subject (str): subject ID
    """
    to_test = get_combinations()

    #
    all_probs = []
    for d,d_t,d_i in to_test:
        time_kws = get_discrimination_period(d_t)
        df,_,_ = load_trial_data(subject,date,
                                    load_clusters=False,**time_kws).values()

        df = get_included_trials(df,which=d_i)

        prob = cp.run_combined_condition_U(df,to_discriminate=d)
        prob = cp.check_cp_significance(prob, to_discriminate=d, p_thr= 0.01,std_thr=0.05)
        stub = f'{d_t}_{d_i}'
        # add stub to all columns except 'neuronID'
        prob.columns = [f'{c}_{stub}' if c!='neuronID' else c for c in prob.columns]

        all_probs.append(prob)

    all_probs = pd.concat(all_probs, axis=1)
    
    all_probs.reset_index(inplace=True)

    all_probs.rename(columns={'index': 'neuronID'}, inplace=True)
    all_probs.rename(columns={'index': 'neuronID'}, inplace=True)
    return all_probs

def process_session(subject,date,session_path):
    """process one session to get ccCP results and selected psths

    Args:
        subject (str): subject ID
        date (str): session date
    """
    # load cluster data for the session as we are going to stitch to that
    _,clusters,_ = load_trial_data(subject,date,
                                load_clusters=True,load_raster=None).values()


    # clusters with ccCP results

    all_probs = get_cps_for_session(subject,date)
    clusters = clusters.merge(all_probs,on='neuronID',how='left')
    clusters['subject'] = subject
    clusters['date'] = date
    clusters['session'] = f'{subject}_{date}'
    clusters.to_csv(session_path / 'clusters_ccCP.csv',index=False)

def run_all_sessions(recompute=False):
    """run all sessions to get ccCP results and selected psths

    Args:
        subject (str): subject ID
    """
    # load data for the session

    df = get_ephys_dataset()
    sessions = df[['subject','date']].drop_duplicates().values
    savepath = Path(r'D:\AV_Neural_Data_Sept2025\ccCP_results')
    for subject,date in sessions:
        session_path = savepath / f'{subject}_{date}'

        if (not session_path.exists()) or recompute:
            print(f'processing {subject} {date}')
            try:
                session_path.mkdir(exist_ok=True,parents=True)
                process_session(subject,date,session_path)
            except Exception as e:
                print(f'error processing {subject} {date}: {e}')
                with open(session_path / 'error_log.txt', 'w') as f:
                    f.write(str(e))
        else:
            print(f'skipping {subject} {date} because already processed')
            continue

run_all_sessions(recompute=False)
