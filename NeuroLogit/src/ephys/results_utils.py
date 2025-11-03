import pandas as pd 
import numpy as np 
from pathlib import Path


UNIQUE_SESSIONS = [
    'AV005_2022-05-12','AV005_2022-05-16',
    'AV007_2022-04-04',
    'AV008_2022-03-09','AV008_2022-03-11', 
    'AV013_2022-05-23', 'AV013_2022-06-10',
    'AV014_2022-06-17',
    'AV023_2022-12-12',
    'AV025_2022-11-08',
    'AV030_2022-12-07',
    'AV034_2022-12-07','AV034_2022-12-08',
    'FT031_2021-12-03',
    'FT030_2021-12-03',
    'FT032_2021-12-13','FT032_2021-12-14']



def read_files(which_result = 'ccCP_results',filestub='clusters',extension='csv', sessions = None, psth_dat_type = None, psth_condition = None):
    """
    sesssions is either 'unique'
    psth_dat_type is one of the keys in the npz file if extension is npz -- either mean,sem or tscale atm
    psth_condition is a list of tuples (vis_stim,vis_stim,choice) to index the psth dict in the npz file
    e.g. (0,0,'Right') for psth for trials with no visual or auditory stimulus and a right choice. The requested conditions will be averaged together. 
    """
    result_path = Path(rf'D:\AV_Neural_Data_Sept2025\{which_result}')
    files = list(result_path.rglob(f'*{filestub}*.{extension}'))

    if sessions=='unique':
        files = [f for f in files if f.parent.stem in UNIQUE_SESSIONS]
    elif sessions is not None:
        files = [f for f in files if f.parent.stem in sessions]

    if extension=='csv':
        dfs = pd.concat([pd.read_csv(f) for f in files])
        dfs = dfs.reset_index(drop=True)
    elif extension=='npy':
        dfs = np.concatenate([np.load(f,allow_pickle=True) for f in files])
    elif extension=='npz':

        if psth_dat_type=='tscale': # only one tscale per session so just get the first one 
            f = files[0]
            tscale = np.load(f,allow_pickle=True)[psth_dat_type]
            dfs = tscale
        elif (psth_dat_type=='hemis') or (psth_dat_type=='cIDs'): # only one hemi per session so just get the first one
            dfs = []
            for f in files:
                array_data = np.load(f,allow_pickle=True)[psth_dat_type]
                dfs.append(array_data)
            dfs = np.concatenate(dfs)
        else: 
            dfs = []
            for f in files:
                psths = np.load(f,allow_pickle=True)[psth_dat_type].item()
                if psth_condition not in psths:
                    #print(f'condition {psth_condition} not found in {f}')
                    # add nans for this file
                    example_shape = list(psths.values())[0].shape
                    dfs.append(np.full(example_shape,np.nan))
                else:
                    dfs.append(psths[psth_condition])
            dfs = np.concatenate(dfs)

    return dfs


def get_brain_region_colors():

    region_colors = {
        'MOs': "#17c071", # orange
        'SCm': "#f738e4", # blue
        'SCs': "#1cc0dd", # green
    }

    return region_colors