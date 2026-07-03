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


def read_files(which_result = 'ccCP_results',filestub='clusters',extension='csv', sessions = None, npz_dat_type = None, psth_condition = None):
    """
    sesssions is either 'unique', None or a list of strings that correspond to the session folder names to load. Use None if  you want to load all sessions.
    psth_dat_type is one of the keys in the npz file if extension is npz -- either mean,sem or tscale atm
    psth_condition is a list of tuples (vis_stim,vis_stim,choice) to index the psth dict in the npz file
    e.g. (0,0,'Right') for psth for trials with no visual or auditory stimulus and a right choice. The requested conditions will be averaged together. 
    """
    result_path = Path(rf'D:\AV_Neural_Data_Sept2025\{which_result}')
    # maybe I should name my files better -- becaus this doesn't work for decoding but if I change it it will read in Ridge 10 and 100 equally...
    
    files = list(result_path.rglob(f'*{filestub}.{extension}'))
    if len(files) == 0:
        print('trying with allowing the * filestub *, caution... ')
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

            if (npz_dat_type=='tscale') or (npz_dat_type=='feature_column_dict') or (npz_dat_type=='feature_tscale_dict'): # only one tscale per session so just get the first one 
                f = files[0]
                tscale = np.load(f,allow_pickle=True)[npz_dat_type]
                dfs = tscale
            elif (npz_dat_type=='hemis') or (npz_dat_type=='cIDs') or (npz_dat_type=='coefficients'): # only one hemi per session so just get the first one
                dfs = []
                for f in files:
                    array_data = np.load(f,allow_pickle=True)[npz_dat_type]
                    dfs.append(array_data)
                dfs = np.concatenate(dfs)
            
            else: 
                dfs = []
                for f in files:
                    psths = np.load(f,allow_pickle=True)[npz_dat_type].item()
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
        # 'MOs': "#17c071", # orange
        # 'SCm': "#f738e4", # blue
        # 'SCs': "#1cc0dd", # green
        'MOs': "#6f916f", # orange
        'SCm': "#7137c8ff", # blue
        'SCs': "#55ddffff", # green
    }

    return region_colors