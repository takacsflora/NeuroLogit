import os
import sys
import itertools
import numpy as np
from src.batch.encoding import fit_session, get_time_params
from utils.av_dat_utils import get_ephys_dataset

from pathlib import Path

def train_linears(rank=1):
    """
    primarily a parallelisable finction to fit ddms
    
    """
    rank = float(rank)
    mycwd = Path(os.getcwd()) # this I might change ... 

    fit_type = 'passive' # could make this in loop
    dataset = 'all'  # could make this in loop?
    time_window = 'stim_bin' # could make this in loop
    bin_size = 0.1 
    pre_times = np.arange(-0.2,0.5,bin_size)

    sessions = get_ephys_dataset(dataset)

    for i,(pre_time,(_,args)) in enumerate(itertools.product(pre_times,sessions[['subject','date']].iterrows())):
        print(i,pre_time,args)
        post_time = pre_time+bin_size        

        # if 'bin' in time_window:
        #     timings = f'{time_window}'#_pre_{pre_time}_post_{post_time}'
        # else:
        #     timings = f'{time_window}'    

        # # get the folders for saving the results
        # savepath = mycwd / 'linear_fit_results'/f'{dataset}_{fit_type}_{timings}'
        # savepath.mkdir(parents=True,exist_ok=True)        
        # sessionID = f'{args.subject}_{args.date}'
        # result_coef_path  = savepath / f'{sessionID}.csv'

        # if (i==(rank-1)) and not result_coef_path.is_file():
        #     time_kwargs = get_time_params(time_window,pre_time=abs(pre_time),post_time=post_time)  
        #     print(f'fitting {sessionID} ...')      
        #     # coefs = fit_session(fit_type = fit_type,**args,**time_kwargs)
        #     print(f'saving results to {result_coef_path}')
        #     # coefs.to_csv(result_coef_path)

if __name__ == "__main__":  
   train_linears(rank=sys.argv[1]) 
