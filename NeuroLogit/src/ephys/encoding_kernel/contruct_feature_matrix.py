
import numpy as np

def get_kernel_params(ev,which='aud_onset',max_pre_time=0.2,max_post_time=0.6,recorded_hemisphere = 'right', support_end = 'move'): 
    n_trials = len(ev)
    all_trials = np.ones(n_trials) # for trials where the support is the same on all trials
    ev = ev.copy()
    ev['rt_max'] = ev.rt.fillna(max_post_time)
    ev['choice_regressed'] = ev['timeline_choiceMoveDir'].replace({2:1,1:-1})

    if recorded_hemisphere == 'left':
        pass
    elif recorded_hemisphere == 'right':
        ev.visDiff_categorical *=-1
        ev.audDiff_categorical *=-1
        ev.choice_regressed *=-1

    if which=='b': 
        params =  {'pre':all_trials*max_pre_time,'fill_value':all_trials*1,'is_temporal':False, 'trial_type':all_trials==1}
    elif which=='aud_onset': 
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':(ev.stim_audAmplitude>0).values}
    elif which=='engagement':
        params =  {'pre':all_trials*max_pre_time,'fill_value':all_trials*1,'is_temporal':False, 'trial_type':(ev.session=='active').values}
    elif which=='aud_ipsi':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':((ev.stim_audAmplitude>0) & (ev.audDiff_categorical==-1)).values}
    elif which=='aud_contra':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':((ev.stim_audAmplitude>0) & (ev.audDiff_categorical==1)).values}
    elif which=='vis_ipsi_1.0':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':((ev.visDiff_categorical==-1)).values}
    elif which=='vis_ipsi_0.5':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':((ev.visDiff_categorical==-0.5)).values}
    elif which=='vis_ipsi_0.25':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':((ev.visDiff_categorical==-0.25)).values}
    elif which=='vis_contra_0.25':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':((ev.visDiff_categorical==0.25)).values}
    elif which=='vis_contra_0.5':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':((ev.visDiff_categorical==0.5)).values}
    elif which=='vis_contra_1.0':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':((ev.visDiff_categorical==1)).values}
    elif which=='action':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type': ~(np.isnan(ev.choice_regressed)).values}
    elif which=='choice_ipsi':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':((ev.choice_regressed==-1)).values}
    elif which=='choice_contra':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':((ev.choice_regressed==1)).values}

    if support_end=='move':
        params['post'] = ev.rt_max.values
    else:
        params['post'] = all_trials*max_post_time

    # whether the kernel varies per each trial
    params['is_varying_fill'] = True if len(np.unique(params['fill_value']))>1 else False 
    params['is_varying_time'] = True if len(np.unique(params['post']))>1 else False

    return params

def contrstruct_toeplitz(ev,tscale,**kernel_kws):

    # for each kernel, we need to construct the toeplitz across all trials
    # identity matrix for temporal kernels 
    n_timepoints = len(tscale)
    n_trials = len(ev)
    p = get_kernel_params(ev=ev,**kernel_kws)


    if p['is_temporal']:
        toeplitz = np.zeros((n_trials,n_timepoints,n_timepoints)) # n_trials x n_timepoints x n_timepoints
    else:
        toeplitz = np.zeros((n_trials,1,n_timepoints)) # n_trials x n_timepoints

    full_t = np.eye(n_timepoints) # full temporal identity matrix
    full_row = np.ones(n_timepoints) # full row of ones for non-temporal kernels

    for i in range(n_trials):
        if p['trial_type'][i]:
            fill_value = p['fill_value'][i]  
            pre_support = p['pre'][i]
            post_support = p['post'][i]      
            # determine supported time indices 
            support_t_idx = (tscale>=-pre_support) & (tscale<=post_support)
            if p['is_temporal']:
                toeplitz[i,support_t_idx,support_t_idx] = full_t[support_t_idx,support_t_idx]*fill_value
            else:
                toeplitz[i,0,support_t_idx] = full_row[support_t_idx]*fill_value



    return toeplitz