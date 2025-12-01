


import numpy as np

def get_kernel_params(ev,which='aud_onset',max_pre_time=0.2,max_post_time=0.6,kernel_support_end = 'move', choice_pre_time=0.2,choice_post_time=0.1): 
    n_trials = len(ev)
    all_trials = np.ones(n_trials) # for trials where the support is the same on all trials
    
    ev = ev.copy()
    ev['rt_max'] = ev.rt.fillna(max_post_time)
    # needs to be minimally 0 as we only support chouce prior to stim onset.
    ev['pre_choice_support'] = (ev.rt_max - choice_pre_time).clip(lower=0)


    assert choice_pre_time <= max_pre_time, "choice pre time cannot be longer than max pre time"

    if which=='b': 
        params =  {'pre':all_trials*max_pre_time,'fill_value':all_trials*1,'is_temporal':False, 'trial_type':all_trials==1,}
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
    elif which=='action_onset':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True, 'trial_type': ~(np.isnan(ev.choice_regressed)).values}
    elif which=='action_stim_contra':
        params =  {'pre':all_trials*0,'fill_value':all_trials*1,'is_temporal':True,'trial_type':(~(np.isnan(ev.choice_regressed)) & (ev.visDiff_categorical>0) & (ev.audDiff_categorical>0)).values}
    elif which=='choice_ipsi':
        params =  {'pre':-ev.pre_choice_support.values,'fill_value':all_trials*1,'is_temporal':True,
                    'trial_type':((ev.choice_regressed==-1)).values}
    elif which=='choice_contra':
        params =  {'pre':-ev.pre_choice_support.values,'fill_value':all_trials*1,'is_temporal':True, 'trial_type':((ev.choice_regressed==1)).values}

    if kernel_support_end=='move':    
        params['post'] = np.min([all_trials*max_post_time, ev.rt_max.values + choice_post_time],axis=0)

        # maybe I will move action onset into the else category. 
        # one option is also to have a stim contra and stim ipsi modulation
        if (which=='b') or (which=='engagement'): #or (which=='action_onset'):
            # for baseline and engagement kernels we want full post time
            params['post'] = ev.rt_max.values + choice_post_time
        if (which=='action_onset')|(which=='action_stim_contra'):
            # for action onset we want to limit to rt max
            params['post'] = all_trials*0.2  
    else:
        params['post'] = all_trials*max_post_time


    if (which=='action') or (which=='choice_ipsi') or (which=='choice_contra'):
        # overwrite post-time
        params['post'] = ev.rt_max.values + choice_post_time
        params['support_rel_to'] = 'move'
        params['support_pre'] = choice_pre_time
        params['support_post'] = choice_post_time
    else:
        params['support_rel_to'] = 'stim'
        params['support_pre'] = max(params['pre'])
        params['support_post'] = max(params['post'])

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

    support_k_t_idx = (tscale>=-p['support_pre']) & (tscale<=(p['support_post']+0.01)) # better to have even more tolerance
    kernel_tscale = tscale[support_k_t_idx]
    n_kernel_timepoints = len(kernel_tscale)

    if p['is_temporal']:
        # this needs to be corrected: should be n_trials x n_kernel_support_timepoints x n_timepoints
        full_toeplitz = np.zeros((n_trials,n_timepoints,n_timepoints)) # n_trials x n_timepoints x n_timepoints       
        toeplitz = np.zeros((n_trials,n_kernel_timepoints,n_timepoints)) # n_trials x n_kernel_support_timepoints x n_timepoints

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
                full_toeplitz[i,support_t_idx,support_t_idx] = full_t[support_t_idx,support_t_idx]*fill_value

                current_toeplitz = full_toeplitz[i,:,:]

                # now we will cut the toeplitz to only the supported times
                if p['support_rel_to' ]=='stim': 
                    toeplitz[i,:, :] = current_toeplitz[support_k_t_idx, :]

                elif p['support_rel_to'] =='move':
                    # remove zero rows from current toeplitz
                    support_move_idx = current_toeplitz.sum(axis=1)!=0

                    # there is some noise in the timing so we need to be a bit flexible with the indices
                    supported_toeplitz = current_toeplitz[support_move_idx,:]
                    n_bins_needed = supported_toeplitz.shape[0]
                    
                    # on passive_active fits the choice kernel is not always supported on all trials
                    if n_bins_needed>0:                      
                        toeplitz[i,-n_bins_needed:,:] = supported_toeplitz
                
                # need to remove along one axis what is not supported 
            else:
                toeplitz[i,0,support_t_idx] = full_row[support_t_idx]*fill_value



    return (toeplitz,kernel_tscale)

def construct_feature_matrix(ev,tscale,kernel_names=['b','aud_onset'],**kernel_kws):
    kernels = {name: contrstruct_toeplitz(ev,tscale,which=name,**kernel_kws) for name in kernel_names}

    # calculate the indices in whicj a particlar kernel is supported
    i = 0
    feature_column_dict = {}
    feature_tscale_dict = {}
    for name in kernel_names:
        k_matrix, k_tscale = kernels[name]
        n_features = k_matrix.shape[1]
        feature_column_dict[name] = np.arange(i,i+n_features)
        feature_tscale_dict[name] = k_tscale
        i += n_features


    # concatenate to trials x features x timepoints
    feature_matrix = np.concatenate([kernels[name][0] for name in kernel_names],axis=1) # n_trials x n_features x n_timepoints

    return feature_matrix,feature_column_dict,feature_tscale_dict