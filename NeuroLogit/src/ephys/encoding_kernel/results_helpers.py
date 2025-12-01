import numpy as np

from NeuroLogit.src.ephys.results_utils import read_files


def read_responses(model_type,result_type='actual_mean',vis_conds = [-1,0,1],aud_conds = ['no_sound',-1,0,1],choice_conds = ['passive'],**kws):
    conds = [(v,a,c) for v in vis_conds for a in aud_conds for c in choice_conds]

    all_responses = {
        cond: read_files(which_result = 'encoding_kernel_results', filestub = f'predictions_{model_type}',
                                               extension='npz', npz_dat_type=result_type,psth_condition=cond, **kws)
        for cond in conds
    }

    return all_responses

def add_info_to_clusters(clusters,kernels,feature_column_dict,VE_threshold=0.001,noise_std=50):


    # adding things about the location of the units
    clusters['ml_gauss'] = clusters['ml'] + np.random.normal(0, noise_std, size=len(clusters))
    clusters['ap_gauss'] = clusters['ap'] + np.random.normal(0, noise_std, size=len(clusters))
      
    clusters['SC_pos'] = clusters['brainLocationAcronyms_ccf_2017'].copy()
    clusters['SC_pos'] = clusters['SC_pos'].replace({'SCzo': 'SCs', 'SCop': 'SCs', 'SCsg': 'SCs', 
                                                    'SCdg': 'SCd', 'SCdw': 'SCd',
                                                    'SCiw': 'SCiw', 'SCig': 'SCig'})




    is_active  = 'engagement' in feature_column_dict.keys()

    clusters['VE_vis_spatial'] = clusters[[f'VE_vis_contra_0.5',f'VE_vis_contra_1.0']].sum(axis=1)
    clusters['VE_aud_spatial'] = clusters[[f'VE_aud_onset',f'VE_aud_ipsi',f'VE_aud_contra']].sum(axis=1)
    clusters['VE_aud_total'] = clusters[[f'VE_aud_onset',f'VE_aud_ipsi',f'VE_aud_contra']].sum(axis=1)


    #% the amp of the kernel that I want to plot
    clusters['amp_vis_spatial'] = kernels[:,feature_column_dict['vis_contra_1.0']].mean(axis=1)
    clusters['amp_aud_spatial'] = (kernels[:,feature_column_dict['aud_contra']]-
                                kernels[:,feature_column_dict['aud_ipsi']]).mean(axis=1)

    clusters['amp_aud_total']=clusters['amp_aud_spatial']
    clusters['amp_aud_total_contra'] = (kernels[:,feature_column_dict['aud_contra']]+
                                kernels[:,feature_column_dict['aud_onset']]).mean(axis=1)
    
    clusters['amp_aud_onset'] = kernels[:,feature_column_dict['aud_onset']].mean(axis=1)

    if is_active:
        clusters['VE_choice'] = clusters[[f'VE_choice_ipsi',f'VE_choice_contra']].sum(axis=1)
        clusters['amp_choice'] = (kernels[:,feature_column_dict['choice_contra']]-
                                kernels[:,feature_column_dict['choice_ipsi']]).mean(axis=1)
        
        clusters['amp_choice_ipsi'] = kernels[:,feature_column_dict['choice_ipsi']].mean(axis=1)
        clusters['amp_choice_contra'] = kernels[:,feature_column_dict['choice_contra']].mean(axis=1)

        clusters['amp_engagement'] = kernels[:,feature_column_dict['engagement']].mean(axis=1)
        
        if 'action_onset' in feature_column_dict.keys():
            action_key = 'action_onset'
        else:
            action_key = 'action_stim_contra'
        
        clusters[f'amp_{action_key}'] = kernels[:,feature_column_dict[action_key]].mean(axis=1)

        clusters['VE_task'] = clusters[[f'VE_engagement',f'VE_{action_key}']].sum(axis=1)
        clusters['amp_task'] = clusters['amp_engagement'] + clusters[f'amp_{action_key}']

    VE_columns = [col for col in clusters.columns if col.startswith('VE_')]


    for col in VE_columns:
        param_name = col.split('VE_')[1]
        clusters[f'significant_{param_name}'] = clusters[col]>VE_threshold

    # label the cells 

    clusters['is_visual'] = clusters['significant_vis_spatial'] & (~clusters['significant_aud_total'])
    clusters['is_auditory'] = clusters['significant_aud_total'] & (~clusters['significant_vis_spatial'])
    clusters['is_AV'] = clusters['significant_vis_spatial'] & clusters['significant_aud_total']
    clusters['VE_visual'] = clusters['VE_vis_spatial']
    clusters['VE_auditory'] = clusters['VE_aud_total']
    clusters['VE_AV'] = clusters['VE_vis_spatial'] + clusters['VE_aud_total']
    clusters['amp_visual'] = clusters['amp_vis_spatial']
    clusters['amp_auditory'] = clusters['amp_aud_total']
    clusters['amp_AV'] = clusters['amp_vis_spatial'] + clusters['amp_aud_total']

    if is_active:
        clusters['is_task'] = clusters['significant_task'] 
        clusters['is_choice'] = clusters['significant_choice']

    clusters['functional_type'] = 'non_significant'
    clusters.loc[clusters['is_visual'], 'functional_type'] = 'visual'
    clusters.loc[clusters['is_auditory'], 'functional_type'] = 'auditory'
    clusters.loc[clusters['is_AV'], 'functional_type'] = 'AV'



def make_kernel_dict(kernels, feature_column_dict, kernel_tscale):
    kernels_dict = {}
    for k in feature_column_dict.keys():
        kernel_inds = feature_column_dict[k]
        kernels_dict[k] = {
            'coefficients': kernels[:,kernel_inds],
            'tscale': kernel_tscale[k]
        }

    if 'engagement' in feature_column_dict.keys():
        
        engagement = kernels[:, feature_column_dict['engagement']]

        tscale_engagement = kernel_tscale['engagement']
        
        if 'action_onset' in feature_column_dict.keys():
            action_key = 'action_onset'
        else:
            action_key = 'action_stim_contra'
        

        action_onset = kernels[:, feature_column_dict[action_key]]
        tscale_action_onset = kernel_tscale[action_key]


        # actually, we will just add the engagement kernel at the beginning. 
        t_scale_engagement_shown = tscale_engagement[(tscale_engagement<0.005) & (tscale_engagement>-0.1)]
        tscle_engagement_only = np.setdiff1d(t_scale_engagement_shown, tscale_action_onset)
        coef_engaement = np.tile(engagement, (1, len(tscle_engagement_only)))

        k_tscale = np.sort(np.concatenate([tscle_engagement_only,tscale_action_onset]), axis=0)
        coeffs = np.concatenate([coef_engaement,action_onset+engagement], axis=1)

        kernels_dict['task'] = {
            'coefficients': coeffs,
            'tscale': k_tscale
        }

    # add the zero padding to the sensory kernels

    sensory_kernels = ['vis_contra_0.25','vis_contra_0.5','vis_contra_1.0',
                       'aud_onset','aud_ipsi','aud_contra']
    
    for k in sensory_kernels:
        kernel_data = kernels_dict[k]
        tscale = kernel_data['tscale']
        coefficients = kernel_data['coefficients']


        baseline_coefs = kernels_dict['b']['coefficients']
        tscale_baseline = kernels_dict['b']['tscale']

        tscale_baseline_shown = tscale_baseline[(tscale_baseline<0.005) & (tscale_baseline>-0.1)]
        coef_baseline = np.tile(baseline_coefs, (1, len(tscale_baseline_shown)))

        # create the new coefficients with zeros at the beginning
        new_tscale = np.sort(np.concatenate([tscale_baseline_shown, tscale]), axis=0)
        new_coefficients = np.concatenate((coef_baseline, coefficients), axis=1)

        # update the dictionary
        kernels_dict[f'{k}_baseline_padded'] = {
            'tscale': new_tscale,
            'coefficients': new_coefficients
        }

    return kernels_dict

def add_correct_chocice_option(psth):

    # we don't want to add this to the no sound option... because it doesn't exist.
    audstim = [-1,0,1]
    visstim = np.unique(np.array([k[0] for k in psth.keys()]))

    # get the means on correct trials 
    for i,aud in enumerate(audstim):
        for j,vis in enumerate(visstim):

            if (np.sign(vis)==-1) & (np.sign(aud)<=0):
                
                resp = psth[(vis,aud,'ipsi')]

            elif (np.sign(vis)==1) & (np.sign(aud)>=0):
                resp = psth[(vis,aud,'contra')]
            elif (vis==0) & (aud==1):
                resp = psth[(vis,aud,'contra')]
            elif (vis==0) & (aud==-1):
                resp = psth[(vis,aud,'ipsi')]
            else:
                contra = psth.get((vis,aud,'contra'), None)
                ipsi = psth.get((vis,aud,'ipsi'), None)

                resp = np.zeros_like(contra)
                for idx,(contra, ipsi) in enumerate(zip(contra, ipsi)):
                    contra_nan = np.isnan(contra).any()
                    ipsi_nan = np.isnan(ipsi).any()
                    if (not contra_nan) and (ipsi_nan):
                        resp[idx] = contra
                    elif (contra_nan) and (not ipsi_nan):
                        resp[idx] = ipsi
                    else: 
                        random_idx = np.random.choice([0,1])
                        resp[idx] = contra if random_idx==0 else ipsi
            

            

            psth[(vis,aud,'correct')] = resp

    return psth

def read_all_results(model_type='passive_active_Ridge100', sessions='unique',trig_kws=None,**cluster_kws):
    
    output = {}
    kernels = read_files(which_result = 'encoding_kernel_results', filestub=f'model_{model_type}', extension='npz', npz_dat_type='coefficients', sessions=sessions)
    clusters = read_files(which_result = 'encoding_kernel_results', filestub=f'clusters_{model_type}', extension='csv', sessions=sessions)
    feature_column_dict = read_files(which_result = 'encoding_kernel_results', filestub = f'model_{model_type}',extension='npz', npz_dat_type='feature_column_dict', sessions=sessions).item()
    kernel_tscale = read_files(which_result = 'encoding_kernel_results', filestub = f'model_{model_type}',extension='npz', npz_dat_type='feature_tscale_dict', sessions=sessions).item()
    # this is the data tscale

    # add the exta info to clusters
    add_info_to_clusters(clusters,kernels,feature_column_dict,**cluster_kws)
    kernel_dict = make_kernel_dict(kernels, feature_column_dict, kernel_tscale)


    response_tscale  = read_files(which_result = 'encoding_kernel_results', filestub = f'model_{model_type}',extension='npz', npz_dat_type='tscale', sessions=sessions)#

    output.update({'kernels': kernels, 'clusters': clusters, 'feature_column_dict': feature_column_dict, 'kernel_tscale': kernel_tscale, 'response_tscale': response_tscale,'kernel_dict': kernel_dict})

    if trig_kws is not None:
        trig_kws.update({'sessions': sessions})

        actual_mean = read_responses(model_type,result_type='actual_mean',**trig_kws)
        predicted_mean = read_responses(model_type,result_type='predicted_mean',**trig_kws)
        actual_sem = read_responses(model_type,result_type='actual_sem',**trig_kws)
        predicted_sem = read_responses(model_type,result_type='predicted_sem',**trig_kws)

        psth_dict = {'actual_mean': actual_mean, 'predicted_mean': predicted_mean, 'actual_sem': actual_sem, 'predicted_sem': predicted_sem}

        is_active = actual_mean.get((0,0,'ipsi'), None) is not None

        if is_active:
            for key in psth_dict.keys():
                psth_dict[key] = add_correct_chocice_option(psth_dict[key])

        output.update(psth_dict)

    return output


