
#%%
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import r2_score,explained_variance_score


from NeuroLogit.src.ephys.dat_utils import load_trial_data,smooth_raster,get_ephys_dataset
from NeuroLogit.src.ephys.encoding_kernel.feature_matrix import construct_feature_matrix
from NeuroLogit.src.ephys.encoding_kernel.kernel_utilities import ReducedRankRegressor

from NeuroLogit.src.ephys.psth_visualise.batch_proc import raster_to_mean_sem
#%%

RESULTS_PATH = Path(r'D:\AV_Neural_Data_Sept2025\encoding_kernel_results')
META_PATH = Path(r'D:\AV_Neural_Data_Sept2025\meta_info\encoding_kernel')

MODEL_ZOO = {
    'ReducedRank': ReducedRankRegressor,
    'Ridge': Ridge,
    'Linear': LinearRegression,
    'Lasso': Lasso
}

def _create_model(model_type: str, **model_kws):
    """Helper function to instantiate a model from the zoo."""
    if model_type not in MODEL_ZOO:
        raise ValueError(f"Model type '{model_type}' not recognized. Available: {list(MODEL_ZOO.keys())}")
    return MODEL_ZOO[model_type](**model_kws)

class reshaper():
    """
    utility class that allows to reshape the data matrices between
    different formats: trials x features x timepoints  --> (trials*timepoints) x features
    trials x neurons x timepoints  --> (trials*timepoints) x neurons    
    """
    def __init__(self,r,X):
        self.X = X.copy()
        self.r = r.copy()

        assert r.ndim ==3, "Response matrix must be trials x neurons x timepoints"
        assert X.ndim ==3, "Predictor matrix must be trials x features x timepoints"

        assert r.shape[0] == X.shape[0], "Number of trials must be the same in response and predictor matrices"
        assert r.shape[2] == X.shape[2], "Number of timepoints must be the same in response and predictor matrices"

        self.n_trials = r.shape[0]
        self.n_neurons = r.shape[1]
        self.n_features = X.shape[1]
        self.n_timepoints = r.shape[2]

    def feature_x_trialtime(self):
            
        X_reshaped = self.X.transpose(0,2,1).reshape(-1,self.n_features) # (n_trials*n_timepoints) x n_features
        return X_reshaped

    def neuron_x_trialtime(self):
        r_reshaped = self.r.transpose(0,2,1).reshape(-1,self.n_neurons) # (n_trials*n_timepoints) x n_neurons
        return r_reshaped

    def trial_x_neuron_x_time(self,r_reshaped=None):
        if r_reshaped is None:
            r_reshaped = self.neuron_x_trialtime()
            
        y = r_reshaped.reshape(self.n_trials,self.n_timepoints,self.n_neurons).transpose(0,2,1) # n_trials x n_neurons x timepoints
        return y

def filter_data(ev,clusters,r,tscale,is_valid_trial,pre_time=0.2,post_time=0.6,variance_threshold=1,hemisphere='right',zscore=True):
    
    # tscale = raster['tscale']

    # # some time could be saved here by smoothing only valid trials and clusters. especauuly that we don't do bl subrraction here..
    # # maybe later.... or save it...
    # r = smooth_raster(raster['data_binned'],tscale,smoothing=temp_smoothing,
    #                 kernel_dir='forward',baseline_subtract=None,zscore=False) 
    # but this zscore might not be the best: at this stage thre are actve and passive trials too

    
    # r is trials x neurons x timepoints 

    # filter along time axis
    ev = ev.copy()
    valid_t_idx = (tscale>-pre_time) & (tscale<post_time)

    # filter from trials 
    non_nan_trials = ~np.isnan(r).all(axis=(1,2))
    valid_trials = non_nan_trials & is_valid_trial # or other criteria # probabily need to refine

    # filter along neurons this is used of cells are not zscored
    cluster_variances = np.nanvar(r[:,:,tscale<0].mean(axis=2), axis=0)
    ok_variance_clusters = cluster_variances > variance_threshold

    # calculate mean baseline and std 
    baselines = np.nanmean(r[:, :, tscale < 0], axis=(0,-1), keepdims=True)  # trials x neurons x 1
    baseline_stds = np.nanstd(r[:, :, tscale < 0], axis=(0,-1), keepdims=True)  # trials x neurons x 1


    valid_clusters = (ok_variance_clusters & 
                    (clusters['bombcell_class']!='noise') & 
                    (clusters['hemi']== hemisphere)  # fitting kernel one hemishere at a time
                    )

    # reduce everything to the valid indices
    tscale = tscale[valid_t_idx]
    ev = ev[valid_trials].reset_index(drop=True)
    clusters = clusters[valid_clusters].reset_index(drop=True)

    raster_kept_idx = np.ix_(valid_trials,valid_clusters,valid_t_idx)
    
    r = r[raster_kept_idx]
    baselines = baselines[:, valid_clusters, :]
    baseline_stds = baseline_stds[:, valid_clusters, :]

    if zscore:
        r = (r - baselines) / baseline_stds


    # flip important events related to sounds
    ev['choice_regressed'] = ev['timeline_choiceMoveDir'].replace({2:1,1:-1}) 

    if hemisphere == 'left':
        ev['choice_categorical'] = ev['choice_categorical'].replace({'right': 'contra', 'left': 'ipsi'})
    elif hemisphere == 'right':
        ev.visDiff_categorical *=-1
        ev.audDiff_categorical *=-1
        ev.choice_regressed *=-1
        ev['choice_categorical'] = ev['choice_categorical'].replace({'right': 'ipsi', 'left': 'contra'})
    
    return ev,clusters,r,tscale

def fit_evaluate(ev,clusters,r,feature_matrix,feature_column_dict,model_type = 'ReducedRank',**model_kws):
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    train_idx, test_idx = next(sss.split(ev, ev['choice_categorical']))

    r_train = r[train_idx,:,:].copy()
    r_test = r[test_idx,:,:].copy()
    X_train = feature_matrix[train_idx,:,:].copy()
    X_test = feature_matrix[test_idx,:,:].copy()

    train = reshaper(r_train,X_train)
    test = reshaper(r_test,X_test)

    X_train = train.feature_x_trialtime() # (n_trials*n_timepoints) x n_features
    y_train = train.neuron_x_trialtime()  # (n_trials*n_timepoints)
    X_test = test.feature_x_trialtime()
    y_test = test.neuron_x_trialtime()


    # remove timepoints where we don't support any kernel (mostly afer )
    non_zero_train = ~np.all(X_train == 0, axis=1)
    X_train = X_train[non_zero_train]
    y_train = y_train[non_zero_train]

    non_zero_test = ~np.all(X_test == 0, axis=1)
    X_test = X_test[non_zero_test]
    y_test = y_test[non_zero_test]


    m = _create_model(model_type, **model_kws)
    m.fit(X_train, y_train) 

    # evaluation on test set
    y_pred = m.predict(X_test) # (n_trials*n_timepoints) x n_neurons
    clusters['r2_tot'] = r2_score(y_test,y_pred,multioutput='raw_values')
    clusters['VE_tot'] = explained_variance_score(y_test,y_pred,multioutput='raw_values')

    #% 2) for each kernel we also calculate the r2 on the residual
    for event in feature_column_dict.keys():

        X_train_leave_one = X_train.copy()
        X_test_leave_one = X_test.copy()
        X_train_leave_one[:, feature_column_dict[event]] = 0
        X_test_leave_one[:, feature_column_dict[event]] = 0

        m_all_but_one = _create_model(model_type, **model_kws)
        m_one = _create_model(model_type, **model_kws)
        
        m_all_but_one.fit(X_train_leave_one, y_train)
        train_prediction = m_all_but_one.predict(X_train_leave_one)
        train_diff = y_train - train_prediction

        test_prediction = m_all_but_one.predict(X_test_leave_one)
        test_diff = y_test - test_prediction

        # Train error on the one kernel feature matrix, and evaluate it's performance on the test set residuals
        one_kernel_X_train = X_train - X_train_leave_one

        m_one.fit(one_kernel_X_train, train_diff)

        one_kernel_X_test = X_test - X_test_leave_one
        one_kernel_test_prediction = m_one.predict(one_kernel_X_test)

        kernel_score = explained_variance_score(test_diff,one_kernel_test_prediction,multioutput='raw_values')
        clusters[f'VE_{event}'] = kernel_score

    return m,clusters

def get_valid_trials(ev,valid_trial_ID='passive'):
    
    if valid_trial_ID=='passive':
        is_valid_trial = (ev['session']=='passive')
    if valid_trial_ID =='active_passive':        
        is_valid_trial = (((ev.rt > 0.05).values & (ev.rt < 1).values & ((ev.timeline_choiceMoveOn-ev.timeline_firstMoveOn)<0.05).values)| # we exclude super short RTs, and when the animal had been wiggling; as well as too long RTs (non-repeatable for stim kernels)
                        (ev.choice_categorical.isin(['passive','NoGo'])).values)

    return is_valid_trial

def get_model_param_combos(model_type='passive_only_RR10'):
    
    if 'passive_only' in model_type:
        valid_trial_ID = 'passive'
        preproc_kws ={
            'pre_time':0.2,
            'post_time':0.65,
            'variance_threshold':1,
            'zscore':True
        }

        kernel_kws = {
            'kernel_names': ['b','aud_onset','aud_ipsi','aud_contra','vis_contra_0.25','vis_contra_0.5','vis_contra_1.0'],
            'max_pre_time': preproc_kws['pre_time'],
            'max_post_time': preproc_kws['post_time'],
            'kernel_support_end': 'stim'
        }
    


    elif 'passive_active' in model_type:
        valid_trial_ID = 'active_passive'
        preproc_kws ={
            'pre_time':0.4,
            'post_time':0.55,
            'variance_threshold':1,
            'zscore':True
        }

        kernel_kws = {
            'kernel_names': ['b','aud_onset','aud_ipsi','aud_contra','vis_contra_0.25','vis_contra_0.5','vis_contra_1.0',
                        'choice_ipsi','choice_contra','engagement','action_onset'],
            'max_pre_time': 0.4,
            'max_post_time': 0.45,
            'choice_pre_time': 0.15,
            'choice_post_time': 0.1,
            'kernel_support_end': 'move'
        }

    # max_post_time _ choice_post_time should be equal or more than post time. I think. else the very short rt sessions will not have the same kernel lenghts for sensory ? messing stuff up potentially (with the kernels that end at move onset (e.g. engagement etc.))

    
    if 'Ridge10' in model_type:
        fitting_kws ={
            'model_type':'Ridge',
            'alpha':10
        }

    elif 'Ridge50' in model_type:
        fitting_kws ={
            'model_type':'Ridge',
            'alpha':50
        }
    
    elif 'Ridge100' in model_type:
        fitting_kws ={
            'model_type':'Ridge',
            'alpha':100
        }

    elif 'Ridge500' in model_type:
        fitting_kws ={
            'model_type':'Ridge',
            'alpha':500
        }

    elif 'RR10' in model_type:
        fitting_kws ={
            'model_type':'ReducedRank',
            'regressor':'Ridge',
            'rank':40,
            'alpha':10
        }

    elif 'RR10' in model_type:
        fitting_kws ={
            'model_type':'ReducedRank',
            'regressor':'Ridge',
            'rank':40,
            'alpha':100
        }

    if fitting_kws['model_type']=='Ridge':
        fitting_kws['fit_intercept'] = False
    



    return valid_trial_ID,kernel_kws,preproc_kws,fitting_kws

def get_psths_per_cond_(df,raster,raster_tscale,groupby=['visDiff_categorical','audDiff_categorical','choice_categorical']): 
    grouped_indices = df.groupby(groupby).indices
    
    fr_mean,fr_sem = {},{}
    for group, indices in grouped_indices.items():
        fr_mean[group],fr_sem[group] = raster_to_mean_sem(raster[indices,:,:],raster_tscale,smoothing=0.025)

    # we might need to insert nans for missing conditions    
    aud_conds = df['audDiff_categorical'].unique()
    vis_conds = df['visDiff_categorical'].unique()
    choice_conds = df['choice_categorical'].unique()

    all_expected_conds = [(v,a,c) for v in vis_conds for a in aud_conds for c in choice_conds]

    missing_conds = set(all_expected_conds) - set(fr_mean.keys())
    for cond in missing_conds:
        n_neurons = raster.shape[1]
        n_timepoints = raster.shape[2]
        fr_mean[cond] = np.full((n_neurons,n_timepoints),np.nan)
        fr_sem[cond] = np.full((n_neurons,n_timepoints),np.nan)

    
    return fr_mean,fr_sem

def fit_all_models_per_session(subject,date,model_IDs = ['passive_RR10']):
    print(f"Fitting session: {subject}_{date} ...")
    ev,clusters,rasters_stim = load_trial_data(subject,date,
                                load_clusters=True,load_raster='prestim',include_no_sound_trials=True).values()
    
    # smooth data
    tscale = rasters_stim['tscale']

    # # some time could be saved here by smoothing only valid trials and clusters. especauuly that we don't do bl subrraction here..
    # # maybe later.... or save it...
    r = smooth_raster(rasters_stim['data_binned'],tscale,smoothing=0.025,
                    kernel_dir='forward',baseline_subtract=None,zscore=False) 
    

    # smooth the raster

    # Replace hemisphere values in clusters
    clusters['hemi'] = clusters['hemi'].replace({-1: 'right', 1: 'left'})
    unique_hemispheres = clusters['hemi'].dropna().unique() #

    # remove nan from unique hemispheres

    model_results = {}
    
    for model_ID in model_IDs:
        valid_trial_ID,kernel_kws,preproc_kws,fitting_kws = get_model_param_combos(model_type=model_ID)
        is_valid_trial = get_valid_trials(ev, valid_trial_ID = valid_trial_ID)

        all_clusters_fitted = []
        coefficients = []
        model_info = {}

        model_predictions_per_cond = {}
        for hemisphere in unique_hemispheres:
            print(f"Fitting model {model_ID} on {hemisphere} hemisphere...")

            ev_p,clusters_p,r_p,tscale_p = filter_data(ev,clusters,r,tscale,is_valid_trial,
                                                   hemisphere=hemisphere,**preproc_kws)
            feature_matrix,feature_column_dict,feature_tscale_dict = construct_feature_matrix(ev_p,tscale_p,**kernel_kws)
            
            m,clusters_fitted = fit_evaluate(ev_p,clusters_p,r_p,feature_matrix,feature_column_dict,**fitting_kws)

            clusters_fitted['hemisphere'] = hemisphere
            clusters_fitted['subject'] = subject
            clusters_fitted['date'] = date

            all_clusters_fitted.append(clusters_fitted)
            coefficients.append(m.coef_)

            # save the actual features matrix for reference
            model_info[f'feature_matrix_{hemisphere}'] = feature_matrix
            
            # as well as the total data and prediction on all trials
            reshaped = reshaper(r_p,feature_matrix)
            X_all = reshaped.feature_x_trialtime()

            pred_all = m.predict(X_all)
            
            # okay, issue with this is that atm this is hard to concatenate across sessions as different sessions have different number of trials etc. 
            
            pred_all = reshaped.trial_x_neuron_x_time(r_reshaped=pred_all)
            model_info[f'actual_responses_{hemisphere}'] = r_p    
            model_info[f'predicted_responses_{hemisphere}'] = pred_all


            # replace 
            ev_p.loc[ev_p.stim_audAmplitude==0,'audDiff_categorical'] = 'no_sound'

            actual_psths,actual_psths_sem  = get_psths_per_cond_(ev_p,r_p,tscale_p)
            preducted_psths,preducted_psths_sem  = get_psths_per_cond_(ev_p,pred_all,tscale_p)
            
            #model_info[f'events_{hemisphere}'] = ev_p.reset_index(drop=True)
            # I think ev_p is flipped so we might be able to potentially concatenatate the results... 
            model_predictions_per_cond[f'actual_mean_{hemisphere}'] = actual_psths
            model_predictions_per_cond[f'actual_sem_{hemisphere}'] = actual_psths_sem
            model_predictions_per_cond[f'predicted_mean_{hemisphere}'] = preducted_psths
            model_predictions_per_cond[f'predicted_sem_{hemisphere}'] = preducted_psths_sem



        all_clusters_fitted = pd.concat(all_clusters_fitted,ignore_index=True)
        coefficients = np.concatenate(coefficients,axis=0)

        # concatenate left and right model predictions 

        all_predictions = {}
        for key in model_predictions_per_cond.keys():
            key_base = key.rsplit('_',1)[0]
            if unique_hemispheres.size ==2:
                resp1 = model_predictions_per_cond[f'{key_base}_{unique_hemispheres[0]}']
                resp2 = model_predictions_per_cond[f'{key_base}_{unique_hemispheres[1]}']

                
                
                all_predictions[key_base] = {k: np.concatenate([resp1[k],resp2[k]],axis=0) for k in resp1.keys()}
            else:
                all_predictions[key_base] = model_predictions_per_cond[f'{key_base}_{unique_hemispheres[0]}']


        model_info['coefficients'] = coefficients
        model_info['feature_column_dict'] = feature_column_dict
        model_info['feature_tscale_dict'] = feature_tscale_dict
        model_info['fitting_kws'] = fitting_kws
        model_info['preproc_kws'] = preproc_kws
        model_info['model_type'] = model_ID
        model_info['kernel_names'] = kernel_kws['kernel_names']
        model_info['tscale'] = tscale_p

        model_results[model_ID] = (all_clusters_fitted, model_info,all_predictions)

    return model_results

def fit_dataset(model_names = ['passive_RR10','passive_Ridge10'],dataset_kwargs={'set_name':'all'},recompute=False):

    sessions = get_ephys_dataset(**dataset_kwargs)
    for _,args in sessions[['subject','date']].iterrows():
            subject = args['subject']
            date = args['date']

            session_stub = f"{subject}_{date}"
            session_path = RESULTS_PATH / session_stub
            session_path.mkdir(parents=True, exist_ok=True)

            
            savefile_names = [f"clusters_{m}.csv" for m in model_names]
            # check if all files exist
            all_exist = all((session_path / fname).exists() for fname in savefile_names)

            if (not all_exist) or recompute:
                try:
                    models_results = fit_all_models_per_session(subject,date,model_IDs=model_names)
                    
                    for m in model_names:
                        clusters_results = models_results[m][0]
                        model_info = models_results[m][1]
                        model_predictions_per_cond = models_results[m][2]

                        # save clusters results
                        clusters_results.to_csv(session_path / f"clusters_{m}.csv",index=False)

                        # save model info
                        np.savez(session_path / f"model_{m}.npz", **model_info)        
                        np.savez(session_path / f"predictions_{m}.npz", **model_predictions_per_cond)        

                    # save results 
                    print(f"Saving results for session: {session_stub} ...")

                except Exception as e:
                    print(f"Error processing session {session_stub}: {e}")


if __name__ == "__main__":
    fit_dataset(model_names = ['passive_only_Ridge10','passive_active_Ridge100','passive_active_Ridge50','passive_active_Ridge10','passive_active_Ridge500'],
                dataset_kwargs={'set_name':'all', 'subset':''},recompute=True)
 
# ['passive_active_RR10','passive_active_Ridge100','passive_active_Ridge50','passive_active_Ridge10','passive_active_Ridge500','passive_active_RR100']
