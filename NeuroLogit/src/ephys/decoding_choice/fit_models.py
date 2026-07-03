#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss


from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel,VarianceThreshold

from NeuroLogit.src.ephys.dat_utils import load_trial_data
from NeuroLogit.src.ephys.results_utils import read_files

from NeuroLogit.src.ephys.decoding_choice.fit_helpers import select_clusters_from_main_region,pairwise_auc_roc,add_behav_to_ev
from NeuroLogit.src.ephys.decoding_choice.av_multi_neurmodels import av_multi_neural_unilateral as av_multi_neural
from NeuroLogit.src.models.av_models_multi import av_multi_symmetric_audio


def load_and_format_session(subject,date,time_kws,VE_choice_thr=0.005,encoder_model_type='passive_active_Ridge10',encoder_based_seletion ='choice'):

    ev,clusters,_ = load_trial_data(subject,date,load_clusters=True,**time_kws).values()
    clusters.hemi.replace({-1:'right',1:'left'},inplace=True)
    clusters.hemi.fillna('void',inplace=True)

    ev = add_behav_to_ev(ev,subject,date) # this might be different for 3-choice

    # load the result of kernel regression to get choice neurons only    
    # 
    ev = ev[(ev.choice!=-2)].copy()
    # I think in my multiclass regression I set up chocies as 0.1.2 for nogo, left, right
    ev.choice+=1

    neuron_cols = [col for col in ev.columns if 'neuron_' in col]
    ev[neuron_cols] = StandardScaler().fit_transform(ev[neuron_cols])  # this is the other thing that I think we probably should move to after the training set to prevent leakage
    # Rename neuron columns with neuronID_hemi names
    new_neuron_cols = (clusters.neuronID + '_' + clusters.hemi).tolist()
    ev.rename(columns=dict(zip(neuron_cols, new_neuron_cols)), inplace=True)

    # Standardize the neural columns to have zero mean and unit variance
    # maybe instead of all of this, I should do an l1 feature selection on the neural weights directly?

    clusters_encoder = read_files(which_result = 'encoding_kernel_results', 
                                filestub=f'clusters_{encoder_model_type}', 
                                extension='csv', sessions=f'{subject}_{date}')
    
    # select good and mua and neurons from main region only
    clusters_encoder, roi = select_clusters_from_main_region(clusters_encoder)

    clusters_encoder['neuronID_hemi'] = clusters_encoder['neuronID'] + '_' + clusters_encoder['hemi'].astype(str)

    # select only action/choice related activity  # now that I implemented l1, maybe I should not do this step?

    clusters_encoder['VE_choice'] = clusters_encoder[['VE_choice_contra','VE_choice_ipsi']].sum(axis=1)
    clusters_encoder['VE_task'] = clusters_encoder[['VE_choice_contra','VE_choice_ipsi','VE_engagement']].sum(axis=1)


    if encoder_based_seletion == 'choice':
        selected_clusters = clusters_encoder[(clusters_encoder.VE_choice>=VE_choice_thr)]
    elif encoder_based_seletion == 'taskorchoice':
        selected_clusters = clusters_encoder[(clusters_encoder.VE_task>=VE_choice_thr) | (clusters_encoder.VE_choice>=VE_choice_thr)]
    elif encoder_based_seletion == 'r2':
        selected_clusters = clusters_encoder[(clusters_encoder.r2_tot>=10e-5)]
    elif encoder_based_seletion == 'task':
        selected_clusters = clusters_encoder[(clusters_encoder.VE_task>=VE_choice_thr)]
    elif encoder_based_seletion == 'topchoice':
        n_select = 10
        selected_clusters = clusters_encoder.nlargest(n_select,'VE_choice')
    elif encoder_based_seletion == 'random10':
        selected_clusters = clusters_encoder[(clusters_encoder.r2_tot>=10e-5)]
        n_select = 10
        selected_clusters = selected_clusters.sample(n_select,random_state=42)
    elif encoder_based_seletion == 'random20':
        selected_clusters = clusters_encoder[(clusters_encoder.r2_tot>=10e-5)]
        n_select = 20
        selected_clusters = selected_clusters.sample(n_select,random_state=42)

    elif encoder_based_seletion == 'random30':
        selected_clusters = clusters_encoder[(clusters_encoder.r2_tot>=10e-5)]
        n_select = 30
        selected_clusters = selected_clusters.sample(n_select,random_state=42)

    elif encoder_based_seletion == 'random40':
        selected_clusters = clusters_encoder[(clusters_encoder.r2_tot>=10e-5)]
        n_select = 40
        selected_clusters = selected_clusters.sample(n_select,random_state=42)

    elif encoder_based_seletion == 'random50':
        selected_clusters = clusters_encoder[(clusters_encoder.r2_tot>=10e-5)]
        n_select = 50
        selected_clusters = selected_clusters.sample(n_select,random_state=42)
    elif encoder_based_seletion == 'all':
        selected_clusters = clusters_encoder[(clusters_encoder.r2_tot>=10e-5)]

    # okay this makes things extra complicated... because everything is set up assuming we have left and right hemisphere neurons
    # so this would need to be dome per 
    neural_predictors = selected_clusters.neuronID_hemi.tolist()

    # potentially this selectior should only be on the training set or perhaps none at all..?

    ev['roi_fitted'] = roi    
    ev['is_right_hemisphere'] = selected_clusters.hemi.isin(['right']).any()
    ev['is_left_hemisphere'] = selected_clusters.hemi.isin(['left']).any()
    
    return {'ev':ev,
            'neuronIDs':neural_predictors,
            'roi':roi,
            'class_idx_to_name': {0:'NoGo',1:'Left',2:'Right'}
}

def assess_logLoss_and_brier_score_per_choice(choices, probs):
    """
    Decomposes Log-Likelihood for a 3-choice model (0: NoGo, 1: Left, 2: Right).
    
    Args:
        choices: np.array (N,) of ints {0, 1, 2}
        probs:   np.array (N, 3) where:
                 Col 0 = P(NoGo)
                 Col 1 = P(Left)
                 Col 2 = P(Right)
    """

    # 1. Numerical Stability: Clip probabilities to avoid log(0)
    # 1e-15 is standard machine epsilon safety
    probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
    
    # ---------------------------------------------------------
    # PART A: Detection (Go vs NoGo)
    # ---------------------------------------------------------
    # We collapse L and R into a single "Go" probability
    p_nogo = probs[:, 0]
    p_go   = probs[:, 1] + probs[:, 2]
    
    # Boolean mask: Did the subject actually go?
    # We treat 1 (L) and 2 (R) as "Go" (True), 0 as "NoGo" (False)
    is_go_trial = (choices > 0)
    
    # Compute Log-Likelihood for binary detection
    # If subject went: log(P_go). If subject stayed: log(P_nogo)
    ll_go_nogo = np.sum(np.log(p_go[is_go_trial])) + \
                 np.sum(np.log(p_nogo[~is_go_trial]))
                 
    avg_ll_detect = ll_go_nogo / len(choices)


    brier_detect = brier_score_loss(is_go_trial, p_go)


    # ---------------------------------------------------------
    # PART B: Discrimination (Left vs Right)
    # ---------------------------------------------------------
    # We ONLY look at trials where a movement happened
    go_indices = np.where(choices > 0)[0]
    
    if len(go_indices) == 0:
        ll_lr = 0.0
        avg_ll_discrim = 0.0
    else:
        # Filter data to only Go trials
        choices_lr = choices[go_indices] # 1 or 2
        probs_lr   = probs[go_indices]   # Relevant rows
        
        # Renormalize: P(L|Go) = P(L) / (P(L) + P(R))
        # This isolates direction choice from the decision to move
        sum_p = probs_lr[:, 1] + probs_lr[:, 2]
        p_L_cond = probs_lr[:, 1] / sum_p
        p_R_cond = probs_lr[:, 2] / sum_p
        
        # Clip again just in case the sum was tiny
        p_L_cond = np.clip(p_L_cond, 1e-15, 1.0)
        p_R_cond = np.clip(p_R_cond, 1e-15, 1.0)
        
        # Compute LL for discrimination
        # If choice=1 (L), take log(P_L_cond). If choice=2 (R), take log(P_R_cond)
        ll_L = np.sum(np.log(p_L_cond[choices_lr == 1]))
        ll_R = np.sum(np.log(p_R_cond[choices_lr == 2]))
        
        ll_lr = ll_L + ll_R
        avg_ll_discrim = ll_lr / len(go_indices)

        brier_discrim = brier_score_loss(choices_lr == 1, p_L_cond)

    return {
        "Detection_LL_Total": ll_go_nogo,
        "Detection_LL_PerTrial": avg_ll_detect,
        "Discrim_LL_Total": ll_lr,
        "Discrim_LL_PerTrial": avg_ll_discrim,
        "Detection_Brier": brier_detect,
        "Discrim_Brier": brier_discrim
    }

def format_model_output_scipy(model, ev_df, X_data, model_prefix, class_idx_to_name):
    """
    Processes the output of a fitted multi-class logistic regression model.

    Args:
        model: A fitted scikit-learn classifier.
        ev_df: The DataFrame of events to add prediction columns to.
        X_data: The feature matrix to generate predictions for.
        test_idx: Indices of the test set within ev_df & X_data.
        model_prefix (str): A prefix for the new columns (e.g., 'l1_0.05', 'stim').
        feature_names (list): The names of the features in X_data.
        class_mapping (dict): Mapping from integer class label to string name.

    Returns:
        tuple: (weights_df, updated_ev_df)
    """
    # compute some utilities

    # --- 1. Process Coefficients ---
    name_to_idx = {i[1]: i[0] for i in (class_idx_to_name.items())}

    # basucally we might change things so that we compare the various models... let's see
    coef_df = pd.DataFrame({'feature': model.params.keys(),'weight_{}'.format(model_prefix): model.params.values()})
    
    # Create an inverse mapping from class index to string name for safety

    # --- 2. Process Predictions ---
    updated_ev_df = ev_df.copy()
    zL,zR = model.predict_log_proba(X_data)
    probas = model.predict_proba(X_data)
    
    # probabilities for each class
    for i, class_label in enumerate(class_idx_to_name.keys()):
        class_name = class_idx_to_name[class_label]
        updated_ev_df[f'proba_{class_name}_{model_prefix}'] = probas[:, i]

    updated_ev_df[f'proba_Go_{model_prefix}'] = updated_ev_df[f'proba_Right_{model_prefix}'] + updated_ev_df[f'proba_Left_{model_prefix}']

    # Add the overall predicted choice
    updated_ev_df[f'predicted_choice_{model_prefix}'] = model.predict(X_data)

    # log-odds of choice pairs
    updated_ev_df[f'logOdds_R_vs_NoGo_{model_prefix}'] = zR.flatten()
    updated_ev_df[f'logOdds_L_vs_NoGo_{model_prefix}'] = zL.flatten()
    updated_ev_df[f'logOdds_L_vs_R_{model_prefix}'] = zL.flatten() - zR.flatten()
    updated_ev_df[f'logOdds_Go_vs_NoGo_{model_prefix}'] = np.log(np.exp(zR) + np.exp(zL))
    
    # model performance
    ev_test_idx = updated_ev_df['is_test_set']
    y_true_train  = updated_ev_df.loc[~ev_test_idx, 'choice']
    y_true_test  = updated_ev_df.loc[ev_test_idx, 'choice']
    probas_train = probas[~ev_test_idx]
    probas_test = probas[ev_test_idx]
    zR_train  = updated_ev_df.loc[~ev_test_idx, f'logOdds_R_vs_NoGo_{model_prefix}']
    zR_test  = updated_ev_df.loc[ev_test_idx, f'logOdds_R_vs_NoGo_{model_prefix}']
    zL_train  = updated_ev_df.loc[~ev_test_idx, f'logOdds_L_vs_NoGo_{model_prefix}']
    zL_test  = updated_ev_df.loc[ev_test_idx, f'logOdds_L_vs_NoGo_{model_prefix}']


    log_loss_train = log_loss(y_true_train,probas_train, normalize=True)
    log_loss_test = log_loss(y_true_test,probas_test, normalize=True)
    auc_roc_train = roc_auc_score(y_true_train,probas_train, multi_class='ovr',average=None)
    auc_roc_test = roc_auc_score(y_true_test,probas_test, multi_class='ovr',average=None)

    # caclulate the logLikelihoods of Go vs NoGo and Left vs Right specifically... 
    scores_per_choicecat_train = assess_logLoss_and_brier_score_per_choice(y_true_train.values, probas_train)
    scores_per_choicecat_test = assess_logLoss_and_brier_score_per_choice(y_true_test.values, probas_test)

    # now we will also calculate the auc_roc pairwise
    auc_roc_train_L_vs_Nogo = pairwise_auc_roc(y_true_train,zL_train,pair=[name_to_idx['Left'],name_to_idx['NoGo']])
    auc_roc_train_R_vs_Nogo = pairwise_auc_roc(y_true_train,zR_train,pair=[name_to_idx['Right'],name_to_idx['NoGo']])
    auc_roc_test_L_vs_Nogo = pairwise_auc_roc(y_true_test,zL_test,pair=[name_to_idx['Left'],name_to_idx['NoGo']])
    auc_roc_test_R_vs_Nogo = pairwise_auc_roc(y_true_test,zR_test,pair=[name_to_idx['Right'],name_to_idx['NoGo']])
    auc_roc_train_R_vs_L = pairwise_auc_roc(y_true_train, zR_train - zL_train, pair=[name_to_idx['Right'],name_to_idx['Left']])
    auc_roc_test_R_vs_L = pairwise_auc_roc(y_true_test, zR_test - zL_test, pair=[name_to_idx['Right'],name_to_idx['Left']])


    right_hemisphere = ev_df.is_right_hemisphere.any()   & (not ev_df.is_left_hemisphere.any())
    left_hemisphere = ev_df.is_left_hemisphere.any() & (not ev_df.is_right_hemisphere.any())
    bilateral = ev_df.is_left_hemisphere.any() & ev_df.is_right_hemisphere.any()
    if right_hemisphere:
        hemisphere = 'right'
        logOdds_contra = updated_ev_df[f'logOdds_L_vs_NoGo_{model_prefix}']
        logOdds_ipsi = updated_ev_df[f'logOdds_R_vs_NoGo_{model_prefix}']
        auc_roc_test_contra_vs_Nogo = auc_roc_test_L_vs_Nogo
        auc_roc_train_contra_vs_Nogo = auc_roc_train_L_vs_Nogo
        auc_roc_test_ipsi_vs_Nogo = auc_roc_test_R_vs_Nogo
        auc_roc_train_ipsi_vs_Nogo = auc_roc_train_R_vs_Nogo
    elif left_hemisphere:
        hemisphere = 'left'
        logOdds_contra = updated_ev_df[f'logOdds_R_vs_NoGo_{model_prefix}']
        logOdds_ipsi = updated_ev_df[f'logOdds_L_vs_NoGo_{model_prefix}']
        auc_roc_test_contra_vs_Nogo = auc_roc_test_R_vs_Nogo
        auc_roc_train_contra_vs_Nogo = auc_roc_train_R_vs_Nogo
        auc_roc_test_ipsi_vs_Nogo = auc_roc_test_L_vs_Nogo
        auc_roc_train_ipsi_vs_Nogo = auc_roc_train_L_vs_Nogo
    else: 
        hemisphere = 'both'
        logOdds_contra = np.nan
        logOdds_ipsi = np.nan
        auc_roc_test_contra_vs_Nogo = np.nan
        auc_roc_train_contra_vs_Nogo = np.nan
        auc_roc_test_ipsi_vs_Nogo = np.nan
        auc_roc_train_ipsi_vs_Nogo = np.nan
    
    
    # format this to a df where each row will be a model after concatenation
    # you can later merge this with other model scores
    metrics_df = pd.DataFrame({
        'model': [model_prefix],
        'logLik_train': [-log_loss_train],
        'logLik_test': [-log_loss_test],
        'logLik_detect_train_total': [scores_per_choicecat_train['Detection_LL_Total']],
        'logLik_detect_test_total': [scores_per_choicecat_test['Detection_LL_Total']],
        'logLik_discrim_train_total': [scores_per_choicecat_train['Discrim_LL_Total']],
        'logLik_discrim_test_total': [scores_per_choicecat_test['Discrim_LL_Total']],
        'logLik_detect_train_avg_per_trial': [scores_per_choicecat_train['Detection_LL_PerTrial']],
        'logLik_detect_test_avg_per_trial': [scores_per_choicecat_test['Detection_LL_PerTrial']],
        'logLik_discrim_train_avg_per_trial': [scores_per_choicecat_train['Discrim_LL_PerTrial']],
        'logLik_discrim_test_avg_per_trial': [scores_per_choicecat_test['Discrim_LL_PerTrial']],
        'brier_detect_train': [scores_per_choicecat_train['Detection_Brier']],
        'brier_detect_test': [scores_per_choicecat_test['Detection_Brier']],
        'brier_discrim_train': [scores_per_choicecat_train['Discrim_Brier']],
        'brier_discrim_test': [scores_per_choicecat_test['Discrim_Brier']],
        'auc_roc_NoGo_vs_rest_train': [auc_roc_train[0]],
        'auc_roc_Left_vs_rest_train': [auc_roc_train[1]],
        'auc_roc_Right_vs_rest_train': [auc_roc_train[2]],
        'auc_roc_NoGo_vs_rest_test': [auc_roc_test[0]],
        'auc_roc_Left_vs_rest_test': [auc_roc_test[1]],
        'auc_roc_Right_vs_rest_test': [auc_roc_test[2]],
        'auc_roc_L_vs_Nogo_train': [auc_roc_train_L_vs_Nogo],
        'auc_roc_R_vs_Nogo_train': [auc_roc_train_R_vs_Nogo],
        'auc_roc_L_vs_Nogo_test': [auc_roc_test_L_vs_Nogo],
        'auc_roc_R_vs_Nogo_test': [auc_roc_test_R_vs_Nogo],
        'auc_roc_R_vs_L_train': [auc_roc_train_R_vs_L], 
        'auc_roc_R_vs_L_test': [auc_roc_test_R_vs_L],
        'hemisphere_fitted': [hemisphere],
        'auc_roc_contra_vs_Nogo_test': [auc_roc_test_contra_vs_Nogo],
        'auc_roc_contra_vs_Nogo_train': [auc_roc_train_contra_vs_Nogo],
        'auc_roc_ipsi_vs_Nogo_test': [auc_roc_test_ipsi_vs_Nogo],
        'auc_roc_ipsi_vs_Nogo_train': [auc_roc_train_ipsi_vs_Nogo],
        
    })    

    return coef_df, updated_ev_df, metrics_df

def fit_session_scipy(subject,date,time_kws,feature_based_selection='none',**choice_neur_selection_kws):

    session_data = load_and_format_session(subject,date,time_kws,**choice_neur_selection_kws)

    neural_predictors = session_data['neuronIDs']
    ev = session_data['ev']

    neural_predictors_contra = [f'{p}_on_contra' for p in neural_predictors]
    neural_predictors_ipsi = [f'{p}_on_ipsi' for p in neural_predictors]

    neural_weights = neural_predictors_contra + neural_predictors_ipsi

    model_init_params = {
        'extra_param_names': neural_weights,
        'extra_param_init': {unit: 0 for unit in neural_weights},
        'extra_param_bounds': {unit: (-5,5) for unit in neural_weights} 
    }


    gamma = ev.gamma.unique()[0]
    all_predictors = ["visL", "visR", "audL", "audR"] + neural_predictors


    train_ev, test_ev = train_test_split(ev, test_size=0.33, random_state=42)
    # Add a column to indicate whether a trial is in the test set
    ev['is_test_set'] = ev.index.isin(test_ev.index)

    X_all = ev[all_predictors]
    y_all = ev['choice']

    X_train = X_all.loc[~ev['is_test_set']].copy()
    y_train = y_all.loc[~ev['is_test_set']].copy()

    if feature_based_selection == 'l1':
        selector = SelectFromModel(estimator = LogisticRegression(penalty="l1", solver="saga", C=.05, max_iter=10000))
        selector.fit(X_train[neural_predictors],y_train)
        
        # reefine the predictors based on the selected features
        neural_predictors = list(np.array(neural_predictors)[selector.get_support()])
        all_predictors = ["visL", "visR", "audL", "audR"] + neural_predictors
        X_all = ev[all_predictors]
        X_train = X_all.loc[~ev['is_test_set']].copy()


    # should apply the standard scaler here to avoid data leakage
    # ohlala this is not done backwards for X_all 

    # option 1, we fit everything
    model_full = av_multi_neural(**model_init_params)
    model_full.fit(X_train, y_train,fixed_params={'gamma':gamma})

    weights, ev, metrics = format_model_output_scipy(
        model_full, ev, X_all, model_prefix='full',
        class_idx_to_name=session_data['class_idx_to_name'])
    
    # now we can also produce the model output when we kindof inactivatate neurons in the X_data
    
    # for k in ['left','right']:
    #     hemi_columns = [col for col in neural_predictors if col.endswith(f'_{k}')]
        
    #     # multiply left hemi columns by 0.2 in X_all
    #     X_all_inactivated = X_all.copy()
    #     X_all_inactivated[hemi_columns] = X_all_inactivated[hemi_columns] -1 

    #     _, ev, metrics_hemi = format_model_output_scipy(
    #         model_full, ev, X_all_inactivated, model_prefix=f'full_neural_{k}_inactivated',
    #         class_idx_to_name=session_data['class_idx_to_name'])
        
    #     metrics = metrics.append(metrics_hemi, ignore_index=True)
    
    # # bilatearlly inactivate neurons
    # X_all_inactivated = X_all.copy()
    # X_all_inactivated[neural_predictors] = X_all_inactivated[neural_predictors] -1
    # _, ev, metrics_bilateral = format_model_output_scipy(
    #     model_full, ev, X_all_inactivated, model_prefix=f'full_neural_bilateral_inactivated',
    #     class_idx_to_name=session_data['class_idx_to_name'])
    # metrics = metrics.append(metrics_bilateral, ignore_index=True)

    # model without the biasR and biasL terms
    model_no_bias = av_multi_neural(**model_init_params)
    model_no_bias.fit(X_train, y_train,fixed_params={'biasR':0, 'biasL':0, 'gamma':gamma})

    weights_no_bias, ev, metrics_no_bias = format_model_output_scipy(
        model_no_bias, ev, X_all, model_prefix='no_bias',
        class_idx_to_name=session_data['class_idx_to_name'])
    metrics = metrics.append(metrics_no_bias, ignore_index=True)
    weights = weights.merge(weights_no_bias,on='feature',how='outer')

    # model with only contralateral neural weights
    model_contra = av_multi_neural(**model_init_params)
    fixed_ipsi_weights = {f'{unit}_on_ipsi':0 for unit in neural_predictors}
    model_contra.fit(X_train, y_train,fixed_params={**fixed_ipsi_weights,'gamma':gamma})

    weights_contra, ev, metrics_contra = format_model_output_scipy(
        model_contra, ev, X_all, model_prefix='contra',
        class_idx_to_name=session_data['class_idx_to_name'])
    metrics = metrics.append(metrics_contra, ignore_index=True)
    weights = weights.merge(weights_contra,on='feature',how='outer')

    model_contra_no_bias = av_multi_neural(**model_init_params)
    model_contra_no_bias.fit(X_train, y_train,fixed_params={**fixed_ipsi_weights,'biasR':0, 'biasL':0,'gamma':gamma})
    weights_contra_no_bias, ev, metrics_contra_no_bias = format_model_output_scipy(
        model_contra_no_bias, ev, X_all, model_prefix='contra_no_bias',
        class_idx_to_name=session_data['class_idx_to_name'])
    metrics = metrics.append(metrics_contra_no_bias, ignore_index=True)
    weights = weights.merge(weights_contra_no_bias,on='feature',how='outer')

    # neural + bias only

    model_neural_bias_only = av_multi_neural(**model_init_params)
    fixed_sensory_weights = {'audR':0,'audL':0,'visR':0,'visL':0}
    model_neural_bias_only .fit(X_train, y_train,fixed_params={**fixed_sensory_weights,'gamma':gamma})
    weights_neural_bias_only, ev, metrics_neural_bias_only = format_model_output_scipy(
        model_neural_bias_only , ev, X_all, model_prefix='neural_bias_only',
        class_idx_to_name=session_data['class_idx_to_name'])
    metrics = metrics.append(metrics_neural_bias_only, ignore_index=True)
    weights = weights.merge(weights_neural_bias_only,on='feature',how='outer')


    # neural only model
    model_neural_only = av_multi_neural(**model_init_params)
    model_neural_only.fit(X_train, y_train,fixed_params={**fixed_sensory_weights,'biasR':0, 'biasL':0,'gamma':gamma})
    weights_neural_only, ev, metrics_neural_only = format_model_output_scipy(
        model_neural_only, ev, X_all, model_prefix='neural_only',   
        class_idx_to_name=session_data['class_idx_to_name'])
    metrics = metrics.append(metrics_neural_only, ignore_index=True)
    weights = weights.merge(weights_neural_only,on='feature',how='outer')

    # # stimulus only model
    model_sensory = av_multi_symmetric_audio()
    model_sensory.fit(X_train, y_train,fixed_params={'gamma':gamma})

    weights_sensory, ev, metrics_sensory = format_model_output_scipy(
        model_sensory, ev, X_all, model_prefix='stim',
        class_idx_to_name=session_data['class_idx_to_name'])
    weights = weights.merge(weights_sensory,on='feature',how='outer')
    metrics = metrics.append(metrics_sensory, ignore_index=True)

    model_bias_only = av_multi_symmetric_audio()
    model_bias_only.fit(X_train, y_train,fixed_params={'audR':0,'audL':0,'visR':0,'visL':0,'gamma':gamma})
    weights_bias_only, ev, metrics_bias_only = format_model_output_scipy(
        model_bias_only, ev, X_all, model_prefix='bias_only',
        class_idx_to_name=session_data['class_idx_to_name'])
    
    metrics = metrics.append(metrics_bias_only, ignore_index=True)
    weights = weights.merge(weights_bias_only,on='feature',how='outer')

    model_zero_pred  = av_multi_symmetric_audio()
    model_zero_pred.fit(X_train,y_train,fixed_params ={'audR':0,'audL':0,'visR':0,'visL':0,'biasL':0,'biasR':0})
    weights_zero_pred, ev, metrics_zero_pred = format_model_output_scipy(
        model_zero_pred, ev, X_all, model_prefix='zero_pred',
        class_idx_to_name=session_data['class_idx_to_name'])
    metrics = metrics.append(metrics_zero_pred, ignore_index=True)
    weights = weights.merge(weights_zero_pred,on='feature',how='outer')

    return ev, weights, metrics

#%%

