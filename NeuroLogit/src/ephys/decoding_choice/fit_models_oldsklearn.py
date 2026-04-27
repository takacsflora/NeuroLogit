# here the idea is that gamma is very hard to fit even session to session
# so we fit one gamma per mice, then we obtain the sensory parameters on a session by session basis
# this is done in src.behav.av_logit_per_session_ephys.py

# now we just use these as stimOdds to fit the neural data
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_selection import VarianceThreshold,SelectFromModel
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.metrics import log_loss,roc_auc_score


from src.ephys.dat_utils import load_trial_data
from NeuroLogit.src.ephys.decoding_choice.fit_helpers import select_clusters_from_main_region,pairwise_auc_roc,add_behav_to_ev

def format_model_output(model, ev_df, X_data, model_prefix, feature_names, class_idx_to_name):
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
    # --- 1. Process Coefficients ---
    coef_df = pd.DataFrame({'feature': feature_names})
    
    # Create an inverse mapping from class index to string name for safety
    idx_to_name = {i: class_idx_to_name[c] for i, c in enumerate(model.classes_)}
    
    # Store the weights for each class
    for i, class_label in enumerate(model.classes_):
        class_name = idx_to_name[i]
        coef_df[f'weight_{class_name}'] = model.coef_[i]

    # --- 2. Process Predictions ---
    updated_ev_df = ev_df.copy()
    scores = model.decision_function(X_data)
    probas = model.predict_proba(X_data)
    
    # Store scores and probabilities for each class
    for i, class_label in enumerate(model.classes_):
        class_name = idx_to_name[i]
        updated_ev_df[f'score_{class_name}_{model_prefix}'] = scores[:, i]
        updated_ev_df[f'proba_{class_name}_{model_prefix}'] = probas[:, i]

    # Add the overall predicted choice
    updated_ev_df[f'predicted_choice_{model_prefix}'] = model.predict(X_data)

    # --- 3. Calculate and Add Reference-Based Metrics (Your Goal) ---
    # Find the indices for our specific classes of interest
    name_to_idx = {v: k for k, v in class_idx_to_name.items()}

    class_list = model.classes_.tolist()
    if name_to_idx['Right'] in class_list and name_to_idx['NoGo'] in class_list:
        right_idx = class_list.index(name_to_idx['Right'])
        nogo_idx = class_list.index(name_to_idx['NoGo'])
        
        # Derived log-odds from scores: score_R - score_NoGo
        updated_ev_df[f'logOdds_R_vs_NoGo_{model_prefix}'] = scores[:, right_idx] - scores[:, nogo_idx]
        
        # Derived weights from coefficients: w_R - w_NoGo
        coef_df['weight_R_vs_NoGo'] = coef_df['weight_Right'] - coef_df['weight_NoGo']

    if name_to_idx['Left'] in class_list and name_to_idx['NoGo'] in class_list:
        left_idx = class_list.index(name_to_idx['Left'])
        nogo_idx = class_list.index(name_to_idx['NoGo'])
        updated_ev_df[f'logOdds_L_vs_NoGo_{model_prefix}'] = scores[:, left_idx] - scores[:, nogo_idx]
        coef_df['weight_L_vs_NoGo'] = coef_df['weight_Left'] - coef_df['weight_NoGo']

    if name_to_idx['NoGo'] in class_list: 
        updated_ev_df[f'logOdds_L_vs_R_{model_prefix}'] = (
            scores[:, left_idx] - scores[:, right_idx]
        )

        # are you sure this is the formula? 
        updated_ev_df[f'logOdds_Go_vs_NoGo_{model_prefix}'] = np.log(
            np.exp(updated_ev_df[f'logOdds_L_vs_NoGo_{model_prefix}']) + np.exp(updated_ev_df[f'logOdds_R_vs_NoGo_{model_prefix}'])
        )

    # For backward compatibility with any downstream code, you can keep proba_right
    if (name_to_idx['Right'] in class_list) and not (name_to_idx['NoGo'] in class_list):
        right_idx = class_list.index(name_to_idx['Right'])
        updated_ev_df[f'logOdds_{model_prefix}'] = scores[:, right_idx]
        updated_ev_df[f'proba_right_{model_prefix}'] = probas[:, right_idx]

    
    coef_df = coef_df.rename(columns=lambda c: f"{c}_{model_prefix}" if c != 'feature' else c)

    # probably it is also good to add metrics of model performance here?
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

    # now we will also calculate the auc_roc pairwise
    # now we will also calculate the auc_roc pairwise
    auc_roc_train_L_vs_Nogo = pairwise_auc_roc(y_true_train,zL_train,pair=[name_to_idx['Left'],name_to_idx['NoGo']])
    auc_roc_train_R_vs_Nogo = pairwise_auc_roc(y_true_train,zR_train,pair=[name_to_idx['Right'],name_to_idx['NoGo']])
    auc_roc_test_L_vs_Nogo = pairwise_auc_roc(y_true_test,zL_test,pair=[name_to_idx['Left'],name_to_idx['NoGo']])
    auc_roc_test_R_vs_Nogo = pairwise_auc_roc(y_true_test,zR_test,pair=[name_to_idx['Right'],name_to_idx['NoGo']])
    auc_roc_train_R_vs_L = pairwise_auc_roc(y_true_train, zR_train - zL_train, pair=[name_to_idx['Right'],name_to_idx['Left']])
    auc_roc_test_R_vs_L = pairwise_auc_roc(y_true_test, zR_test - zL_test, pair=[name_to_idx['Right'],name_to_idx['Left']])



    # format this to a df where each row will be a model after concatenation
    # you can later merge this with other model scores
    metrics_df = pd.DataFrame({
        'model': [model_prefix],
        'log_loss_train': [log_loss_train],
        'log_loss_test': [log_loss_test],
        'auc_roc_Left_vs_rest_train': [auc_roc_train[0]],
        'auc_roc_Right_vs_rest_train': [auc_roc_train[1]],
        'auc_roc_NoGo_vs_rest_train': [auc_roc_train[2]],
        'auc_roc_Left_vs_rest_test': [auc_roc_test[0]],
        'auc_roc_Right_vs_rest_test': [auc_roc_test[1]],
        'auc_roc_NoGo_vs_rest_test': [auc_roc_test[2]],
        'auc_roc_L_vs_Nogo_train': [auc_roc_train_L_vs_Nogo],
        'auc_roc_R_vs_Nogo_train': [auc_roc_train_R_vs_Nogo],
        'auc_roc_L_vs_Nogo_test': [auc_roc_test_L_vs_Nogo],
        'auc_roc_R_vs_Nogo_test': [auc_roc_test_R_vs_Nogo],
        'auc_roc_R_vs_L_train': [auc_roc_train_R_vs_L], 
        'auc_roc_R_vs_L_test': [auc_roc_test_R_vs_L],

    })    

    return coef_df, updated_ev_df, metrics_df
    
def fit_session_sklearn(subject,date,multi_class = 'multinomial',**time_kws):
    # load relevant events and spiking data
    ev,clusters,_ = load_trial_data(subject,date,load_clusters=True,**time_kws).values()

    # filter differently if multi class or binary (but many things are the same so we will not write a separate function)
    # otherwise no need to write the multiclass into the logistic because it is supposed to automatically detect it
    if multi_class=='multinomial':
        ev = ev[ev.choice!=-2].copy()
        # Replace -1 or -2 choice values with 2
        ev.loc[ev.choice.isin([-1, -2]), 'choice'] = 2
    else:
        ev  = ev[~ev.timeline_choiceMoveDir.isna()].copy() # fit only for trials with a valid choice    

    ev = add_behav_to_ev(ev,subject,date) # this might be different for 3-choice
    ev['bias'] = 1
    sensory_predictors = ['visR_gamma','visL_gamma','audR','audL','bias']
    choice_name_mapping = {  0:'Left', 1:'Right', 2:'NoGo' }

    # neural predictors
    clusters_sel,top_region = select_clusters_from_main_region(clusters)
    neuron_cols = clusters_sel.neuronID.tolist()    
    left_neurons = clusters_sel[clusters_sel.hemi==1].neuronID.tolist()
    right_neurons = clusters_sel[clusters_sel.hemi==-1].neuronID.tolist()

    all_predictors = sensory_predictors + neuron_cols

    # Split the data into training and testing sets
    train_ev, test_ev = train_test_split(ev, test_size=0.33, random_state=42)
    # Add a column to indicate whether a trial is in the test set
    ev['is_test_set'] = ev.index.isin(test_ev.index)
    ev['roi_fitted'] = top_region
    X_all = ev[all_predictors].copy()
    X_train = train_ev[all_predictors].copy()
    y_train = train_ev.choice.copy()

    X_test = test_ev[all_predictors].copy()
    y_test = test_ev.choice.copy()

    weights = pd.DataFrame(columns=['feature']) # just initialized weights
    metrics = pd.DataFrame()  # initialize metrics dataframe

    ### train the neural models ###
    #  fit a bunch of different models with different neuron selection strategies
    base_clf = LogisticRegression(penalty=None,solver='lbfgs',max_iter=10000,fit_intercept=False)
    neuron_selectors = {
        "l1_0.05": SelectFromModel(
            LogisticRegression(penalty="l1", solver="saga", C=.05, max_iter=10000)
        ),
        # "forward": SequentialFeatureSelector( # takes 30 s to fit  -- ok but actually doesn't seem to do much better than the others
        #     base_clf,
        #     n_features_to_select=10,   # adjust depending on dataset
        #     direction="forward",
        #     cv=2,tol=0.01,
        #     n_jobs=-1
        # ),
        'pca_5': PCA(n_components=5),  # top 5 PCs
        #"pca_10": PCA(n_components=10)  # top 10 PCs
    }

    sensory_no_bias_predictors = [p for p in sensory_predictors if p != 'bias']
    for model_name in neuron_selectors.keys():

        # # each hemisphere is preprocessed differently as we are flipping the activity of the right hemisphere neurons
        left_pipeline = Pipeline([
            ('var_thresh', VarianceThreshold(threshold=1)),
            ('scaler', StandardScaler())]) 
        
        right_pipeline = Pipeline([
            ('var_thresh', VarianceThreshold(threshold=1)),
            ('scaler', StandardScaler()),
            ('negate', FunctionTransformer(lambda x: -x, feature_names_out='one-to-one'))])  # negate the features for right hemisphere neurons
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('left', left_pipeline, left_neurons),
                ('right', right_pipeline, right_neurons)
            ]
        )

        # neural_feature selection 
        neural_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection',  neuron_selectors[model_name])
        ])

        full_preprocessor = ColumnTransformer(
            transformers=[
                ('neural_feature_proc', neural_pipeline, neuron_cols),
                ('sensory_features', 'passthrough', sensory_no_bias_predictors)
            ]
        )

        pipe= Pipeline([
            ('preproc', full_preprocessor),
            ('log_reg', base_clf)
        ])
        try:
            all_except_bias_predictors = [col for col in all_predictors if col != 'bias']
            pipe.fit(X_train[all_except_bias_predictors],y_train)

            coef_names_raw = pipe.named_steps['preproc'].get_feature_names_out()
            coef_names = [name.split('__')[-1] for name in coef_names_raw]
            
            coef_df_full, ev, metrics_full = format_model_output(
                model=pipe.named_steps['log_reg'],
                ev_df=ev,
                X_data=pipe.named_steps['preproc'].transform(X_all[all_except_bias_predictors]),
                model_prefix=model_name,
                feature_names=coef_names,
                class_idx_to_name=choice_name_mapping
            )

            weights = weights.merge(coef_df_full,on='feature',how='outer')
            metrics = metrics.append(metrics_full, ignore_index=True)

            neural_coefs_idx = np.array([i for i,c in enumerate(coef_names) if c not in sensory_predictors])
            
            if len(neural_coefs_idx)>0:
                X_train_preproc = pipe.named_steps['preproc'].transform(X_train)
                X_all_preproc = pipe.named_steps['preproc'].transform(X_all)
                
                X_train_preproc_neur = X_train_preproc[:,neural_coefs_idx]
                X_all_preproc_neur = X_all_preproc[:,neural_coefs_idx]
                # retrain the neuron-only model
                neural_clf = LogisticRegression(penalty=None,solver='lbfgs',max_iter=10000,fit_intercept=False)
                neural_clf.fit(X_train_preproc_neur, y_train)

                
                coef_df_neural, ev, metrics_neural = format_model_output(
                    model=neural_clf,
                    ev_df=ev,
                    X_data=X_all_preproc_neur,
                    model_prefix='neural_'+model_name,
                    feature_names=[coef_names[i] for i in neural_coefs_idx],
                    class_idx_to_name=choice_name_mapping
                )
                metrics = metrics.append(metrics_neural, ignore_index=True)

        except Exception as e:
            print(f"Could not fit model {model_name} for {subject} {date}: {e}")
            ev['proba_right_'+model_name] = np.nan
            ev[f'predicted_choice_{model_name}'] = np.nan
            ev['logOdds_'+model_name] = np.nan

    ### train the stimulus only model ###
    behav_clf = LogisticRegression(penalty=None,solver='lbfgs',max_iter=10000,fit_intercept=False)
    behav_clf.fit(X_train[sensory_predictors], y_train)

    coef_df_behav, ev, metrics_behav = format_model_output(
        model=behav_clf,
        ev_df=ev,
        X_data=X_all[sensory_predictors],
        model_prefix='stim',
        feature_names=sensory_predictors,
        class_idx_to_name=choice_name_mapping
    )
    weights = weights.merge(coef_df_behav,on='feature',how='outer')
    metrics = metrics.append(metrics_behav, ignore_index=True)


    # train the intercept only model
    bias_predictor = ['bias']
    intercept_clf = LogisticRegression(penalty=None,solver='lbfgs',max_iter=10000,fit_intercept=False)
    intercept_clf.fit(X_train[bias_predictor], y_train)

    coef_df_intercept, ev, metrics_intercept = format_model_output(
        model=intercept_clf,
        ev_df=ev,
        X_data=X_all[bias_predictor],
        model_prefix='intercept',
        feature_names=bias_predictor,
        class_idx_to_name=choice_name_mapping
    )
    weights = weights.merge(coef_df_intercept,on='feature',how='outer')
    metrics = metrics.append(metrics_intercept, ignore_index=True)

    return ev,weights, metrics