# here the idea is that gamma is very hard to fit even session to session
# so we fit one gamma per mice, then we obtain the sensory parameters on a session by session basis
# this is done in src.behav.av_logit_per_session_ephys.py

# now we just use these as stimOdds to fit the neural data
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_selection import VarianceThreshold,SequentialFeatureSelector,SelectFromModel
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


from src.ephys.dat_utils import load_trial_data,get_ephys_dataset


def raise_to_sigm(logOdds):
    "function to calculate pR from logOdds"
    return np.exp(logOdds) / (1 + np.exp(logOdds))

def add_behav_to_ev(ev,subject,date):
    """add the behavioural model parameters to the event dataframe

    Args:
        ev (pd.DataFrame): event dataframe
        subject (str): subject ID
        date (str): date of the session
    """

    behav_params = pd.read_csv(r'C:\Users\Flora\Documents\Github\NeuroLogit\data\behaviour\logit_params_per_session_ephys.csv')
    behav_params = behav_params[(behav_params.subject==subject) & (behav_params.date==date)].iloc[0]
    ev['visDiff_gamma'] = ev.visR**behav_params.gamma - ev.visL**behav_params.gamma

    ev['visR_gamma'] = ev.visR**behav_params.gamma
    ev['visL_gamma'] = ev.visL**behav_params.gamma

    ev['logOdds_stim'] = (
        ev.visR**behav_params.gamma * behav_params.visR -
        ev.visL**behav_params.gamma * behav_params.visL +
        ev.audR * behav_params.audR -
        ev.audL * behav_params.audL 
    )

    ev['logOdds_behav'] = ev['logOdds_stim'] + (behav_params.biasR - behav_params.biasL)
    ev['proba_right_behav'] = raise_to_sigm(ev['logOdds_behav'])
    ev['predicted_choice_behav'] = ev['proba_right_behav'].apply(lambda x: 1 if x > 0.5 else 0)
    ev['logOdds_stim_category'] = pd.cut(ev.logOdds_stim, bins=[-np.inf, -1, -0.5, 0.5, 1,np.inf], labels=np.linspace(-1,1,5))
    return ev

def select_clusters_from_main_region(clusters):
    """
    function filters out noise and non-somatic spikes (keeps mua and good)
    and select the region with the most neurons (to avoid mixing regions) for the session
    """
    clusters_sel = clusters[
        (clusters.bombcell_class.isin(['mua','good'])) & 
        (clusters.BerylAcronym.isin(['SCm','SCs','MOs'])) 
    ].copy()


    # Determine which BerylAcronym has the highest counts
    top_acronym = clusters_sel['BerylAcronym'].value_counts().idxmax()

    # Filter clusters to only include those with the top BerylAcronym
    clusters_sel = clusters_sel[clusters_sel['BerylAcronym'] == top_acronym]
    
    return clusters_sel,top_acronym

def fit_session(subject,date,**time_kws):
    # load relevant events and spiking data
    ev,clusters,_ = load_trial_data(subject,date,load_clusters=True,**time_kws).values()
    ev  =ev[~ev.timeline_choiceMoveDir.isna()].copy() # fit only for trials with a valid choice 
    
    ev = add_behav_to_ev(ev,subject,date)
    ev['bias'] = 1
    sensory_predictors = ['visR_gamma','visL_gamma','audR','audL','bias']

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

    weights = pd.DataFrame(columns=['feature'])


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
                ('sensory_features', 'passthrough', sensory_predictors)
            ]
        )

        pipe= Pipeline([
            ('preproc', full_preprocessor),
            ('log_reg', base_clf)
        ])
        try:
            pipe.fit(X_train,y_train)

            # get the weights of each feature
            coef = pipe.named_steps['log_reg'].coef_.flatten()
            coef_names = pipe.named_steps['preproc'].get_feature_names_out()
            coef_names = [name.split('__')[-1] for name in coef_names]
            coef_df = pd.DataFrame({'feature':coef_names,f'weight_{model_name}':coef})
            weights = weights.merge(coef_df,on='feature',how='outer')

            # get the predictions on all trials
            ev['logOdds_'+model_name] = pipe.decision_function(X_all)
            ev['proba_right_'+model_name] = pipe.predict_proba(X_all)[:,1]
            ev['predicted_choice_'+model_name] = pipe.predict(X_all)

            neural_coefs_idx = np.array([i for i,c in enumerate(coef_names) if c not in sensory_predictors])
            
            if len(neural_coefs_idx)>0:
                X_train_preproc = pipe.named_steps['preproc'].transform(X_train)
                X_all_preproc = pipe.named_steps['preproc'].transform(X_all)
                
                X_train_preproc_neur = X_train_preproc[:,neural_coefs_idx]
                X_all_preproc_neur = X_all_preproc[:,neural_coefs_idx]
                # retrain the neuron-only model
                neural_clf = LogisticRegression(penalty=None,solver='lbfgs',max_iter=10000,fit_intercept=False)
                neural_clf.fit(X_train_preproc_neur, y_train)

                ev['logOdds_neural_'+model_name] = neural_clf.decision_function(X_all_preproc_neur)
                ev['proba_right_neural_'+model_name] = neural_clf.predict_proba(X_all_preproc_neur)[:,1]
                ev['predicted_choice_neural_'+model_name] = neural_clf.predict(X_all_preproc_neur)

            # also calculate the logOdds etc by the neural features only (without the sensory predictors)

        except Exception as e:
            print(f"Could not fit model {model_name} for {subject} {date}: {e}")
            ev['proba_right_'+model_name] = np.nan
            ev[f'predicted_choice_{model_name}'] = np.nan
            ev['logOdds_'+model_name] = np.nan

    ### train the stimulus only model ###
    behav_clf = LogisticRegression(penalty=None,solver='lbfgs',max_iter=10000,fit_intercept=False)
    behav_clf.fit(X_train[sensory_predictors], y_train)

    # get the weights of the sensory predictors
    coef = behav_clf.coef_.flatten()
    coef_df = pd.DataFrame({'feature':sensory_predictors,'weight_stim':coef})
    weights = weights.merge(coef_df,on='feature',how='outer')

    # get the predictions on all trials
    ev['logOdds_stim'] = behav_clf.decision_function(ev[sensory_predictors])
    ev['proba_right_stim'] = behav_clf.predict_proba(ev[sensory_predictors])[:,1]
    ev['predicted_choice_stim'] = behav_clf.predict(ev[sensory_predictors])


    # train the intercept only model
    bias_predictor = ['bias']
    intercept_clf = LogisticRegression(penalty=None,solver='lbfgs',max_iter=10000,fit_intercept=False)
    intercept_clf.fit(X_train[bias_predictor], y_train)

    coef = intercept_clf.coef_.flatten()
    coef_df = pd.DataFrame({'feature':bias_predictor,'weight_bias':coef})
    weights = weights.merge(coef_df,on='feature',how='outer')

    # get the predictions of the bias predictor
    ev['logOdds_bias'] = intercept_clf.decision_function(ev[bias_predictor])
    ev['proba_right_bias'] = intercept_clf.predict_proba(ev[bias_predictor])[:,1]
    ev['predicted_choice_bias'] = intercept_clf.predict(ev[bias_predictor])


    return ev,weights

def get_time_kws(which):
    if which == 'prestim':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.2,'post_time':0.0}}
    elif which == 'stim':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.0,'post_time':0.2}}
    elif which == 'choice':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.2,'post_time':0.0}}    
    elif which == 'prestim0':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0,'post_time':0.05}}
    elif which == 'prestim1':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.05,'post_time':0.0}}
    elif which == 'prestim2':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.1,'post_time':-0.05}}
    elif which == 'prestim3':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.15,'post_time':-0.1}}
    elif which == 'prestim4':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.2,'post_time':-0.15}}
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
    elif which == 'choice7':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.3,'post_time':-0.25}}
    elif which == 'choice8':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.35,'post_time':-0.3}}
    elif which == 'choice9':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.4,'post_time':-0.35}}
    else:
        raise ValueError("Unknown time window")

def batch_fit_sessions(recompute=False):
    """fit all sessions and save the results"""

    save_path = Path(r'D:\AV_Neural_Data_Sept2025\decoding_choice_results')
    sessions = get_ephys_dataset(set_name='all', subset='')
    for i,rec in sessions.iterrows():
        session_path = save_path / f"{rec.subject}_{rec.date}"

        if (not session_path.exists()) or recompute:
            try: 
                print(f"fitting {rec.subject} {rec.date}")

                session_path.mkdir(exist_ok=True,parents=True)
                #times = ['prestim','stim','choice']
                #times += [f'choice{i}' for i in range(9)] # finer bins around choice
                times = [f'prestim{i}' for i in range(4)] # finer bins around prestim
                for t in times: 
                    time_kws = get_time_kws(t)
                    task_ev,weights = fit_session(rec.subject,rec.date,**time_kws)
                    # no need to save all the neuron columns as they will vary across sessions
                    keep_cols = [c for c in task_ev.columns if 'neuron_' not in c]
                    task_ev = task_ev[keep_cols].copy()

                    # add subject, date and session ID to the df
                    task_ev['subject'] = rec.subject
                    task_ev['date'] = rec.date
                    task_ev['sessionID'] = f"{rec.subject}_{rec.date}"

                    weights['subject'] = rec.subject
                    weights['date'] = rec.date
                    weights['sessionID'] = f"{rec.subject}_{rec.date}"
                    
                    # save the results
                    timing_stub = f'{t}_pre{time_kws["avg_kwargs"]["pre_time"]}_post{time_kws["avg_kwargs"]["post_time"]}'
                    task_ev.to_csv(session_path / f'task_ev_{timing_stub}.csv',index=True)
                    weights.to_csv(session_path / f'weights_{timing_stub}.csv',index=True)
            except Exception as e:
                print(f'error processing {rec.subject} {rec.date}: {e}')
                with open(session_path / 'error_log.txt', 'w') as f:
                    f.write(str(e))


if __name__ == "__main__":
    batch_fit_sessions(recompute=True)
