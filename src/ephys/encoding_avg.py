#%%

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


from src.ephys.dat_utils import load_trial_data,get_ephys_dataset


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

# linear regression with gridsearching gamma 

######## functions for model selection  ########
def get_predictors(fit_type='vis'):
    # Define the predictors based on the fit type
    if fit_type == 'vis':
        predictors = ['visC']
    elif fit_type == 'vis_bilateral':
        predictors = ['visC','visI']
    elif fit_type == 'aud':
        predictors = ['audC']
    elif fit_type == 'aud_bilateral':
        predictors = ['audC', 'audI']
    elif fit_type == 'av':
        predictors = ['visC', 'audC']
    elif fit_type == 'av_aud_bilateral':
        predictors = ['visC', 'audC', 'audI']
    elif fit_type == 'av_bilateral':
        predictors = ['visC', 'audC', 'visI', 'audI']
    elif fit_type == 'av_multiplicative':
        pass
    elif fit_type == 'task':
        predictors = ['task']
    elif fit_type == 'choice':
        predictors = ['task','choice']
    elif fit_type == 'vis_task':
        predictors = ['visC','task']
    elif fit_type == 'aud_task':
        predictors = ['audC','task']
    elif fit_type == 'aud_bilateral_task':
        predictors = ['audC','audI','task']
    elif fit_type == 'av_task_bilateral':
        predictors = ['visC','audC','audI','task']
    elif fit_type == 'av_task':
        predictors = ['visC','audC','task']
    elif fit_type == 'vis_choice':
        predictors = ['visC','task','choice']
    elif fit_type == 'aud_choice':
        predictors = ['audC','task','choice']
    elif fit_type == 'aud_bilateral_choice':
        predictors = ['audC','audI','task','choice']
    elif fit_type == 'av_choice':
        predictors = ['visC','audC','task','choice']
    elif fit_type == 'av_aud_bilateral_choice':
        predictors = ['visC','audC','audI','task','choice']
    elif fit_type == 'baseline':
        predictors = []

    predictors+= ['baseline']
    
    return predictors

def get_tested_models(fit_type = 'passive'):
    # Define the models we want to test -- this is where I am going to introduce fit_type as an arg
    # models = ['vis','aud','av','baseline']

    if fit_type == 'passive':
        models = [
            'vis',
            'vis_bilateral',
            'aud',
            'aud_bilateral',
            'av',
            'av_aud_bilateral',
            'av_bilateral',
            'baseline',

        ]
    elif fit_type == 'active':
        models = [
            'vis',
            'aud',
            'aud_bilateral',
            'av',
            'av_aud_bilateral',
            'baseline',
            'task',
            'choice',
            'vis_task',
            'aud_task',
            'av_task',
            'aud_bilateral_task',
            'av_task_bilateral',
            'vis_choice',
            'aud_choice',
            'av_choice', 
            'aud_bilateral_choice',
            'av_aud_bilateral_choice',

        ]

    return models

def get_visual_models(model_list):
    """get the visual models from the list of models
    works by identifying which model uses the visC predictor

    Args:
        model_list (list): list of model names  (needs to call get_model)

    Returns:
        list: list of visual models
    """

    return [m for m in model_list if'visC' in get_predictors(m)]

def get_model_gamma_combinations(**kwargs):
    # Define the gamma values for each model
    # vis and av models will have a range of gammas, while the rest will be fixed at 1
    # this is based on the observation that in the previous analysis,

    # most gammas appear to fall between 0 and 2 with occasionally going to 3,4

    gamma_grid = np.concatenate([
        np.round(np.arange(0.1, 2.1, 0.1),2), 
        np.array([2.5])
    ])

    tested_models = get_tested_models(**kwargs)
    visual_models = get_visual_models(tested_models)



    model_gamma_combinations = [
        (model, gamma) if model in visual_models else (model, 1)
        for model in tested_models
        for gamma in (gamma_grid if model in visual_models else [1])
    ]

    return model_gamma_combinations

####### functions for preprocessing the predictors #######
def get_predictor_matrix(df,hemi=1):
    # Create a new DataFrame for the predictors
    X = df[['choice']].copy()
    X['baseline'] = 1
    visDiff_hemispheric = df.visDiff * hemi
    audDiff_hemispheric = df.audDiff * hemi
    X['visC'] = np.abs(visDiff_hemispheric) * (visDiff_hemispheric>0)
    X['visI'] = np.abs(visDiff_hemispheric) * (visDiff_hemispheric<0)
    X['audC'] = np.abs(audDiff_hemispheric) * (audDiff_hemispheric>0)
    X['audI'] = np.abs(audDiff_hemispheric) * (audDiff_hemispheric<0)

    # add multiplicative terms for audiovisual 

    # in the input choice is 1 for right, 0 for left, 0 for nogo and -2 for passive. 
    # But for the predictor matrix we want to have 1 for right, -1 for left, 0 for nogo and 0 for passive too.
    # thenm we need to multiply by hemi to make +ve values contra choice tuning, essentially.

    X['choice'] = df['choice'].replace({1: 1, 0: -1, -1: 0, -2: 0})
    X['choice'] *= hemi

    X['task'] = (df.session=='active').astype('int')

    return X

def gamma_transform(X,gamma=1):
    # columns in X that contain vis will be raised to the power of gamma
    # and the rest will be left as is
    X = X.copy()
    for col in X.columns:
        if 'vis' in col:
            X[col] = X[col]**gamma
    return X

def filt_trials(df,fit_type = 'passive'): 
    """essentially filter the rows of the df (= trials) based on what we want to fit. Options atm are: 
    passive, all.
    Args:
        df (pd.df): pandas dataframe with the trial data
        fit_type (str, optional): indentifier indicating how to filter. Defaults to 'passive'.

    Returns:
        _type_: _description_
    """

    if fit_type == 'passive':
        df = df[df.session=='passive'].copy()
    elif fit_type == 'active':
        # will get rid of the NoGo trials atm because they are a bit of a mess? 
        # in general here, I might not be filtering everything...
        df = df[df.choice_categorical!='NoGo'].copy()

    else:
        raise ValueError('fit_type not recognized')
    
    return df

###### functions for model fitting and evaluation #######
def get_scores_per_neuron(y_actual,y_pred,metric = 'r2_score'):
    """Evaluate each row in the predicted matrix separately (i.e. each neuron separately)

    Args:
        y_actual (np.ndarray): predicted matrix (trials x nrns)
        y_pred (np.ndarray): acutal response matrix (trials x nrns)

    Returns:
        _type_: _description_
    """
    nrns = y_actual.columns
    scorer = getattr(metrics, metric)
    scores = [scorer(y_actual[neuron], y_pred[:, nrns.get_loc(neuron)]) for neuron in nrns]

    return  pd.DataFrame({'neuronID':nrns,metric:scores})

def calculate_adjusted_r2(r2, n, p):
    """calculating the adjusted r2 score, i.e. coefficient of determination
    https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2

    Args:
        r2 (np.ndarray): r2 score of the model
        n (int): number of samples the r2 was calculated on
        p (int): number of predictors,excluding the intercept

    Returns:
        float: adjusted r2 score
    """
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

def train_and_evaluate_model(df,clusters,fit_type = 'passive',hemi=1):
    """model fitting for neurons with a common predictor matrix 
    (i.e. neurons on the same probe).

    Args:
        df (pd.df): trial data 
        clusters (_type_): _description_
        hemi (int, optional): whether the neurons are on the left or right (1 or -1) hemisphere. Defaults to 1. 
            Calculates contra and ipsi predictors from left and right stimuli, essentially. 

    Returns:
        pd.df: predictor weights and scores for each neuron
    """
    # Get the predictor matrix and the target variable
    X_all = get_predictor_matrix(df,hemi=hemi)
    selected_nrns = clusters[clusters.hemi==hemi].neuronID
    y = df[selected_nrns]



    models_to_fit = get_model_gamma_combinations(fit_type=fit_type)

    all_fits = []
    for model_name, gamma in models_to_fit:
        predictors = get_predictors(fit_type=model_name)
        X = X_all[predictors].copy()
        X = gamma_transform(X,gamma=gamma)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Perform linear regression
        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)
        r2_score = get_scores_per_neuron(y_test,y_pred,metric='r2_score')

        # adjust the r2 score for the number of predictors
        n_test_trials = X_test.shape[0]
        n_predictors = len(predictors) - 1  # exclude intercept 
        # if the model is visual we include gamma though
        if 'visC' in predictors:
            n_predictors += 1
        adj_r2 = calculate_adjusted_r2(r2_score['r2_score'], n_test_trials, n_predictors)
        r2_score['adj_r2'] = adj_r2



        mse = get_scores_per_neuron(y_test,y_pred,metric='mean_squared_error')
        explained_variance = get_scores_per_neuron(y_test,y_pred,metric='explained_variance_score')

        # Merge r2_score, mse, and explained_variance on neuronID without setting it as the index
        scores = r2_score.merge(mse, on='neuronID').merge(explained_variance, on='neuronID')

        # also score each parameter -- 
        # retrain the model, 
        # subtract the prediction from y_test to get the residuals
        # the score on the residual by each preictor
        tested_predictors = [p for p in predictors if p!= 'baseline']
        residual_metric = 'explained_variance_score'

        for tested_predictor in tested_predictors:
            remaining_predictors = [p for p in predictors if p!= tested_predictor]
            X_train_leave_one = X_train[remaining_predictors].copy()
            X_test_leave_one = X_test[remaining_predictors].copy()

            model_leave_one = LinearRegression(fit_intercept=False)
            model_leave_one.fit(X_train_leave_one, y_train)
            train_pred_leave_one = model_leave_one.predict(X_train_leave_one)
            train_diff = y_train - train_pred_leave_one
            test_pred_leave_one = model_leave_one.predict(X_test_leave_one)
            test_diff = y_test - test_pred_leave_one

            one_train = X_train[[tested_predictor]].copy()
            one_test = X_test[[tested_predictor]].copy()
            model_residual = LinearRegression(fit_intercept=False)
            model_residual.fit(one_train, train_diff)
            test_pred_residual = model_residual.predict(one_test)
            
            score_test_predictor  = get_scores_per_neuron(test_diff,
                                                        test_pred_residual,metric=residual_metric)
            score_test_predictor.rename(columns={f'{residual_metric}': f"{residual_metric}_{tested_predictor}"}, inplace=True)

            scores = scores.merge(score_test_predictor, on='neuronID')




        # add also the coefs 
        coefs = pd.DataFrame(model.coef_,columns=X_train.columns)
        coefs['gamma'] = gamma
        coefs['model'] = model_name
        all_fits.append(pd.concat([scores,coefs],axis=1))

    return pd.concat(all_fits,axis=0).assign(hemi=hemi)

def fit_session(df,clusters,fit_type = 'passive'):
    """fitting the modelto all neurons in a session.
    This is done by calling the train_and_evaluate_model function for each hemisphere separately.

    Args:
        df (pd.df): trial data
        clusters (pd.df): cluster data 

    Returns:
        pd.df: model fits for each neuron in the session
    """
    unique_hemis = clusters.hemi.dropna().unique()
    model_fits = pd.concat([train_and_evaluate_model(df,clusters,hemi=hemi,fit_type=fit_type) for hemi in unique_hemis])


    clusters_added_columns = ['neuronID','BerylAcronym','bombcell_class','is_good','ml','ap','dv']
    model_fits = model_fits.merge(clusters[clusters_added_columns], on='neuronID', how='left')
    return model_fits

############ functions for processing the whole dataset ############
def get_time_params(time_window,pre_time = None,post_time = None):

    if time_window == 'prestim':
        kwargs = {
            'load_raster': 'stim',
            'avg_kwargs': {'pre_time': 0.15, 'post_time': 0.0}
        }

    elif time_window == 'stim':
        kwargs = {
            'load_raster': 'stim',
            'avg_kwargs': {'pre_time': 0.0, 'post_time': 0.15}
        }

    elif time_window == 'choice':
        kwargs = {
            'load_raster': 'choice',
            'avg_kwargs': {'pre_time': 0.15, 'post_time': 0.0}
        }

    elif 'stim_bin' in time_window: 
        kwargs = {
            'load_raster': 'stim',
            'avg_kwargs': {'pre_time': pre_time, 'post_time': post_time}
        }

    elif 'choice_bin' in time_window: 
        kwargs = {
            'load_raster': 'choice',
            'avg_kwargs': {'pre_time': pre_time, 'post_time': post_time}
        }

    return kwargs

def fit_dataset(fit_type='passive',
        dataset_kwargs={'set_name':'all'},
        time_kwargs={'time_window':'stim','pre_time':0.0,'post_time':0.15}
        ):
    """_summary_

    Args:
        dataset_kwargs (dict, optional): _description_. Defaults to {'set_name':'all'}.
        time_kwargs (dict, optional): _description_. Defaults to {'time_window':'stim','pre_time':0.0,'post_time':0.15}.
    """

    sessions = get_ephys_dataset(**dataset_kwargs)
    time_params = get_time_params(**time_kwargs)
    coefs = []
    for _,args in sessions[['subject','date']].iterrows():
            df,clusters,_  = load_trial_data(**args,**time_params).values()

            if df is not None:
                ## to rewrite this so that we allow multiple tested model sets
                tested_df = filt_trials(df,fit_type=fit_type)
                model_fits = fit_session(tested_df,clusters,fit_type=fit_type)
                model_fits['subject'] = args['subject']
                model_fits['date'] = args['date']
                coefs.append(model_fits)

    # create a new column that is fitID, and its is a string of subject date and neuronID
    coefs = pd.concat(coefs)
    coefs['fitID'] = coefs['subject'] + '_' + coefs['date'] + '_' + coefs['neuronID']

    return coefs.reset_index(drop=True)

def get_winning_model(model_fits,thr_scorer='adj_r2',thr=0):
    model_fits = model_fits.reset_index(drop=True)
    best_model = model_fits.loc[model_fits.groupby('fitID')[thr_scorer].idxmax()]

    # Replace rows in best_model with score < 0 with the baseline model from model_fits
    baseline_model = model_fits[model_fits['model'] == 'baseline']
    best_model = best_model.apply(
        lambda row: baseline_model[baseline_model['fitID'] == row['fitID']].iloc[0]
        if row[thr_scorer] < thr else row, axis=1
    )
    return best_model


#%%

def compute_means_and_errors(df):
        means = df.groupby(['visDiff', 'audDiff','task','choice'])['response'].mean().reset_index()
        errors = df.groupby(['visDiff', 'audDiff','task','choice'])['response'].sem().reset_index()
        return means, errors

def generate_pseudo():
    n_points =  600
    visDiff = np.linspace(-1,1,n_points)
    visC = np.abs(visDiff) * (visDiff>0)
    visI = np.abs(visDiff) * (visDiff<0)

    audCs = [1,0,0]
    audIs = [0,0,1]

    baseline = np.ones(n_points) * 1

    pseudo = []
    for audC,audI in zip(audCs,audIs):
        audC = np.ones(n_points) * audC
        audI = np.ones(n_points) * audI

        pseudo.append(pd.DataFrame({
            'visC': visC,
            'visI': visI,
            'audC': audC,
            'audI': audI,
            'baseline': baseline
        }))

    return pseudo

def plot_prediction(df,nrn_model):
    print(nrn_model)
    # to plot the actual points 
    nrn = nrn_model.neuronID.values[0]
    nrn_best = nrn_model.model.values[0]
    nrn_hemi = nrn_model.hemi.values[0]
    nrn_gamma = nrn_model.gamma.values[0]

    nrn_df = get_predictor_matrix(df,hemi=nrn_hemi)
    nrn_df = gamma_transform(nrn_df,gamma=nrn_gamma)



    nrn_df['response'] = df[nrn].copy()
    nrn_df['visDiff'] = nrn_df['visC'] - nrn_df['visI']
    nrn_df['audDiff'] = nrn_df['audC'] - nrn_df['audI']
    
    means,errors = compute_means_and_errors(nrn_df)


    task_choice_combinations = nrn_df.groupby(['task', 'choice']).size().reset_index().rename(columns={0: 'count'})

    unique_audDiffs = [1,0,-1]
    colors = ['red','gray','blue']


    predictors = get_predictors(nrn_best)
    weights = nrn_model[predictors]
    pseudo = generate_pseudo()


    # Loop over unique combinations of task and choice for subplots
    n_combinations = len(task_choice_combinations)
    fig, axes = plt.subplots(1, n_combinations, figsize=(2*n_combinations, 2), dpi=300, sharex=True, sharey=True)
    
    if n_combinations == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (task, choice) in enumerate(task_choice_combinations[['task', 'choice']].values):
        ax = axes[i]
        means_subset = means[(means.task == task) & (means.choice == choice)]
        errors_subset = errors[(errors.task == task) & (errors.choice == choice)]

        for audDiff, current_pseudo, color in zip(unique_audDiffs, pseudo, colors):
            means_ = means_subset[means_subset.audDiff == audDiff]
            errors_ = errors_subset[errors_subset.audDiff == audDiff]
            ax.errorbar(
                x=means_.visDiff,
                y=means_.response,
                yerr=errors_.response,
                label=f'{audDiff}',
                color=color,
                fmt='o',
                zorder=2,
                markersize=6,
                markeredgecolor='black',
                markeredgewidth=0.5,
            )
            # ax.scatter(
            #     x=means_.visDiff,
            #     y=means_.response,
            #     color=color,
            #     s=1,
            #     zorder=3,
            # )

            # Plot the prediction
            current_pseudo = gamma_transform(current_pseudo, gamma=nrn_gamma)
            visDiff = current_pseudo['visC'] - current_pseudo['visI']

            # update the pseudo with the current task and choice
            current_pseudo['task'] = task
            current_pseudo['choice'] = choice

            predictorM = current_pseudo[predictors].copy()
            pred = np.dot(weights.values, predictorM.values.T).flatten()
            ax.plot(visDiff, pred, color=color, lw=2, linestyle='-')

        ax.set_title(f"Task: {task}, Choice: {choice}")
        ax.axhline(weights.baseline.values, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("visDiff")
        ax.set_ylabel("Response")


    # Add legend and adjust layout
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="audDiff", bbox_to_anchor=(1.4, .8))
    fig.tight_layout()
