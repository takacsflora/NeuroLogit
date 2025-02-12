

from joblib import Parallel, delayed
import pandas as pd 
import numpy as np

# models
# the scipy framwework
import src.models.av_models_ephys as model_set

# the sklearn framework
from src.models.mode_base_sklearn import fit_linear_regression_neural
from src.models.mode_base_sklearn import get_weights

# model fitting
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from utils.fit_utils import calculate_adjusted_r2 # this adjust the r2 score for the number of predictors
from utils.av_dat_utils import load_trial_data,prepare_for_fit 

# plotting 
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.av_models_opto import av_pseudoPlotter


def plot_prediction(m,df,predictors,neuron,ax=None,extra_predictors = None):

    pred_name = f'{neuron}_pred'
    df[pred_name] = m.predict(df[predictors])

    # check whether the predictors are the same/correct ones


    def compute_means_and_errors(df, neuron, pred_name):
        means = df.groupby(['visDiff', 'audDiff'])[[neuron, pred_name]].mean().reset_index()
        errors = df.groupby(['visDiff', 'audDiff'])[[neuron, pred_name]].sem().reset_index()
        return means, errors

    # here we need to filter df based on the extra predictors

    if extra_predictors is not None:
        query_string = ' & '.join([f'({k} == {v})' for k,v in extra_predictors.items()])
        df = df.query(query_string)
        pass

    means, errors = compute_means_and_errors(df, neuron, pred_name)

    # Plotting the means and errors
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    unique_audDiffs = np.sort(df.audDiff.unique())
    colors = sns.color_palette('coolwarm', n_colors=len(unique_audDiffs))

    for i, audDiff in enumerate(unique_audDiffs):
        means_audDiff = means[means.audDiff == audDiff]
        errors_audDiff = errors[errors.audDiff == audDiff]
        ax.errorbar(
            x=means_audDiff.visDiff,
            y=means_audDiff[neuron],
            yerr=errors_audDiff[neuron],
            label=f'{audDiff}',
            color=colors[i],
            fmt='o',  # use 'o' to plot only dots without connecting lines
        )
        ax.scatter(
            x=means_audDiff.visDiff,
            y=means_audDiff[neuron],
            color=colors[i],
            s=50,  # size of the dot
            zorder=3  # ensure dots are on top of error bars
        )


    # plot the prediction
    pseudo_plotter = av_pseudoPlotter()
    pseudo_plotter.update_pseudo(extra_predictors)
    pseudo_data = pseudo_plotter.pseudo

    # add other pseudo-columns to the pseudo pdata


    for i,p in enumerate(pseudo_data):
        visDiff = p['visR'] - p['visL']
        y_pred = m.predict(p)
        ax.plot(visDiff,y_pred,color = colors[i],lw = 2,linestyle = '--')


    ax.set_xlabel('Visual Evidence (R-L)') 
    ax.set_ylabel('Response (z-score)')
    ax.set_title('Neuron Response and Prediction')
    
    ax.legend(title='Aud Azimuth (norm.)')

    return ax

def fit_nrn(df,neuron,predictors,model_name = None, fitter = 'sklearn',return_model = False):
    y = df[neuron]

# check variance 

    if (y.var() < 0.05) | y.isna().any():
        print(f'neuron {neuron} has low variance or missing values')

        weights = None
    else:
        X = df[predictors] # this can potentially be avoided by using scipy. Also the entire splotting can be done once... maybe

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if fitter == 'sklearn':
            m = fit_linear_regression_neural(X_train, y_train, gridCV_vis=True) # it is just here where we need to modufiy the function

            params = get_weights(m)
            gamma = params['hyperparameters']['features__vis__power']
            weights = params['weights']
            weights['gamma'] = gamma

         
        elif fitter == 'scipy':
            m = getattr(model_set,model_name)
            m = m()
            m.fit(X_train,y_train)
            weights = pd.DataFrame.from_dict(m.params,orient='index').T

        weights['neuronID'] = neuron
        weights['model_type'] = model_name

        y_pred = m.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        weights['r2'] = r2
        weights['mse'] = mse

        n_trials = X_test.shape[0]
        n_predictors = len(predictors)
        adj_r2 = calculate_adjusted_r2(r2,n_trials,n_predictors)
        weights['adj_r2'] = adj_r2

        

    if return_model:
        return weights,m
    else:
        return weights

def get_model_set(fit_type=None):
    """this basically is a parameter calling function that calls the collections of models we fit for partiucular analysis. 

    Args:
        fit_type (str, optional): the identifier of the analysis. Defaults to None.
    Returns:
        dict: a dictionary of the models we want to fit, wwhith the model names and the corresponding predictors
    """
    
    if fit_type == 'passive':
        models  = {
            'audiovisual': ['visL','visR', 'audL','audR','baseline'],
            'aud': ['audL','audR','baseline'],
            'vis' : ['visL','visR','baseline'], 
            'baseline': ['baseline'],

        }
    elif fit_type == 'engagement':
        models = {
            'audiovisual_engagement': ['visL','visR', 'audL','audR','baseline','is_active'],
            'aud_engagement': ['audL','audR','baseline','is_active'],
            'vis_engagement' : ['visL','visR','baseline','is_active'], 
            'audiovisual_engagement_gain': ['visL','visR', 'audL','audR','baseline','is_active'],
            'aud_engagement_gain': ['audL','audR','baseline','is_active'],
            'vis_engagement_gain' : ['visL','visR','baseline','is_active'], 
            'baseline_engagement': ['baseline','is_active'],
            'audiovisual': ['visL','visR', 'audL','audR','baseline'],
            'aud': ['audL','audR','baseline'],
            'vis' : ['visL','visR','baseline'], 
            'baseline': ['baseline'],
        }

    if (fit_type == 'choice'):
        models  = {
            'audiovisual': ['visL','visR', 'audL','audR','baseline'],
            'aud': ['audL','audR','baseline'],
            'vis' : ['visL','visR','baseline'], 
            'baseline': ['baseline'],
            'audiovisual_choice': ['visL','visR', 'audL','audR','baseline','choice'],
            'aud_choice': ['audL','audR','baseline','choice'],
            'vis_choice' : ['visL','visR','baseline','choice'], 
            'baseline_choice': ['baseline','choice'],
        }     

    # this is potentially the final model set except if we introduce a join fitting over time ...
    if fit_type == 'choice_engagement': 
        models  = {
            'audiovisual': ['visL','visR', 'audL','audR','baseline'],
            'aud': ['audL','audR','baseline'],
            'vis' : ['visL','visR','baseline'], 
            'baseline': ['baseline'],
            'audiovisual_choice_engaged': ['visL','visR', 'audL','audR','baseline','is_active','choice'],
            'aud_choice_engaged': ['audL','audR','baseline','is_active','choice'],
            'vis_choice_engaged' : ['visL','visR','baseline','is_active','choice'], 
            'baseline_choice_engaged': ['baseline','is_active','choice'],
            'audiovisual_engagement': ['visL','visR', 'audL','audR','baseline','is_active'],
            'aud_engagement': ['audL','audR','baseline','is_active'],
            'vis_engagement' : ['visL','visR','baseline','is_active'], 
            'audiovisual_engagement_gain': ['visL','visR', 'audL','audR','baseline','is_active'],
            'aud_engagement_gain': ['audL','audR','baseline','is_active'],
            'vis_engagement_gain' : ['visL','visR','baseline','is_active'], 
            'baseline_engagement': ['baseline','is_active'],
        }
    
    else:
        print('fit type not recognised')


    return models 

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

def fit_all_neurons(df,fit_type='passive'):
    df = prepare_for_fit(df,fit_type=fit_type)

    nrns_extracted = np.array([k for k in df.columns if 'neuron' in k]) 

    print('I even got here')
    models = get_model_set(fit_type=fit_type)
    results = Parallel(n_jobs=-1)(
        delayed(fit_nrn)(df,neuron,models[model],model_name = model,fitter='scipy',return_model=False)
        for neuron in nrns_extracted
        for model in models
    )
    coefs  = pd.concat(results)

    return coefs

def fit_session(fit_type='passive',**args):
    # try:
    df,clusters,_  = load_trial_data(**args).values()
    kept_clusInfo  = ['neuronID', 'hemi', 'bombcell_class','is_good','BerylAcronym','ml', 'ap', 'dv','probeID','firing_rate','amp_median']
    print('loaded, fitting session now ...')
    coefs  = fit_all_neurons(df,fit_type = fit_type)
    coefs = coefs.merge(clusters[kept_clusInfo], on='neuronID', how='left')

    # add subject and date to coefs dataframe
    coefs['subject'] = args['subject']
    coefs['date'] = args['date']

    return coefs
    # except:
    #     print(f'Error in session {args}')
    #     return None
    
