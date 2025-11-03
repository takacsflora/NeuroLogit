

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


def plot_prediction(m,df,predictors,neuron,hemisphere=1,ax=None,extra_predictors = None):

    pred_name = f'{neuron}_pred'
    df['hemi'] = hemisphere 
    df[pred_name] = m.predict(df[predictors + ['hemi']])

    # check whether the predictors are the same/correct ones


    def compute_means_and_errors(df, neuron, pred_name):
        means = df.groupby(['visDiff', 'audDiff'])[[neuron, pred_name]].mean().reset_index()
        errors = df.groupby(['visDiff', 'audDiff'])[[neuron, pred_name]].sem().reset_index()
        return means, errors

    # here we need to filter df based on the extra predictors

    if extra_predictors is not None:
        query_string = ' & '.join([f'({k} == {v})' for k,v in extra_predictors.items()])
        df = df.query(query_string)
        # we laso append the hemi to the extra predictors dict
        extra_predictors.update({'hemi': hemisphere})
    else:
        extra_predictors = {'hemi': hemisphere}

    means, errors = compute_means_and_errors(df, neuron, pred_name)

    # Plotting the means and errors
    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3),dpi=300)

    unique_audDiffs = np.sort(df.audDiff.unique())

                    # Create a custom colormap with darker grey in the middle
    colors = [(0, 0, 1), (0.5,0.5, 0.5), (1, 0, 0)]  # Blue -> Grey -> Red
    # positions = [0, 0.5, 1]  # Positions for the colors
    # custom_cmap = LinearSegmentedColormap.from_list("custom_coolwarm", list(zip(positions, colors)))
    # colors = sns.color_palette('coolwarm', n_colors=len(unique_audDiffs))

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
            s=36,  # size of the dot
            zorder=3,
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
    

    return ax

def fit_nrn(df,neuron,predictors,hemisphere = 1,model_name = None,return_model = False):
    

    y = df[neuron].copy()


    # if the neuron is just shit
    if (y.var() < 0.05) | y.isna().any():
        #print(f'neuron {neuron} has low variance or missing values')

        weights = None
    else:
        


        X = df[predictors].copy()

        # this is not entirely correct. we can always identify the hemisphere.... but this happens I believe to neurons who are void. 
        if np.isnan(hemisphere): 
            X['hemi'] = 1
        else:
            X['hemi'] = hemisphere

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                 
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
        n_predictors = len(m.get_params())
        adj_r2 = calculate_adjusted_r2(r2,n_trials,n_predictors)
        weights['adj_r2'] = adj_r2

    
    if return_model:
        return weights,m
    else:
        return weights

## to clean this up later ...
def get_model_set(fit_type=None):
    """this basically is a parameter calling function that calls the collections of models we fit for partiucular analysis. 

    Args:
        fit_type (str, optional): the identifier of the analysis. Defaults to None.
    Returns:
        dict: a dictionary of the models we want to fit, wwhith the model names and the corresponding predictors
    """
    
    if fit_type == 'passive':
        models  = ['baseline','vis','aud','audiovisual']
        predictors = ['visL','visR', 'audL','audR','baseline']

    elif fit_type == 'task':
        models =[
            'baseline','vis','aud','audiovisual',
            'baseline_task','vis_task','aud_task','audiovisual_task',
            'vis_task_gain','aud_task_gain','audiovisual_task_gain',   
            'baseline_choice','vis_choice','aud_choice','audiovisual_choice'

        ]

        predictors = ['visL','visR', 'audL','audR','baseline','choice','is_active']

    return models,predictors

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

def fit_all_neurons(df,clusters,fit_type='passive'):
    df = prepare_for_fit(df,fit_type=fit_type)

    nrns_extracted = np.array([k for k in df.columns if 'neuron' in k]) 

    nrn_hemisphere = [clusters[clusters.neuronID==neuron].hemi.values[0] for neuron in nrns_extracted]

    models,predictors = get_model_set(fit_type=fit_type)
   

    # loop for debugging
    # results = []
    # for neuron,hemi in zip(nrns_extracted,nrn_hemisphere):
    #     for model in models:
    #         results.append(fit_nrn(df,neuron,predictors,model_name = model,return_model=False,hemisphere=hemi))

    results = Parallel(n_jobs=8)(
        delayed(fit_nrn)(df,neuron,predictors,model_name = model,return_model=False,hemisphere=hemi)
        for neuron,hemi in zip(nrns_extracted,nrn_hemisphere)
        for model in models
    )


    coefs  = pd.concat(results)

    return coefs

def fit_session(fit_type='passive',**args):
    try:
        df,clusters,_  = load_trial_data(**args).values()
        kept_clusInfo  = ['neuronID', 'hemi', 'bombcell_class','is_good','BerylAcronym','ml', 'ap', 'dv','probeID','firing_rate','amp_median']
        print('loaded, fitting session now ...')
        coefs  = fit_all_neurons(df,clusters,fit_type = fit_type)
        coefs = coefs.merge(clusters[kept_clusInfo], on='neuronID', how='left')

        # add subject and date to coefs dataframe
        coefs['subject'] = args['subject']
        coefs['date'] = args['date']

        return coefs
    except:
        print(f'Error in session {args}')
        return None
    


