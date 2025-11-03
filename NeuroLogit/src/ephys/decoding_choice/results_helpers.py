
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from floras_helpers.plotting import off_axes

from sklearn.metrics import log_loss, roc_auc_score

def get_scores(task_ev,models= ['l1_0.1','l1_0.05','pca_10','behav']):
    """
    Computes the scores (logLoss and AUC) for each model in `models` on both training and test sets,
    both the total scores and scores grouped by stimulus odds category.
    Might need to input per session data if some models didn't dit as atm if there is any nan in the probabilities it will skip the whole model
    """

    scores = []
    models_to_evaluate = list(models)
    
    tot_col_to_groupby = ['is_test_set','sessionID','subject','date','roi_fitted']
    for model_name in models_to_evaluate:

        probabilities = f'proba_right_{model_name}'
        pred_choices = f'predicted_choice_{model_name}'

        if task_ev[probabilities].isna().any():
            print(f"Skipping model {model_name} for session {task_ev.sessionID.unique()} due to all NaN probabilities. Model probably didn't fit.")
            continue

        grouped_scores = task_ev.groupby(['logOdds_stim_category'] +tot_col_to_groupby).apply(
                lambda group: pd.Series({
                    'log_loss': log_loss(group['choice'], group[probabilities],normalize=True) if len(group['choice'].unique()) > 1 else np.nan,
                    'auc': roc_auc_score(group['choice'], group[pred_choices]) if len(group['choice'].unique()) > 1 else np.nan
                })).unstack('is_test_set')

        grouped_scores.columns = [f"{col}_{'train' if idx == 0 else 'test'}" for col, idx in grouped_scores.columns]
        grouped_scores = grouped_scores.reset_index()


        total_score = task_ev.groupby(tot_col_to_groupby).apply(
                lambda group: pd.Series({
                    'log_loss': log_loss(group['choice'], group[probabilities],normalize=True) if len(group['choice'].unique()) > 1 else np.nan,
                    'auc': roc_auc_score(group['choice'], group[pred_choices]) if len(group['choice'].unique()) > 1 else np.nan
                })).unstack('is_test_set')
        total_score.columns = [f"{col}_{'train' if idx == 0 else 'test'}" for col, idx in total_score.columns]
        total_score = total_score.reset_index()
        total_score['logOdds_stim_category'] = 'all'
        grouped_scores = pd.concat([grouped_scores,total_score],ignore_index=True)
        grouped_scores['model'] = model_name
        scores.append(grouped_scores)

    scores = pd.concat(scores)

    return scores

def plot_logOdds_comparison(task_ev,model_name = 'pca_10',ax=None):
    """
    compares the odds of the behavioural model with the neural models
    """
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150,sharex=False,sharey=False)

    sns.scatterplot(data =task_ev, x='logOdds_stim',y='logOdds_'+model_name,hue='choice',alpha=0.5,s=3,
                    edgecolor='k',palette={0:'blue',1:'red'},ax=ax,legend=False)
    # sns.kdeplot(data =task_ev, x='logOdds_behav',y='logOdds_'+model_name,hue='choice',levels=5,
    #             ax=ax,palette={0:'blue',1:'red'},alpha=0.5,legend=False)
    ax.axhline(0,color='k',ls=':',alpha=0.3)
    ax.axvline(0,color='k',ls=':',alpha=0.3)
    ax.axline((0,0),slope=1,color='k',ls=':',alpha=0.3)

def plot_psycho_logOdds(task_ev,models = ['l1_0.1','l1_0.05','pca_10','behav'],ax=None):
     if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150,sharex=True,sharey=True)
     for i,model_name in enumerate(models):
         sns.pointplot(data=task_ev, x='logOdds_stim_category', y='proba_right_'+model_name, label=None,ax=ax,alpha=0.7)
 
         ax.set_title(model_name)
         ax.axhline(0.5,color='k',ls=':',alpha=0.3)
         off_axes(ax,which='top')
         ax.set_ylabel('p(right)')
         ax.set_xticklabels('')

def plot_psycho_stim(task_ev,model_name = 'pca_10',ax=None):
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150,sharex=True,sharey=True)

    sns.pointplot(data=task_ev, x='visDiff_categorical', y='proba_right_'+model_name, hue='audDiff', palette='coolwarm',ax=ax,legend=None)
    ax.set_title(model_name)
    ax.axhline(0.5,color='k',ls=':',alpha=0.3)
    off_axes(ax,which='top')
    ax.set_ylabel('p(right)')
    ax.set_xticklabels('')
