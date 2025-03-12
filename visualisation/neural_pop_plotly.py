#%%

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import ttest_rel
from itertools import combinations

from utils.neural_results_utils import load_results,select_best_models

def plot_param_changes(coefs, active_params):
    #coefs = coefs[coefs.model_type==model_type]

    n_params = len(active_params)


    fig = make_subplots(rows=1, cols=n_params, shared_xaxes=True, shared_yaxes=True, subplot_titles=active_params)
    line_style = dict(color="black", width=1, dash="dash")
    for i, param in enumerate(active_params):
        base_param = param.replace('_active', '')
        combined_param = f'{base_param}_combined'
        coefs[combined_param] = coefs[base_param] + coefs[param]

        scatter = px.scatter(coefs, x=base_param, y=combined_param, color='session', hover_data=['neuronID'])
        
        for trace in scatter['data']:
            fig.add_trace(trace, row=1, col=i+1)

        x_min, x_max = coefs[base_param].min(), coefs[base_param].max()
        y_min, y_max = coefs[combined_param].min(), coefs[combined_param].max()
        
        min_val = min(x_min, y_min)
        max_val = max(x_max, y_max)
        
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=line_style, row=1, col=i+1)
        fig.add_shape(type="line", x0=min_val, y0=0, x1=max_val, y1=0, line=line_style, row=1, col=i+1)
        fig.add_shape(type="line", x0=0, y0=min_val, x1=0, y1=max_val, line=line_style, row=1, col=i+1)


        fig.update_xaxes(title_text=base_param, row=1, col=i+1)
        fig.update_yaxes(title_text=f'{base_param} + {param}', row=1, col=i+1)

    fig.update_layout(height=400, width=400*n_params, title_text="Neural Population Activity")
    fig.show()


def plot_param_covariation(coefs, params1,params2,fig=None):
    #coefs = coefs[coefs.model_type==model_type]

    n_params = len(params1)

    if fig is None: 
        fig = make_subplots(rows=1, cols=n_params, shared_xaxes=True, shared_yaxes=True)
    
    
    line_style = dict(color="black", width=1, dash="dash")

    for i, (p1,p2) in enumerate(zip(params1,params2)):


        scatter = px.scatter(coefs, x=p1, y=p2, color='session', hover_data=['neuronID','model_type'])
        
        # Compute the correlation between the values
        correlation = coefs[[p1, p2]].corr().iloc[0, 1]
        fig.add_annotation(dict(text=f"r = {correlation:.2f}", xref="x domain", yref="y domain", x=1.05, y=1.05, showarrow=False), row=1, col=i+1)
        #fig.update_layout(annotations=[dict(text=f" r= {correlation:.2f}", xref="x domain", yref="y domain", x=0.5, y=1.15, showarrow=False)], row=1, col=i+1)
        #fig.add_annotation(dict(text=f" r= {correlation:.2f}", xref="paper", yref="paper", x=0.5, y=1.1, showarrow=False), row=1, col=i+1)

        for trace in scatter['data']:
            fig.add_trace(trace, row=1, col=i+1)

        x_min, x_max = coefs[p1].min(), coefs[p1].max()
        y_min, y_max = coefs[p2].min(), coefs[p2].max() 
        min_val = min(x_min, y_min)
        max_val = max(x_max, y_max)

        #fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=line_style, row=1, col=i+1)
        fig.add_shape(type="line", x0=min_val, y0=0, x1=max_val, y1=0, line=line_style, row=1, col=i+1)
        fig.add_shape(type="line", x0=0, y0=min_val, x1=0, y1=max_val, line=line_style, row=1, col=i+1)
        
        fig.update_xaxes(title_text=p1, row=1, col=i+1)
        fig.update_yaxes(title_text=p2, row=1, col=i+1)

    fig.update_layout(height=450, width=450*n_params)
    fig.show()

def plot_adj_r2_scores(coefs, model_types):
  
    pairs = list(combinations(model_types, 2))
    n_pairs = len(pairs)

    fig = make_subplots(rows=1, cols=n_pairs, shared_xaxes=False, shared_yaxes=False)


    for i, (x_plotted, y_plotted) in enumerate(pairs):
        r2_pivot = coefs.pivot_table(index=['neuronID', 'session','subject'], columns='model_type', values='adj_r2')
        r2_pivot.reset_index(inplace=True)
    
        scatter = px.scatter(r2_pivot, x=x_plotted, y=y_plotted,color='subject', hover_data=['neuronID','session'], labels={
            x_plotted: f'R2 {x_plotted}',
            y_plotted: f'R2 {y_plotted}', 
        })
        scatter.update_traces(marker=dict(line=dict(width=1, color='black'), opacity=0.6))

        if i == 0:
            scatter.update_traces(showlegend=True)
        else:
            scatter.update_traces(showlegend=False)

        for trace in scatter['data']:
            fig.add_trace(trace, row=1, col=i+1)

        fig.add_shape(type="line", x0=-0.3, y0=-0.3, x1=1, y1=1, line=dict(color="Black", dash="dash"), row=1, col=i+1)

        fig.update_xaxes(range=[-0.3, 1], row=1, col=i+1)
        fig.update_yaxes(range=[-0.3, 1], row=1, col=i+1)

        fig.update_xaxes(title_text=x_plotted, row=1, col=i+1)
        fig.update_yaxes(title_text=y_plotted, row=1, col=i+1)

        # Perform paired t-test between the two sets of R2 scores
        t_stat, p_value = ttest_rel(r2_pivot[x_plotted], r2_pivot[y_plotted])

        fig.update_layout(height=400, width=400*n_pairs, title_text=f"p-value: {p_value:.2f}")

    fig.show()


# load all neurons

coefs = load_results(dataset=None,region = 'SCs',fit_type = 'passive',time_bin = 'stim_bin_pre_0.00_post_0.20')


#%%
coefs_selected = select_best_models(coefs)
coefs_selected  = coefs_selected.fillna(0)
#%%

model_types = ['vis_engagement','vis_engagement_gain','vis']
model_types = ['baseline_engagement','baseline_choice_engaged','vis_engagement_gain','aud_engagement_gain']
plot_adj_r2_scores(coefs, model_types)





#%%
coefs_selected = select_best_models(coefs)
coefs_selected  = coefs_selected.fillna(0)
# how much engagement modulates the params
active_params = ['vis_ipsi_active', 'vis_contra_active', 'baseline_active']
active_params = ['vis_ipsi_active', 'vis_contra_active','aud_ipsi_active', 'aud_contra_active', 'baseline_active']

plot_param_changes(coefs_selected,  active_params) 







pairs = [
    ('baseline_active', 'vis_contra_active'),
    ('baseline_active', 'vis_ipsi_active'),
    ('baseline_active', 'aud_ipsi_active'),
    ('baseline_active', 'aud_contra_active'),
]

params1, params2 = zip(*pairs)
plot_param_covariation(coefs_selected, params1, params2)


pairs = [
    ('vis_contra', 'aud_contra'),
    ('tot_vis_contra', 'tot_aud_contra'),
    ('tot_vis_contra', 'choice_contra'),
    ('tot_aud_contra', 'choice_contra'),
    ('baseline_active', 'choice_contra')
]

params1, params2 = zip(*pairs)
plot_param_covariation(coefs_selected, params1, params2)


pairs = [
    ('baseline_active', 'tot_vis_contra'),
    ('baseline_active', 'tot_vis_ipsi'),
    ('baseline_active', 'tot_aud_contra'),
    ('baseline_active', 'tot_aud_ipsi'),
]

params1, params2 = zip(*pairs)
plot_param_covariation(coefs_selected, params1, params2)

#%%

# %%
