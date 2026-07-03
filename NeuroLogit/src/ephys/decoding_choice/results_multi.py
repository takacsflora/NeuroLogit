#%%

from pathlib import Path
import pandas as pd 
import numpy as np 

from NeuroLogit.src.ephys.results_utils import read_files,get_brain_region_colors
from results_helpers import format_scores

import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import roc_auc_score




SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Choice')

plt.rcParams.update({'font.size': 8,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})

#SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\conferences\Cosyne2026\Poster figures')

# plt.rcParams.update({'font.size': 24,'font.family':'Calibri','axes.linewidth':1,'axes.spines.top':False,'axes.spines.right':False,
#                      'axes.spines.left':True,'axes.spines.bottom':True,
#                      'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})

# %

time_stub = 'prestim_pre0.2_post0.0'
time_stub = 'choice_pre0.2_post0.0'
model_type = 'scipy_taskorchoice_l1'  # model types to load results for
which_sesh = None # list of sessionIDs to load, or None to load all'
ev = read_files(which_result = 'decoding_choice_results', filestub=f'task_ev_{model_type}_{time_stub}', extension='csv', sessions=which_sesh)
weights = read_files(which_result = 'decoding_choice_results', filestub=f'weights_{model_type}_{time_stub}', extension='csv', sessions=which_sesh)
metrics = read_files(which_result = 'decoding_choice_results', filestub=f'metrics_{model_type}_{time_stub}', extension='csv', sessions=which_sesh)

#
# in ev there is a column called sessionID and another one called roi_fitted
metrics = metrics.merge(ev[['sessionID','roi_fitted']].drop_duplicates(),on=['sessionID'],how='left')
weights = weights.merge(ev[['sessionID','roi_fitted']].drop_duplicates(),on=['sessionID'],how='left')
metrics['stub'] = time_stub
ev['visDiff_gamma'] = ev.visR**ev.gamma - ev.visL**ev.gamma
ev['GoNoGo_categorical'] = ev['choice_categorical'].replace({'left':'Go','right':'Go','NoGo':'NoGo'})


unique_models = metrics['model'].unique()
for model in unique_models:
    ev['logOdds_R_vs_L_'+model] = ev[f'logOdds_L_vs_R_{model}'] * -1

metrics = format_scores(metrics)
#%%
auc_roc_test_cols = [col for col in metrics.columns if col.startswith('auc_roc') and col.endswith('_test')] 
# plot the various auc_roc_metrics for a given model in different brain regions

stim_model_name = 'stim'
model_name = 'full'


fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150)
sns.lineplot(data=metrics, x='model', y='delta_logLik_test',
            hue='roi_fitted',ax=ax,legend = True)


ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


#%%
# t-test for full vs no_bias model
full_metrics = metrics[metrics.model=='full']
no_bias_metrics = metrics[metrics.model=='no_bias']
from scipy.stats import ttest_rel

metric = 'delta_logLik_test'
ttest_res = ttest_rel(full_metrics[metric], 
                      no_bias_metrics[metric])
print(f"T-test full vs no_bias delta logLik: t={ttest_res.statistic   :.2f}, p={ttest_res.pvalue:.4f}")



#ax.set_ylim([-1,1])
# %%

# region ev 

roi,subject = 'SCm','AV008'
model_name_x = 'stim'
model_name_y = 'full'
comparison_to_plot = 'R_vs_L'

region_ev = ev[(ev.roi_fitted==roi) & (ev.is_test_set)].copy()

proba_cols = [col for col in ev.columns if col.startswith('proba')]
fig,axs = plt.subplots(2,2,figsize=(2.2,2.5),dpi=150,gridspec_kw={'width_ratios':[3,1], 'height_ratios':[1,3]})

fig.subplots_adjust(hspace=0.5, wspace=0.5)

if comparison_to_plot == 'R_vs_L':
    metric_x, metric_y = 'R_vs_L', 'R_vs_L'
    palette_color = {'left':'magenta','right':'green'}
    hue_order = ['left','right']
    hue_var = 'choice_categorical'
    auc_mapping = {'left':0,'right':1}
    lims = [-10.5,10.5]
    ticks = [5,0,-5]
    ticklabels = [ '10⁵', '1', '10⁻⁵']

elif comparison_to_plot == 'Go_vs_NoGo':
    metric_x, metric_y = 'Go_vs_NoGo', 'Go_vs_NoGo'    
    palette_color = {'NoGo':'orange','Go':'violet'}
    hue_order = ['NoGo','Go']
    hue_var = 'GoNoGo_categorical'
    auc_mapping = {'NoGo':0,'Go':1}
    lims = [-2.5,15]
    ticks = [10,0]
    ticklabels = ['10¹⁰','1']





scatterax = axs[1,0]

region_ev_sel_choice = region_ev[region_ev[hue_var].isin(hue_order)]


sns.scatterplot(data=region_ev_sel_choice,
                 x=f'logOdds_{metric_x}_{model_name_x}', 
                y=f'logOdds_{metric_y}_{model_name_y}', 
                hue=hue_var,hue_order=hue_order,alpha=0.5,
                palette=palette_color,ax=scatterax,s=4,
                edgecolor='black',linewidth=0.2,legend=False)


scatterax.axhline(0,color='gray',ls=':',linewidth=0.5)
scatterax.axvline(0,color='gray',ls=':',linewidth=0.5)
scatterax.axline((0,0),slope=1,color='gray',ls=':',linewidth=0.5)


stim_kde_ax = axs[0,0]

sns.kdeplot(data=region_ev_sel_choice, x=f'logOdds_{metric_x}_{model_name_x}',hue=hue_var,
            fill=True,alpha=0.5,linewidth=1,
            palette=palette_color,ax=stim_kde_ax,legend=False,common_norm=False)

auc_x = roc_auc_score(region_ev_sel_choice[hue_var].map(auc_mapping),
              region_ev_sel_choice[f'logOdds_{metric_x}_{model_name_x}'])

stim_kde_ax.set_title(f'AUC: {auc_x:.2f}') 

neural_kde_ax = axs[1,1]    
sns.kdeplot(data=region_ev_sel_choice, y=f'logOdds_{metric_y}_{model_name_y}',hue=hue_var,
            fill=True,alpha=0.5,linewidth=1,
            palette=palette_color,ax=neural_kde_ax,legend=False,common_norm=False)


auc_y = roc_auc_score(region_ev_sel_choice[hue_var].map(auc_mapping), 
              region_ev_sel_choice[f'logOdds_{metric_y}_{model_name_y}'])
neural_kde_ax.set_title(f'AUC: {auc_y:.2f}')


scatterax.set_xlim(lims)
scatterax.set_ylim(lims) 
neural_kde_ax.set_ylim(lims)
stim_kde_ax.set_xlim(lims)

neural_kde_ax.set_ylabel('')
neural_kde_ax.set_yticklabels([])

stim_kde_ax.set_xlabel('')
stim_kde_ax.set_xticklabels([])

scatterax.set_xlabel('Odds \n stimulus model')

scatterax.set_ylabel('Odds \n stim. + neurons')



scatterax.set_xticks(ticks)
scatterax.set_yticks(ticks)

scatterax.set_xticklabels(ticklabels)
scatterax.set_yticklabels(ticklabels)
fig.delaxes(axs[0, 1])

fig.tight_layout()
fig.savefig(SAVE_PATH / f'scatterplot_logOdds_comparison_{roi}_{model_name_x}_vs_{model_name_y}_{comparison_to_plot}.svg', dpi=300)


# #%% scatterplot without the kde plots
# #fig,ax = plt.subplots(1,1,figsize=(1.3,1.3),dpi=150)
# fig,ax = plt.subplots(1,1,figsize=(4.5,4),dpi=150)

# roi,subject = 'MOs','AV008'
# model_name_x = 'stim'
# model_name_y = 'full'
# comparison_to_plot = 'R_vs_L'

# region_ev = ev[(ev.roi_fitted==roi) & (ev.is_test_set)].copy()

# proba_cols = [col for col in ev.columns if col.startswith('proba')]

# if comparison_to_plot == 'R_vs_L':
#     metric_x, metric_y = 'R_vs_L', 'R_vs_L'
#     palette_color = {'left':'magenta','right':'green'}
#     hue_order = ['left','right']
#     hue_var = 'choice_categorical'
#     auc_mapping = {'left':1,'right':0}
#     lims = [-10.5,10.5]
#     ticks = [5,0,-5]
#     ticklabels = [ '10⁵', '1', '10⁻⁵']

# elif comparison_to_plot == 'Go_vs_NoGo':
#     metric_x, metric_y = 'Go_vs_NoGo', 'Go_vs_NoGo'    
#     palette_color = {'NoGo':'orange','Go':'violet'}
#     hue_order = ['NoGo','Go']
#     hue_var = 'GoNoGo_categorical'
#     auc_mapping = {'NoGo':0,'Go':1}
#     lims = [-2.5,15]
#     ticks = [10,0]
#     ticklabels = ['10¹⁰','1']


# region_ev_sel_choice = region_ev[region_ev[hue_var].isin(hue_order)]

# print(f"Number of trials for {roi}: {len(region_ev_sel_choice)}", 
#       f'no of subjects: {region_ev_sel_choice.subject.nunique()}',
#       f'no of sessions: {region_ev_sel_choice.sessionID.nunique()}')


# sns.scatterplot(data=region_ev_sel_choice,
#                  x=f'logOdds_{metric_x}_{model_name_x}', 
#                 y=f'logOdds_{metric_y}_{model_name_y}', 
#                 hue=hue_var,hue_order=hue_order,alpha=0.7,
#                 palette=palette_color,ax=ax,s=10,
#                 edgecolor='black',linewidth=0.2,legend=False)

# lw=1.5 
# ax.axhline(0,color='gray',ls=':',linewidth=lw)
# ax.axvline(0,color='gray',ls=':',linewidth=lw)
# ax.axline((0,0),slope=1,color='gray',ls=':',linewidth=lw)

# ax.set_xlabel('Odds \n stimulus model')

# ax.set_ylabel('Odds \n stim. + neurons')
# ax.set_xlim(lims)
# ax.set_ylim(lims) 

# ax.set_xticks(ticks)
# ax.set_yticks(ticks)

# ax.set_xticklabels(ticklabels)
# ax.set_yticklabels(ticklabels)

# fig.tight_layout()

# fig.savefig(SAVE_PATH / f'scatterplot_logOdds_comparison_{roi}_{model_name_x}_vs_{model_name_y}_{comparison_to_plot}.png', dpi=300)


# #%%auc_roc_cols = [col for col in metrics.columns if col.startswith('auc_roc')]


# for col in auc_roc_test_cols:
#     metrics[f'delta_{col}'] = metrics[col] - metrics[f'{stim_model_name}_{col}']

# delta_auc_roc_test_cols = [f'delta_{col}' for col in auc_roc_test_cols]


# metrics_melted = metrics[metrics.model==model_name].melt(id_vars=['sessionID','roi_fitted'], 
#                                                          value_vars=delta_auc_roc_test_cols,
#                                                           var_name='auc_roc_metric',
#                                                             value_name='auc_roc_value')


# fig,ax = plt.subplots(1,1,figsize=(4,4),dpi=150)
# sns.boxplot(data=metrics_melted, x='auc_roc_metric', y='auc_roc_value',
#              hue='roi_fitted', ax=ax)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
# ax.set_ylabel('Delta AUC ROC (test)')
# # ax.axhline(0.5,color='gray',ls='--')
# # ax.axhline(1,color='gray',ls='--')

# ax.axhline(0,color='gray',ls='--')


# %%


neuron_features = weights[weights.feature.str.contains('neuron_')]
unique_neurons = np.unique(np.array([f.split('_on')[0] for f in neuron_features['feature'].unique()]))

contra_weights = [f'{f}_on_contra' for f in unique_neurons]
ipsi_weights = [f'{f}_on_ipsi' for f in unique_neurons]



# %%
fig,ax = plt.subplots(1,1,figsize=(4,4),dpi=150)
sns.scatterplot(data = weights[weights.feature.isin(contra_weights)],
                x='weight_full',
                y='weight_contra',ax=ax)
corr = np.corrcoef(weights[weights.feature.isin(contra_weights)]['weight_full'],
            weights[weights.feature.isin(contra_weights)]['weight_contra'])[0,1]
ax.set_title(f'Correlation: {corr:.2f}')
ax.axline((0,0),slope=1,color='gray',ls='--')
ax.axhline(0,color='gray',ls='--')
ax.axvline(0,color='gray',ls='--')
border = 5.5
ax.set_xlim([-border,border])
ax.set_ylim([-border,border])
# %%


fig,ax = plt.subplots(1,1,figsize=(4,4),dpi=150)

contra_full_model = weights[weights.feature.isin(contra_weights)]['weight_full'] 
ipsi_full_model = weights[weights.feature.isin(ipsi_weights)]['weight_full']   
rois = weights[weights.feature.isin(contra_weights)]['roi_fitted']

weight_data = pd.DataFrame({
    'contra_weight': contra_full_model.values,  
    'ipsi_weight': ipsi_full_model.values,
    'roi': rois.values
})

sns.scatterplot(data=weight_data[weight_data.roi=='SCm'], x='contra_weight', y='ipsi_weight', hue='roi', ax=ax)

#sns.kdeplot(data=weight_data[weight_data.roi=='MOs'], x='contra_weight', y='ipsi_weight', ax=ax, levels=3, fill=False, alpha=0.5, linewidth=1)
corr = np.corrcoef(contra_full_model.values,
            ipsi_full_model.values)[0,1]
ax.set_title(f'Correlation: {corr:.2f}')
ax.axline((0,0),slope=1,color='gray',ls='--')
ax.axhline(0,color='gray',ls='--')
ax.axvline(0,color='gray',ls='--')

# ax.set_xlim([-border,border])

ax.set_xlabel('Contra weight full')
ax.set_ylabel('Ipsi weight full')
# ax.set_ylim([-border,border])

# %%
import pandas as pd

region = 'SCm'
region_ev = ev[(ev.roi_fitted==region) & (ev.is_test_set)].copy()

fig,axs = plt.subplots(2,1,figsize=(1.2,1.5),dpi=150,height_ratios=[3,1],sharex=True)


which_logOdds= 'L_vs_R'
which_fractions = 'left'
which_probas = 'Left'
for model_type,color in zip(['no_bias','stim'],['cyan','grey']):

    logOdds = region_ev[f'logOdds_{which_logOdds}_{model_type}']
    bins = np.linspace(-15,15,30)
    axs[1].hist(logOdds, bins=bins, color=color, edgecolor=color,
                alpha=0.9,histtype='step')
    axs[1].set_ylabel(' trials')
    axs[1].set_xlabel(f'logOdds {which_logOdds}')

    # Categorize ev according to the logOdds bin
    region_ev = region_ev.assign(logOdds_bin=pd.cut(logOdds, bins=bins, labels=bins[:-1].astype(str)))
    # Calculate the average fraction of choices per bin

    fractions_per_bin = region_ev.groupby('logOdds_bin')['choice_categorical'].apply(lambda x: x.value_counts(normalize=True))

    # Calculate the fraction of left choices per logOdds bin
    fractions_left = fractions_per_bin.unstack().fillna(0)[which_fractions]

    proba_NoGo = region_ev.groupby('logOdds_bin')[f'proba_{which_probas}_{model_type}'].mean()

    # Plot the fraction of left choices per logOdds bin
    axs[0].plot(fractions_left.index.astype(float), fractions_left.values, marker='o',
                markeredgecolor='black', markersize=2.5,markeredgewidth=0.2,
                color=color, label=None, linestyle='',alpha=0.9)

    axs[0].plot(proba_NoGo.index.astype(float), proba_NoGo.values, 
                color=color, label=None, linestyle='-',alpha=0.5)

    axs[0].set_ylabel(f'P({which_fractions})')
    #axs[0].set_xlabel(f'logOdds {which_logOdds} Bin')
    axs[0].legend()
    axs[0].grid(False)
    axs[0].axvline(0,color='gray',ls=':',linewidth=1)
    axs[1].axvline(0,color='gray',ls=':',linewidth=1)
    #axs[0].axhline(0.5,color='gray',ls=':',linewidth=1)


fig.tight_layout()
fig.savefig(SAVE_PATH / f'logOddsbins_vs_choice_fractions_{region}_{which_logOdds}.svg', dpi=300)



# %% new plot which multiplies the odds with the choice (-1 or 1) to establish accuracy

# appears to be called signed log-odds



def plot_signed_logOdds_comparison(ev, roi, model_name_x, model_name_y, comparison_to_plot,ax=None):  

    region_ev = ev[(ev.roi_fitted==roi) & (ev.is_test_set)].copy()


    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(1.3,1.25),dpi=150)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
    
    scatterax = ax


    if comparison_to_plot == 'R_vs_L':
        metric_x, metric_y = 'R_vs_L', 'R_vs_L'
        palette_color = {'left':'magenta','right':'green'}
        hue_order = ['left','right']
        hue_var = 'choice_categorical'
        auc_mapping = {'left':0,'right':1}
        signed_odds_mapping = {'left':-1,'right':1}
        lims = [-7,18]
        ticks = [-5,0,5,10,15]
        ticklabels = [ '-5','0', '5','10','15' ]

    elif comparison_to_plot == 'Go_vs_NoGo':
        metric_x, metric_y = 'Go_vs_NoGo', 'Go_vs_NoGo'    
        palette_color = {'NoGo':'orange','Go':'violet'}
        hue_order = ['NoGo','Go']
        hue_var = 'GoNoGo_categorical'
        auc_mapping = {'NoGo':0,'Go':1}
        signed_odds_mapping = {'NoGo':-1,'Go':1}
        lims = [-7,18]
        ticks = [-5,0,5,10,15]
        ticklabels =  [ '-5','0', '5','10','15' ]


    #scatterax = axs[1,0]
    region_ev_sel_choice = region_ev[region_ev[hue_var].isin(hue_order)]

    print(f"Number of trials for {roi}, {comparison_to_plot}: {len(region_ev_sel_choice)}, no of subjects: {region_ev_sel_choice.subject.nunique()}, no of sessions: {region_ev_sel_choice.sessionID.nunique()}")


    region_ev_sel_choice = region_ev_sel_choice.assign(
        signed_logOdds_x = region_ev_sel_choice[f'logOdds_{metric_x}_{model_name_x}'] * region_ev_sel_choice[hue_var].map(signed_odds_mapping),
        signed_logOdds_y = region_ev_sel_choice[f'logOdds_{metric_y}_{model_name_y}'] * region_ev_sel_choice[hue_var].map(signed_odds_mapping)
    )

    sns.scatterplot(data=region_ev_sel_choice,
                    x='signed_logOdds_x', 
                    y='signed_logOdds_y',alpha=0.3,
                    hue=hue_var,hue_order=hue_order,palette=palette_color,ax=scatterax,s=5,
                    edgecolor='black',linewidth=0.3,legend=False)

    # bins = np.linspace(-10.5,10.5,33)
    # sns.histplot(data=region_ev_sel_choice, x='signed_logOdds_x',y='signed_logOdds_y',
    #                 bins=(bins, bins), pthresh=0.05, cmap='Blues', ax=scatterax, legend=False)
    # sns.kdeplot(data=region_ev_sel_choice, x='signed_logOdds_x',y='signed_logOdds_y',
    #             levels=3,hue=hue_var,hue_order=hue_order,
    #             palette=palette_color, ax=scatterax,
    #             fill=False,alpha=0.5,linewidth=1,legend=False)


    scatterax.axhline(0,color='gray',ls=':',linewidth=1)
    scatterax.axvline(0,color='gray',ls=':',linewidth=1)
    scatterax.axline((0,0),slope=1,color='gray',ls=':',linewidth=1)


    scatterax.set_xlim(lims)
    scatterax.set_ylim(lims) 

    scatterax.set_xticks(ticks)
    scatterax.set_yticks(ticks)

    scatterax.set_xticklabels(ticklabels)
    scatterax.set_yticklabels(ticklabels)

    scatterax.set_xlabel(f'Signed odds,{model_name_x} model') 
    scatterax.set_ylabel(f'Signed odds,{model_name_y} model')



rois = ['MOs','MOs']
comparisons = ['R_vs_L','Go_vs_NoGo']
fig,ax = plt.subplots(1,len(rois),figsize=(len(rois)*1.4,1.5),dpi=150)
fig.subplots_adjust(wspace=0.3)

for i, (roi, comparison) in enumerate(zip(rois, comparisons)):
    plot_signed_logOdds_comparison(ev, roi, model_name_x, model_name_y, comparison, ax=ax[i])
    ax[i].set_title(f'{roi} {comparison}')
    ax[i].grid(False)
    if i>0:
        ax[i].set_ylabel('')
        ax[i].set_yticklabels([])
    ax[i].set_title('')
    ax[i].set_xlabel('')
    ax[i].set_ylabel('')
    
plt.tight_layout()

fig.savefig(SAVE_PATH / f'scatterplot_signed_logOdds_comparison_{roi}_{model_name_x}_vs_{model_name_y}_{time_stub}.png', dpi=400)


# %%

region_colors = get_brain_region_colors()



rois = ['SCs','SCm','MOs']


tested_metrics = [
   # 'logLik_test',
    # 'logLik_discrim_test_avg_per_trial',
    # 'logLik_detect_test_avg_per_trial',
    # 'auc_roc_R_vs_L_test', 
    # 'auc_roc_NoGo_vs_rest_test', 
    'brier_discrim_test',
    'brier_detect_test',

]


def stat_tests(metrics_df,model_x='stim',model_y='full'):
    # input the metrics_pivot df essentially
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings
    
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    
    metrics_df['delta_metric'] = metrics_df[f'{model_y}'] - metrics_df[f'{model_x}']

    regions = metrics_df['roi_fitted'].unique()
    for region in regions:
        region_data = metrics_df[metrics_df['roi_fitted'] == region]
        try: 
            model = smf.mixedlm("delta_metric ~ 1", region_data, groups=region_data["subject"])
            result = model.fit()

            intercept = result.params['Intercept']
            p_two_tailed = result.pvalues['Intercept']

            if intercept > 0:
                p_one_tailed = p_two_tailed / 2
            else:
                p_one_tailed = 1 - (p_two_tailed / 2)
            
            print(
                f"Region: {region}, "
                f"Coefficient: {intercept:.3f}, "
                f"p-value, one sided for {model_y} > {model_x}: {p_one_tailed:.4f}"
            )
        except Exception as e:
            print(f"Region: {region}, model fit failed: {e}")


    print('SCm vs MOs comparison:')
    metrics_SCmMOs = metrics_df[metrics_df['roi_fitted'].isin(['SCm', 'MOs'])].copy()
    model = smf.mixedlm("delta_metric ~ roi_fitted", metrics_SCmMOs, groups=metrics_SCmMOs["subject"])
    result = model.fit()

    print(result.summary())



n_plots = len(tested_metrics) * len(rois)
fig,axs = plt.subplots(1,n_plots,figsize=(n_plots*1,1.5),dpi=150,sharex=True,sharey=True)


for m in tested_metrics:
    print(f"Testing metric: {m}")
    metrics_subset = metrics[['sessionID','roi_fitted','subject', 'model', m]].copy()

    metrics_pivot = (
        metrics_subset
        .pivot_table(
            index=['sessionID','roi_fitted','subject'],
            columns='model',          # gives one column per model (full, stim, etc.)
            values=m,
            aggfunc='first'
                                # or 'mean' if you have duplicates per session/model
        )
        .reset_index()
    ) 


    stat_tests(metrics_pivot, model_x='full', model_y='stim')



import itertools

for i, (roi, m) in enumerate(itertools.product(rois, tested_metrics)):
    ax = axs[i]


    metrics_subset = metrics[['sessionID','roi_fitted','subject', 'model', m]].copy()

    metrics_pivot = (
        metrics_subset
        .pivot_table(
            index=['sessionID','roi_fitted','subject'],
            columns='model',          # gives one column per model (full, stim, etc.)
            values=m,
            aggfunc='first'
                                # or 'mean' if you have duplicates per session/model
        )
        .reset_index()
    ) 

    # average metrics_pivot across sessions for each subject and roi_fitted and also get error bars for each subject (e.g. by taking the standard deviation across sessions)
    metrics_pivot_means = (
        metrics_pivot
        .groupby(['roi_fitted','subject'])
        .mean()
        .reset_index()
    )

    metrics_pivot_stds = (
        metrics_pivot
        .groupby(['roi_fitted','subject'])
        .std()
        .reset_index()
    )

    # sns.kdeplot(data=metrics_pivot,
    #     x ='stim',
    #     y='full',hue='roi_fitted',
    #     levels=2,palette = region_colors, ax=ax,legend = False,alpha=0.5,linewidth=1)
    
    # sns.scatterplot(data=metrics_pivot_means[metrics_pivot_means.roi_fitted==roi],
    #     x ='stim',
    #     y='full',hue='roi_fitted',s=8,
    #     legend=False,palette=region_colors,ax=ax,edgecolor='k')

    plot_df = metrics_pivot_means[metrics_pivot_means.roi_fitted == roi].copy()
    err_df = metrics_pivot_stds[metrics_pivot_stds.roi_fitted == roi].copy()
    plot_df = plot_df.merge(
        err_df[['subject', 'stim', 'full']].rename(
            columns={'stim': 'stim_std', 'full': 'full_std'}
        ),
        on='subject',
        how='left'
    )

    ax.errorbar(
        plot_df['stim'],
        plot_df['full'],
        xerr=plot_df['stim_std'].fillna(0),
        yerr=plot_df['full_std'].fillna(0),
        fmt='none',
        ecolor=region_colors[roi],
        elinewidth=1.5,
        alpha=1,
        capsize=0,
        zorder=1,
    )
    ax.scatter(
        plot_df['stim'],
        plot_df['full'],
        s=16,
        color=region_colors[roi],
        edgecolor='k',
        linewidth=0.4,
        alpha=0.95,
        zorder=2,
    )

    ax.set_title(f'{roi}\n{m.replace("_test_avg_per_trial", "").replace("logLik_", "")}', fontsize=7)
    if i == 0:
        ax.set_xlabel('stim')
        ax.set_ylabel('full')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')

    

    mean_val_at_stim = metrics_pivot.stim.mean()
    ax.axline((mean_val_at_stim,mean_val_at_stim),slope=1,color='gray',ls=':')

    ax.set_xlim([0.01,0.2])
    ax.set_ylim([0.01,0.2])

plt.tight_layout()

fig.savefig(SAVE_PATH / f'model_comparison_metrics_scatter_full_vs_stim_{time_stub}.svg', dpi=150)




# statistical test. 


# %% reliability diagram
from sklearn.calibration import calibration_curve
roi = 'MOs'

region_ev = ev[(ev.roi_fitted==roi) & (ev.is_test_set)].copy()


def plot_calibration_curve(df,which='Go', model_name='full', ax=None,**pkws):
    if ax is None: 
        fig,ax = plt.subplots(1, 2, figsize=(2, 2)) 
    
    if which=='Go':
        y_true = df['choice_categorical'].isin(['left', 'right']).astype(int)
        y_prob = df[f'proba_Go_{model_name}']

    elif which =='NoGo':
        y_true = (df['choice_categorical'] == 'NoGo').astype(int)
        y_prob = df[f'proba_NoGo_{model_name}']

    else: 
        df_go = df[df['choice_categorical'].isin(['left', 'right'])].copy()

        if which == 'Right':
            y_true = (df_go['choice_categorical'] == 'right').astype(int)
        elif which == 'Left':
            y_true = (df_go['choice_categorical'] == 'left').astype(int)

        y_prob_3_choice = df_go[f'proba_{which}_{model_name}']
        y_prob = y_prob_3_choice / (df_go[f'proba_Go_{model_name}'])


    y_prob = np.clip(y_prob, 1e-15, 1.0)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8,strategy='quantile')

    ax.plot(prob_pred, prob_true, '-',markeredgecolor='k', **pkws)
    ax.plot([0, 1], [0, 1], ':', color='gray',linewidth=0.5)



fig,axs = plt.subplots(1,2,figsize=(3.5,1.5),dpi=150)
fig.subplots_adjust(wspace=0.5)

for model,my_color in zip(['full','stim','neural_only'],['cyan','grey','magenta']):
    plot_calibration_curve(region_ev,which='Go', model_name=model, ax=axs[0], color=my_color,linewidth=2.5)
    plot_calibration_curve(region_ev,which='Right', model_name=model, ax=axs[1], color=my_color,linewidth=2.5)

axs[0].set_title('Go vs NoGo')
axs[1].set_title('Right vs Left')  

axs[0].set_xlabel('Predicted Probability of Go')
axs[0].set_ylabel('True Probability of Go')
axs[1].set_xlabel('Predicted Probability of Right (given Go)')
axs[1].set_ylabel('True Probability of Right (given Go)')




# %%
