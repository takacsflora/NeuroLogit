#%% 


import pandas as pd 
import numpy as np 

from src.ephys.results_utils import read_files 
import src.ephys.decoding_choice.results_helpers as hP

import matplotlib.pyplot as plt
import seaborn as sns
from floras_helpers.plotting import off_axes
from sklearn.metrics import roc_auc_score
# %

unique_only = True
time_stub = 'choice_pre0.2_post0.0'
ev = read_files(which_result = 'decoding_choice_results', filestub=f'task_ev_{time_stub}', extension='csv', sessions='unique')
weights = read_files(which_result = 'decoding_choice_results', filestub=f'weights_{time_stub}', extension='csv', sessions='unique')
# %%

# select sessions from a particular region 

#%%

# for each session plot the logOdds comparison for one model 
unique_sess = ev.sessionID.unique()
n_sess = len(unique_sess)
n_col = int(np.ceil(np.sqrt(n_sess)))
n_row = int(np.ceil(n_sess/n_col))

fig,axs = plt.subplots(n_row,n_col,figsize=(1.5*n_col,2*n_row),dpi=150,sharex=False,sharey=True)
fig.subplots_adjust(hspace=0.02,wspace=0.02)
axs = axs.flatten()
for i,sess in enumerate(unique_sess):
    sess_ev = ev[(ev.sessionID==sess) & (ev.is_test_set==True)].copy()
    #plot_logOdds_comparison(sess_ev,model_name = 'l1_0.05',ax=axs[i])
    #plot_psycho_logOdds(task_ev= sess_ev,models = ['l1_0.05','pca_10','stim'],ax=axs[i])
    hP.plot_psycho_stim(task_ev= sess_ev,model_name = 'l1_0.05',ax=axs[i])
    title = f"{sess_ev.subject.values[0]} {sess_ev.date.values[0]} {sess_ev.roi_fitted.values[0]}"
    axs[i].set_title(title,fontsize=5)
    off_axes(axs[i],which='all')

for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

#%% now plot the same thing for each subject

# compute scores per mice
# %%
neuronal_models = ['l1_0.05','pca_5','neural_l1_0.05']
all_models = list(neuronal_models) + ['stim','bias','behav']

sess_scores = pd.concat([hP.get_scores(sess_ev,models=all_models) for _,sess_ev in ev.groupby('sessionID')])
# get average scores per subject and roi for each model (so just grop by subject and roi and average everything else)
avg_sess_scores = sess_scores.groupby(['subject','roi_fitted','model','logOdds_stim_category']).mean().reset_index()
# %%
# Define the order of models for consistent plotting
model_order = ['stim', 'l1_0.05', 'pca_5','neural_l1_0.05']
avg_sess_scores['model'] = pd.Categorical(avg_sess_scores['model'], categories=model_order, ordered=True)
# Update the lineplot to use the defined model order
sns.lineplot(data=avg_sess_scores[(avg_sess_scores.logOdds_stim_category == 'all')], 
             x='model', y='log_loss_test', hue='roi_fitted', markers=True, dashes=False)

plt.legend(bbox_to_anchor=(-0.2, 1), loc='upper left', borderaxespad=0.)

# %%
sns.lineplot(data=avg_sess_scores[(avg_sess_scores.logOdds_stim_category!='all')& (avg_sess_scores.model== 'l1_0.05')], 
             x='logOdds_stim_category', y='auc_test',hue='roi_fitted')
# %%
fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150,sharex=True,sharey=True)
for i,model_name in enumerate(all_models):
    sns.pointplot(data=sess_ev, x='logOdds_stim_category', y='proba_right_'+model_name, label=None,ax=ax,alpha=0.7)

    ax.set_title(model_name)
    ax.axhline(0.5,color='k',ls=':',alpha=0.3)
    off_axes(ax,which='top')
    ax.set_ylabel('p(right)')
    ax.set_xticklabels('')
# %%
# subsample ev so sthat each subject has the same number of trials 



rois = ev.roi_fitted.unique()
models = ['neural_l1_0.05','neural_pca_5','stim']
n_rois = len(rois)
n_models = len(models)
fig,axs = plt.subplots(n_rois,n_models,figsize=(2*n_models,2*n_rois),dpi=150,sharex=True,sharey=True)
fig.subplots_adjust(hspace=0.3,wspace=0.3)
for i,roi in enumerate(rois):
    roi_ev = ev[ev.roi_fitted==roi]
    min_trials = roi_ev.subject.value_counts().min()
    balanced_ev = pd.concat([sub_ev.sample(n=min_trials, random_state=42) for _,sub_ev in roi_ev.groupby('subject')])
    
    for j,model_name in enumerate(models):
        ax = axs[i,j] if n_models>1 else axs[i]

    
        hP.plot_psycho_stim(task_ev=balanced_ev[(balanced_ev.is_test_set==True)],model_name = model_name,ax=ax)
        if j==0:
            ax.set_ylabel(roi)
        if i==0:
            ax.set_title(model_name)

# %%
fig,axs = plt.subplots(1,n_rois,figsize=(2*n_rois,2),dpi=150,sharex=True,sharey=True)
fig.subplots_adjust(hspace=0.3,wspace=0.3)
for i,roi in enumerate(rois):
    roi_ev = ev[ev.roi_fitted==roi]
    min_trials = roi_ev.subject.value_counts().min()
    balanced_ev = pd.concat([sub_ev.sample(n=min_trials, random_state=42) for _,sub_ev in roi_ev.groupby('subject')])
    
    ax = axs[i] if n_rois>1 else axs
    hP.plot_logOdds_comparison(task_ev=balanced_ev[(balanced_ev.is_test_set==True)],model_name = 'l1_0.05',ax=ax)
    #plot_psycho_logOdds(task_ev=balanced_ev[(balanced_ev.is_test_set==True)],models=models,ax=ax)
    ax.set_title(roi)
    off_axes(ax,which='top')
    if i==0:
        ax.set_ylabel('logOdds l1_0.05')
    ax.set_ylim([-10,10])
        
# %%

# separate stuff for each subject
ev['roi_subject'] = ev.subject + '_' + ev.roi_fitted
subjects = ev['roi_subject'].unique()

n_subj = len(subjects)
n_row = int(np.ceil(np.sqrt(n_subj)))
n_col = int(np.ceil(n_subj/n_row))
fig,axs = plt.subplots(n_row,n_col,figsize=(1.5*n_col,1.5*n_row),dpi=150,sharex=False,sharey=False)
fig.subplots_adjust(hspace=0.3,wspace=0.3)
axs = axs.flatten()
for i,subj in enumerate(subjects):
    subj_ev = ev[(ev.roi_subject==subj) & (ev.is_test_set==True)].copy()


    hP.plot_logOdds_comparison(subj_ev,model_name = 'l1_0.05',ax=axs[i])
    #plot_psycho_logOdds(task_ev= subj_ev,models = ['l1_0.05','pca_10','behav'],ax=axs[i])
    #plot_psycho_stim(task_ev= subj_ev,model_name = 'forward',ax=axs[i])
    axs[i].set_title(subj,fontsize=5)
    off_axes(axs[i],which='all')

for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])
# %%
fig,ax = plt.subplots(1,1,figsize=(2,2),dpi=150,sharex=False,sharey=False)
sns.histplot(data=ev[(ev.is_test_set==True) & (ev.roi_fitted=='MOs')], x = 'logOdds_neural_l1_0.05', hue='choice', element='step',fill=False,
             legend=False,bins=np.linspace(-250,250,550),
             stat='density',cumulative=False, common_norm=False, palette={0:'blue',1:'red'},ax=ax)

sns.histplot(data=ev[(ev.is_test_set==True) & (ev.roi_fitted=='MOs')], x = 'logOdds_stim', hue='choice', element='step',fill=True,legend=False,
             bins=np.linspace(-250,250,550),
             stat='density',cumulative=False, common_norm=False, palette={0:'blue',1:'red'},ax=ax)

ax.axvline(0,color='k',ls=':',alpha=0.3)
off_axes(ax,which='top')
ax.set_xscale('symlog', linthresh=5)


# %%


fig,axs = plt.subplots(2,2,dpi=150,sharex=False,sharey=False,gridspec_kw={'height_ratios':[1,4],'width_ratios':[4,1]},figsize=(2,2))
fig.subplots_adjust(hspace=0.05,wspace=0.05)

plt.rcParams.update({'font.size': 8})
roi='AV030_SCm'
roi_ev = ev[ev.roi_subject==roi]
min_trials = roi_ev.subject.value_counts().min()
balanced_ev = pd.concat([sub_ev.sample(n=min_trials, random_state=42) for _,sub_ev in roi_ev.groupby('subject')])
b_test = balanced_ev[(balanced_ev.is_test_set==True)].copy()


sns.scatterplot(data =b_test, x='logOdds_stim',y='logOdds_neural_l1_0.05',hue='choice',alpha=.6,s=15,
                edgecolor='k',palette={0:'blue',1:'red'},ax=axs[1,0],legend=False)

sns.kdeplot(data=b_test,y='logOdds_neural_l1_0.05',hue='choice',ax=axs[1,1],fill=True,palette={0:'blue',1:'red'},legend=False)
sns.kdeplot(data=b_test,x='logOdds_stim',hue='choice',ax=axs[0,0],fill=True,palette={0:'blue',1:'red'},legend=False)
# Compute AUC score for the 'logOdds_neural_l1_0.05' model
y_true = b_test['choice']
y_score_neur = b_test['logOdds_neural_l1_0.05']
y_score_stim = b_test['logOdds_stim']
auc_score_neur = roc_auc_score(y_true, y_score_neur)
auc_score_stim = roc_auc_score(y_true, y_score_stim)

# Make ax[0,0] equal axes
axs[1,0].set_xlim([-10,10])
axs[1,0].set_ylim([-10,10])
axs[0,0].set_xlim([-10,10])
axs[1,1].set_ylim([-10,10])

# Add AUC scores as text in the middle of the density plots
axs[0, 0].text(0.5, 1.05, f"AUC: {auc_score_stim:.2f}", 
               fontsize=8, color='black', ha='center', va='center', transform=axs[0, 0].transAxes)
axs[1, 1].text(1.05, 0.5, f"AUC: {auc_score_neur:.3f}", 
               fontsize=8, color='black', ha='center', va='center', transform=axs[1, 1].transAxes, rotation=270)

print(f"AUC Score (Neural): {auc_score_neur:.3f}")
print(f"AUC Score (Stim): {auc_score_stim:.3f}")


off_axes(axs[0,0],which='all')
off_axes(axs[1,1],which='all')
off_axes(axs[1,0],which='top')



# Remove the first two axes from the figure
fig.delaxes(axs[0, 1])

axs[1,0].axhline(0,color='k',ls=':',alpha=0.3)
axs[1,0].axvline(0,color='k',ls=':',alpha=0.3)
axs[1,0].set_xlabel('logOdds, stimulus model')
axs[1,0].set_ylabel('logOdds, \n neural  + stimulus model')


axs[1,0].axline((0,0),slope=1,color='k',ls=':',alpha=0.3)


axs[1,1].set_ylabel('')
axs[0,0].set_ylabel('')
axs[1,1].axhline(0,color='k',ls=':',alpha=0.3)
axs[0,0].axvline(0,color='k',ls=':',alpha=0.3)

axs[1,0].set_xticks([-10,0,10])
axs[1,0].set_yticks([-10,0,10])

# Add transparent grey squares to the 2nd and 4th quadrants
axs[1, 0].add_patch(plt.Rectangle((-10, 0), 10, 10, color='grey', alpha=0.2, zorder=0))
axs[1, 0].add_patch(plt.Rectangle((0, -10), 10, 10, color='grey', alpha=0.2, zorder=0))

savepath = r'C:\Users\Flora\OneDrive - University College London\Cortexlab\conferences\Cosyne2026\Raw_figures'
fig.savefig(f'{savepath}/Decoding_choice_SCm_logOdds_scatter.svg',dpi=300,bbox_inches='tight',transparent=True)
# %%
