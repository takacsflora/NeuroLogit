#%%

# fit all the behavioural models ... 
from src.ephys.dat_utils import get_ephys_dataset, load_trial_data
from src.models.av_models_multi import av_multi_symmetric_audio,av_multi_asymetric
from src.models.av_models_opto import av_split  


from src.models.visualise import plot_psychometric,plot_psychometric_multi
import numpy as np
from matplotlib.gridspec import GridSpec
import pandas as pd
import matplotlib.pyplot as plt



# load data
sessions = get_ephys_dataset(subset='AV030')
sessions['session'] = sessions['subject'] + '_' + sessions['date']


sessions = sessions[sessions.subject.isin(
    ['AV005', 'AV008', 'AV014','FT030','FT031','FT032','AV025', 'AV030', 'AV034','AV007','AV013','AV023'])].copy()

lowSPL = ['AV025', 'AV030', 'AV034','AV023']
highSPL = ['AV005', 'AV008', 'AV014','FT030','FT031','FT032','AV007','AV013']

# Add SPL column to sessions DataFrame
sessions['SPL'] = np.where(sessions['subject'].isin(lowSPL), 'low', 'high')

# Read in and concatenate all behavioural data

all_behav_dfs = []
for _, args in sessions[['subject', 'date']].iterrows():
    try:
        df, _, _ = load_trial_data(**args, load_clusters=False, load_raster=None).values()
        df_behav = df[df.choice > -2].copy()
        df_behav['subject'] = args['subject']
        df_behav['date'] = args['date']
        df_behav['session'] = f"{args['subject']}_{args['date']}"
        all_behav_dfs.append(df_behav)
    except (FileNotFoundError, AttributeError):
        print(f"Data for {args['subject']} on {args['date']} not found. Skipping.")
        continue

trials = pd.concat(all_behav_dfs, ignore_index=True)
trials['SPL'] = np.where(trials['subject'].isin(lowSPL), 'low', 'high')




# atm this contains trials when the animal started moving at the time of the stimulus onset
# maybe we need to throw these out ..? 

# also will stick to fitting with av_split not hte multi model so removing noGos as well...

#trials = trials[(trials.choice != -1)].copy()


# %%
# Group sessions by subject
# Group sessions by subject and prepare for plotting

subjects = trials['subject'].unique()
n_rows = len(subjects)
sessions_per_subject = [sessions[sessions['subject'] == subj].session for subj in subjects]
max_sessions = max(len(s) for s in sessions_per_subject)
n_cols = max_sessions


#%%

fig = plt.figure(figsize=(n_cols * 1, n_rows * 2),dpi=150)
gs = GridSpec(nrows=n_rows * 2, ncols=n_cols, height_ratios=[2.5, 1] * n_rows)
plt.rcParams.update({'font.size': 6})

params_per_session = []
predictors = ['visL', 'visR', 'audL', 'audR']
for subj_idx, subj_sessions in enumerate(sessions_per_subject):
    df_mouse = trials[trials['subject'] == subjects[subj_idx]].copy()


    # m_subject = av_split()
    # m_subject.fit(df_mouse[predictors], df_mouse.choice)
    # gamma = m_subject.params['gamma']

    m_subject = av_multi_symmetric_audio()
    m_subject.fit(df_mouse[predictors], df_mouse['choice']+1)
    gamma = m_subject.params['gamma']

    for sess_idx, session in enumerate(subj_sessions):
        df_session = df_mouse[df_mouse['session'] == session].copy()
        if df_session.empty:
            continue
        
        # m = av_split()
        # m.fit(df_session[predictors], 
        #       df_session['choice'], 
        #       fixed_params={'gamma': gamma})


        # m = av_multi_symmetric_audio()
        # m.fit(df_session[predictors], 
        #       df_session['choice']+1,
        #       fixed_params={'gamma': gamma})  
        
        m = av_multi_asymetric()
        m.fit(df_session[predictors], 
              df_session['choice']+1,
              fixed_params={'gamma': gamma})
        
        
        y_pred = m.predict(df_session[predictors])
        y_true = df_session['choice'] + 1

        # confusion matrix
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{df_session.subject.iloc[0]} {df_session.date.iloc[0]}")
        plt.show()




        ax_psy = fig.add_subplot(gs[subj_idx * 2, sess_idx])
        ax_nogo = fig.add_subplot(gs[subj_idx * 2 + 1, sess_idx])
        plot_psychometric_multi(df_session, m, space='exp', ax=[ax_psy, ax_nogo])
        ax_psy.set_title(f"{df_session.date.iloc[0]}")

        ax_psy.set_axis_off()
        ax_nogo.set_axis_off()

        if sess_idx == 0:
            fig.text(
                0.01, 
                1 - ((subj_idx * 2 + 1) / (n_rows * 2)), 
                f"{subjects[subj_idx]}", 
                va='center', ha='left', rotation=90, fontsize=7, fontweight='bold'
            )
    

        params = pd.DataFrame(m.params, index=[session])
        params['subject'] = df_session['subject'].iloc[0]
        params['date'] = df_session['date'].iloc[0]
        params['sessionID'] = session
        params['SPL'] = df_session['SPL'].iloc[0]

        params_per_session.append(params)

params_per_session = pd.concat(params_per_session, ignore_index=True)


plt.tight_layout()

#
#params_per_session.to_csv(r'C:\Users\Flora\Documents\Github\NeuroLogit\data\behaviour\logit_params_per_session_ephys.csv',index=False)


# %%

import seaborn as sns


g  = sns.pairplot(
    params_per_session,
    vars=['visL', 'visR', 'audL', 'audR', 'biasL','biasR'],
    hue='SPL',
    kind='scatter',
    plot_kws={'alpha': .7, 's': 30,'edgecolor':'k','linewidth':0.5},
    diag_kind='kde',
    height=1,
    aspect=1
)
# Seaborn's pairplot does not have a direct argument to share axes.
# However, you can access the axes after creation and set the limits manually if needed.
# Example (optional, not required for most use cases):

# g = sns.pairplot(...)
common_xmin = -1 
common_xmax = 8
common_ymin = -1
common_ymax = 8
for ax in g.axes.flatten():
    ax.set_xlim(common_xmin, common_xmax)
    ax.set_ylim(common_ymin, common_ymax)

# There is no 'sharex' or 'sharey' argument in pairplot as of seaborn v0.12.
plt.show()


# %%


# now we will test if we get the same parameters if instead we fit with sklearn

# check the confusion matrix as well
from sklearn.linear_model import LogisticRegression


df_mouse['visR_gamma'] = df_mouse['visR'] ** gamma
df_mouse['visL_gamma'] = df_mouse['visL'] ** gamma
df_mouse['bias'] = 1 

df_session = df_mouse[df_mouse['session'] == session].copy()



# okay I think the results are quite similar now! 

sensory_pred_no_intercept = ['visR_gamma','visL_gamma','audR','audL', 'bias']

model = LogisticRegression(penalty=None,solver='lbfgs',C=1e9,multi_class='multinomial',
                           fit_intercept=False,max_iter=1000000)
model.fit(df_session[sensory_pred_no_intercept],df_session['choice']+1)

coef = model.coef_

coefR = coef[2]-coef[0]
coefL = coef[1]-coef[0]
# %%
