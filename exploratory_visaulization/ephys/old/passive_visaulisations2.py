#%%

import pandas as pd
import matplotlib.pyplot as plt


coefs = pd.read_csv(r'D:\AV_Neural_Data\fit_results\all_stim_passive.csv')


# coefs = coefs[
#     (coefs.is_good) &
#     ((coefs.BerylAcronym == 'SCs')|(coefs.BerylAcronym=='SCm')) #& 
#     #(coefs.model_type == 'audiovisual') 
#       ].copy()

# Plot the counts of each BerylAcronym
#%%
# beryl_counts = coefs['BerylAcronym'].value_counts()
# plt.figure(figsize=(10, 6))
# beryl_counts = beryl_counts[beryl_counts > 800]
# beryl_counts.plot(kind='bar', color='skyblue', edgecolor='black')
# plt.title('Counts of Each BerylAcronym')
# plt.xlabel('BerylAcronym')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

#%%
coefs = coefs[
    (coefs.is_good) &
    ((coefs.BerylAcronym == 'MOs')) #& 
    #(coefs.model_type == 'audiovisual') 
      ].copy()


#%%
coefs['session'] = coefs['subject'] + '_' + coefs['date']
coefs = coefs.loc[coefs.groupby(['session', 'neuronID'])['adj_r2'].idxmax()]
# also keep only neurons where the adj_r2 is above 0.1
coefs = coefs[coefs['adj_r2'] > -0.01]
coefs = coefs.fillna(0)

#%%

average_dv_per_session = coefs.groupby(['subject', 'date'])['dv'].mean().reset_index()

# Select sessions within each subject that are at least 600 dv apart
selected_sessions = []
for subject, group in average_dv_per_session.groupby('subject'):
    group = group.sort_values('dv')
    selected = []
    for _, row in group.iterrows():
        if not selected or all(abs(row['dv'] - s['dv']) >= 600 for s in selected):
            selected.append(row)
    selected_sessions.extend(selected)

selected_sessions = pd.DataFrame(selected_sessions)

# Filter selected_coefs to keep only neurons from selected_sessions
selected_coefs = coefs.merge(
    selected_sessions[['subject', 'date']],
    on=['subject', 'date'],
    how='inner'
)

#%%

from floras_helpers.anat_plots import anatomy_plotter

anat = anatomy_plotter()
fig,axs = plt.subplots(1,5,figsize=(18, 10),dpi=300,sharey=True,width_ratios=[.5,1,1,1,1])
fig.subplots_adjust(wspace=0.05)
#% ratios of winning models per dv bin 
selected_coefs['-dv'] = selected_coefs.dv *-1 + 332 
selected_coefs['dv_bin'] = pd.cut(selected_coefs['-dv'], bins=10)
selected_coefs['dv_bin_mean'] = selected_coefs['dv_bin'].apply(lambda x: x.mid if pd.notnull(x) else None)
model_counts = selected_coefs.groupby(['dv_bin_mean', 'model_type']).size().unstack(fill_value=0)
model_counts = model_counts.div(model_counts.sum(axis=1), axis=0) * 100

color_map = {'baseline': 'grey', 'vis': 'black', 'aud': 'red', 'audiovisual': 'brown'}


for i, model in enumerate(['baseline','vis','aud','audiovisual']):
    ax = axs[0]
    color = color_map[model]
    ax.plot(model_counts[model], model_counts.index,lw=5,color=color)

ax.axvline(5, color='k', linestyle='--', linewidth=1)
#
params = ['visC','audC','audI','baseline']
for i,param in enumerate(params):
    ax = axs[i+1]
    anat.plot_anat_canvas(ax=ax,coord = 3800, axis='ap')
    anat.plot_points(selected_coefs['ml'],selected_coefs['dv'],unilateral=True,c = 'grey',alpha=0.2,marker = '.',s=100,edgecolor=None)

    if 'vis' in param:
        cc = selected_coefs[(selected_coefs['model_type'] != 'baseline') & (selected_coefs['model_type'] != 'aud')].copy()
    elif 'aud' in param:
        cc = selected_coefs[(selected_coefs['model_type'] != 'baseline') & (selected_coefs['model_type'] != 'vis')].copy()
    else:
        cc = selected_coefs.copy()


    anat.plot_points(cc['ml'],cc['dv'],unilateral=True,c = cc[param],alpha=1,marker = '.',s=200,edgecolor='k',cmap='coolwarm',vmin=-20,vmax=20)

    ax.set_xlim([-2200, -200])
    ax.set_ylim([-6000, -500])
    ax.invert_xaxis()


# Remove the top and right spines from all subplots
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    # Save the figure as an SVG file
#fig.savefig('figure.svg', format='svg')


# %%
plt.rcParams.update({'font.size': 6})

fig,axs = plt.subplots(2,1,figsize=(1,2),dpi=300,sharex=True,sharey=True)
for i,param in enumerate(['visC','audI']):
    ax = axs[i]

    ax.scatter(selected_coefs['audC'],selected_coefs[param],marker = '.',s=30,edgecolor='k',c='lightgrey',alpha=1,linewidth=0.3)
    # Add horizontal and vertical lines at 0
    ax.axhline(0, color='k', linestyle='--', linewidth=.3)
    ax.axvline(0, color='k', linestyle='--', linewidth=.3)


    # Turn off the spines on the top and right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
# %%
import numpy as np
from src.models.av_models_multi import av_multi_asymetric_audio
from src.models.av_models_opto import av_pseudoPlotter


av_nrns = selected_coefs[selected_coefs['model_type'] == 'audiovisual'].copy()
vC = selected_coefs['visC'].mean()
aC = selected_coefs['audC'].mean()
aI = selected_coefs['audI'].mean()
bl = selected_coefs['baseline'].mean()

def plot_prediction(m,ax=None):

    ps = av_pseudoPlotter()
    if ax is None:
        fig, ax  = plt.subplots(2,1,figsize=(1.5,2),dpi=300,height_ratios=[2.5,1],sharex=True,sharey=False)

    unique_aud = np.array([-1,0,1])
    n_aud = len(ps.pseudo)
    colors = plt.cm.coolwarm(np.linspace(0,1,n_aud))

    gamma = m.params['gamma']


    opto = [0]
    opto_linestyle =['-']
    
    for opto_,linestyle in zip(opto,opto_linestyle):
        ps.update_pseudo(extra_predictors={'bias_opto':opto_})
        for i,(a,c) in enumerate(zip(unique_aud,colors)):
            #predicted 

            pkws={
            'color':c,
            'linestyle':linestyle,
            'linewidth':1+opto_*.4,
            
            }
            matrix = ps.pseudo[i]

            visDiff = matrix.visR - matrix.visL
            visDiff_gamma = matrix.visR**gamma -matrix.visL **gamma
            zL,zR = m.predict_log_proba(matrix)
            Nogo_Go_pred = -np.log(np.exp(zR) +np.exp(zL))

            pNoGo = 1 / (1+np.exp(zR) + np.exp(zL))
            pR = np.exp(zR) / (1+np.exp(zR) + np.exp(zL))
            pL = np.exp(zR) / (1+np.exp(zR) + np.exp(zL))

            ax[0].plot(visDiff_gamma,zR-zL,**pkws)
            ax[1].plot(visDiff_gamma,Nogo_Go_pred,**pkws)
        
            # Turn off the spines on the top and right
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)


m= av_multi_asymetric_audio()
m.set_params(params = {'audR': aC,
 'audL': aC,
 'visR': vC,
 'visL': vC,
 'gamma': 1,
 'biasL': 0,
 'biasR': 0,
 'audR_onL': 0,
 'audL_onR': 0,}
 )
plot_prediction(m)




# %%
