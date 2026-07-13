#%%

# How does the best unisensory model compare to the best multisensory model re model performance?
# in neurons that are not best fitted by the baseline model

# Q2 is the best additive model better than the best multiplicative model?


from NeuroLogit.src.ephys.encoding_avg.results_helpers import read_in_all_coefs,get_winning_model
import pandas as pd

coefs,models = read_in_all_coefs(fit_type='passive', subset='', recompute=False, add_behav_params=True, get_best=True)

#%%
models = models[models.is_good & (models.BerylAcronym.isin(['SCm','SCs']))].copy()
coefs = coefs[coefs.is_good & (coefs.BerylAcronym.isin(['SCm','SCs']))].copy()


# select units that are not best fitted by the baseline model
responding_neurons = models[models.model!='baseline'].copy()

coefs_responding = coefs[coefs.fitID.isin(responding_neurons.fitID)].copy()

unisensory_models = ['vis','vis_bilateral','aud','aud_ipsi','aud_bilateral','baseline']
multisensory_models = ['av','av_aud_bilateral','av_bilateral',
                       'av_multiplicative','av_bilateral_multiplicative','a_vPres','baseline']


best_unisensory = get_winning_model(coefs_responding[coefs_responding.model.isin(unisensory_models)],
                        thr_scorer='adj_r2', 
                        thr=0.005)

best_multisensory = get_winning_model(coefs_responding[coefs_responding.model.isin(multisensory_models)],
                        thr_scorer='adj_r2', 
                        thr=0.005)


# %%
import seaborn as sns
import matplotlib.pyplot as plt
unisensory_adj_r2 = best_unisensory[['fitID', 'adj_r2']].rename(columns={'adj_r2': 'best_unisensory_adj_r2'})
multisensory_adj_r2 = best_multisensory[['fitID', 'adj_r2']].rename(columns={'adj_r2': 'best_multisensory_adj_r2'})

best_adj_r2_df = pd.merge(unisensory_adj_r2, multisensory_adj_r2, on='fitID', how='inner')

fig,ax = plt.subplots(figsize=(6, 6))
# 2d histogram/desity plot 
sns.scatterplot(data=best_adj_r2_df, x='best_unisensory_adj_r2',
              y='best_multisensory_adj_r2',#,fill=True,thresh=0.05,#binwidth=0.025,stat='frequency',
              ax=ax
              )
            # Plot unity line
ax.axline((0, 0), slope=1, color='k', linestyle='--', linewidth=1)


ax.set_xlim([0,0.4])
ax.set_ylim([0,0.4])


# %%

# now we will do the same excercise but for the best multiplicative model vs the best additive model

additive_models = ['av','av_aud_bilateral','av_bilateral'] + unisensory_models
multiplicative_models = ['av_multiplicative','av_bilateral_multiplicative','a_vPres','baseline']

best_additive = get_winning_model(coefs_responding[coefs_responding.model.isin(additive_models)],
                        thr_scorer='adj_r2', 
                        thr=0.005)
best_multiplicative = get_winning_model(coefs_responding[coefs_responding.model.isin(multiplicative_models)],
                        thr_scorer='adj_r2', 
                        thr=0.005)


additive_adj_r2 = best_additive[['fitID', 'adj_r2']].rename(columns={'adj_r2': 'best_additive_adj_r2'})
multiplicative_adj_r2 = best_multiplicative[['fitID', 'adj_r2']].rename(columns={'adj_r2': 'best_multiplicative_adj_r2'})
best_additive_multiplicative_df = pd.merge(additive_adj_r2, multiplicative_adj_r2, on='fitID', how='inner')
fig,ax = plt.subplots(figsize=(6, 6))
# 2d histogram/desity plot
sns.scatterplot(data=best_additive_multiplicative_df, x='best_additive_adj_r2',
              y='best_multiplicative_adj_r2',#fill=True,thresh=0.05,#binwidth=0.025,stat='frequency',
              ax=ax
              )
# Plot unity line
ax.axline((0, 0), slope=1, color='k', linestyle='--', linewidth=1)

ax.set_xlim([0,0.4])
ax.set_ylim([0,0.4])


# %%
