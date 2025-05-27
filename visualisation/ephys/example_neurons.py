

#%%

from src.ephys.encoding_avg import fit_dataset, get_winning_model, plot_prediction,filt_trials,get_time_params
from src.ephys.dat_utils import load_trial_data

timing = {'time_window':'stim','pre_time':0.0,'post_time':0.15}

fit_type = 'passive'

# load the models 
subject = 'AV030'
date = '2022-12-06'


coefs = fit_dataset(fit_type = fit_type,
    dataset_kwargs={'set_name':'all', 'subset':f'{subject}_{date}'},
    time_kwargs=timing
)

models = get_winning_model(coefs,thr_scorer='adj_r2',thr=0)




#%%

import matplotlib.pyplot as plt
import seaborn as sns


# reload the data
sess_params ={
    'subject':subject,
    'date':date
}
time_params = get_time_params(**timing)
df,clusters,_ = load_trial_data(**sess_params,**time_params).values()


#%%

nrn = 'neuron_1310'

nrn_fitID = f"{sess_params['subject']}_{sess_params['date']}_{nrn}"
# find neuron in the best_mode_df
nrn_model = models[models.fitID==nrn_fitID].copy()

print(nrn_model.BerylAcronym.values[0],nrn_model.model.values[0],nrn_model.bombcell_class.values[0])
#nrn_model = coefs[(coefs.fitID==nrn_fitID) & (coefs.model=='av') & (coefs.gamma==2)].copy()

plot_prediction(filt_trials(df,fit_type),nrn_model)

plt.suptitle(nrn)

# %%

models[(models.model=='vis_task') & models.is_good]

# %%

import seaborn as sns
goodClus  = models[models.is_good]


# Count and calculate percentages
model_counts = goodClus['model'].value_counts()
model_percentages = (model_counts / model_counts.sum()) * 100

# Create a DataFrame for plotting
model_stats = model_counts.to_frame(name='count')
model_stats['percentage'] = model_percentages
model_stats = model_stats.sort_values(by='count', ascending=False)

# Plot as a barplot
sns.barplot(x=model_stats.index, y=model_stats['percentage'], order=model_stats.index)
plt.ylabel('Count')
plt.title('Model Counts and Percentages')

plt.xticks(rotation=90)
# %%

fullm = coefs[(coefs.model == 'av_aud_bilateral_choice')].copy()
fullm = get_winning_model(fullm,thr_scorer='explained_variance_score',thr=-1)

evthr = 0.02
fullm['is_audC'] = fullm['explained_variance_score_audC'] > evthr
fullm['is_visC'] = fullm['explained_variance_score_visC'] > evthr
fullm['is_audI'] = fullm['explained_variance_score_audI'] > evthr
fullm['is_task'] = fullm['explained_variance_score_task'] > evthr
fullm['is_choice'] = fullm['explained_variance_score_choice'] > evthr

fullm = fullm[fullm.is_good].copy()

ps = ['audC','visC','audI','task','choice']

explained_vars = [f'explained_variance_score_{p}' for p in ps]

sns.scatterplot(data=fullm,x=explained_vars[1],y=explained_vars[4],hue='date',palette='Set2')



# %%
# %%
