

#%%
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

region = 'SCm'
fit_type = 'passive'
time_period = 'stim'
df = pd.read_csv(f'D:/AVTrialData/{region}_{time_period}/fit_results/{fit_type}.csv')
# filter for neruons that are good and are from the ROI

df =df[(df.BerylAcronym == region) & df.is_good]
# combine subject and date into a single string named session
df['session'] = df['subject'] + '_' + df['date']

coefs = df.copy()
# %%
# flip to ipsi and contralateral 

coefs['vis_ipsi'] = coefs.apply(lambda row: row['visR'] if row['hemi'] == -1 else row['visL'], axis=1)
coefs['vis_contra'] = coefs.apply(lambda row: row['visL'] if row['hemi'] == -1 else row['visR'], axis=1)
coefs['aud_ipsi'] = coefs.apply(lambda row: row['audR'] if row['hemi'] == -1 else row['audL'], axis=1)
coefs['aud_contra'] = coefs.apply(lambda row: row['audL'] if row['hemi'] == -1 else row['audR'], axis=1)

if 'engagement' in fit_type:
    coefs['vis_ipsi_active'] = coefs.apply(lambda row: row['visR_active'] if row['hemi'] == -1 else row['visL_active'], axis=1)
    coefs['vis_contra_active'] = coefs.apply(lambda row: row['visL_active'] if row['hemi'] == -1 else row['visR_active'], axis=1)
    coefs['aud_ipsi_active'] = coefs.apply(lambda row: row['audR_active'] if row['hemi'] == -1 else row['audL_active'], axis=1)
    coefs['aud_contra_active'] = coefs.apply(lambda row: row['audL_active'] if row['hemi'] == -1 else row['audR_active'], axis=1)

if fit_type == 'choice':
    coefs['choice_ipsi'] = coefs.apply(lambda row: row['choice_right'] if row['hemi'] == -1 else row['choice_left'], axis=1)
    coefs['choice_contra'] = coefs.apply(lambda row: row['choice_left'] if row['hemi'] == -1 else row['choice_right'], axis=1)
    coefs['choice_diff'] = coefs['choice_contra'] - coefs['choice_ipsi']

#%%

# I also want to add anoter columnt to coefs_melted, the hemissphere of the neuron

param_values = ['visL', 'visR', 'audL', 'audR','gamma','baseline']
param_values = ['vis_ipsi', 'vis_contra', 'aud_ipsi', 'aud_contra','gamma','baseline']


if fit_type == 'engagement':
    active_params = ['vis_ipsi_active', 'vis_contra_active', 'aud_ipsi_active', 'aud_contra_active', 'baseline_active']
    param_values.extend(active_params)

if fit_type == 'choice':
    active_params = ['choice_diff']
    param_values.extend(active_params)

if fit_type == 'choice_engagement':
    active_params = ['baseline_active','choice']
    param_values.extend(active_params)

coefs_melted = coefs[coefs.model_type=='audiovisual_choice_engaged'].melt(id_vars=['neuronID','hemi','subject'], value_vars=param_values, var_name='predictor', value_name='coef')

# Displace along the x axis by hemi
coefs_melted['hemisphere'] = coefs_melted['hemi'].apply(lambda hemi: 'right' if hemi == 1 else 'left')


# Plotting with seaborn

plt.figure(figsize=(10, 4))
sns.stripplot(data=coefs_melted, 
              x='predictor', y='coef',hue='subject', log_scale=False,
              edgecolor='black', jitter=True, linewidth=.3,
              dodge=True,palette='coolwarm')
plt.title('Coefficient values per predictor')
plt.xlabel('Predictor')
plt.ylabel('Coefficient value')
plt.legend(title='hemisphere', bbox_to_anchor=(1.05, 1), loc='upper left')  
plt.xticks(rotation=45)
#plt.ylim([-30,30])

#plt.yscale('log')
plt.axhline(0, color='black', linewidth=1, linestyle='--')

plt.show()


#%%
# Create individual boxenplots for each predictor

# Create individual boxenplots for each predictor with aligned 0s and variable ylims

predictors = coefs_melted['predictor'].unique()
n_predictors = len(predictors)
fig, axes = plt.subplots(1, n_predictors, figsize=(4 * n_predictors, 6), sharey=True)

for i, predictor in enumerate(predictors):
    sns.stripplot(data=coefs_melted[coefs_melted['predictor'] == predictor], 
                  x='predictor', y='coef', ax=axes[i])
    axes[i].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[i].set_title(predictor)
    axes[i].set_xlabel('Predictor')
    axes[i].set_ylabel('Coefficient value')
    y_min, y_max = coefs_melted[coefs_melted['predictor'] == predictor]['coef'].min(), coefs_melted[coefs_melted['predictor'] == predictor]['coef'].max()
    axes[i].set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))

plt.tight_layout()
plt.show()



#%%

sns.lineplot(data=coefs_melted, x='predictor', y='coef', ci='sd')
plt.axhline(0, color='black', linewidth=1, linestyle='--')
#%%

# Plot each neuron's parameter against its parameter + parameter_active for engagement fit type
if fit_type == 'engagement':
    n_params = len(active_params)

    fig,ax = plt.subplots(1,n_params,figsize = (4*n_params,4),sharex=True,sharey=True)

    for i,param in enumerate(active_params):
        base_param = param.replace('_active', '')
        combined_param = f'{base_param}_combined'
        coefs[combined_param] = coefs[base_param] + coefs[param]

        
        
        sns.scatterplot(data=coefs, x=base_param, y=combined_param, hue='session', palette='tab10',ax=ax[i])

        #sns.kdeplot(data=coefs, x=base_param, y=combined_param, fill=True, palette='viridis', levels=10, thresh=0.05)
        #sns.jointplot(data=coefs, x=base_param, y=combined_param, hue= 'subject',palette='tab10', kind='kde',levels=10, thresh=0.05)
        ax[i].axline((0, 0), slope=1, linestyle='--', color='k')  # Diagonal line for reference
        ax[i].axhline(0, linestyle='--', color='gray')  # Horizontal line at y=0
        ax[i].axvline(0, linestyle='--', color='gray')  # Vertical line at x=0
        ax[i].set_xlabel(base_param)
        ax[i].set_ylabel(f'{base_param} + {param}')
        if i == n_params - 1:
            ax[i].legend(title='session', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax[i].get_legend().remove()

    plt.show()



#%%

# Pivot the DataFrame to see the r2 score for each neuron for each model type
# Pivot the DataFrame to see the r2 score for each neuron for each model type, including session
r2_pivot = coefs.pivot_table(index=['neuronID', 'session'], columns='model_type', values='adj_r2')
# Normalize each row in the r2 pivot table to the audiovisual column
r2_pivot_normalized = r2_pivot.div(r2_pivot['audiovisual'], axis=0)

# Plotting the pivoted DataFrame
plt.figure(figsize=(12, 8))
sns.heatmap(r2_pivot_normalized, annot=False, cmap='coolwarm', cbar_kws={'label': 'R2 Score'},vmin=-1, vmax=1)
plt.title('R2 Score for Each Neuron Across Different Model Types')
plt.xlabel('Model Type')
plt.ylabel('Neuron ID')
plt.show()



#%%
x_plotted, y_plotted = 'vis_engagement','vis_engagement_gain'

x_plotted, y_plotted = 'baseline_choice_engaged','aud_engagement_gain'

#x_plotted, y_plotted = 'vis','audiovisual'

#x_plotted, y_plotted = 'vis','aud'

fig,ax = plt.subplots(1,1,figsize = (4,4))

plt.plot(r2_pivot[x_plotted],r2_pivot[y_plotted],'o')
plt.plot([-.3,1],[-.3,1],'k--')
plt.xlim([-.3,1])
plt.ylim([-.3,1])
plt.xlabel(f'R2 {x_plotted}')
plt.ylabel(f'R2 {y_plotted}')

# Perform paired t-test between the two sets of R2 scores
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(r2_pivot[x_plotted], r2_pivot[y_plotted])

# Print the results
print(f'Paired t-test results between {x_plotted} and {y_plotted}:')
print(f't-statistic: {t_stat:.3f}, p-value: {p_value:.3e}')

# %%
# Calculate the winning model for each neuron



winning_models = r2_pivot.idxmax(axis=1)
is_bad  = (r2_pivot<-.1).all(axis=1)
winning_models[is_bad] = 'None'
winning_model_fractions_per_session = winning_models.groupby(level='session').value_counts(normalize=True).unstack(fill_value=0).reset_index()
winning_model_fractions_per_session['subject'] = winning_model_fractions_per_session['session'].apply(lambda x: x.split('_')[0])
winning_model_fractions_per_session['date'] = pd.to_datetime(winning_model_fractions_per_session['session'].apply(lambda x: x.split('_')[1]))
# Calculate the first date for each animal
first_dates = winning_model_fractions_per_session.groupby('subject')['date'].transform('min')
# Calculate the days since the first date
winning_model_fractions_per_session['days_since_first'] = (winning_model_fractions_per_session['date'] - first_dates).dt.days+1 



# Plot the fraction of neurons against model type with a stripplot
winning_model_melted  = winning_model_fractions_per_session.melt(id_vars=['session', 'subject', 'date', 'days_since_first'], var_name='model_type', value_name='fraction')

# also plot the average per eavh model type
# Calculate the average fraction per model type
average_fraction = winning_model_melted.groupby('model_type')['fraction'].mean().reset_index()

# Plot the average fraction per model type as a line
plt.figure(figsize=(10, 6))
sns.stripplot(data=winning_model_melted,
              x='model_type', y='fraction', hue='subject', jitter=False, dodge=True, palette='tab10')
sns.lineplot(data=average_fraction, x='model_type', y='fraction', color='black', marker='o', label='Average')
plt.title('Fraction of Neurons per Model Type')
plt.xlabel('Model Type')
plt.ylabel('Fraction of Neurons')
plt.legend(title='subject', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=90)

plt.show()

# %%
# Calculate the average fraction per model type and filter for those > 5%
average_fraction_filtered = average_fraction[average_fraction['fraction'] > 0.05].sort_values(by='fraction', ascending=False)
average_fraction_filtered = average_fraction_filtered[average_fraction_filtered['model_type'] != 'baseline']

# Plot the average fraction per model type as a line
plt.figure(figsize=(10, 6))
sns.lineplot(data=average_fraction_filtered, x='model_type', y='fraction', color='black', marker='o', label='Average')
plt.title('Average Fraction of Neurons per Model Type (> 5%)')
plt.xlabel('Model Type')
plt.ylabel('Fraction of Neurons')
plt.xticks(rotation=40)
plt.legend(title='Average', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

#%%
# Print the top 5 neurons for each model type based on adjusted R2 score
# Identify the top 10 neurons for each filtered model type based on adjusted R2 score
filtered_models = average_fraction_filtered['model_type'].tolist()
# Print the top 10 neurons for each filtered model type based on non-normalized adjusted R2 score
top_neurons_non_normalized = r2_pivot[filtered_models].apply(lambda x: x.nlargest(10).index.tolist(), axis=0)
for model_type, neurons in top_neurons_non_normalized.items():
    print(f'Top 10 neurons for model type {model_type} (non-normalized):')
    for neuron in neurons:
        r2_score = r2_pivot.loc[neuron, model_type]
        print(f'Neuron: {neuron}, R2 Score: {r2_score:.3f}')
    print()




  # %% 

# the relationship between coefficients in the particular set of com
from src.batch.encoding import get_model_set  
models = get_model_set(fit_type=fit_type)
model_type  = 'audiovisual_choice'
av_neurons = winning_models[winning_models == model_type].index
av_coefs = coefs[(coefs.model_type == model_type) & (coefs.neuronID.isin(av_neurons))]



sns.pairplot(
    data = av_coefs,
    x_vars = param_values,
    y_vars = param_values,
    )
plt.xticks(rotation=45)

# %%
