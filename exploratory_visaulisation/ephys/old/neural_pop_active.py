#%%
import pandas as pd 
from utils.neural_results_utils import load_results,select_best_models


regions = ['SCs','SCm']
#coefs = pd.concat([load_results(dataset='all',region = r,fit_type = 'passive',time_bin = 'stim_bin_pre_0.40_post_0.60') for r in regions])
coefs = pd.concat([load_results(dataset=None,region = r,fit_type = 'choice',time_bin = 'poststim') for r in regions])

#%%

#%%|
selected_coefs = select_best_models(coefs,thr=-0.01)

selected_coefs = selected_coefs.fillna(0)


#%%
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

pairs = [
    ('vis_contra', 'tot_vis_contra'),
    ('vis_ipsi', 'tot_vis_ipsi'),
    ('aud_contra', 'tot_aud_contra'),
    ('aud_ipsi', 'tot_aud_ipsi'),
    ('baseline', 'tot_baseline'),

]


pairs = [
    ('baseline_active', 'tot_vis_contra'),
    ('baseline_active', 'tot_vis_ipsi'),
    ('baseline_active', 'tot_aud_contra'),
    ('baseline_active', 'tot_aud_ipsi'),
    ('baseline_active', 'choice_contra'),

]

pairs = [
    ('baseline_active', 'vis_contra_active'),
    ('baseline_active', 'vis_ipsi_active'),
    ('baseline_active', 'aud_contra_active'),
    ('baseline_active', 'aud_ipsi_active'),
    ('baseline_active', 'choice_contra'),

]


pairs = [
    ('baseline_active', 'vis_contra'),
    ('baseline_active', 'vis_ipsi'),
    ('baseline_active', 'aud_contra'),
    ('baseline_active', 'aud_ipsi'),
    ('baseline_active', 'choice_contra'),

]


pairs = [
    ('baseline', 'vis_contra'),
    ('baseline', 'vis_ipsi'),
    ('baseline', 'aud_contra'),
    ('baseline', 'aud_ipsi'),

]

parameters = ['vis_contra', 'vis_ipsi', 'aud_contra', 'aud_ipsi', 'baseline']
filtered_coefs = selected_coefs.copy()
for param in parameters:
    lower_bound = selected_coefs[param].quantile(0.00)
    upper_bound = selected_coefs[param].quantile(1)
    filtered_coefs = filtered_coefs[(filtered_coefs[param] >= lower_bound) & (filtered_coefs[param] <= upper_bound)]

to_correlate = False
to_fit = False
params1, params2 = zip(*pairs)
   
unique_acronyms = selected_coefs['BerylAcronym'].unique()
fig, axes = plt.subplots(len(unique_acronyms), len(pairs), figsize=(3* len(pairs), 2.5 * len(unique_acronyms)), sharex=True, sharey=True)

for row, acronym in enumerate(unique_acronyms):
    acronym_data = filtered_coefs[filtered_coefs['BerylAcronym'] == acronym]
    
       
    for col, (param1, param2) in enumerate(pairs):
        ax = axes[row, col]


        sns.scatterplot(data=acronym_data, x=param1, y=param2, alpha=0.5, ax=ax,color='blue')
        #sns.histplot(data=acronym_data, x=param1, y=param2, bins=10, pmax=0.9, cmap="viridis", ax=ax)
        #sns.kdeplot(data=acronym_data, x=param1, y=param2, fill=True, cmap="viridis", ax=ax,levels=30,thresh=0.05)

        if to_fit:
            # Fit a linear mixed effects model
            model = smf.mixedlm(f"{param2} ~ {param1}", acronym_data, groups=acronym_data["session"])
            result = model.fit()

            # Extract the fixed effects line
            intercept, slope = result.fe_params

            # Plot the regression line
            x_vals = acronym_data[param1]
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals, y_vals, color='red', linewidth=1)
            ax.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1)  # Unity line

        if to_correlate:
            # Calculate Spearman correlation
            spearman_corr = acronym_data[[param1, param2]].corr(method='spearman').iloc[0, 1]
            ax.set_title(f'Spearman r={spearman_corr:.2f}')


        #sns.regplot(data=acronym_data, x=param1, y=param2, scatter=False, ax=ax, color='blue', line_kws={"linewidth": 1})
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)  # Horizontal zero line
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)  # Vertical zero line
        # if row == 0:

        if col == 0:
            ax.set_ylabel(acronym)
    
plt.suptitle('Combined Plots for All Acronyms')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.5, hspace=0.2)
plt.show()



# %%
import seaborn as sns
unique_acronyms = selected_coefs['BerylAcronym'].unique()

# separate plot for separate acronyms
# separate plot for separate acronyms
for acronym in unique_acronyms:
    acronym_data = selected_coefs[selected_coefs['BerylAcronym'] == acronym]

    parameters = ['vis_contra', 'vis_ipsi','gamma', 'aud_contra', 'aud_ipsi', 'baseline']
    g = sns.PairGrid(acronym_data, vars=parameters, hue="BerylAcronym")
    g.fig.set_size_inches(6,5)

    g = g.map_upper(sns.scatterplot)
    #g = g.map_lower(sns.scatterplot)

    #g = g.map_lower(sns.kdeplot, cmap="viridis", fill=True)
    g = g.map_diag(sns.histplot, kde_kws={"color": "k"})
    g.add_legend()

    # Ensure ylims are the same across subplots
    # for ax in g.axes.flatten():
    #     ax.set_ylim(acronym_data[parameters].min().min(), acronym_data[parameters].max().max())
    #     # Ensure xlims are the same across subplots
    #     ax.set_xlim(acronym_data[parameters].min().min(), acronym_data[parameters].max().max())


#%%
import seaborn as sns
import matplotlib.pyplot as plt
unique_acronyms = selected_coefs['BerylAcronym'].unique()
parameters = ['vis_contra', 'vis_ipsi','gamma', 'aud_contra', 'aud_ipsi', 'baseline']

# Filter data points within the 10-90 percentile for each parameter
filtered_coefs = selected_coefs.copy()
for param in parameters:
    lower_bound = selected_coefs[param].quantile(0.10)
    upper_bound = selected_coefs[param].quantile(0.90)
    filtered_coefs = filtered_coefs[(filtered_coefs[param] >= lower_bound) & (filtered_coefs[param] <= upper_bound)]
    # Separate pairplots for each brain region
for acronym in unique_acronyms:
    region_data = filtered_coefs[filtered_coefs['BerylAcronym'] == acronym]
    sns.pairplot(region_data, vars=parameters, hue="BerylAcronym")
    plt.suptitle(f'Pairplot of Selected Parameters within 5-95 Percentile for {acronym}')
    plt.ylim([0,300])

plt.show()

#%%
# Plot histograms for each parameter on top of each other
for acronym in unique_acronyms:
    region_data = filtered_coefs[filtered_coefs['BerylAcronym'] == acronym]
    fig,ax = plt.subplots(1,len(parameters),figsize=(3*len(parameters),3),sharex=False,sharey=True)
    for cax,param in zip(ax,parameters):
        sns.histplot(region_data[param], kde=True, label=param, bins=30, alpha=0.5,ax=cax)

plt.legend()
plt.title('Histograms of Selected Parameters')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


# Adjust subplot sizes
#%%

for acronym in unique_acronyms:
    acronym_data = selected_coefs[selected_coefs['BerylAcronym'] == acronym]
    sns.pairplot(acronym_data[parameters])
    plt.suptitle(f'Pairplot of Selected Parameters for {acronym}')
    plt.show()
# Classic pairplot with linked x and y axes
plt.suptitle('Pairplot of Selected Parameters')
plt.show()
parameters = ['vis_contra', 'vis_ipsi', 'aud_contra', 'aud_ipsi', 'baseline']
g = sns.PairGrid(selected_coefs, vars=parameters, hue="BerylAcronym")
g = g.map_upper(sns.scatterplot)
#g = g.map_lower(sns.kdeplot, cmap="Blues_d")
g = g.map_diag(sns.histplot, kde=True, bins=20, kde_kws={"color": "k"})
g.add_legend()


#%%

for acronym in unique_acronyms:
    acronym_data = selected_coefs[selected_coefs['BerylAcronym'] == acronym]
    sns.pairplot(acronym_data[parameters])
    plt.suptitle(f'Pairplot of Selected Parameters for {acronym}')
    plt.show()
# Classic pairplot with linked x and y axes
plt.suptitle('Pairplot of Selected Parameters')
plt.show()

# %%


