#%% imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


from NeuroLogit.src.ephys.encoding_kernel.results_helpers import read_all_results  
from floras_helpers.plotting import off_axes
from floras_helpers.anat_plots import anatomy_plotter
from scipy.stats import pearsonr,spearmanr
import statsmodels.formula.api as smf

from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Passive')

plt.rcParams.update({'font.size': 8,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})


#%% load  cluster results
results = read_all_results(model_type='passive_active_Ridge10', 
                           sessions='unique',
                            trig_kws=None, VE_threshold = 0.005)


clusters = results['clusters']
clusters['session'] = clusters['subject'] + '_' + clusters['date'].astype(str)

clusters['is_sensory'] = clusters['is_visual'] | clusters['is_auditory']| clusters['is_AV']


clusters['is_task_and_sensory'] = clusters['is_task'] & clusters['is_sensory']

clusters['is_choice_and_task'] = clusters['is_choice'] & clusters['is_task']
clusters['is_choice_and_sensory'] = clusters['is_choice'] & clusters['is_sensory']

# %%

def get_across_all_neuron_measures(sel_clusters):
    print('from brain reigons: ' + ', '.join(sel_clusters['BerylAcronym'].unique()))
    print(f"Total number of neurons: {len(sel_clusters)}")

    percentage_r2_tot = (sel_clusters['r2_tot'] > 10e-3).mean() * 100
    print(f"Percentage of sel_clusters where r2_tot > 10e-3: {percentage_r2_tot:.2f}%")

    # Calculate the percentage of neurons for each functional type
    percentages = sel_clusters['functional_type'].value_counts(normalize=True) * 100

    print(percentages)

    vis_neurons = sel_clusters[sel_clusters['is_visual']]
    percentage_positive_amp_visual = (vis_neurons['amp_visual'] > 0).mean() * 100
    print(f"Percentage of visual neurons where amp_visual > 0: {percentage_positive_amp_visual:.2f}%")


    aud_neurons = sel_clusters[sel_clusters['is_auditory']]
    percentage_positive_amp_auditory = (aud_neurons['amp_auditory'] > 0).mean() * 100
    print(f"Percentage of auditory neurons where amp_auditory > 0: {percentage_positive_amp_auditory:.2f}%")

    task_neurons = sel_clusters[sel_clusters['is_task']]
    percentage_positive_amp_task = (task_neurons['amp_engagement'] > 0).mean() * 100
    print(f"Percentage of task neurons where amp_engagement > 0: {percentage_positive_amp_task:.2f}%")

    choice_neurons = sel_clusters[sel_clusters['is_choice']]
    percentage_positive_amp_choice_contra = (choice_neurons['amp_choice_contra'] > 0).mean() * 100
    percentage_positive_amp_choice_ipsi = (choice_neurons['amp_choice_ipsi'] > 0).mean() * 100
    print(f"Percentage of choice neurons where amp_choice_ipsi > 0: {percentage_positive_amp_choice_ipsi:.2f}%")
    print(f"Percentage of choice neurons where amp_choice_contra > 0: {percentage_positive_amp_choice_contra:.2f}%")

    # amp contra bigger than amp ipsi
    percentage_choice_contra_greater_ipsi = (choice_neurons['amp_choice_contra'] > choice_neurons['amp_choice_ipsi']).mean() * 100
    print(f"Percentage of choice neurons where amp_choice_contra > amp_choice_ipsi: {percentage_choice_contra_greater_ipsi:.2f}%")

    # number of mice and sessions
    n_mice = sel_clusters['subject'].nunique()
    n_sessions = sel_clusters['session'].nunique()
    print (f"Number of mice: {n_mice}, Number of sessions: {n_sessions}")

def fraction_significant_neurons_across_subjects(sel_clusters, d_string='visual',min_region_count=5):
    subjects = sel_clusters['subject'].unique()
    regions = sel_clusters['BerylAcronym'].unique()
    fractions = []
    total_fractions = []
    for subject in subjects:
        subject_clusters = sel_clusters[sel_clusters['subject'] == subject]
        region_counts = subject_clusters.groupby('BerylAcronym').size()
        region_counts[region_counts < min_region_count] = np.nan
        
        counts = subject_clusters[subject_clusters[f'is_{d_string}']].groupby('BerylAcronym').size()
        # because if 0 pd makes this a nan apparently 
        counts = counts.reindex(region_counts.index, fill_value=0)
        frac = (counts / region_counts)
        fractions.append(frac)
        sig_count = counts.reindex(regions, fill_value=0).sum()
        total_count = region_counts.reindex(regions).sum()
        total_fraction = sig_count / total_count if total_count and not np.isnan(total_count) else np.nan
        total_fractions.append(total_fraction)

    fractions_df = pd.DataFrame(fractions, index=subjects)
    fractions_df['across_region_total_fraction'] = total_fractions
    return fractions_df

def average_measure_across_subjects(sel_clusters, measure= 'VE', d_string='visual', min_region_count=5):
    subjects = sel_clusters['subject'].unique()
    regions = sel_clusters['BerylAcronym'].unique()
    measure_means = []
    for subject in subjects:
        subject_clusters = sel_clusters[sel_clusters['subject'] == subject]
        region_counts = subject_clusters.groupby('BerylAcronym').size()
        region_counts[region_counts < min_region_count] = np.nan
        
        measure_mean = subject_clusters.groupby('BerylAcronym')[f'{measure}_{d_string}'].mean()
        measure_mean = measure_mean.reindex(region_counts.index)
        measure_mean_all_regions = subject_clusters[f'{measure}_{d_string}'].mean()
        measure_mean['across_region_total_fraction'] = measure_mean_all_regions
        measure_means.append(measure_mean)

    measure_means_df = pd.DataFrame(measure_means, index=subjects)
    return measure_means_df

def compare_two_regions_with_lme(sel_clusters,d_string='visual',region1='SCs',region2='SCm',**fraction_kws):
    "test compare two regions with LME and also print the modeled percentages for the paper text"
    "sel_clusters: dataframe of clusters with columns for subject, BerylAcronym, and is_{d_string} (e.g. is_visual)"
    "d_string: which functional type to compare (e.g. 'visual', 'auditory', 'task', 'choice')"
    "region1, region2: which two regions to compare (e.g. 'SCs', 'SCm') -- test compares hypothesis region2 > region1"
    
    fractions_df = fraction_significant_neurons_across_subjects(sel_clusters,d_string=d_string, **fraction_kws)
    lme_data = fractions_df[[region1, region2]].copy()
    lme_data.index.name = 'subject'
    lme_data = lme_data.reset_index().melt(id_vars='subject', var_name='region', value_name='fraction').dropna()
    lme_data['region_code'] = (lme_data['region'] == region2).astype(float)  # region2=1, region1=0
    try:
        lme_model = smf.mixedlm('fraction ~ region_code', lme_data, groups=lme_data['subject']).fit()
        print(f'\n--- LME: {region1} vs {region2} ({d_string}) ---')
        print(lme_model.summary())
        # Wald test: coefficient of region_code > 0 means region2 > region1
        wald_stat = lme_model.tvalues['region_code']
        two_sided_p = lme_model.pvalues['region_code']  # one-sided: region1 < region2

        if wald_stat > 0:
            wald_pval = two_sided_p / 2
        else:
            wald_pval = 1 - (two_sided_p / 2)

        print(f'Wald test (one-sided, {region1} < {region2}): t={wald_stat:.3f}, p={wald_pval:.4f}')


        # NEW: Print clean model-derived percentages for your paper
        # -------------------------------------------------------------
        scs_pct = lme_model.params["Intercept"] * 100
        scm_pct = (lme_model.params["Intercept"] + lme_model.params["region_code"]) * 100
        print(f"--> Modeled Percentages for Text: {region1} = {scs_pct:.1f}%, {region2} = {scm_pct:.1f}%")
        print(f"Number of subjects: {lme_data['subject'].nunique()}")
    except Exception as e:
        print(f'LME failed for {d_string}: {e}')
    fig,ax = plt.subplots(1,1,figsize=(1,1),dpi=150)
    sns.boxplot(data=fractions_df[[region1, region2]], palette='Set2', ax=ax)

    ax.axhline(0.05, color='k', linestyle=':',alpha=1,linewidth=0.6)
    ax.set_ylabel(f'Fraction of {d_string} neurons')
    off_axes(ax, which='top')
    print('across regions average fraction of significant neurons for',d_string,':{:.3f}'.format(fractions_df['across_region_total_fraction'].mean()))

def between_kernels_VE_correlations_lme(sel_clusters,x_var = 'vis_spatial', y_var = 'aud_onset'):
    col_x = f'VE_{x_var}'
    col_y = f'VE_{y_var}'
    fig,ax = plt.subplots(1,1,figsize=(1,1),dpi=150)
    sns.scatterplot(data=sel_clusters, x=col_x, y=col_y,s=8,
                    color='#A7A5A596',
                    edgecolor='k',linewidth=0.3,alpha=0.7,ax=ax)


    lme_data = sel_clusters[[col_x, col_y, 'subject']].copy().dropna()
    lme_data['rank_x'] = lme_data[col_x].rank()
    lme_data['rank_y'] = lme_data[col_y].rank()

    try: 
        lme_model = smf.mixedlm('rank_y ~ rank_x', lme_data, groups=lme_data['subject']).fit()
        print(f'\n--- LME: Correlation between {col_x} and {col_y} ---')
        print(lme_model.summary())

        df = lme_model.nobs - len(lme_model.params)  # degrees of freedom
        t_stat = lme_model.tvalues['rank_x']
        pval = lme_model.pvalues['rank_x']

        hierarchical_rho = np.sign(t_stat) * (np.abs(t_stat) / np.sqrt(t_stat**2 + df))
        # give p value in 10e-4 format if very small
        if pval < 0.0001:
            pval_str = f"{pval:.2e}"
        else:            
            pval_str = f"{pval:.4f}"
        print(f"Hierarchical Spearman's rho: {hierarchical_rho:.3f}, p={pval_str}")
    except Exception as e:
        print(f'LME failed for correlation between {col_x} and {col_y}: {e}')

# %%

sel_clusters = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCm'])) 
].copy()

get_across_all_neuron_measures(sel_clusters)
#%%


sel_clusters = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCs', 'SCm'])) 
].copy()

compare_two_regions_with_lme(sel_clusters, d_string='visual', region1='SCm', region2='SCs', min_region_count=10)
# %%
compare_two_regions_with_lme(sel_clusters, d_string='auditory', region1='SCs', region2='SCm', min_region_count=10)
# %%
compare_two_regions_with_lme(sel_clusters, d_string='AV', region1='SCs', region2='SCm', min_region_count=10)

# %%
compare_two_regions_with_lme(sel_clusters, d_string='task', region1='SCs', region2='SCm', min_region_count=10)

#%%
compare_two_regions_with_lme(sel_clusters, d_string='choice', region1='SCs', region2='SCm', min_region_count=10)

#%%

compare_two_regions_with_lme(sel_clusters, d_string='sensory', region1='SCs', region2='SCm', min_region_count=10)

#%%
compare_two_regions_with_lme(sel_clusters, d_string='task_and_sensory', region1='SCs', region2='SCm', min_region_count=10)

#%%
compare_two_regions_with_lme(sel_clusters, d_string='choice_and_sensory', region1='SCs', region2='SCm', min_region_count=10)

#%%
compare_two_regions_with_lme(sel_clusters, d_string='choice_and_task', region1='SCs', region2='SCm', min_region_count=10)

# %%
sel_clusters = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['MOs'])) 
].copy()

between_kernels_VE_correlations_lme(sel_clusters, x_var = 'vis_spatial', y_var = 'aud_onset')

between_kernels_VE_correlations_lme(sel_clusters, x_var = 'vis_spatial', y_var = 'aud_spatial')

between_kernels_VE_correlations_lme(sel_clusters, x_var = 'aud_onset', y_var = 'aud_spatial')

between_kernels_VE_correlations_lme(sel_clusters, x_var = 'aud_onset', y_var = 'aud_spatial')

# %%

between_kernels_VE_correlations_lme(sel_clusters, x_var = 'choice_ipsi', y_var = 'choice_contra')
#%%
between_kernels_VE_correlations_lme(sel_clusters, x_var = 'task', y_var = 'choice')
# %%
between_kernels_VE_correlations_lme(sel_clusters, x_var = 'task', y_var = 'vis_spatial')
# %%
between_kernels_VE_correlations_lme(sel_clusters, x_var = 'task', y_var = 'aud_onset')
# %%
between_kernels_VE_correlations_lme(sel_clusters, x_var = 'task', y_var = 'aud_spatial')

02# %%
between_kernels_VE_correlations_lme(sel_clusters, x_var = 'choice', y_var = 'vis_spatial')
# %%

between_kernels_VE_correlations_lme(sel_clusters, x_var = 'choice', y_var = 'aud_onset')
# %%

between_kernels_VE_correlations_lme(sel_clusters, x_var = 'choice', y_var = 'aud_spatial')
# %%

sel_clusters = clusters[
    (clusters.bombcell_class=='good') &
    (clusters.BerylAcronym.isin(['SCs', 'SCm'])) 
].copy()


VE_means_df = average_measure_across_subjects(sel_clusters, measure='VE', d_string='action_onset', min_region_count=10)

print('across regions average VE for engagement:{:.3f}'.format(VE_means_df['across_region_total_fraction'].mean()))

# %%
# 
amp_choicediff_df = average_measure_across_subjects(sel_clusters, measure='amp', d_string='choice', min_region_count=10)

from scipy import stats
t_stat, p_val = stats.ttest_1samp(amp_choicediff_df['across_region_total_fraction'].dropna(), 0)
print('t-test against 0: t={:.3f}, p={:.4f}'.format(t_stat, p_val))

# %%
