# utility functions for neural data visualisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def revert_per_hemisphere(coefs):
        coefs['vis_ipsi'] = coefs.apply(lambda row: row['visR'] if row['hemi'] == -1 else row['visL'], axis=1)
        coefs['vis_contra'] = coefs.apply(lambda row: row['visL'] if row['hemi'] == -1 else row['visR'], axis=1)
        coefs['aud_ipsi'] = coefs.apply(lambda row: row['audR'] if row['hemi'] == -1 else row['audL'], axis=1)
        coefs['aud_contra'] = coefs.apply(lambda row: row['audL'] if row['hemi'] == -1 else row['audR'], axis=1)

        
        if 'baseline_active' in coefs.columns:
            coefs['vis_ipsi_active'] = coefs.apply(lambda row: row['visR_active'] if row['hemi'] == -1 else row['visL_active'], axis=1)
            coefs['vis_contra_active'] = coefs.apply(lambda row: row['visL_active'] if row['hemi'] == -1 else row['visR_active'], axis=1)
            coefs['aud_ipsi_active'] = coefs.apply(lambda row: row['audR_active'] if row['hemi'] == -1 else row['audL_active'], axis=1)
            coefs['aud_contra_active'] = coefs.apply(lambda row: row['audL_active'] if row['hemi'] == -1 else row['audR_active'], axis=1)


def add_active_to_base(df):
        active_columns = [col for col in df.columns if 'active' in col]
        if len(active_columns) > 0:
            for col in active_columns:
                base_col = col.replace('_active', '')
                active_col = col    
                new_col_name = f'tot_{base_col}'

                base = df[base_col].copy()
                active = df[active_col].copy()

                active = active.fillna(0)

                tot = base + active

                df[new_col_name] = tot

        return df

def load_results(region = 'SCm',fit_type = 'engagement',time_bin = 'poststim',revert_coefs = True):
        df = pd.read_csv(f'D:/AVTrialData/{region}_{time_bin}/fit_results/{fit_type}.csv')
        # filter for neruons that are good and are from the ROI

        df =df[(df.BerylAcronym == region) & df.is_good]
        # combine subject and date into a single string named session
        df['session'] = df['subject'] + '_' + df['date']


        coefs = df.copy()

        if revert_coefs:
            revert_per_hemisphere(coefs)


        coefs = add_active_to_base(coefs)

        coefs['choice_contra']=(coefs.hemi*coefs.choice)

        return coefs


def select_best_models(coefs):
    # select the best models
    coefs_kept = coefs.loc[coefs.groupby(['session', 'neuronID'])['adj_r2'].idxmax()]
    # also keep only neurons where the adj_r2 is above 0.1
    coefs_kept = coefs_kept[coefs_kept['adj_r2'] > 0.05]


    return coefs_kept

def get_region_params(region,random_state = 1,plot_dist = True,set = 'active'):

    coefs = load_results(region = region,fit_type = 'choice_engagement',time_bin = 'poststim')

    # keep neurons

    coefs_kept = select_best_models(coefs)
    # replace NaNs with 0
    

    #coefs_kept = coefs_kept.fillna(0)
    if plot_dist:

        params_to_plot = ['tot_vis_contra', 'tot_vis_ipsi', 'tot_aud_contra', 'tot_aud_ipsi', 'gamma', 'baseline_active','baseline','choice_contra']
        n_params = len(params_to_plot)
        _, axs = plt.subplots(1, n_params, figsize=(n_params*3, 3))

        # Plot tot_vis_contra
        
        def hist_plotter(x, ax):
            ax.hist(x, bins=50)
            ax.plot(x.median(), 2, marker='v', color='r', markersize=10)
            #ax.axvline(x.median(), color='g', linestyle='dashed', linewidth=1)
            ax.axvline(0, color='k', linestyle='dashed', linewidth=1)

        for i, param in enumerate(params_to_plot):
            hist_plotter(coefs_kept[param], axs[i])
            axs[i].set_title(f'{param}, median: {coefs_kept[param].median():.2f}')

        plt.tight_layout()
        plt.show()


    # randomly sample 10 neurons from each session
    coefs_sess_subsampled = coefs_kept.groupby('session').apply(lambda x: x.sample(n=min(10, len(x)), random_state=random_state)).reset_index(drop=True)
    # randomly sample 50 neurons in total from the sess subsampled one
    coefs_kept = coefs_sess_subsampled.sample(n=min(50, len(coefs_sess_subsampled)), random_state=random_state)
    

    if set == 'active':
        avg_coefs = {
            'vis_contra':coefs_kept['tot_vis_contra'].sum(),
            'vis_ipsi':coefs_kept['tot_vis_ipsi'].sum(),
            'aud_contra':coefs_kept['tot_aud_contra'].sum(),
            'aud_ipsi':coefs_kept['tot_aud_ipsi'].sum(),
            'gamma':coefs_kept['gamma'].median(),
            'baseline':coefs_kept['baseline_active'].sum()
        }

    elif set == 'choice':
        avg_coefs = {
            'vis_contra':coefs_kept['tot_vis_contra'].sum(),
            'vis_ipsi':coefs_kept['tot_vis_ipsi'].sum(),
            'aud_contra':coefs_kept['tot_aud_contra'].sum(),
            'aud_ipsi':coefs_kept['tot_aud_ipsi'].sum(),
            'gamma':coefs_kept['gamma'].median(),
            'baseline':coefs_kept['baseline_active'].sum(),
            'choice_contra':coefs_kept['choice_contra'].sum(),
        }  

    elif set == 'passive':
        avg_coefs = {
            'vis_contra':coefs_kept['vis_contra'].sum(),
            'vis_ipsi':coefs_kept['vis_ipsi'].sum(),
            'aud_contra':coefs_kept['aud_contra'].sum(),
            'aud_ipsi':coefs_kept['aud_ipsi'].sum(),
            'gamma':coefs_kept['gamma'].median(),
            'baseline':coefs_kept['baseline'].sum()
        }

    return avg_coefs