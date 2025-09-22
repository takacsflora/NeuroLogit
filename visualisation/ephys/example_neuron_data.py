# %%

import numpy as np
from src.ephys.dat_utils import load_trial_data
from src.ephys.encoding_avg import filt_trials, get_time_params

import matplotlib.pyplot as plt 
import seaborn as sns
# 
# test things with example neuron 


fit_type = 'active_choice'
time_window = 'stim_bin'

time_params = get_time_params(time_window=time_window, pre_time=0.15, post_time=0.4)

#%%
ev,clusters,rasters = load_trial_data('AV030','2022-12-08',
                            load_clusters=True,**time_params).values()

ev = filt_trials(ev,fit_type = fit_type)

rasters = rasters[np.isin(rasters.Trial,ev.index)].copy()


ev['RT'] = ev.timeline_choiceMoveOn-ev.timeline_audPeriodOn


added_conditions = ['choice','visDiff','audDiff']
rasters = rasters.merge(ev[added_conditions], left_on='Trial', right_index=True, how='left')








#%%


#%%

# plot neuron location 

# Prepare data for plotting
feature_to_plot = 'neuron_1061'


clus_features = clusters[clusters.neuronID==feature_to_plot]
#
# plot the example neurons location
from floras_helpers.anat_plots import anatomy_plotter
from matplotlib.colors import LinearSegmentedColormap

anat = anatomy_plotter()
fig,ax = plt.subplots(1,1,figsize=(4, 2.5),dpi=300)
anat.plot_anat_canvas(ax=ax,coord = 3800, axis='ap')
goodClus = clusters[(clusters.is_good) & (clusters.BerylAcronym!='void')].copy()

anat.plot_points(goodClus.ml,goodClus.dv,unilateral=True,c = 'grey',alpha=0.2,marker = '.',s=150,edgecolor=None)

anat.plot_points(clus_features.ml,clus_features.dv,unilateral=True,c = 'red',alpha=1,marker = '.',s=300,edgecolor='k')


ax.set_xlim([-2200, 2200])
ax.set_ylim([-3000, -500])
ax.set_title(f'{feature_to_plot}, {clus_features.bombcell_class.values[0]}') 

ax.invert_xaxis()



# %%

on_x = 'visDiff'
on_y = 'choice'
on_hue  = 'audDiff'



# Average the responses across trials for each combination of conditions
average_responses = rasters[(rasters.Feature == feature_to_plot) & (rasters.choice==1)].groupby(
    [on_y, on_x, on_hue, 'Time']
)['Response'].mean().reset_index()


# # # we multiply visDiff and audDiff by hemi to always get the contralateral response
# average_responses['visDiff'] = average_responses['visDiff']*clus_features['hemi'].values[0] *-1
# average_responses['audDiff'] = average_responses['audDiff']*clus_features['hemi'].values[0] * -1 

# Adjust the spacing between subplots

g = sns.FacetGrid(average_responses, 
                  row=on_y, col=on_x, margin_titles=True, 
                  palette='coolwarm',
                  despine=True, height=4, aspect=.5)
g.fig.subplots_adjust(hspace=0.2, wspace=0.02)

                # Create a custom colormap with darker grey in the middle
colors = [(0, 0, 1), (0.5,0.5, 0.5), (1, 0, 0)]  # Blue -> Grey -> Red
positions = [0, 0.5, 1]  # Positions for the colors
custom_cmap = LinearSegmentedColormap.from_list("custom_coolwarm", list(zip(positions, colors)))

# Pass the custom colormap to the FacetGrid
g.map_dataframe(sns.lineplot, 'Time', 'Response', hue=on_hue, linewidth=4, alpha=1, hue_norm=(-1, 1), palette=custom_cmap)

# Add horizontal lines at y=0 for each subplot using the map function
#g.map(plt.axvline, x=0,ymin=.9,ymax=1, color='k', linestyle='-', linewidth=1, alpha=0.7)
# Adjust the y-axis limits to ensure the plot always includes the bottom of the frame
min_resp = average_responses['Response'].min() - np.ptp(average_responses['Response'])*.1
g.map(plt.hlines, y=min_resp, xmin=.15, xmax=0.20, color='k', linestyle='-', linewidth=1, alpha=1)
#g.map(plt.vlines, x=-.07, ymin=min_resp+10, ymax=min_resp+20, color='k', linestyle='-', linewidth=1, alpha=1)


# Remove x-axis labels and ticks
g.set_axis_labels("", "")
g.set(xticks=[])
g.set(yticks=[])


# Remove subplot titles
for ax in g.axes.flat:
    ax.set_title("")

# Remove bottom spines
for ax in g.axes.flat:
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)



plt.show()





#%%
vis_vals = np.sort(rasters.visDiff.unique())
aud_vals = np.sort(rasters.audDiff.unique())

colors = [(1,1,1),(0, 0, 1)]  # Blue -> Grey -> Red
positions = [0, 1]  # Positions for the colors
custom_blue = LinearSegmentedColormap.from_list("custom_blues", list(zip(positions, colors)))

colors = [(1,1,1),(1, 0, 0)]  # Blue -> Grey -> Red
positions = [0, 1]  # Positions for the colors
custom_red = LinearSegmentedColormap.from_list("custom_red", list(zip(positions, colors)))

aud_palettes= [custom_blue,'Greys',custom_red]
choice = 1
r_neur = rasters[(rasters.Feature == feature_to_plot) & (rasters.choice==choice)]

fig, axs = plt.subplots(len(aud_vals), len(vis_vals), figsize=(6, 3), dpi=300, sharex=True, sharey=False)
fig.subplots_adjust(hspace=0.1, wspace=0.1)
#

for i, aud_val in enumerate(aud_vals):
    for j,vis_val in enumerate(vis_vals):
        # Add the indented block of code here
        ax = axs[i, j]
        r_neur_trialtype = r_neur[(r_neur.visDiff == vis_val) & (r_neur.audDiff == aud_val)]

        response_matrix = r_neur_trialtype.pivot(index='Trial', columns='Time', values='Response')
        
        if (not response_matrix.empty) & (choice!=-2):
            rts = ev.loc[r_neur_trialtype.Trial.unique()].rt.values

            rts_sorted  = np.sort(rts)
            response_matrix_sorted = response_matrix.iloc[np.argsort(rts)]

            cax = ax.matshow(response_matrix_sorted, aspect='auto', cmap=aud_palettes[i])
            
            if time_window == 'choice_bin':
                dots = np.array(
                    [np.argmin(np.abs(-response_matrix_sorted.columns-rt)) for rt in rts_sorted]
                )

            else: 
                dots = np.array(
                    [np.argmin(np.abs(response_matrix_sorted.columns-rt)) for rt in rts_sorted]
                )

            ax.scatter(dots, np.arange(len(dots)), color='k', s=10, alpha=1, marker='o', edgecolor='none')
        elif choice==-2: 
            cax = ax.matshow(response_matrix, aspect='auto', cmap=aud_palettes[i])


        # Remove ticks and spines
        
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(left=False, bottom=False, top=False, right=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)


#%%
# also plot all the non-noise neurons on an example trial 

nrn_names = clusters[clusters.is_good].neuronID.unique()

trial_data = rasters[rasters.Trial == 5].copy()
trial_data = trial_data[trial_data.Feature.isin(nrn_names)].copy()


response_matrix = trial_data.pivot(index='Feature', columns='Time', values='Response')

plt.matshow(response_matrix, aspect='auto', cmap='coolwarm', vmin=-200, vmax=200)
#%%



# Add colorbar



# g = sns.FacetGrid(rasters[rasters.Feature==feature_to_plot], 
#                   row=on_y, col=on_x, hue=on_hue, margin_titles=True, palette='coolwarm',
#                   despine=False, height=1, aspect=1)

# g.map(sns.lineplot, 'Time', 'Response')

# # Add legend
# g.add_legend(title=on_hue)
# plt.show()

# %%
# Add a colorbar to the last plot
# Generate random data for the plot
random_data = np.random.rand(10, 10)

# Create a plot with the 'coolwarm' colormap
plt.figure(figsize=(6, 5))
cax = plt.imshow(random_data, cmap='coolwarm', aspect='auto')

# Add a colorbar
cbar = plt.colorbar(cax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Random Intensity', rotation=270, labelpad=15)

plt.title('Random Coolwarm Plot')
plt.show()


# %%
