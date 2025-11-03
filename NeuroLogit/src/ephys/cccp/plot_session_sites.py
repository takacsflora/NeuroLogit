# %%


from src.ephys.cccp.results_helpers import read_files

from floras_helpers.anat_plots import anatomy_plotter
from floras_helpers.plotting import off_axes
import matplotlib.pyplot as plt

df = read_files(filestub='clusters',extension='csv')

coord = 3800

anat = anatomy_plotter()

subjects = df.subject.unique()

#subjects = ['AV005','AV007']
n_subjects = len(subjects)
# Find the maximum number of dates per subject
max_dates_per_subject = df.groupby('subject')['date'].nunique().max()
print(f"Maximum number of dates per subject: {max_dates_per_subject}")
fig,axs = plt.subplots(n_subjects,max_dates_per_subject,
                       figsize=(max_dates_per_subject*1.75, n_subjects*1),
                       dpi=300,sharex=True,sharey=True)
fig.subplots_adjust(wspace=0.1,hspace=0.3)
for i, subject in enumerate(subjects):
    subject_df = df[df.subject==subject].copy()
    dates = subject_df.date.unique()
    coord = subject_df.ap.mean()-5600
    if coord>0:
        coord = 3800
    else:
        coord = -1500

    for j, date in enumerate(dates):
        ax = axs[i,j]
        anat.plot_anat_canvas(ax=ax, coord=coord, axis='ap')

        anat.plot_points(subject_df['ml'], subject_df['dv'], unilateral=False, c='grey', alpha=0.2, marker='.', s=5, edgecolor=None)

        on_date = subject_df[subject_df.date==date].copy()

        anat.plot_points(on_date['ml'], on_date['dv'], unilateral=False, c='red',
                        alpha=0.8, marker='.', s=50, edgecolor='white', cmap='coolwarm', vmin=.1, vmax=.9)

        ax.set_xlim([-2200, 2200])
        ax.set_ylim([-3000, -500])
        ax.set_title(f'{date}', fontsize=8)
    
    axs[i,0].set_ylabel(f'{subject}', fontsize=10)
        
for ax in axs.flatten():
    off_axes(ax)


# %%
