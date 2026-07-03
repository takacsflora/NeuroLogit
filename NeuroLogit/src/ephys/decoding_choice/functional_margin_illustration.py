#%%
from pathlib import Path
import matplotlib.pyplot as plt 




SAVE_PATH = Path(r'C:\Users\Flora\OneDrive - University College London\Cortexlab\papers\SCpaper_v2025Dec\raw plots\Choice')

plt.rcParams.update({'font.size': 8,'font.family':'Calibri','axes.linewidth':0.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':True,'axes.spines.bottom':True,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':2,'ytick.major.size':2})


fig,ax = plt.subplots(1,1,figsize=(1,1),dpi=150)

ax.plot([-10,10],[-10,10],color='g',linewidth=1)
ax.plot([10,-10],[-10,10],color='m',linewidth=1)

ax.axhline(0,color='k',linewidth=0.5,linestyle=':')

# ax.set_xticks([-10,-5,0,5,10])
# ax.set_yticks([-10,-5,0,5,10])

fig.savefig(SAVE_PATH / 'functional_margin_illustration.svg',bbox_inches='tight',dpi=300)
# %%
