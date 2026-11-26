"""Presentation-only correction: move legend clear of the historical bars."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
names=['fraction_0.5','fraction_0.75','base','history_504']
data=pd.read_csv(ROOT/'data/research-v2/tables/exp_learning.csv')
z=data[data.variant.isin(names)].groupby('variant')[['train_log_loss','validation_log_loss']].mean().reindex(names)
z.index=['50% fitting rows','75% fitting rows','Full 1,008 window','504 window']
z.columns=['Training','Outer validation']
fig,ax=plt.subplots(figsize=(7,4.4))
z.plot.bar(ax=ax,color=['#167d8d','#d78348'],rot=12)
ax.set_ylabel('Log loss');ax.set_xlabel('Chronological training design')
ax.set_ylim(0,float(z.max().max())*1.18)
ax.legend(loc='upper center',ncol=2,frameon=False)
ax.set_title('Training fit versus outer validation')
ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(ROOT/'data/research-v2/figures/exp_learning_v2.png',dpi=180);plt.close(fig)
