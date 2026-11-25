"""Static, exportable architecture and information-timing diagrams."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/research-v2/figures'

def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4.8));ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
    boxes=[(.2,3.4,2.3,'Public completed bars\nRaw response + adjustments\nSession / freshness checks'),(3.05,3.4,2.7,'Frozen causal model policy\nPast-only fit / calibration / gate\nRegistered refit anchor'),(6.4,3.4,3.3,'Immutable daily plan\nProbabilities / target deltas\nInput, model, source hashes'),(.2,1.35,2.3,'Versioned historical evidence\nRegistered trials + forecasts\nExploratory diagnostics'),(3.05,1.35,2.7,'Shared execution contract\nCash / reservations / fills / fees\nDeployment cash + trend shadow'),(6.4,1.35,3.3,'Durable state + receipts\nLock / recovery / backups\nCalendar scheduler + watchdog')]
    for x,y,w,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,1.15,boxstyle='round,pad=0.06',facecolor='#e8f0f6',edgecolor='#244b6c',linewidth=1.3));ax.text(x+w/2,y+.58,label,ha='center',va='center',fontsize=10)
    for start,end in [((2.5,4),(3,4)),((5.75,4),(6.35,4)),((8,3.35),(8,2.55)),((6.3,1.93),(5.85,1.93)),((3,1.93),(2.6,1.93)),((4.4,3.35),(4.4,2.58))]:
        ax.annotate('',end,start,arrowprops={'arrowstyle':'->','color':'#264a61','lw':1.8})
    ax.text(5,.55,'All surfaces invoke one sealed runtime. Research development uses separate state. External orders are disabled.',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'system_architecture.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.7));ax.set(xlim=(-1,8),ylim=(-1,3));ax.axis('off')
    ax.annotate('',(7.5,1),(0,1),arrowprops={'arrowstyle':'->','lw':2,'color':'#244b6c'})
    for x,label,desc in [(0,'t','Completed close\nfeatures available'),(1,'t+1','First permitted fill\nreturn denominator'),(6,'t+6','Outcome endpoint\nlabel becomes observed')]:
        ax.plot([x,x],[.87,1.13],color='#244b6c',lw=2);ax.text(x,1.45,label,ha='center',fontsize=12,weight='bold');ax.text(x,.5,desc,ha='center',va='top',fontsize=10)
    ax.annotate('',(6,2.15),(1,2.15),arrowprops={'arrowstyle':'<->','lw':1.3,'color':'#348b82'})
    ax.text(3.5,2.45,'Five-session target return; adjacent labels overlap',ha='center',fontsize=11)
    ax.text(3.5,-.65,'Model fitting uses only labels with endpoint strictly before its decision. Future endpoints remain missing.',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'information_timeline.png',dpi=200);plt.close(fig)
    (OUT/'diagram_sources.json').write_text(json.dumps({'architecture':'docs/research-v2/manuscript.template.md sections4/9 and runtime_v2 policy','timeline':'data/research-v2/protocol.json labels','kind':'conceptual diagrams; no measured values'},indent=2))
if __name__=='__main__':main()
