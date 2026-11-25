"""Descriptive postprocessing of frozen predictions; no fitting or selection."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'data/research-v2/experiments'
TAB=ROOT/'data/research-v2/tables'
p=pd.read_parquet(OUT/'predictions.parquet')
selected=p[p.seed.eq(42)]
base=selected[selected.variant.eq('base')]
seed=[]
for (s,f),g in p[p.variant.eq('base')&p.model.eq('calibrated')].groupby(['symbol','outer_fold']):
 if g.seed.nunique()<3:continue
 a=g[g.seed.eq(42)].sort_values('date')
 losses=[];maxdelta=0.
 for sv,v in g.groupby('seed'):
  v=v.sort_values('date');maxdelta=max(maxdelta,float(np.abs(a[['p0','p1','p2']].to_numpy()-v[['p0','p1','p2']].to_numpy()).max()));losses.append(float(v.log_loss.mean()))
 seed.append(dict(symbol=s,fold=f,n_seeds=g.seed.nunique(),n_dates=len(a),maximum_probability_difference=maxdelta,mean_seed_log_loss=np.mean(losses),seed_log_loss_sd=np.std(losses,ddof=1)))
pd.DataFrame(seed).to_csv(TAB/'exp_seed_variance.csv',index=False)
rows=[];conf=[]
for (variant,symbol),g in selected[selected.variant.isin(['base','transfer'])&selected.model.eq('calibrated')].groupby(['variant','symbol']):
 y=g.label.to_numpy();guess=g[['p0','p1','p2']].to_numpy().argmax(1)
 precision,recall,f1,support=precision_recall_fscore_support(y,guess,labels=[0,1,2],zero_division=0)
 cm=confusion_matrix(y,guess,labels=[0,1,2])
 for c in range(3):
  rows.append(dict(variant=variant,symbol=symbol,class_id=c,n_rows=len(g),support=int(support[c]),precision=precision[c],recall=recall[c],f1=f1[c]))
  for z in range(3):conf.append(dict(variant=variant,symbol=symbol,true_class=c,predicted_class=z,count=int(cm[c,z])))
pd.DataFrame(rows).to_csv(TAB/'exp_transfer_class_metrics.csv',index=False)
pd.DataFrame(conf).to_csv(TAB/'exp_transfer_confusion.csv',index=False)
l=base[base.model.eq('linear')].copy();t=base[base.model.eq('tree')].copy()
l['linear_class']=l[['p0','p1','p2']].to_numpy().argmax(1);t['tree_class']=t[['p0','p1','p2']].to_numpy().argmax(1)
a=l[['date','symbol','linear_class']].merge(t[['date','symbol','tree_class']],on=['date','symbol'],validate='one_to_one')
a['disagree']=a.linear_class.ne(a.tree_class)
joined=base.merge(a,on=['date','symbol'],validate='many_to_one')
joined.groupby(['model','disagree','gate_separate']).agg(n_rows=('label','size'),n_dates=('date','nunique'),log_loss=('log_loss','mean')).reset_index().to_csv(TAB/'exp_disagreement.csv',index=False)
coverage=[]
for m,g in base.groupby('model'):
 prob=g[['p0','p1','p2']].to_numpy();confidence=prob.max(1);guess=prob.argmax(1)
 for threshold in [0,.4,.5,.6,.7,.8,.9]:
  mask=confidence>=threshold
  coverage.append(dict(model=m,threshold=threshold,n_total=len(g),n_retained=int(mask.sum()),n_dates=int(g.loc[mask,'date'].nunique()),coverage=float(mask.mean()),accuracy=float(np.mean(guess[mask]==g.label.to_numpy()[mask])) if mask.any() else None,log_loss=float(g.loc[mask,'log_loss'].mean()) if mask.any() else None))
pd.DataFrame(coverage).to_csv(TAB/'exp_confidence_coverage.csv',index=False)
# Calibration bands use ratio of total successes/total eligible forecasts under
# date-block resampling, including zero-bin-count dates; every symbol stays on date.
cal=[];rng=np.random.default_rng(20260928)
for model,g in base[base.model.isin(['raw','calibrated','gate_separate'])].groupby('model'):
 dates=pd.DatetimeIndex(sorted(g.date.unique()));datefold=g.groupby('date').outer_fold.first().reindex(dates).to_numpy()
 for cl in [0,1,2]:
  confidence=g['p'+str(cl)];bins=np.minimum((confidence*10).astype(int),9)
  for b in range(10):
   m=bins.eq(b)
   if not m.any():continue
   count=g.loc[m].groupby('date').size().reindex(dates,fill_value=0).to_numpy()
   hit=g.loc[m].assign(hit=g.loc[m,'label'].eq(cl)).groupby('date').hit.sum().reindex(dates,fill_value=0).to_numpy()
   sum_count=np.zeros(2000);sum_hit=np.zeros(2000)
   for f in np.unique(datefold):
    ix=np.flatnonzero(datefold==f);n=len(ix);starts=rng.integers(0,n,size=(2000,int(np.ceil(n/10))))
    sample=ix[((starts[:,:,None]+np.arange(10))%n).reshape(2000,-1)[:,:n]]
    sum_count+=count[sample].sum(1);sum_hit+=hit[sample].sum(1)
   valid=sum_count>0;frequency=sum_hit[valid]/sum_count[valid]
   cal.append(dict(model=model,class_id=cl,bin=b,n=int(m.sum()),n_dates_with_support=int((count>0).sum()),mean_probability=float(confidence[m].mean()),observed_frequency=float(g.loc[m,'label'].eq(cl).mean()),frequency_ci_low=float(np.quantile(frequency,.025)),frequency_ci_high=float(np.quantile(frequency,.975)),valid_bootstrap_replicates=int(valid.sum()),block=10))
pd.DataFrame(cal).to_csv(TAB/'exp_reliability_uncertainty.csv',index=False)
# The figure uses exactly this table; bin intervals are descriptive, not a new
# family of hypothesis tests, and are not adjusted simultaneous bands.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(10,3.6),sharex=True,sharey=True)
cal=pd.DataFrame(cal)
for cl,ax in enumerate(axs):
 for model,color in [('raw','#cc793d'),('calibrated','#177b87')]:
  z=cal[cal.class_id.eq(cl)&cal.model.eq(model)].sort_values('mean_probability')
  ax.plot(z.mean_probability,z.observed_frequency,'o-',label=model,color=color,ms=3)
  ax.fill_between(z.mean_probability,z.frequency_ci_low,z.frequency_ci_high,alpha=.16,color=color)
 ax.plot([0,1],[0,1],'--',c='#999',lw=.8);ax.set_title(['DOWN','FLAT','UP'][cl]);ax.set_xlabel('Forecast probability')
axs[0].set_ylabel('Observed frequency');axs[-1].legend(fontsize=8);fig.suptitle('Calibration: descriptive 95% date-block intervals');fig.tight_layout();fig.savefig(ROOT/'data/research-v2/figures/exp_calibration_uncertainty.png',dpi=180);plt.close(fig)
files=list(TAB.glob('exp_*.csv'))+list((ROOT/'data/research-v2/figures').glob('exp_*.png'))
report=dict(role='Descriptive postprocessing only; no fitting or winner selection',input_sha256=hashlib.sha256((OUT/'predictions.parquet').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),expert_disagreement_rate=float(a.disagree.mean()),n_base_symbol_dates=len(a),maximum_seed_probability_difference=float(pd.DataFrame(seed).maximum_probability_difference.max()),files={str(z.relative_to(ROOT)):hashlib.sha256(z.read_bytes()).hexdigest() for z in files})
(OUT/'supplement.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['n_base_symbol_dates','expert_disagreement_rate','maximum_seed_probability_difference']}))
