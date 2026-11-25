"""Registered exploratory development experiments; never edits sealed releases.

Run pilot, register, then run. Each completed fit is an immutable checkpoint.
Models use current-vintage adjusted OHLCV and may not be interpreted as broker shares.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.models.adaptive import features, execution_labels, probabilities

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/research-v2/experiments'
MARKET = ROOT / 'data/releases/2026-09-28-adaptive/market'
FAMILIES = {
    'momentum': ['return_1', 'return_5', 'return_21', 'return_63', 'return_126', 'return_252', 'momentum_12_1'],
    'trend': ['average_20', 'average_50', 'average_200'],
    'risk': ['vol_21', 'vol_63', 'gap', 'range'],
    'context_volume': ['relative_21', 'relative_63', 'volume_ratio'],
}
REPRESENTATIVES = ['AAPL', 'JPM', 'GLD', 'XLE', 'TLT', 'IWM']
BLOCK_STARTS = ['2024-01-02', '2024-07-01', '2025-01-02', '2025-07-01']


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT / 'src/models/adaptive.py', ROOT / 'src/execution/account.py', ROOT / 'src/backtest/portfolio.py', ROOT / 'src/research/adaptive_portfolio.py']}


def load_data():
    manifest = json.loads((MARKET / 'manifest.json').read_text())
    result = {}
    for symbol, meta in sorted(manifest['data'].items()):
        path = MARKET / 'adjusted' / (symbol + '.parquet')
        if digest(path) != meta['sha256']:
            raise ValueError('Dataset hash mismatch: ' + symbol)
        result[symbol] = pd.read_parquet(path)
    return result


def make_panel(data):
    result = {}
    for symbol, raw in data.items():
        x = features(raw, data['SPY'])
        y, r = execution_labels(raw.Close)
        endpoints = pd.Series(x.index, index=x.index).shift(-6)
        result[symbol] = (x, y, r, endpoints)
    return result


@dataclass(frozen=True)
class Config:
    history: int = 1008
    fraction: float = 1.
    remove: str = ''
    seed: int = 42
    calibration: int = 63
    gate: int = 63
    cadence: int = 63


def partition(x, y, endpoints, decision, cfg):
    """Split by actual endpoint timestamps, never row-distance assumptions."""
    if not x.index.is_unique or not x.index.is_monotonic_increasing or not x.index.equals(y.index) or not x.index.equals(endpoints.index):
        raise ValueError('Ordered aligned data required')
    decision = pd.Timestamp(decision)
    eligible = x.notna().all(axis=1) & y.ge(0) & endpoints.lt(decision)
    ids = np.flatnonzero(eligible.to_numpy())[-cfg.history:]
    if len(ids) < cfg.calibration + cfg.gate + 150:
        raise ValueError('Insufficient eligible observations')
    gate = ids[-cfg.gate:]
    cal_eligible = ids[endpoints.iloc[ids].lt(x.index[gate[0]]).to_numpy()]
    cal = cal_eligible[-cfg.calibration:]
    fit = ids[endpoints.iloc[ids].lt(x.index[cal[0]]).to_numpy()]
    fit = fit[-max(1, int(len(fit) * cfg.fraction)):]
    if len(fit) < 120 or y.iloc[fit].nunique() < 2:
        raise ValueError('Insufficient fit observations/classes')
    assert endpoints.iloc[fit].max() < x.index[cal[0]]
    assert endpoints.iloc[cal].max() < x.index[gate[0]]
    assert endpoints.iloc[gate].max() < decision
    return fit, cal, gate


def temp(p, value):
    return softmax(np.log(np.clip(p, 1e-12, 1.)) / value, axis=1)


def row_loss(y, p):
    return -np.log(np.clip(p[np.arange(len(y)), np.asarray(y, dtype=int)], 1e-12, 1.))


class ControlledModel:
    def __init__(self, cfg=Config()):
        self.cfg = cfg

    def fit(self, panel, symbols, decision):
        self.symbols = list(symbols)
        cfg = self.cfg
        self.columns = [c for c in panel[symbols[0]][0].columns if c not in FAMILIES.get(cfg.remove, [])]
        chunks = {'fit': [], 'cal': [], 'gate': []}
        boundaries = []
        for s in symbols:
            x, y, r, end = panel[s]
            ids = partition(x, y, end, decision, cfg)
            for name, ix in zip(chunks, ids):
                frame = x.iloc[ix][self.columns].copy()
                frame['label'] = y.iloc[ix].to_numpy()
                frame['return'] = r.iloc[ix].to_numpy()
                frame['symbol'] = s
                chunks[name].append(frame)
            f, c, g = ids
            boundaries.append(dict(symbol=s, fit_start=str(x.index[f[0]]), fit_end=str(x.index[f[-1]]), fit_endpoint=str(end.iloc[f].max()), calibration_start=str(x.index[c[0]]), calibration_end=str(x.index[c[-1]]), calibration_endpoint=str(end.iloc[c].max()), gate_start=str(x.index[g[0]]), gate_end=str(x.index[g[-1]]), gate_endpoint=str(end.iloc[g].max()), n_fit=len(f), n_cal=len(c), n_gate=len(g)))
        frames = {n: pd.concat(v).sort_index(kind='stable') for n, v in chunks.items()}
        xf, yf = frames['fit'][self.columns], frames['fit'].label
        self.linear = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500, random_state=cfg.seed))
        self.tree = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=10, learning_rate=.05, min_samples_leaf=40, early_stopping=False, random_state=cfg.seed)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            self.linear.fit(xf, yf)
            self.tree.fit(xf, yf)
            self.prior = (np.bincount(yf, minlength=3) + 1) / (len(yf) + 3)
            self.class_returns = np.array([frames['fit'].loc[yf.eq(c), 'return'].mean() if yf.eq(c).any() else 0. for c in range(3)])
            pc = (probabilities(self.linear, frames['cal'][self.columns]) + probabilities(self.tree, frames['cal'][self.columns])) / 2
            self.temperature = min([.75, 1., 1.5, 2., 3.], key=lambda v: log_loss(frames['cal'].label, temp(pc, v), labels=[0, 1, 2]))
            same_loss = log_loss(frames['cal'].label, temp(pc, self.temperature), labels=[0, 1, 2])
            same_prior = log_loss(frames['cal'].label, np.tile(self.prior, (len(pc), 1)), labels=[0, 1, 2])
            pg = temp((probabilities(self.linear, frames['gate'][self.columns]) + probabilities(self.tree, frames['gate'][self.columns])) / 2, self.temperature)
            sep_loss = log_loss(frames['gate'].label, pg, labels=[0, 1, 2])
            sep_prior = log_loss(frames['gate'].label, np.tile(self.prior, (len(pg), 1)), labels=[0, 1, 2])
            self.gate_same = bool(same_loss + .01 < same_prior)
            self.gate_separate = bool(sep_loss + .01 < sep_prior)
            ptr = temp((probabilities(self.linear, xf) + probabilities(self.tree, xf)) / 2, self.temperature)
        self.metadata = dict(config=asdict(cfg), decision=str(decision), fitting_symbols=list(symbols), boundaries=boundaries, n_fit=len(xf), n_cal=len(pc), n_gate=len(pg), temperature=self.temperature, gate_same=self.gate_same, gate_separate=self.gate_separate, calibration_loss=float(same_loss), calibration_prior_loss=float(same_prior), gate_loss=float(sep_loss), gate_prior_loss=float(sep_prior), train_log_loss=float(log_loss(yf, ptr, labels=[0, 1, 2])), train_accuracy=float(accuracy_score(yf, ptr.argmax(1))), fit_seconds=time.perf_counter()-started, training_hash=hashlib.sha256(pd.util.hash_pandas_object(pd.concat(frames.values()), index=True).values.tobytes()).hexdigest())
        return self

    def predict(self, x):
        if list(x.columns) != self.columns or not np.isfinite(x.to_numpy()).all():
            raise ValueError('Feature schema/finiteness mismatch')
        if (x.index < pd.Timestamp(self.metadata['decision'])).any():
            raise ValueError('Cannot infer before fitting decision')
        with threadpool_limits(limits=1):
            pl, pt = probabilities(self.linear, x), probabilities(self.tree, x)
        pr = np.tile(self.prior, (len(x), 1))
        raw = (pl+pt)/2
        cal = temp(raw, self.temperature)
        return dict(prior=pr, linear=pl, tree=pt, raw=raw, calibrated=cal, gate_same=cal if self.gate_same else pr, gate_separate=cal if self.gate_separate else pr)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.mkdir(parents=True)
        joblib.dump(self, path / 'model.joblib', compress=3)
        write_json(path / 'manifest.json', dict(metadata=self.metadata, sha256=digest(path/'model.joblib')))

    @staticmethod
    def load(path):
        path = Path(path)
        m = json.loads((path/'manifest.json').read_text())
        if digest(path/'model.joblib') != m['sha256']:
            raise ValueError('Model checksum mismatch')
        obj = joblib.load(path/'model.joblib')
        if obj.metadata != m['metadata']:
            raise ValueError('Model metadata mismatch')
        return obj


def pilot():
    if (OUT/'pilot.json').exists():
        raise FileExistsError('Pilot already recorded')
    t = time.perf_counter()
    data = load_data()
    panel = make_panel(data)
    rows = []
    for s in ['AAPL', 'SPY']:
        model = ControlledModel().fit(panel, [s], pd.Timestamp('2024-01-02', tz='UTC'))
        buf = io.BytesIO()
        joblib.dump(model, buf, compress=3)
        rows.append(dict(symbol=s, fit_seconds=model.metadata['fit_seconds'], n_fit=model.metadata['n_fit'], artifact_bytes=buf.tell()))
    write_json(OUT/'pilot.json', dict(registered_role='development runtime only; no predictive outcomes used', recorded_at=now(), symbols=['AAPL','SPY'], fits=4, rows=rows, elapsed_seconds=time.perf_counter()-t, rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, hardware=platform.platform(), python=sys.version, source_hashes=source_hashes()))
    print(json.dumps(rows), flush=True)


def register():
    if (OUT/'design.json').exists():
        raise FileExistsError('Frozen design already exists')
    p = json.loads((OUT/'pilot.json').read_text())
    if p['elapsed_seconds'] > 120:
        raise ValueError('Pilot runtime exceeded protocol')
    data = load_data()
    symbols = sorted(data)
    variants = [dict(name='base', family='R4', **asdict(Config()))]
    variants += [dict(name='without_'+f, family='R5', **asdict(Config(remove=f))) for f in FAMILIES]
    variants += [dict(name='fraction_'+str(v), family='R6', **asdict(Config(fraction=v))) for v in [.5,.75]]
    variants += [dict(name='history_504', family='R6', **asdict(Config(history=504)))]
    variants += [dict(name='cadence_21', family='R6', **asdict(Config(cadence=21)))]
    trials = []
    for v in variants:
        universe = REPRESENTATIVES if v['name'] == 'cadence_21' else symbols
        for s in universe:
            for i, start in enumerate(BLOCK_STARTS):
                dates = data[s].loc[start:].index[:63]
                for j in range(0,63,v['cadence']):
                    trials.append(dict(id=f"{v['name']}-{s}-F{i}-R{j}-S42", variant=v['name'], family=v['family'], symbol=s, fit_symbols=[s], fold=i, start=str(dates[j]), stop=str(dates[min(62,j+v['cadence']-1)]), cfg={k:v[k] for k in asdict(Config())}))
    for seed in [137,2026]:
        for s in REPRESENTATIVES:
            for i,start in enumerate(BLOCK_STARTS):
                dates=data[s].loc[start:].index[:63]
                trials.append(dict(id=f'base-{s}-F{i}-R0-S{seed}',variant='base',family='R4',symbol=s,fit_symbols=[s],fold=i,start=str(dates[0]),stop=str(dates[-1]),cfg=asdict(Config(seed=seed))))
    for s in REPRESENTATIVES:
        for i,start in enumerate(BLOCK_STARTS):
            dates=data[s].loc[start:].index[:63]
            trials.append(dict(id=f'transfer-{s}-F{i}-S42',variant='transfer',family='R7',symbol=s,fit_symbols=[z for z in symbols if z != s],fold=i,start=str(dates[0]),stop=str(dates[-1]),cfg=asdict(Config())))
    contrasts = [dict(id='R3-calibration',family='R3',a='base/calibrated',b='base/raw'),dict(id='R3-gate-separation',family='R3',a='base/gate_separate',b='base/gate_same'),dict(id='R3-gate-off',family='R3',a='base/gate_separate',b='base/calibrated')]
    contrasts += [dict(id='R4-'+v,family='R4',a='base/'+v,b='base/prior') for v in ['linear','tree','raw','calibrated','gate_separate']]
    contrasts += [dict(id='R5-'+f,family='R5',a='without_'+f+'/calibrated',b='base/calibrated') for f in FAMILIES]
    contrasts += [dict(id='R6-'+v,family='R6',a=v+'/calibrated',b='base/calibrated') for v in ['fraction_0.5','fraction_0.75','history_504','cadence_21']]
    contrasts += [dict(id='R7-transfer',family='R7',a='transfer/calibrated',b='base/calibrated')]
    design = dict(version='controlled-development-v1',registered_at=now(),protocol_sha256=digest(ROOT/'data/research-v2/protocol.json'),pilot_sha256=digest(OUT/'pilot.json'),dataset_manifest_sha256=digest(MARKET/'manifest.json'),source_hashes=source_hashes(),outer_rule='First 63 exchange observations in each 2024/2025 half-year, no 2026 evaluation; all history exploratory',gate_design='Shared frozen estimators; temperature on 63-session calibration; reused calibration gate versus distinct purged 63-session gate, improvement=.01; gates select calibrated forecast vs trailing prior',fractions='Most recent registered fraction of eligible fit rows; no random shuffling',transfer='Each target absent from all fitting, scaling, calibration, gating and class-return estimation. SPY contemporaneous covariate retained; ETF economic overlap acknowledged.',selection='No hyperparameter or winner search; fixed logistic and HGB settings; temperature candidate choice uses calibration only; no outer tuning',expected_estimator_fits=len(trials)*2,maximum_estimator_fits=1796,maximum_minutes=60,seeds=[42,137,2026],stochastic_note='HGB early_stopping=False with no row subsampling; seed replay may be identical and is not independent temporal evidence',trials=trials,predictive_contrasts=contrasts,economic_contrasts=[{'id':'E-'+v,'a':v,'b':'equal_weight'} for v in ['linear','tree','raw','calibrated','fixed_mix','same_gate','separate_gate','trend']],inference=dict(replicates=2000,block_lengths=[6,10,21],primary_block=10,seed=20260928,holm_family='All 17 registered predictive contrasts, separately all 8 economic contrasts',resampling='Mean across symbols per session then circular contiguous blocks within each 63-session outer block; outer blocks remain separate'),exclusions={'XGB/LGBM/LSTM/MoE':'Existing legacy bundles lack evidence binding their transformations, training endpoints, calibration roles and execution labels to these new outer partitions. They are not admitted via saved scores; reengineering all candidate contracts exceeds fixed 1800-fit comparison budget. Fixed logistic and nonlinear HGB provide eligible component controls. No claim to superiority over excluded architectures.', 'macro_news_crypto':'No historical as-of vintage/content availability record; excluded rather than filled with synthetic zeros.'})
    if len(trials)*2 >1796:
        raise ValueError('Fit budget exceeded at registration')
    write_json(OUT/'design.json',design)
    for family in ['R3','R4','R5','R6','R7']:
        path=ROOT/'data/research-v2/registry'/f'{family}.json'
        meta=json.loads(path.read_text())
        meta['child_trials']=[t['id'] for t in trials if t['family']==family]
        if family=='R3': meta['child_trials']=[t['id'] for t in trials if t['variant']=='base']
        meta['controlled_design']='data/research-v2/experiments/design.json'
        meta['controlled_design_sha256']=digest(OUT/'design.json')
        meta['predictive_contrasts']=[c['id'] for c in contrasts if c['family']==family]
        write_json(path,meta)
    print(f'Registered {len(trials)} trials, {len(trials)*2} estimator fits, {len(contrasts)} predictive contrasts',flush=True)


def append_event(event):
    with (OUT/'trials.jsonl').open('a') as f:
        f.write(json.dumps(dict(timestamp=now(),**event),default=str)+'\n')
        f.flush()
        os.fsync(f.fileno())


def trial_run(trial,panel,design):
    cfg=Config(**trial['cfg'])
    folder=OUT/'checkpoints'/trial['id']
    cache_input=dict(trial=trial,source_hashes=design['source_hashes'],data_hash=design['dataset_manifest_sha256'],design_hash=digest(OUT/'design.json'))
    cache_hash=hashlib.sha256(json.dumps(cache_input,sort_keys=True).encode()).hexdigest()
    if (folder/'complete.json').exists():
        done=json.loads((folder/'complete.json').read_text())
        if done['cache_hash'] !=cache_hash or digest(folder/'predictions.parquet')!=done['predictions_sha256'] or digest(folder/'model/model.joblib')!=done['model_sha256']:
            raise ValueError('Cache provenance mismatch')
        return done
    if folder.exists():
        raise ValueError('Incomplete checkpoint exists; preserve and investigate before retry')
    folder.mkdir(parents=True)
    append_event(dict(id=trial['id'],status='started',cache_hash=cache_hash))
    t=time.perf_counter()
    try:
        decision=pd.Timestamp(trial['start'])
        model=ControlledModel(cfg).fit(panel,trial['fit_symbols'],decision)
        if trial['variant']=='transfer' and trial['symbol'] in model.symbols:
            raise AssertionError('Held target entered fitting')
        x,y,r,end=panel[trial['symbol']]
        xx=x.loc[trial['start']:trial['stop'],model.columns]
        infer_t=time.perf_counter()
        predicted=model.predict(xx)
        infer_seconds=time.perf_counter()-infer_t
        frames=[]
        for name,p in predicted.items():
            frame=pd.DataFrame(dict(trial=trial['id'],variant=trial['variant'],model=name,symbol=trial['symbol'],date=xx.index,seed=cfg.seed,outer_fold=trial['fold'],p0=p[:,0],p1=p[:,1],p2=p[:,2],label=y.loc[xx.index].to_numpy(),forward_return=r.loc[xx.index].to_numpy(),fit_decision=decision,latest_label_end=max(b['gate_endpoint'] for b in model.metadata['boundaries']),gate_same=model.gate_same,gate_separate=model.gate_separate,expected_return=p@model.class_returns,model_age=np.arange(len(xx)),temperature=model.temperature,train_log_loss=model.metadata['train_log_loss']))
            frame['log_loss']=row_loss(frame.label,p)
            frames.append(frame)
        pred=pd.concat(frames,ignore_index=True)
        pred.to_parquet(folder/'predictions.parquet',index=False)
        model.save(folder/'model')
        done=dict(trial=trial,cache_hash=cache_hash,status='complete',model_metadata=model.metadata,n_decisions=len(xx),prediction_rows=len(pred),estimator_fits=2,seconds=time.perf_counter()-t,inference_seconds=infer_seconds,model_bytes=(folder/'model/model.joblib').stat().st_size,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,predictions_sha256=digest(folder/'predictions.parquet'),model_sha256=digest(folder/'model/model.joblib'))
        write_json(folder/'complete.json',done)
        append_event(dict(id=trial['id'],status='complete',seconds=done['seconds'],estimator_fits=2))
        return done
    except Exception as e:
        append_event(dict(id=trial['id'],status='failed',error=repr(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-t))
        raise


def run():
    design=json.loads((OUT/'design.json').read_text())
    if source_hashes()!=design['source_hashes']:
        raise ValueError('Source changed after registration')
    t=time.perf_counter()
    data=load_data()
    ft=time.perf_counter()
    panel=make_panel(data)
    feature_seconds=time.perf_counter()-ft
    completed=[]
    for i,trial in enumerate(design['trials']):
        if time.perf_counter()-t>design['maximum_minutes']*60:
            append_event(dict(id=trial['id'],status='stopped',reason='elapsed budget'))
            break
        completed.append(trial_run(trial,panel,design))
        if (i+1)%20==0: print(f'Completed {i+1}/{len(design["trials"])} trials in {time.perf_counter()-t:.1f}s',flush=True)
    write_json(OUT/'run.json',dict(started_source_hashes=source_hashes(),recorded_at=now(),elapsed_seconds=time.perf_counter()-t,feature_generation_seconds=feature_seconds,complete_trials=len(completed),expected_trials=len(design['trials']),estimator_fits=sum(c['estimator_fits'] for c in completed),hardware=platform.platform(),python=sys.version,cpu_count=os.cpu_count(),thread_limit=1,temperature_candidates=[.75,1,1.5,2,3]))
    summarize(design,data,completed)


def bootstrap(values,folds,block,reps=2000,seed=20260928):
    values=np.asarray(values,dtype=float)
    folds=np.asarray(folds)
    if len(values)<2 or not np.isfinite(values).all(): raise ValueError('Invalid bootstrap data')
    rng=np.random.default_rng(seed)
    sampled=np.zeros(reps)
    for f in np.unique(folds):
        a=values[folds==f]
        n=len(a)
        starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
        ids=(starts[:,:,None]+np.arange(block))%n
        means=a[ids.reshape(reps,-1)[:,:n]].mean(1)
        sampled+=means*n/len(values)
    effect=float(values.mean())
    se=float(sampled.std(ddof=1))
    centered=sampled-effect
    p=float((1+np.sum(np.abs(centered)>=abs(effect)))/(reps+1))
    return dict(effect=effect,ci_low=float(np.quantile(sampled,.025)),ci_high=float(np.quantile(sampled,.975)),standard_error=se,p_value=p,minimum_detectable_effect_80pct=2.802*se,n_dates=len(values),block=block)


def holm(p):
    p=np.asarray(p,float)
    order=np.argsort(p)
    out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out


def summarize(design,data,done):
    predictions=pd.concat([pd.read_parquet(OUT/'checkpoints'/r['trial']['id']/'predictions.parquet') for r in done],ignore_index=True)
    predictions.to_parquet(OUT/'predictions.parquet',index=False)
    fits=[]
    for d in done:
        t=d['trial'];m=d['model_metadata']
        fits.append(dict(trial=t['id'],variant=t['variant'],symbol=t['symbol'],fold=t['fold'],seed=t['cfg']['seed'],n_fit=m['n_fit'],n_cal=m['n_cal'],n_gate=m['n_gate'],temperature=m['temperature'],gate_same=m['gate_same'],gate_separate=m['gate_separate'],train_log_loss=m['train_log_loss'],train_accuracy=m['train_accuracy'],calibration_loss=m['calibration_loss'],gate_loss=m['gate_loss'],fit_seconds=m['fit_seconds'],inference_seconds=d['inference_seconds'],total_seconds=d['seconds'],artifact_bytes=d['model_bytes'],rss_peak_bytes=d['rss_peak_bytes']))
    fits=pd.DataFrame(fits)
    fits.to_parquet(OUT/'fits.parquet',index=False)
    selected=predictions[predictions.seed.eq(42)]
    metrics=[]
    for (v,m),g in selected.groupby(['variant','model']):
        p=g[['p0','p1','p2']].to_numpy(); y=g.label.to_numpy()
        metrics.append(dict(variant=v,model=m,n_rows=len(g),n_dates=g.date.nunique(),n_symbols=g.symbol.nunique(),log_loss=float(g.log_loss.mean()),accuracy=float(accuracy_score(y,p.argmax(1))),balanced_accuracy=float(balanced_accuracy_score(y,p.argmax(1))),brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),gate_same_rate=float(g.gate_same.mean()),gate_separate_rate=float(g.gate_separate.mean())))
    metrics=pd.DataFrame(metrics)
    metrics.to_csv(ROOT/'tables/exp_predictive.csv',index=False)
    contrasts=[]
    for c in design['predictive_contrasts']:
        va,ma=c['a'].split('/');vb,mb=c['b'].split('/')
        a=selected[selected.variant.eq(va)&selected.model.eq(ma)]
        b=selected[selected.variant.eq(vb)&selected.model.eq(mb)]
        pair=a.merge(b,on=['date','symbol','outer_fold'],suffixes=('_a','_b'),validate='one_to_one')
        pair['delta']=pair.log_loss_a-pair.log_loss_b
        daily=pair.groupby(['date','outer_fold']).delta.mean().reset_index()
        daily.to_csv(OUT/(c['id']+'-daily.csv'),index=False)
        for block in design['inference']['block_lengths']:
            contrasts.append(dict(**c,**bootstrap(daily.delta,daily.outer_fold,block),n_rows=len(pair),n_symbols=pair.symbol.nunique()))
    contrasts=pd.DataFrame(contrasts)
    contrasts['p_holm']=np.nan
    for block in design['inference']['block_lengths']:
        mask=contrasts.block.eq(block)
        if mask.sum()!=len(design['predictive_contrasts']): raise AssertionError('Incomplete inference family')
        contrasts.loc[mask,'p_holm']=holm(contrasts.loc[mask,'p_value'])
    contrasts.to_csv(ROOT/'tables/exp_contrasts.csv',index=False)
    seed_metrics=[]
    for (symbol,fold,seed),g in predictions[predictions.variant.eq('base')&predictions.model.eq('calibrated')&predictions.symbol.isin(REPRESENTATIVES)].groupby(['symbol','outer_fold','seed']):
        seed_metrics.append(dict(symbol=symbol,fold=fold,seed=seed,log_loss=g.log_loss.mean(),n_rows=len(g)))
    pd.DataFrame(seed_metrics).to_csv(ROOT/'tables/exp_seeds.csv',index=False)
    learning=selected[selected.model.eq('calibrated')].groupby(['variant','outer_fold']).agg(n_rows=('label','size'),n_symbols=('symbol','nunique'),train_log_loss=('train_log_loss','mean'),validation_log_loss=('log_loss','mean'),model_age_mean=('model_age','mean')).reset_index()
    learning.to_csv(ROOT/'tables/exp_learning.csv',index=False)
    per_symbol=selected[selected.model.eq('calibrated')].groupby(['variant','symbol','outer_fold']).agg(n_rows=('label','size'),log_loss=('log_loss','mean')).reset_index()
    per_symbol.to_csv(ROOT/'tables/exp_transfer.csv',index=False)
    fits.to_csv(ROOT/'tables/exp_runtime.csv',index=False)
    age=selected[selected.variant.isin(['base','cadence_21'])&selected.model.eq('calibrated')&selected.symbol.isin(REPRESENTATIVES)].copy()
    age['age_bin']=pd.cut(age.model_age,[-1,20,41,62],labels=['0-20','21-41','42-62'])
    age.groupby(['variant','age_bin'],observed=True).agg(log_loss=('log_loss','mean'),n_rows=('label','size')).reset_index().to_csv(ROOT/'tables/exp_model_age.csv',index=False)
    economics=portfolio_experiments(design,data,selected)
    summary=dict(recorded_at=now(),design_sha256=digest(OUT/'design.json'),status='complete' if len(done)==len(design['trials']) else 'partial',complete_trials=len(done),registered_trials=len(design['trials']),estimator_fits=len(done)*2,pilot_estimator_fits=4,prediction_rows=len(predictions),unique_decision_samples=int(selected[selected.variant.eq('base')&selected.model.eq('prior')][['symbol','date']].drop_duplicates().shape[0]),n_outer_dates=int(selected[selected.variant.eq('base')].date.nunique()),outer_date_min=str(selected.date.min()),outer_date_max=str(selected.date.max()),symbols=sorted(data),predictive_contrast_count=len(design['predictive_contrasts']),economic_contrast_count=len(design['economic_contrasts']),base_metrics=metrics[metrics.variant.eq('base')].to_dict('records'),primary_contrasts=contrasts[contrasts.block.eq(10)].to_dict('records'),runtime=dict(fit_p50=float(fits.fit_seconds.quantile(.5)),fit_p95=float(fits.fit_seconds.quantile(.95)),fit_p99=float(fits.fit_seconds.quantile(.99)),total_fit_seconds=float(fits.fit_seconds.sum()),artifact_bytes=int(fits.artifact_bytes.sum()),max_rss_bytes=int(fits.rss_peak_bytes.max()),n_trials=len(fits)),economics=economics,limitations=['All dates are consumed development history and exploratory.','Quarter-length blocks are separated; portfolio capital resets to 10000 for each block.','Pooled holdout excludes target labels but economic exposures overlap across ETFs and stocks.','Current-vintage adjusted OHLCV omits historical vintage and delisted-universe coverage.','Seed repetitions are model-randomness checks, not independent market observations.','No 2026 pilot outcomes or audit outcomes used.'])
    write_json(OUT/'summary.json',summary)
    plot_results(contrasts,learning,fits,metrics,selected)
    for family in ['R3','R4','R5','R6','R7']:
        path=ROOT/'data/research-v2/registry'/f'{family}.json'
        meta=json.loads(path.read_text());meta['status']='completed_exploratory' if summary['status']=='complete' else 'partial';meta['results']='data/research-v2/experiments/summary.json';meta['completed_at']=now();write_json(path,meta)
    print(json.dumps({k:summary[k] for k in ['status','complete_trials','estimator_fits','prediction_rows','n_outer_dates']}),flush=True)


def portfolio_experiments(design,data,predictions):
    from src.backtest.portfolio import replay_portfolio
    from src.research.adaptive_portfolio import constrain_weights,limits
    base=predictions[predictions.variant.eq('base')]
    symbols=sorted(data)
    full_prices=pd.DataFrame({s:d.Close for s,d in data.items()})
    returns=full_prices.pct_change(fill_method=None)
    all_metrics=[];all_returns=[]
    for fold,chunk in base.groupby('outer_fold'):
        index=pd.DatetimeIndex(sorted(chunk.date.unique()))
        trend=pd.DataFrame({s:((features(data[s],data['SPY']).average_200>0)&(features(data[s],data['SPY']).momentum_12_1>0)).astype(float) for s in symbols}).loc[index]
        raws={}
        for m in ['linear','tree','raw','calibrated']:
            panel=chunk[chunk.model.eq(m)].pivot(index='date',columns='symbol',values='expected_return').reindex(index=index,columns=symbols)
            raws[m]=panel.gt(.003).astype(float)
        cal=chunk[chunk.model.eq('calibrated')]
        same=cal.pivot(index='date',columns='symbol',values='gate_same').reindex(index=index,columns=symbols).astype(float)
        separate=cal.pivot(index='date',columns='symbol',values='gate_separate').reindex(index=index,columns=symbols).astype(float)
        raws.update(fixed_mix=.5*trend+.5*raws['calibrated'],same_gate=(1-.5*same)*trend+.5*same*raws['calibrated'],separate_gate=(1-.5*separate)*trend+.5*separate*raws['calibrated'],trend=trend,equal_weight=trend*0+1,cash=trend*0)
        # Actual last exchange observation in each ISO week; holiday Fridays handled.
        calendar=pd.DatetimeIndex(data['SPY'].index)
        week=calendar.tz_localize(None).to_period('W-FRI')
        last_by_week=pd.Series(calendar,index=week).groupby(level=0).max()
        rebalance=set(last_by_week.to_list()) & set(index)
        rebalance.add(index[0])
        for name,intent in raws.items():
            targets=pd.DataFrame(0.,index=index,columns=symbols)
            w=pd.Series(0.,index=symbols)
            for date in index:
                if date in rebalance: w=constrain_weights(intent.loc[date]*(.8/len(symbols)),returns.loc[:date])
                targets.loc[date]=w
            ledger,account=replay_portfolio(full_prices.loc[index],targets,10000.,limits(),rebalance_dates=rebalance)
            folder=OUT/'portfolios'/f'fold-{fold}'
            folder.mkdir(parents=True,exist_ok=True)
            ledger.to_parquet(folder/(name+'-account.parquet'))
            targets.to_parquet(folder/(name+'-targets.parquet'))
            write_json(folder/(name+'-events.json'),account.events)
            equity=ledger.equity
            rr=equity.pct_change(fill_method=None).fillna(equity.iloc[0]/10000-1)
            dd=equity/equity.cummax()-1
            quantities=ledger[[c for c in ledger if c.startswith('quantity_')]].copy();quantities.columns=[c.removeprefix('quantity_') for c in quantities]
            exposure=(quantities*full_prices.loc[index]).sum(1)/equity
            all_metrics.append(dict(outer_fold=int(fold),policy=name,n_sessions=len(index),initial_capital=10000.,final_equity=float(equity.iloc[-1]),net_return=float(equity.iloc[-1]/10000-1),max_drawdown=float(dd.min()),annualized_sharpe=float(rr.mean()/rr.std()*np.sqrt(252)) if rr.std()>0 else None,mean_exposure=float(exposure.mean()),fees=float(account.fees),realized_pnl=float(account.realized),terminal_open_positions=sum(q>0 for q in account.positions.values()),gate_same_rate=float(same.to_numpy().mean()),gate_separate_rate=float(separate.to_numpy().mean())))
            all_returns.extend(dict(date=str(date),outer_fold=int(fold),policy=name,daily_return=float(val)) for date,val in rr.items())
    metrics=pd.DataFrame(all_metrics);daily=pd.DataFrame(all_returns)
    metrics.to_csv(ROOT/'tables/exp_portfolios.csv',index=False);daily.to_csv(OUT/'portfolio_daily.csv',index=False)
    contrasts=[]
    for c in design['economic_contrasts']:
        a=daily[daily.policy.eq(c['a'])];b=daily[daily.policy.eq(c['b'])]
        pair=a.merge(b,on=['date','outer_fold'],suffixes=('_a','_b'),validate='one_to_one')
        for block in design['inference']['block_lengths']:
            contrasts.append(dict(**c,**bootstrap(pair.daily_return_a-pair.daily_return_b,pair.outer_fold,block)))
    contrasts=pd.DataFrame(contrasts);contrasts['p_holm']=np.nan
    for block in design['inference']['block_lengths']:
        ix=contrasts.block.eq(block);contrasts.loc[ix,'p_holm']=holm(contrasts.loc[ix,'p_value'])
    contrasts.to_csv(ROOT/'tables/exp_economic_contrasts.csv',index=False)
    return dict(capital_per_fold=10000,folds=4,policies=10,metrics=metrics.replace({np.nan:None}).to_dict('records'),primary_contrasts=contrasts[contrasts.block.eq(10)].to_dict('records'))


def plot_results(contrasts,learning,fits,metrics,pred):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    c=contrasts[contrasts.block.eq(10)].iloc[::-1]
    fig,ax=plt.subplots(figsize=(9,6))
    ax.errorbar(c.effect,np.arange(len(c)),xerr=[c.effect-c.ci_low,c.ci_high-c.effect],fmt='o',capsize=3,color='#167d8d')
    ax.set_yticks(np.arange(len(c)),c.id);ax.axvline(0,color='#888',ls='--');ax.set_xlabel('Daily paired log-loss difference (candidate minus reference)');ax.set_title('Controlled contrasts: 95% date-block intervals, block=10')
    fig.tight_layout();fig.savefig(ROOT/'figures/exp_contrasts.png',dpi=180);plt.close(fig)
    names=['fraction_0.5','fraction_0.75','base','history_504']
    z=learning[learning.variant.isin(names)].groupby('variant')[['train_log_loss','validation_log_loss']].mean().reindex(names)
    fig,ax=plt.subplots(figsize=(7,4));z.plot.bar(ax=ax,color=['#167d8d','#d78348']);ax.set_ylabel('Log loss');ax.set_xlabel('Training design');ax.set_title('Training fit versus outer validation');ax.tick_params(axis='x',rotation=15);fig.tight_layout();fig.savefig(ROOT/'figures/exp_learning.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));fits.boxplot(column='fit_seconds',by='variant',ax=ax,showfliers=False,rot=30);ax.set_ylabel('Seconds / two-estimator fit');ax.set_title('Measured single-thread fitting latency');fig.suptitle('');fig.tight_layout();fig.savefig(ROOT/'figures/exp_runtime.png',dpi=180);plt.close(fig)
    # Reliability bins retain exact n and empirical frequency; classwise calibration.
    reliability=[]
    for model in ['raw','calibrated','gate_separate']:
        g=pred[pred.variant.eq('base')&pred.model.eq(model)]
        for cl in [0,1,2]:
            p=g['p'+str(cl)];bins=np.minimum((p*10).astype(int),9)
            for b in range(10):
                ix=bins.eq(b)
                if ix.any():reliability.append(dict(model=model,class_id=cl,bin=b,n=int(ix.sum()),mean_probability=float(p[ix].mean()),observed_frequency=float(g.loc[ix,'label'].eq(cl).mean())))
    rel=pd.DataFrame(reliability);rel.to_csv(ROOT/'tables/exp_reliability.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(10,3.5),sharex=True,sharey=True)
    for cl,ax in enumerate(axs):
        for m in ['raw','calibrated','gate_separate']:
            g=rel[rel.class_id.eq(cl)&rel.model.eq(m)];ax.plot(g.mean_probability,g.observed_frequency,'o-',label=m,ms=3)
        ax.plot([0,1],[0,1],'--',color='#999');ax.set_title(['DOWN','FLAT','UP'][cl]);ax.set_xlabel('Forecast probability')
    axs[0].set_ylabel('Observed class frequency');axs[-1].legend(fontsize=7);fig.suptitle('Out-of-time reliability; support is retained in source table');fig.tight_layout();fig.savefig(ROOT/'figures/exp_reliability.png',dpi=180);plt.close(fig)


def reproduce():
    """Actually refit three prespecified checkpoints and compare probabilities."""
    design=json.loads((OUT/'design.json').read_text())
    if source_hashes()!=design['source_hashes']:
        raise ValueError('Reproduction source differs from frozen registration')
    panel=make_panel(load_data())
    ids=['base-AAPL-F0-R0-S42','without_momentum-AAPL-F0-R0-S42','transfer-AAPL-F0-S42']
    rows=[]
    for trial in design['trials']:
        if trial['id'] not in ids: continue
        checkpoint=OUT/'checkpoints'/trial['id']
        done=json.loads((checkpoint/'complete.json').read_text())
        model=ControlledModel(Config(**trial['cfg'])).fit(panel,trial['fit_symbols'],pd.Timestamp(trial['start']))
        x=panel[trial['symbol']][0].loc[trial['start']:trial['stop'],model.columns]
        actual=model.predict(x)
        saved=pd.read_parquet(checkpoint/'predictions.parquet')
        differences={}
        for name,p in actual.items():
            expected=saved[saved.model.eq(name)].sort_values('date')[['p0','p1','p2']].to_numpy()
            delta=float(np.max(np.abs(expected-p)))
            if not np.allclose(expected,p,rtol=1e-10,atol=1e-12):
                raise AssertionError('Reproduction mismatch '+trial['id']+'/'+name)
            differences[name]=delta
        for name in ['boundaries','temperature','gate_same','gate_separate','training_hash']:
            if model.metadata[name]!=done['model_metadata'][name]:
                raise AssertionError('Reproduced metadata mismatch '+name)
        rows.append(dict(trial=trial['id'],probability_rows=sum(len(p) for p in actual.values()),maximum_absolute_error=max(differences.values()),per_model_errors=differences,n_fitting_symbols=len(model.symbols),temperature=model.temperature,gate_same=model.gate_same,gate_separate=model.gate_separate))
    report=dict(recorded_at=now(),status='passed',runtime=sys.executable,python=sys.version,design_sha256=digest(OUT/'design.json'),source_hashes=source_hashes(),rtol=1e-10,atol=1e-12,actual_refits=len(rows)*2,trials=rows)
    destination=Path(os.environ.get('EXPERIMENT_REPRODUCTION_OUTPUT',str(OUT/'reproduction.json')))
    if destination.exists():raise FileExistsError(destination)
    write_json(destination,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['pilot','register','run','summarize','reproduce']);a=parser.parse_args()
    if a.command=='pilot':pilot()
    elif a.command=='register':register()
    elif a.command=='run':run()
    elif a.command=='reproduce':reproduce()
    else:
        d=json.loads((OUT/'design.json').read_text());done=[json.loads((OUT/'checkpoints'/t['id']/'complete.json').read_text()) for t in d['trials'] if (OUT/'checkpoints'/t['id']/'complete.json').exists()];summarize(d,load_data(),done)
