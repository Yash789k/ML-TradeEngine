"""Registered delayed-feedback controllers over immutable OOF forecasts.

Run ``register`` before ``run``. No downloads, base-estimator fits, or orders.
The experiment deliberately reuses consumed history and is exploratory only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import signal
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'data/releases/2026-09-28-adaptive'
OUT = ROOT / 'data/research-controller-v1/20261030-screen-v1'
EXPERTS = ('prior', 'linear', 'tree')
CONTROLLERS = ('hedge', 'stack')
BASELINES = ('fixed_mixture', 'prior', 'uniform3')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(obj, handle, indent=2, allow_nan=False, default=str)
        handle.write('\n')


def checked(path, expected):
    if not Path(path).is_file() or digest(path) != expected:
        raise ValueError(f'Checksum mismatch: {path}')


def valid_probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim < 2 or p.shape[-1] != 3 or not np.isfinite(p).all():
        raise ValueError('Invalid probability shape/finiteness')
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(-1), 1, atol=1e-10, rtol=0):
        raise ValueError('Invalid probability bounds/normalization')
    return p


def source_hashes():
    paths = sorted((ROOT/'research_controller_v1').glob('*.py'))
    paths += sorted((ROOT/'tests/controller_v1').glob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def inventory():
    """Verify manifests and their already-sealed per-file hashes before reading outcomes."""
    manifest = json.loads((RELEASE/'manifest.json').read_text())
    checked(RELEASE/'market/manifest.json', manifest['artifacts']['market/manifest.json'])
    market = json.loads((RELEASE/'market/manifest.json').read_text())
    symbols = sorted(manifest['symbols'])
    if len(symbols) != 22 or set(symbols) != set(market['data']):
        raise ValueError('Registered 22-symbol universe unavailable')
    result = {str((RELEASE/'manifest.json').relative_to(ROOT)): digest(RELEASE/'manifest.json')}
    result[str((RELEASE/'market/manifest.json').relative_to(ROOT))] = digest(RELEASE/'market/manifest.json')
    for symbol in symbols:
        names = [(f'experiment/{symbol}_predictions.parquet', manifest['artifacts'][f'experiment/{symbol}_predictions.parquet']),
                 (f'market/adjusted/{symbol}.parquet', market['data'][symbol]['sha256']),
                 (f'market/raw/{symbol}.parquet', market['data'][symbol]['raw_sha256'])]
        for name, expected in names:
            path = RELEASE/name
            checked(path, expected)
            result[str(path.relative_to(ROOT))] = expected
    return symbols, result


def register():
    symbols, inputs = inventory()
    write_new(OUT/'protocol.json', dict(
        id='CONTROLLER-OOF-EXPLORATORY-20261030-V1', registered_at=now(),
        role='exploratory feasibility screen; all 2024-2025 history previously inspected',
        inputs=inputs, sources=source_hashes(), symbols=symbols,
        decision_label='next-close execution: adjusted Close[t+6]/Close[t+1]-1; classes < -.01 DOWN, > .01 UP, else FLAT',
        eligibility='feedback label endpoint strictly before decision; full date group before an update; no duplicate feedback',
        periods=dict(warmup='2024 only, OOF base forecasts', evaluation=['2025-01-01..2025-06-30','2025-07-01..2025-12-31']),
        candidates=dict(fixed_mixture='0.5 linear + 0.5 tree', prior='sealed rolling training-frequency prior', uniform3='one-third each prior/linear/tree; isolates adaptation from adding prior',
                        hedge='eta=.1, initially uniform prior/linear/tree weights; multiply by exp(-eta * mean date loss) only after t+6 endpoint is strictly past',
                        stack='StandardScaler + L2 LogisticRegression(C=.1,max_iter=1000,random_state=20261030); nine clipped log probabilities of prior/linear/tree'),
        stack_refits=['first 2025 decision, all mature 2024 OOF rows', 'first July 2025 decision, all mature 2024 and first-half 2025 OOF rows'],
        selection='No tuning, temperature calibration, candidate reselection, or promotion; fixed evaluation of both controllers',
        primary_endpoint='date-average multiclass log loss; lower is better',
        secondary_endpoints=['accuracy', 'Brier score', 'half-year descriptive metrics'],
        contrasts=[f'{c}-vs-{b}' for c in CONTROLLERS for b in BASELINES],
        inference=dict(replicates=2000,seed=20261030,primary_block=10,sensitivity_blocks=[6,21],
                       method='paired date averages; circular contiguous blocks within each half-year; preserve all symbols in each date',
                       multiplicity='Holm across six controller-versus-baseline contrasts separately at each registered block length',
                       limits='Exploratory conditional-on-these-periods intervals; no untouched holdout or prospective claim'),
        budget=dict(base_fits=0,meta_fits_maximum=2,soft_seconds=120,hard_seconds=600,downloads=0),
        external_orders=False, accounts_modified=False,
        limitations=['current-vintage adjusted prices, fixed universe, economically overlapping assets',
                     'base forecasts reused from 2024-2025 rolling experiment, not newly generated predictions',
                     'Hedge sees delayed feedback during evaluation; stack refits once in July using only then-mature observations',
                     'no portfolio/cost/execution comparison; predictive screening does not establish alpha'],
    ))
    return {'status':'registered','path':str(OUT/'protocol.json')}


def load_panel(protocol):
    for name, expected in protocol['inputs'].items():
        checked(ROOT/name, expected)
    frames = []
    for symbol in protocol['symbols']:
        pred = pd.read_parquet(RELEASE/f'experiment/{symbol}_predictions.parquet')
        market = pd.read_parquet(RELEASE/f'market/adjusted/{symbol}.parquet')
        raw = pd.read_parquet(RELEASE/f'market/raw/{symbol}.parquet')
        if not market.index.equals(raw.index) or not market.index.is_unique or not market.index.is_monotonic_increasing:
            raise ValueError('Raw/adjusted market calendar invalid')
        if not pred.index.is_unique or not pred.index.is_monotonic_increasing:
            raise ValueError('Invalid prediction dates')
        pred = pred.loc['2024-01-01':'2025-12-31'].copy()
        endpoint = pd.Series(market.index, index=market.index).shift(-6).reindex(pred.index)
        returns = (market.Close.shift(-6)/market.Close.shift(-1)-1).reindex(pred.index)
        labels = np.where(returns > .01, 2, np.where(returns < -.01, 0, 1))
        if endpoint.isna().any() or returns.isna().any() or not np.array_equal(labels,pred.label):
            raise ValueError('Saved labels differ from execution-aligned market endpoints')
        if not np.allclose(returns,pred.forward_return,rtol=1e-12,atol=1e-14):
            raise ValueError('Saved returns differ from adjusted market')
        fit = pd.to_datetime(pred.fit_decision,utc=True)
        latest = pd.to_datetime(pred.latest_label_end,utc=True)
        if not ((latest < fit) & (fit <= pred.index)).all():
            raise ValueError('Base forecasts lack causal fitting endpoints')
        pred['endpoint'] = endpoint
        pred['symbol'] = symbol
        frames.append(pred.reset_index())
    panel = pd.concat(frames, ignore_index=True).sort_values(['date','symbol']).reset_index(drop=True)
    counts = panel.groupby('date').symbol.nunique()
    if not counts.eq(len(protocol['symbols'])).all() or panel.duplicated(['date','symbol']).any():
        raise ValueError('Incomplete or duplicate date groups')
    if not panel.groupby('date').endpoint.nunique().eq(1).all():
        raise ValueError('Inconsistent cross-sectional label endpoints')
    expert_probabilities(panel)
    return panel


def expert_probabilities(frame):
    return valid_probabilities(np.stack([frame[[f'{e}_{c}' for c in range(3)]].to_numpy() for e in EXPERTS],axis=1))


class DelayedHedge:
    def __init__(self, eta=.1):
        self.eta = eta
        self.log_weight = np.zeros(3)
        self.consumed = set()

    @property
    def weights(self):
        return softmax(self.log_weight)

    def feedback(self, key, endpoint, decision, probabilities, labels):
        if key in self.consumed:
            return False
        if pd.Timestamp(endpoint) >= pd.Timestamp(decision):
            return False
        p = valid_probabilities(probabilities)
        y = np.asarray(labels,dtype=int)
        if p.ndim != 3 or p.shape[1] != 3 or len(y) != len(p) or not np.isin(y,[0,1,2]).all():
            raise ValueError('Invalid feedback')
        losses = -np.log(np.clip(p[np.arange(len(y))[:,None],np.arange(3)[None,:],y[:,None]],1e-12,1))
        # One update per date: symbols are dependent observations, not extra rounds.
        self.log_weight -= self.eta * losses.mean(axis=0)
        self.log_weight -= self.log_weight.max()
        self.consumed.add(key)
        return True


def stack_features(p):
    return np.log(np.clip(valid_probabilities(p),1e-12,1)).reshape(len(p),-1)


def forecast(panel, use_stack=True):
    groups = [(d,g.copy()) for d,g in panel.groupby('date',sort=True)]
    hedge = DelayedHedge()
    feedback_cursor = 0
    model = None
    fit_records, outputs, weights = [], [], []
    for date, group in groups:
        while feedback_cursor < len(groups):
            old_date, old = groups[feedback_cursor]
            endpoint = old.endpoint.iloc[0]
            if endpoint >= date:
                break
            hedge.feedback(str(old_date),endpoint,date,expert_probabilities(old),old.label)
            feedback_cursor += 1
        if date.year != 2025:
            continue
        block = '2025-H1' if date.month <= 6 else '2025-H2'
        if use_stack and (model is None or (date.month >= 7 and len(fit_records)==1)):
            training = panel[(panel.date < date)&(panel.endpoint < date)]
            if len(fit_records)==0:
                training = training[training.date.dt.year==2024]
            else:
                training = training[training.date < pd.Timestamp('2025-07-01',tz='UTC')]
            if training.empty or training.label.nunique() < 2:
                raise ValueError('Insufficient mature stack training')
            model = make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,random_state=20261030))
            with threadpool_limits(limits=1):
                model.fit(stack_features(expert_probabilities(training)),training.label)
            if model[-1].n_iter_.max() >= 1000:
                raise ValueError('Stack did not converge within registered budget')
            fit_records.append(dict(decision=str(date),n_rows=len(training),n_dates=training.date.nunique(),
                                    first_training_date=str(training.date.min()),last_training_date=str(training.date.max()),
                                    latest_endpoint=str(training.endpoint.max()),classes=model.classes_.tolist(),
                                    scaler_mean=model[0].mean_.tolist(),coefficients=model[-1].coef_.tolist(),intercept=model[-1].intercept_.tolist(),
                                    training_hash=hashlib.sha256(pd.util.hash_pandas_object(training,index=False).values.tobytes()).hexdigest()))
        p = expert_probabilities(group)
        result = {'prior':p[:,0], 'fixed_mixture':.5*p[:,1]+.5*p[:,2], 'uniform3':p.mean(axis=1), 'hedge':np.einsum('e,nek->nk',hedge.weights,p)}
        if use_stack:
            pred = np.full((len(group),3),1e-12)
            pred[:,model.classes_.astype(int)] = model.predict_proba(stack_features(p))
            result['stack'] = pred/pred.sum(axis=1,keepdims=True)
        weights.append(dict(date=str(date),block=block,n_feedback_dates=len(hedge.consumed),
                            **dict(zip(EXPERTS,hedge.weights.tolist()))))
        for name, probs in result.items():
            probs=valid_probabilities(probs)
            rows=group[['date','symbol','label','endpoint']].copy()
            rows['model']=name;rows['block']=block
            rows[['p0','p1','p2']]=probs
            labels=rows.label.to_numpy(dtype=int)
            rows['log_loss']=-np.log(np.clip(probs[np.arange(len(rows)),labels],1e-12,1))
            rows['correct']=(probs.argmax(axis=1)==labels).astype(float)
            rows['brier']=((probs-np.eye(3)[labels])**2).sum(axis=1)
            outputs.append(rows)
    return pd.concat(outputs,ignore_index=True),pd.DataFrame(weights),fit_records


def holm(values):
    p=np.asarray(values);order=np.argsort(p);out=np.empty(len(p));prev=0.
    for rank,i in enumerate(order):
        prev=max(prev,min(1.,(len(p)-rank)*p[i]));out[i]=prev
    return out


def bootstrap(values, blocks, length, replicates=2000, seed=20261030):
    values=np.asarray(values,dtype=float);blocks=np.asarray(blocks)
    rng=np.random.default_rng(seed);draws=np.zeros(replicates)
    for block in np.unique(blocks):
        x=values[blocks==block];n=len(x)
        starts=rng.integers(0,n,size=(replicates,int(np.ceil(n/length))))
        ids=(starts[:,:,None]+np.arange(length))%n
        samples=x[ids.reshape(replicates,-1)[:,:n]]
        draws+=samples.sum(axis=1)/len(values)
    effect=float(values.mean());lo,hi=np.quantile(draws,[.025,.975])
    p=(1+int((np.abs(draws-effect)>=abs(effect)).sum()))/(replicates+1)
    return dict(effect=effect,ci_low=float(lo),ci_high=float(hi),p_value=p,n_dates=len(values),block_length=length)


def run():
    start=time.perf_counter()
    protocol=json.loads((OUT/'protocol.json').read_text())
    if protocol['sources']!=source_hashes():
        raise ValueError('Registered source changed; create a new version')
    if (OUT/'run.json').exists():
        raise FileExistsError('Run already attempted; preserve it and use a new version')
    write_new(OUT/'run.json',dict(status='started',started_at=now(),protocol_sha256=digest(OUT/'protocol.json')))
    def timeout(*_):
        raise TimeoutError('Registered 600-second hard runtime budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        panel=load_panel(protocol)
        forecasts,weights,fits=forecast(panel)
        if len(fits)!=2:
            raise ValueError('Expected exactly two registered meta fits')
        daily=forecasts.groupby(['date','block','model'])[['log_loss','correct','brier']].mean().reset_index()
        metrics=[]
        for (block,name),g in daily.groupby(['block','model']):
            metrics.append(dict(period=block,model=name,n_dates=len(g),n_symbol_dates=len(g)*22,log_loss=g.log_loss.mean(),accuracy=g.correct.mean(),brier=g.brier.mean()))
        for name,g in daily.groupby('model'):
            metrics.append(dict(period='2025-all',model=name,n_dates=len(g),n_symbol_dates=len(g)*22,log_loss=g.log_loss.mean(),accuracy=g.correct.mean(),brier=g.brier.mean()))
        wide=daily.pivot(index=['date','block'],columns='model',values='log_loss')
        contrasts=[]
        for length in [10,6,21]:
            family=[]
            for controller in CONTROLLERS:
                for baseline in BASELINES:
                    values=wide[controller]-wide[baseline]
                    family.append(dict(id=f'{controller}-vs-{baseline}',controller=controller,baseline=baseline,
                                       **bootstrap(values,wide.index.get_level_values('block'),length)))
            for row,adjusted in zip(family,holm([r['p_value'] for r in family])):
                row['p_holm']=float(adjusted)
            contrasts+=family
        forecasts.to_parquet(OUT/'forecasts.parquet',index=False)
        daily.to_csv(OUT/'daily_metrics.csv',index=False)
        weights.to_csv(OUT/'hedge_weights.csv',index=False)
        write_new(OUT/'stack_fits.json',fits)
        elapsed=time.perf_counter()-start
        result=dict(status='complete_exploratory',completed_at=now(),protocol_sha256=digest(OUT/'protocol.json'),
                    metrics=metrics,contrasts=contrasts,n_base_forecasts=len(panel),n_evaluation_symbol_dates=len(forecasts)//5,
                    n_evaluation_dates=forecasts.date.nunique(),meta_fits=len(fits),base_fits=0,
                    seconds=elapsed,soft_budget_exceeded=elapsed>120,hard_budget_seconds=600,
                    python=platform.python_version(),platform=platform.platform(),sources=protocol['sources'],inputs=protocol['inputs'],
                    limitations=protocol['limitations'],external_orders=False,accounts_modified=False)
        write_new(OUT/'summary.json',result)
        lines=['# Controller screen: exploratory results','',
               'This is a predictive feasibility screen on previously inspected 2025 history, using immutable rolling forecasts. It is not a fresh holdout or a trading result.','',
               f"Evaluation: {result['n_evaluation_dates']} dates, {result['n_evaluation_symbol_dates']} dependent symbol-date observations, 22 assets. No base models were retrained; the stack used two registered fits.",'',
               '| Model | Log loss ↓ | Accuracy | Brier ↓ |','|---|---:|---:|---:|']
        for row in metrics:
            if row['period']=='2025-all':lines.append(f"| {row['model']} | {row['log_loss']:.6f} | {row['accuracy']:.2%} | {row['brier']:.6f} |")
        lines += ['', 'Primary paired date-block inference (block 10; 2,000 replicates; Holm family of six). Negative effects favor the controller.','',
                  '| Contrast | Δ log loss | 95% interval | Holm p |','|---|---:|---|---:|']
        for row in contrasts:
            if row['block_length']==10:lines.append(f"| {row['id']} | {row['effect']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | {row['p_holm']:.4f} |")
        lines += ['', 'Hedge updates once per matured date using the mean loss across all assets. Endpoints must be strictly before the decision, and each date is consumed once. The logistic stack fits only mature OOF rows in 2024 at the first 2025 decision, then mature 2024/first-half-2025 rows at the first July decision.','',
                  'The source panel predates this screen and was already inspected. The original current-vintage adjustment, universe overlap and survivorship limitations remain. These results do not certify portfolio value, costs, controller integration, profitability, or promotion. The next stage needs a new registered portfolio test with identical capital, execution and risk constraints.','',
                  f'Completed in {elapsed:.3f} seconds. Exact inputs/source hashes, split rules, budget and all six comparisons are in protocol.json; detailed metrics and block-6/21 sensitivity are in summary.json.']
        with (OUT/'REPORT.md').open('x') as handle:handle.write('\n'.join(lines)+'\n')
        files={str(p.relative_to(ROOT)):digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}
        write_new(OUT/'manifest.json',dict(created_at=now(),status='complete',files=files))
        return dict(status=result['status'],output=str(OUT),seconds=elapsed,metrics=[m for m in metrics if m['period']=='2025-all'])
    except BaseException as exc:
        write_new(OUT/'failure.json',dict(at=now(),exception=type(exc).__name__,message=str(exc),seconds=time.perf_counter()-start))
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','run'])
    args=parser.parse_args()
    print(json.dumps(register() if args.command=='register' else run(),indent=2))
