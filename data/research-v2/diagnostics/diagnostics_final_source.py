"""Registered saved-evidence diagnostics; never modifies historical releases.

Execute: python -m research_v2.diagnostics
Probabilities are reused, never retrained. Historical evidence is exploratory.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import platform
import types
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'data/releases/2026-09-28-adaptive'
EXP = RELEASE / 'experiment'
OUT = ROOT / 'data/research-v2'
PERIODS = {'development': ('2024-01-02', '2025-12-31'), 'selection': ('2026-01-02', '2026-07-10'), 'audit': ('2026-07-13', '2026-09-25')}
MODELS = ['linear', 'tree', 'raw_mix', 'hybrid', 'prior', 'always_up']
STRATEGIES = ['trend', 'ml', 'adaptive_hybrid', 'equal_weight', 'cash']
CLASSES = ['DOWN', 'FLAT', 'UP']
SEED = 20260928


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, (pd.Timestamp, Path)):
            return str(x)
        return x
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False))


def table(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / 'tables' / f'diag_{name}.csv', index=False)
    return frame


def block_indices(n, length=10, reps=2000, seed=SEED):
    if n <= 0 or length <= 0:
        raise ValueError('Positive sample and block lengths required')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, (reps, math.ceil(n / length)))
    return ((starts[..., None] + np.arange(length)) % n).reshape(reps, -1)[:, :n]


def block_effect(values, length=10, reps=2000):
    """Paired mean CI; centered circular-block null; two-sided p, no IID rows."""
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all() or not len(values):
        raise ValueError('Finite nonempty daily vector required')
    effect = float(values.mean())
    boot = values[block_indices(len(values), length, reps)].mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    null = boot - effect
    p = (1 + np.count_nonzero(np.abs(null) >= abs(effect) - 1e-15)) / (reps + 1)
    se = float(boot.std(ddof=1))
    return dict(effect=effect, ci_low=float(lo), ci_high=float(hi), bootstrap_se=se,
                p_two_sided=float(p), mde80_approx=2.801585 * se,
                sessions=len(values), block_length=length, approximate_blocks=len(values)/length,
                uncertainty='2000 circular date blocks; cross-sectional daily average; percentile CI')


def paired_difference(candidate, benchmark):
    if not candidate.index.equals(benchmark.index) or not candidate.index.is_unique:
        raise ValueError('Candidate and benchmark must have identical unique sessions')
    if not np.isfinite(candidate.to_numpy()).all() or not np.isfinite(benchmark.to_numpy()).all():
        raise ValueError('Candidate and benchmark must be finite')
    return candidate - benchmark


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1, np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order]))
    result = np.empty(len(p))
    result[order] = adjusted
    return result


def probability_metrics(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if p.shape != (len(y), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid probabilities')
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError('Unmatured labels cannot be scored')
    pred = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return dict(n=len(y), accuracy=float(np.mean(pred == y)), balanced_accuracy=float(recall[support>0].mean()),
                macro_f1=float(f1.mean()), log_loss=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
                brier=float(np.square(p-np.eye(3)[y]).sum(1).mean()),
                predicted_down=int((pred==0).sum()), predicted_flat=int((pred==1).sum()), predicted_up=int((pred==2).sum()))


def verify_evidence():
    inventory = json.loads((OUT/'baseline-2026-09-28/inventory.json').read_text())
    changed = [p for p,h in inventory['sealed_files'].items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
    if changed:
        raise RuntimeError(f'Sealed evidence changed: {changed}')
    m = json.loads((RELEASE/'manifest.json').read_text())
    art_bad = [p for p,h in m['artifacts'].items() if digest(RELEASE/p)!=h]
    source_bad = [p for p,h in m['source'].items() if digest(RELEASE/'source'/p)!=h]
    if art_bad or source_bad:
        raise RuntimeError(f'Release hash mismatch {art_bad} {source_bad}')
    repro_root = ROOT/'data/releases/2026-09-28-reproduction'
    reg = json.loads((repro_root/'verification.json').read_text())
    for name in reg['checked_files']:
        a,b = EXP/name, repro_root/'release/experiment'/name
        if name.endswith('.parquet'):
            pd.testing.assert_frame_equal(pd.read_parquet(a),pd.read_parquet(b),check_exact=True)
        elif name.endswith('.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(a),pd.read_csv(b),check_exact=True)
        else:
            assert json.loads(a.read_text())==json.loads(b.read_text()), name
    report = ET.parse(RELEASE/'test-results.xml').getroot()
    cases = list(report.iter('testcase'))
    counts = dict(total=len(cases), failures=sum(c.find('failure') is not None for c in cases), errors=sum(c.find('error') is not None for c in cases), skipped=sum(c.find('skipped') is not None for c in cases))
    saved = json.loads((RELEASE/'verification.json').read_text())
    assert digest(RELEASE/'test-results.xml')==saved['test_report_sha256']
    tests_log = (RELEASE/'tests.log').read_text()
    excluded_line = [l for l in tests_log.splitlines() if 'deselected' in l][-1:]
    result = dict(sealed_files_checked=len(inventory['sealed_files']),sealed_files_changed=changed,
                  release_artifact_hashes=len(m['artifacts']), archived_source_hashes=len(m['source']),
                  reproduced_files_checked=len(reg['checked_files']), reproduction='exact numeric/text-object comparison repeated against both saved sets; no new model training',
                  saved_junit=counts, saved_test_log_selection=excluded_line,
                  provider_integrations_unrun_saved=saved['provider_integration_tests_unrun'],
                  provider_exclusions_scope='17 recorded in historical verification and test log; not newly executed',
                  engineering_scope='Saved JUnit is historical evidence and does not certify current source',
                  market_manifest_sha256=digest(RELEASE/'market/manifest.json'))
    save_json(OUT/'diagnostics/evidence_verification.json',result)
    return result


def load_predictions_and_data():
    import exchange_calendars as xcals
    from src.models.adaptive import features, execution_labels
    markets = {p.stem:pd.read_parquet(p) for p in sorted((RELEASE/'market/adjusted').glob('*.parquet'))}
    prediction_pieces, fold_rows, coverage = [], [], []
    common = markets['SPY'].index
    for raw in markets.values(): common = common.intersection(raw.index)
    calendar=xcals.get_calendar('XNYS',start=common[0].date(),end=common[-1].date())
    expected=calendar.sessions_in_range(common[0].date(),common[-1].date()).tz_localize('UTC')
    for symbol, raw in markets.items():
        pred = pd.read_parquet(EXP/f'{symbol}_predictions.parquet')
        folds=json.loads((EXP/f'{symbol}_folds.json').read_text())
        x=features(raw, markets['SPY'])
        y,ret=execution_labels(raw.Close)
        assert np.array_equal(pred.label.to_numpy(), y.loc[pred.index].to_numpy()),symbol
        assert np.allclose(pred.forward_return,ret.loc[pred.index],equal_nan=True),symbol
        original=pd.read_parquet(RELEASE/'market/raw'/f'{symbol}.parquet')
        factor=original.Adj_Close/original.Close
        finite=np.isfinite(raw[['Open','High','Low','Close','Volume']]).all(axis=1)
        coverage.append(dict(symbol=symbol,raw_rows=len(original),adjusted_rows=len(raw),start=str(raw.index[0].date()),end=str(raw.index[-1].date()),expected_sessions=len(expected),missing_sessions=len(expected.difference(raw.index)),duplicate_keys=int(raw.index.duplicated().sum()),nonfinite_rows=int((~finite).sum()),nonpositive_prices=int((raw.Close<=0).sum()),zero_volume=int((raw.Volume==0).sum()),feature_complete_rows=int(np.isfinite(x).all(axis=1).sum()),feature_warmup_rows=int((~np.isfinite(x).all(axis=1)).sum()),prediction_rows=len(pred),mature_prediction_rows=int((pred.label>=0).sum()),immature_prediction_rows=int((pred.label<0).sum()),folds=len(folds),adjustment_factor_min=float(factor.min()),adjustment_factor_max=float(factor.max()),corporate_action_events='not archived; provider current-vintage adjustment factors only',raw_sha256=digest(RELEASE/'market/raw'/f'{symbol}.parquet'),adjusted_sha256=digest(RELEASE/'market/adjusted'/f'{symbol}.parquet')))
        pred=pred.copy();pred['date']=pred.index;pred['symbol']=symbol
        pred['period']=np.select([pred.index<=pd.Timestamp('2025-12-31',tz='UTC'),pred.index<=pd.Timestamp('2026-07-10',tz='UTC')],['development','selection'],default='audit')
        ages = {d:i for i,d in enumerate(raw.index)}
        pred['age_sessions']=[ages[d]-ages[pd.Timestamp(f)] for d,f in zip(pred.index,pred.fit_decision)]
        pred['quarter']=pred.index.tz_localize(None).to_period('Q').astype(str)
        for c in range(3): pred[f'raw_mix_{c}']=(pred[f'linear_{c}']+pred[f'tree_{c}'])/2
        for c,pv in enumerate([1e-12,1e-12,1-2e-12]): pred[f'always_up_{c}']=pv
        for i, fold in enumerate(folds):
            assert pd.Timestamp(fold['fit_label_end'])<pd.Timestamp(fold['calibration_start'])
            assert pd.Timestamp(fold['latest_label_end'])<pd.Timestamp(fold['decision'])
            fold_rows.append(dict(symbol=symbol,fold=i,**{k:v for k,v in fold.items() if k not in ['columns','config']},features=len(fold['columns'])))
        prediction_pieces.append(pred)
    frame=pd.concat(prediction_pieces,ignore_index=True).sort_values(['date','symbol']).reset_index(drop=True)
    spy=markets['SPY'].Close
    mom=spy.pct_change(63);vol=spy.pct_change().rolling(21).std()*np.sqrt(252)
    frame['regime']=[('positive_trend' if mom.loc[d]>=0 else 'negative_trend')+('_high_vol' if vol.loc[d]>=.20 else '_low_vol') for d in frame.date]
    frame['expert_disagreement']=frame[[f'linear_{c}' for c in range(3)]].to_numpy().argmax(1)!=frame[[f'tree_{c}' for c in range(3)]].to_numpy().argmax(1)
    cov=table('data_coverage',coverage);table('fold_metadata',fold_rows)
    flow=[dict(stage='raw_symbol_sessions',n=int(cov.raw_rows.sum())),dict(stage='common_market_dates',n=len(common)),dict(stage='feature_complete_symbol_sessions',n=int(cov.feature_complete_rows.sum())),dict(stage='warmup_symbol_sessions',n=int(cov.feature_warmup_rows.sum())),dict(stage='pre_evaluation_feature_complete_symbol_sessions',n=int(cov.feature_complete_rows.sum()-len(frame))),dict(stage='decision_symbol_sessions',n=len(frame)),dict(stage='mature_labeled_symbol_sessions',n=int((frame.label>=0).sum())),dict(stage='unmatured_symbol_sessions',n=int((frame.label<0).sum()))]
    table('dataset_flow',flow)
    table('class_counts',frame.assign(class_name=frame.label.map({-1:'UNMATURED',0:'DOWN',1:'FLAT',2:'UP'})).groupby(['period','class_name']).agg(n=('label','size'),sessions=('date','nunique')).reset_index())
    return frame,markets,dict(instruments=len(markets),raw_symbol_sessions=int(cov.raw_rows.sum()),raw_dates=len(common),decision_dates=frame.date.nunique(),decision_rows=len(frame),mature_rows=int((frame.label>=0).sum()),mature_dates=frame.loc[frame.label>=0,'date'].nunique(),unmatured_rows=int((frame.label<0).sum()),folds=len(fold_rows),flow=flow,coverage=cov.to_dict('records'))


def model_p(df, model): return df[[f'{model}_{c}' for c in range(3)]].to_numpy()


def summarize_group(df, **keys):
    rows=[]
    for m in MODELS:
        rows.append(dict(**keys,model=m,sessions=df.date.nunique(),**probability_metrics(df.label,model_p(df,m))))
    return rows


def forecasts(frame, registration):
    valid=frame.loc[frame.label>=0].copy()
    summaries=[];class_reports=[];confusions=[];daily=[]
    for period,g in valid.groupby('period',sort=False):
        summaries+=summarize_group(g,period=period)
        for model in MODELS:
            p=model_p(g,model);y=g.label.to_numpy(int);pred=p.argmax(1)
            prec,rec,f1,sup=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
            cm=confusion_matrix(y,pred,labels=[0,1,2])
            for c in range(3):
                class_reports.append(dict(period=period,model=model,class_id=c,class_name=CLASSES[c],precision=prec[c],recall=rec[c],f1=f1[c],support=int(sup[c]),predicted_support=int((pred==c).sum())))
                for k in range(3):confusions.append(dict(period=period,model=model,true_class=CLASSES[c],predicted_class=CLASSES[k],count=int(cm[c,k]),row_fraction=float(cm[c,k]/sup[c]) if sup[c] else np.nan))
            part=pd.DataFrame(dict(date=g.date,log_loss=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)),accuracy=(pred==y).astype(float),confidence=p.max(1)))
            d=part.groupby('date').mean().reset_index();d['period']=period;d['model']=model;daily.append(d)
    summary=table('forecast_summary',summaries);table('class_report',class_reports);table('confusion',confusions)
    daily=pd.concat(daily,ignore_index=True);table('forecast_daily',daily)
    inference=[]
    for contrast in registration['predictive_family']:
        period,model,endpoint=contrast['period'],contrast['model'],contrast['endpoint']
        ds=daily.loc[daily.period==period].pivot(index='date',columns='model',values=endpoint)
        for length in registration['block_lengths']:
            inference.append(dict(period=period,model=model,comparator='prior',endpoint=endpoint,positive_direction='negative' if endpoint=='log_loss' else 'positive',**block_effect(paired_difference(ds[model],ds.prior),length)))
    inf=pd.DataFrame(inference)
    for length,g in inf.groupby('block_length'):
        inf.loc[g.index,'holm_p']=holm(g.p_two_sided)
        inf.loc[g.index,'family_size']=len(g)
    table('paired_inference',inf)
    for key,filename in [('symbol','instrument'),('quarter','period'),('regime','regime'),('fit_decision','fold'),('age_sessions','model_age')]:
        rows=[]
        for (period,value),g in valid.groupby(['period',key],sort=True):rows+=summarize_group(g,period=period,**{key:value})
        table(filename,rows)
    gates=[]
    for (period,gate,disagree),g in valid.groupby(['period','gate','expert_disagreement']):
        row=dict(period=period,gate=bool(gate),expert_disagreement=bool(disagree),n=len(g),sessions=g.date.nunique(),fraction_of_period=len(g)/int((valid.period==period).sum()))
        for m in ['hybrid','prior','raw_mix']:
            met=probability_metrics(g.label,model_p(g,m));row.update({f'{m}_{k}':met[k] for k in ['accuracy','log_loss']})
        gates.append(row)
    table('gate',gates)
    calibration(valid)
    failures(valid)
    return dict(summary=summary.to_dict('records'),primary_contrasts=inf.loc[inf.block_length==10].to_dict('records'),predictive_family_size=len(registration['predictive_family']),gate_true_rows=int(valid.gate.sum()),mature_rows=len(valid),disagreement_rows=int(valid.expert_disagreement.sum()),interpretation='All three periods were inspected before this registration; interval and Holm results are exploratory. Calibrated hybrid is temperature-scaled mixture, not the portfolio gate.')


def calibration(valid):
    reliability=[];coverage=[];comparisons=[]
    for period,g in valid.groupby('period',sort=False):
        dates=pd.Index(sorted(g.date.unique()));dateid=dates.get_indexer(g.date);ids=block_indices(len(dates))
        y=g.label.to_numpy(int)
        for model in MODELS:
            p=model_p(g,model);conf=p.max(1);pred=p.argmax(1);correct=pred==y
            # Top-label reliability with date-block ratio uncertainty preserving empty days.
            for b in range(10):
                mask=(conf>=b/10)&(conf<(b+1)/10 if b<9 else conf<=1)
                if not mask.any():
                    reliability.append(dict(period=period,model=model,bin=b,lower=b/10,upper=(b+1)/10,n=0,sessions=0));continue
                sums=np.bincount(dateid[mask],weights=correct[mask],minlength=len(dates));counts=np.bincount(dateid[mask],minlength=len(dates))
                den=counts[ids].sum(1);num=sums[ids].sum(1);boot=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
                lo,hi=np.nanquantile(boot,[.025,.975])
                reliability.append(dict(period=period,model=model,bin=b,lower=b/10,upper=(b+1)/10,n=int(mask.sum()),sessions=int(np.count_nonzero(counts)),confidence=float(conf[mask].mean()),observed_accuracy=float(correct[mask].mean()),ci_low=lo,ci_high=hi,valid_bootstraps=int(np.isfinite(boot).sum())))
            for threshold in [.34,.4,.45,.5,.6,.7,.8,.9]:
                mask=conf>=threshold
                coverage.append(dict(period=period,model=model,threshold=threshold,n=int(mask.sum()),sessions=g.loc[mask,'date'].nunique(),coverage=float(mask.mean()),accuracy=float(correct[mask].mean()) if mask.any() else np.nan,mean_forward_return=float(g.loc[mask,'forward_return'].mean()) if mask.any() else np.nan,note='Descriptive confidence filter, never selected or converted into a trading strategy'))
        raw=model_p(g,'raw_mix');cal=model_p(g,'hybrid');rowlog=-np.log(cal[np.arange(len(y)),y])+np.log(raw[np.arange(len(y)),y]);dd=pd.Series(rowlog,index=g.date).groupby(level=0).mean()
        comparisons.append(dict(period=period,contrast='calibrated_minus_raw_mix_log_loss',same_argmax=bool((cal.argmax(1)==raw.argmax(1)).all()),**block_effect(dd),testing='descriptive calibration intervention; no independently validated gate claim'))
    table('reliability',reliability);table('coverage',coverage);table('calibration_contrast',comparisons)


def failures(valid):
    examples=[]
    for period,g in valid.groupby('period',sort=False):
        g=g.copy();p=model_p(g,'hybrid');y=g.label.to_numpy(int)
        g['predicted_class']=p.argmax(1);g['confidence']=p.max(1);g['log_loss']=-np.log(p[np.arange(len(y)),y]);g['correct']=g.predicted_class==g.label
        chosen=[('worst_log_loss',g.nlargest(20,'log_loss')),('confident_error',g.loc[~g.correct].nlargest(20,'confidence'))]
        for c in range(3):
            err=g.loc[(~g.correct)&(g.label==c)].copy()
            if len(err):
                err['distance_from_median']=(err.log_loss-err.log_loss.median()).abs()
                chosen.append(('representative_median_error_'+CLASSES[c],err.sort_values(['distance_from_median','date','symbol']).head(1)))
        for rule,part in chosen:
            for _,r in part.iterrows(): examples.append(dict(period=period,selection_rule=rule,date=r.date,symbol=r.symbol,true_class=CLASSES[int(r.label)],predicted_class=CLASSES[int(r.predicted_class)],confidence=r.confidence,log_loss=r.log_loss,forward_return=r.forward_return,gate=bool(r.gate),expert_disagreement=bool(r.expert_disagreement),age_sessions=int(r.age_sessions),regime=r.regime,fit_decision=r.fit_decision))
    table('failures',examples)


def drawdown_duration(equity,initial=10000):
    a=np.asarray(equity,float);peak=np.maximum.accumulate(np.r_[initial,a])[1:];under=a<peak-1e-9
    longest=current=0
    for flag in under:
        current=current+1 if flag else 0;longest=max(longest,current)
    return float(np.min(a/peak-1)),int(longest),int(current)


def portfolio_metrics(history,events,initial=10000,scope='fresh_capital'):
    equity=history.equity;rets=equity.pct_change().fillna(equity.iloc[0]/initial-1).to_numpy()
    fills=[e for e in events if e['kind']=='fill'];closed=[e for e in fills if e['side']=='sell']
    turnover=sum(e['qty']*e['price'] for e in fills)/initial
    fees=sum(e['fee'] for e in fills)
    pnl=np.asarray([e['pnl'] for e in closed],float)
    maxdd,duration,terminaldd=drawdown_duration(equity,initial)
    sd=np.std(rets,ddof=1);downside=np.sqrt(np.mean(np.minimum(rets,0)**2))
    return dict(scope=scope,sessions=len(history),initial_equity=initial,terminal_equity=float(equity.iloc[-1]),net_return=float(equity.iloc[-1]/initial-1),gross_fixed_fill_return=float((equity.iloc[-1]+fees)/initial-1),cost_drag_fixed_fill=fees/initial,
                cagr=float((equity.iloc[-1]/initial)**(252/len(history))-1),sharpe_zero_rf=float(rets.mean()/sd*np.sqrt(252)) if sd>1e-15 else np.nan,
                sortino_zero_mar=float(rets.mean()/downside*np.sqrt(252)) if downside>1e-15 else np.nan,
                max_drawdown=maxdd,max_drawdown_duration_sessions=duration,terminal_underwater_sessions=terminaldd,
                mean_gross_exposure=float(history.gross_exposure.mean()),max_gross_exposure=float(history.gross_exposure.max()),
                combined_fees_usd=fees,commission_usd=fees*2/3,slippage_proxy_usd=fees/3,turnover_initial_capital=turnover,
                fills=len(fills),sell_fills=len(closed),net_realized_sell_pnl_usd=float(pnl.sum()) if len(pnl) else 0,
                sell_fill_win_rate=float((pnl>0).mean()) if len(pnl) else np.nan,
                profit_factor_sell_fills=float(pnl[pnl>0].sum()/-pnl[pnl<0].sum()) if (pnl<0).any() else np.nan,
                terminal_positions=sum(abs(float(history.iloc[-1][c]))>1e-8 for c in history if c.startswith('quantity_')),
                rejections=sum(e['kind']=='rejection' for e in events),max_reconciliation_error=float(history.reconciliation_error.abs().max()))


def account_module():
    """Use release-pinned execution source in a private module, never change sys.path."""
    path=RELEASE/'source/src/execution/account.py'
    # Compile bytes explicitly so Python never writes __pycache__ into the seal.
    module=types.ModuleType('_diag_sealed_account');module.__file__=str(path)
    sys.modules[module.__name__]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def replay(prices,targets,module,cost_multiplier=1,delay=0,partial=1,gross=.8,reserve=.03):
    if not prices.index.equals(targets.index) or not prices.index.is_unique or not prices.index.is_monotonic_increasing or not np.isfinite(prices.to_numpy()).all() or (prices.to_numpy()<=0).any() or not np.isfinite(targets.to_numpy()).all():
        raise ValueError('Incomplete or invalid replay panel')
    growth={s:'growth' for s in ['AAPL','AMZN','GOOGL','META','MSFT','NVDA','QQQ','TSLA','XLK']}
    lim=module.Limits(max_gross=gross,max_position=.1,max_turnover=.4,commission=.001*cost_multiplier,slippage=.0005*cost_multiplier,reserve_buffer=reserve,groups=growth,group_limits={'growth':.35})
    acc=module.Account(list(prices.columns),10000,lim)
    if partial<1:
        original=acc.fill
        def fill(order_id,qty,price,asof,prices,fill_id):
            return original(order_id,qty*partial,price,asof,prices,fill_id+':'+str(asof))
        acc.fill=fill
    session=module.ExecutionSession(acc)
    rebalance=np.flatnonzero((prices.index.weekday==4)|(np.arange(len(prices))==0))
    decisions={int(i+delay):targets.iloc[i] for i in rebalance if i+delay<len(prices)}
    rows=[]
    for i,date in enumerate(prices.index):
        weights=decisions[i].copy() if i in decisions else None
        if weights is not None and weights.sum()>gross:weights*=gross/weights.sum()
        result=session.on_close(date,prices.iloc[i].to_dict(),None if weights is None else weights.to_dict())
        rows.append(dict(date=date,**{k:v for k,v in result.items() if k!='positions'},**{f'quantity_{s}':q for s,q in result['positions'].items()}))
    return pd.DataFrame(rows).set_index('date'),acc


def economics(registration):
    module=account_module();prices=pd.read_parquet(EXP/'prices.parquet');returns=pd.read_parquet(EXP/'returns.parquet')
    targets={n:pd.read_parquet(EXP/f'{n}_targets.parquet') for n in STRATEGIES}
    records=[];curves=[];attribution=[];closed_rows=[];fresh_returns={};fresh_history={}
    for name in STRATEGIES:
        hist=pd.read_parquet(EXP/f'{name}_account.parquet');events=json.loads((EXP/f'{name}_events.json').read_text())
        records.append(dict(period='full',strategy=name,**portfolio_metrics(hist,events,scope='continuous_2024_start')))
        assert np.allclose(hist.equity.pct_change().fillna(0),returns[name])
        # Net sell P&L already includes pro-rata acquisition fees in cost basis.
        fills=[e for e in events if e['kind']=='fill'];sales=[e for e in fills if e['side']=='sell']
        assert abs(sum(e['pnl'] for e in sales)-hist.realized_pnl.iloc[-1])<1e-7
        for e in sales: closed_rows.append(dict(strategy=name,**{k:e[k] for k in ['asof','symbol','qty','price','fee','pnl']},unit='sell fill or partial reduction, not necessarily flat-to-flat trade'))
        for symbol in prices:
            sf=[e for e in fills if e['symbol']==symbol];ss=[e for e in sf if e['side']=='sell']
            attribution.append(dict(strategy=name,symbol=symbol,net_realized_usd=sum(e['pnl'] for e in ss),combined_fees_usd=sum(e['fee'] for e in sf),turnover_usd=sum(e['qty']*e['price'] for e in sf),sell_fills=len(ss),losing_sell_fills=sum(e['pnl']<0 for e in ss),negative_sell_pnl_usd=sum(min(0,e['pnl']) for e in ss),terminal_units=float(hist.iloc[-1][f'quantity_{symbol}']),scope='continuous account; net realized is not whole-symbol total return'))
        for period,(start,end) in PERIODS.items():
            h=hist.loc[start:end];before=hist.loc[hist.index<h.index[0]].equity;initial=float(before.iloc[-1]) if len(before) else 10000
            ev=[e for e in events if e.get('asof') and h.index[0]<=pd.Timestamp(e['asof'])<=h.index[-1]]
            records.append(dict(period=period,strategy=name,**portfolio_metrics(h,ev,initial,scope='continuous_account_period')))
            ph,account=replay(prices.loc[start:end],targets[name].loc[start:end],module)
            fresh_history[(period,name)]=ph;fresh_returns[(period,name)]=ph.equity.pct_change().fillna(0)
            records.append(dict(period=period,strategy=name,**portfolio_metrics(ph,account.events,scope='matched_fresh_10000')))
            ph.to_parquet(OUT/'diagnostics'/f'{period}_{name}_fresh_account.parquet')
            save_json(OUT/'diagnostics'/f'{period}_{name}_fresh_events.json',account.events)
        dd=hist.equity/np.maximum.accumulate(np.r_[10000,hist.equity])[1:]-1
        for d in hist.index:curves.append(dict(date=d,strategy=name,equity=hist.at[d,'equity'],drawdown=dd.loc[d],exposure=hist.at[d,'gross_exposure'],combined_fees_usd=hist.at[d,'fees'],scope='continuous_2024_start'))
    inf=[]
    for contrast in registration['economic_family']:
        p,m=contrast['period'],contrast['model'];diff=paired_difference(fresh_returns[(p,m)],fresh_returns[(p,'equal_weight')])
        for length in registration['block_lengths']:
            row=dict(period=p,strategy=m,comparator='equal_weight',endpoint='net_daily_return',scope='matched_fresh_10000',**block_effect(diff,length))
            row.update({f'{k}_annualized_arithmetic':row[k]*252 for k in ['effect','ci_low','ci_high','mde80_approx']});inf.append(row)
    infer=pd.DataFrame(inf)
    for length,g in infer.groupby('block_length'):infer.loc[g.index,'holm_p']=holm(g.p_two_sided);infer.loc[g.index,'family_size']=len(g)
    met=table('portfolio',records);table('economic_inference',infer);table('portfolio_curves',curves);table('attribution',attribution);closed=table('closed_trades',closed_rows)
    if len(closed):table('worst_sell_fills',closed.groupby('strategy',group_keys=False).apply(lambda g:g.nsmallest(20,'pnl'),include_groups=False).reset_index())
    return dict(metrics=met.to_dict('records'),primary_contrasts=infer.loc[infer.block_length==10].to_dict('records'),economic_family_size=len(registration['economic_family']),conventions={'capital':'$10000 fresh per period for new economic comparison; native account continuous start2024 also shown','cost':'10bp commission+5bp additive slippage per traded notional, both sides; fixed-fill gross adds exact debits, not a frictionless rerun','sharpe':'252 sessions/year; zero risk-free; sample standard deviation; undefined for zero variance','sortino':'zero minimum acceptable return; root mean squared negative returns over all sessions; undefined for no downside','turnover':'sum absolute executed notional / initial account capital; no halving','trades':'sell fills include partial reductions; both-side fees are in realized P&L; open terminal holdings retained','spy':'frictionless SPY is descriptive only, excluded from matched-capital inference'})


def stress(registration):
    module=account_module();prices=pd.read_parquet(EXP/'prices.parquet').loc[slice(*PERIODS['audit'])]
    rows=[]
    for scenario in registration['stress']:
        sid=scenario['id']
        for name in STRATEGIES[:-1]:
            targets=pd.read_parquet(EXP/f'{name}_targets.parquet').loc[prices.index];p=prices.copy();kwargs={}
            if sid=='cost_zero':kwargs['cost_multiplier']=0
            if sid=='cost_2x':kwargs['cost_multiplier']=2
            if sid=='cost_4x':kwargs['cost_multiplier']=4
            if sid=='delay_1':kwargs['delay']=1
            if sid=='liquidity_50pct':kwargs['partial']=.5
            if sid=='risk_gross_60pct':kwargs['gross']=.6
            if sid=='reserve_zero':kwargs['reserve']=0
            if sid=='gap_down_10pct':p.iloc[len(p)//2]*=.9
            if sid=='missing_bar':p.iloc[len(p)//2,p.columns.get_loc('SPY')]=np.nan
            t=time.perf_counter()
            try:
                hist,acc=replay(p,targets,module,**kwargs)
                met=portfolio_metrics(hist,acc.events)
                rows.append(dict(scenario=sid,strategy=name,status='completed',elapsed_seconds=time.perf_counter()-t,**met))
                hist.to_parquet(OUT/'diagnostics'/f'stress_{sid}_{name}_account.parquet')
            except ValueError as e:
                if sid!='missing_bar':raise
                rows.append(dict(scenario=sid,strategy=name,status='expected_validation_halt',error=str(e),elapsed_seconds=time.perf_counter()-t))
    df=table('stress',rows)
    return dict(rows=df.to_dict('records'),scope='54 audit dates; each scenario and strategy starts fresh $10000. Frozen target intent, account quantities/fees/rejections change. Synthetic stress is not a probability forecast.',decisions_changed=['delay_1 shifts submission one session','risk_gross_60pct proportionally clips weights; model forecasts unchanged'],missing_bar='All 4 strategies halted before replay; no fill or forward-filled bar')


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','figure.dpi':140,'axes.titlesize':12})
    colors={'linear':'#4776a6','tree':'#c7792d','raw_mix':'#864f8f','hybrid':'#14786e','prior':'#222a35','always_up':'#999999','adaptive_hybrid':'#14786e','trend':'#4776a6','ml':'#c7792d','equal_weight':'#222a35','cash':'#aaaaaa'}
    def read(name):return pd.read_csv(OUT/'tables'/f'diag_{name}.csv')
    def save(fig,name):
        fig.savefig(OUT/'figures'/f'diag_{name}.png',dpi=200);fig.savefig(OUT/'figures'/f'diag_{name}.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,3.7));counts=read('class_counts');pv=counts.pivot(index='period',columns='class_name',values='n').fillna(0).reindex(['development','selection','audit']);pv.plot.bar(stacked=True,ax=ax[0],color=['#5d83a6','#c8b779','#b0b5bb','#3b8878']);ax[0].set_ylabel('Instrument-session rows');ax[0].set_title('Mature labels and withheld endpoints');ax[0].tick_params(axis='x',rotation=0);cov=read('data_coverage');ax[1].barh(['Raw observations','Feature complete','Saved forecasts','Mature forecasts'],[cov.raw_rows.sum(),cov.feature_complete_rows.sum(),cov.prediction_rows.sum(),cov.mature_prediction_rows.sum()],color=['#a5b6c9','#6f8ea9','#3a6f92','#14786e']);ax[1].set_title('Exact retained-data accounting');ax[1].set_xlabel('Instrument-session rows');fig.tight_layout();save(fig,'dataset')
    fig,axes=plt.subplots(1,3,figsize=(11,3.6));cm=read('confusion');
    for ax,m in zip(axes,['hybrid','prior','always_up']):
        c=cm.loc[(cm.period=='audit')&(cm.model==m)].pivot(index='true_class',columns='predicted_class',values='count').reindex(index=CLASSES,columns=CLASSES);im=ax.imshow(c, cmap='Blues');
        for i in range(3):
            for j in range(3):ax.text(j,i,str(int(c.iloc[i,j])),ha='center',va='center',color='white' if c.iloc[i,j]>c.to_numpy().max()*.65 else '#172735')
        ax.set_xticks(range(3),CLASSES);ax.set_yticks(range(3),CLASSES);ax.set_xlabel('Predicted class');ax.set_title(m+f' | audit n={int(c.to_numpy().sum())}')
    axes[0].set_ylabel('True class');fig.tight_layout();save(fig,'confusion')
    rel=read('reliability');cover=read('coverage');fig,ax=plt.subplots(1,2,figsize=(11,3.8))
    for m in ['raw_mix','hybrid','prior']:
        g=rel.loc[(rel.period=='audit')&(rel.model==m)&(rel.n>0)];ax[0].errorbar(g.confidence,g.observed_accuracy,yerr=[g.observed_accuracy-g.ci_low,g.ci_high-g.observed_accuracy],fmt='o-',label=m,color=colors[m],capsize=2)
        g=cover.loc[(cover.period=='audit')&(cover.model==m)&(cover.n>0)];ax[1].plot(g.coverage,g.accuracy,'o-',label=m,color=colors[m])
    ax[0].plot([0,1],[0,1],':',color='gray');ax[0].set(xlabel='Mean top probability',ylabel='Observed accuracy',title='Audit calibration: date-block 95% intervals');ax[1].set(xlabel='Retained fraction',ylabel='Accuracy among retained',title='Confidence threshold sweep: descriptive');ax[0].legend();ax[1].legend();fig.tight_layout();save(fig,'calibration')
    inf=read('paired_inference');fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    for ax,period in zip(axes,PERIODS):
        g=inf.loc[(inf.period==period)&(inf.endpoint=='log_loss')&(inf.block_length==10)].set_index('model').loc[['linear','tree','raw_mix','hybrid']];ys=np.arange(len(g));ax.errorbar(g.effect,ys,xerr=[g.effect-g.ci_low,g.ci_high-g.effect],fmt='o',color='#14786e',capsize=3);ax.axvline(0,color='gray',ls=':');ax.set_yticks(ys,g.index);ax.set_title(period);ax.set_xlabel('Log loss minus prior (lower better)')
    fig.tight_layout();save(fig,'forecast_inference')
    ins=read('instrument');fig,ax=plt.subplots(figsize=(9,5.6));h=ins.loc[ins.model=='hybrid'].pivot(index='symbol',columns='period',values='log_loss').reindex(columns=['development','selection','audit']);b=ins.loc[ins.model=='prior'].pivot(index='symbol',columns='period',values='log_loss').reindex(columns=h.columns);v=h-b;im=ax.imshow(v,aspect='auto',cmap='RdBu_r',vmin=-.2,vmax=.2);ax.set_yticks(range(len(v)),v.index);ax.set_xticks(range(3),v.columns);ax.set_title('Hybrid minus prior log loss by instrument\nExploratory single-instrument models; not transfer learning');fig.colorbar(im,ax=ax,label='Positive = worse probability quality');fig.tight_layout();save(fig,'generalization')
    age=read('model_age');fig,ax=plt.subplots(1,2,figsize=(11,3.7));
    for period in PERIODS:
        a=age.loc[(age.period==period)&(age.model=='hybrid')].set_index('age_sessions');b=age.loc[(age.period==period)&(age.model=='prior')].set_index('age_sessions');ax[0].plot(a.index,a.log_loss-b.log_loss,label=period);ax[1].plot(a.index,a.n,label=period)
    ax[0].axhline(0,color='gray',ls=':');ax[0].set(xlabel='Observed sessions since fitting',ylabel='Hybrid minus prior log loss',title='Model-age association (descriptive)');ax[1].set(xlabel='Observed sessions since fitting',ylabel='Instrument-session support',title='Unequal support near sample end');ax[0].legend();ax[1].legend();fig.tight_layout();save(fig,'model_age')
    curves=read('portfolio_curves');fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for name in STRATEGIES:
        g=curves.loc[curves.strategy==name];d=pd.to_datetime(g.date)
        for ax,col in zip(axes,['equity','drawdown','exposure']):ax.plot(d,g[col],label=name,color=colors[name],lw=1.1)
    axes[0].set_ylabel('Account equity (USD)');axes[0].set_title('Continuous matched policy accounts: $10,000 initial capital');axes[0].legend(ncol=3);axes[1].set_ylabel('Drawdown (fraction)');axes[2].set_ylabel('Gross exposure (fraction)');fig.tight_layout();save(fig,'portfolio')
    met=read('portfolio');g=met.loc[(met.period=='full')].set_index('strategy').loc[STRATEGIES[:-1]];fig,ax=plt.subplots(1,2,figsize=(11,3.8));g[['commission_usd','slippage_proxy_usd']].plot.bar(stacked=True,ax=ax[0],color=['#4776a6','#c7792d']);ax[0].set_ylabel('USD debited on filled notional');ax[0].set_title('Explicit cost attribution');ax[0].tick_params(axis='x',rotation=20);ax[1].bar(g.index,g.turnover_initial_capital,color='#14786e');ax[1].set_ylabel('Executed notional / initial $10,000');ax[1].set_title(f'Two-way turnover, {int(g.sessions.iloc[0])} sessions');ax[1].tick_params(axis='x',rotation=20);fig.tight_layout();save(fig,'costs')
    st=read('stress');g=st.loc[st.status=='completed'].pivot(index='scenario',columns='strategy',values='net_return');base=g.loc['base'];delta=(g-base)*100;fig,ax=plt.subplots(figsize=(9,5));im=ax.imshow(delta,aspect='auto',cmap='RdBu',vmin=-5,vmax=5);ax.set_yticks(range(len(delta)),delta.index);ax.set_xticks(range(len(delta.columns)),delta.columns,rotation=15);ax.set_title('Frozen-target stress: change from fresh audit baseline');fig.colorbar(im,ax=ax,label='Net-return change (percentage points)');
    for i in range(len(delta)):
        for j in range(len(delta.columns)):ax.text(j,i,f'{delta.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9)
    fig.tight_layout();save(fig,'stress')


def main():
    started=time.perf_counter();(OUT/'diagnostics').mkdir(exist_ok=True)
    registration=json.loads((OUT/'diagnostics/registration.json').read_text())
    assert registration['status']=='registered'
    assert registration['protocol_sha256']==digest(OUT/'protocol.json')
    summary=dict(run_id=registration['id'],registered_at=registration['registered_at'],started_at=datetime.now(timezone.utc).isoformat(),source_sha256=digest(__file__),protocol_sha256=digest(OUT/'protocol.json'),registration_sha256=digest(OUT/'diagnostics/registration.json'),historical_exposure='All dates exploratory; all historical partitions previously inspected',environment=dict(python=sys.version,platform=platform.platform(),processor=platform.processor()))
    print('Verifying saved evidence',flush=True);summary['evidence_verification']=verify_evidence()
    print('Auditing data and folds',flush=True);frame,market,summary['dataset']=load_predictions_and_data()
    print('Computing forecasts and diagnostics',flush=True);summary['forecasting']=forecasts(frame,registration)
    print('Computing matched portfolio economics',flush=True);summary['portfolio']=economics(registration)
    print('Replaying registered stresses',flush=True);summary['sensitivity']=stress(registration)
    print('Drawing evidence figures',flush=True);figures()
    summary['elapsed_seconds']=time.perf_counter()-started
    summary['completed_at']=datetime.now(timezone.utc).isoformat()
    summary['limitations']=['No historical untouched holdout remains. These are diagnostics, not new confirmatory evidence.','Price data are current-vintage retrospectively adjusted proxies, with no archived corporate-action event tape, delistings, or historical constituent memberships.','Single-instrument models establish temporal reuse only; held-instrument transfer is a separate registered experiment.','Temperature and legacy gate use the same trailing block; gate slices do not identify independent gate value.','Gross fixed-fill accounting adds debits back to realized net equity; zero-cost replay separately changes quantities and risk rejections.','54 audit account sessions have only 48 mature outcome dates, with strongly overlapping labels. 22 correlated assets are not 22 independent replications.','Fresh-capital audit accounts differ from continuous account period returns due to initial holdings, first rebalance, and fees.','Slippage is an additive fee proxy, not quote-level execution evidence; liquidity and gap stresses are synthetic.','63 prospective sessions and final label maturity must accumulate over time; no backfilled historical rows count.']
    summary['artifacts']={str(p.relative_to(ROOT)):digest(p) for directory,pattern in [(OUT/'tables','diag_*.csv'),(OUT/'figures','diag_*'),(OUT/'diagnostics','*.parquet')] for p in sorted(directory.glob(pattern))}
    save_json(OUT/'diagnostics/summary.json',summary)
    for rid in ['R1','R2','R8','R9','R10']:
        path=OUT/'registry'/f'{rid}.json';reg=json.loads(path.read_text());reg.update(status='completed_exploratory',completed_at=summary['completed_at'],summary='data/research-v2/diagnostics/summary.json',source_sha256=summary['source_sha256']);save_json(path,reg)
    print(json.dumps(dict(run_id=summary['run_id'],elapsed_seconds=summary['elapsed_seconds'],artifacts=len(summary['artifacts']))),flush=True)

if __name__=='__main__':main()


def trade_episodes(events, final_prices, calendar):
    """Account complete flat-to-flat episodes; terminal holdings stay explicitly open."""
    active,finished,counters={},{},{}
    output=[];positions={str(d):i for i,d in enumerate(calendar)}
    for e in events:
        if e['kind']!='fill':continue
        s=e['symbol'];qty=float(e['qty']);notional=qty*float(e['price']);fee=float(e['fee'])
        if s not in active:
            if e['side']!='buy':raise ValueError('Episode cannot start with sale without opening inventory')
            counters[s]=counters.get(s,0)+1
            active[s]=dict(symbol=s,episode=counters[s],opened_at=e['asof'],quantity=0.,cashflow=0.,buy_notional=0.,sell_notional=0.,combined_fees_usd=0.,buy_fills=0,sell_fills=0)
        a=active[s];a['combined_fees_usd']+=fee
        if e['side']=='buy':a['quantity']+=qty;a['cashflow']-=notional+fee;a['buy_notional']+=notional;a['buy_fills']+=1
        else:a['quantity']-=qty;a['cashflow']+=notional-fee;a['sell_notional']+=notional;a['sell_fills']+=1
        if a['quantity'] < -1e-7:raise ValueError('Episode becomes short')
        if abs(a['quantity'])<1e-7:
            a.update(status='closed',closed_at=e['asof'],terminal_inventory_value=0.,net_pnl_usd=a['cashflow'],duration_sessions=positions[str(pd.Timestamp(e['asof']))]-positions[str(pd.Timestamp(a['opened_at']))])
            output.append(a);del active[s]
    for s,a in active.items():
        value=a['quantity']*float(final_prices[s])
        a.update(status='open',closed_at=None,terminal_inventory_value=value,net_pnl_usd=a['cashflow']+value,duration_sessions=len(calendar)-1-positions[str(pd.Timestamp(a['opened_at']))])
        output.append(a)
    return output


def episode_supplement():
    registration=json.loads((OUT/'diagnostics/episode_registration.json').read_text())
    assert registration['parent']=='DIAG-SAVED-20260928-01'
    prices=pd.read_parquet(EXP/'prices.parquet');rows=[];metrics=[];attribution=[]
    for scope,period in [('continuous_2024_start','full')]+[('matched_fresh_10000',p) for p in PERIODS]:
        for name in STRATEGIES:
            if period=='full':
                history=pd.read_parquet(EXP/f'{name}_account.parquet');events=json.loads((EXP/f'{name}_events.json').read_text())
            else:
                history=pd.read_parquet(OUT/'diagnostics'/f'{period}_{name}_fresh_account.parquet');events=json.loads((OUT/'diagnostics'/f'{period}_{name}_fresh_events.json').read_text())
            episodes=trade_episodes(events,prices.loc[history.index[-1]].to_dict(),history.index)
            assert abs(sum(e['net_pnl_usd'] for e in episodes)-(history.equity.iloc[-1]-10000))<1e-6
            closed=[e for e in episodes if e['status']=='closed'];opened=[e for e in episodes if e['status']=='open']
            vals=np.asarray([e['net_pnl_usd'] for e in closed],float)
            metrics.append(dict(scope=scope,period=period,strategy=name,closed_episodes=len(closed),open_episodes=len(opened),closed_episode_net_pnl_usd=sum(vals),closed_episode_win_rate=float((vals>0).mean()) if len(vals) else np.nan,closed_episode_profit_factor=float(vals[vals>0].sum()/-vals[vals<0].sum()) if (vals<0).any() else np.nan,terminal_open_episode_pnl_usd=sum(e['net_pnl_usd'] for e in opened),total_pnl_usd=sum(e['net_pnl_usd'] for e in episodes),reconciliation_error=sum(e['net_pnl_usd'] for e in episodes)-(history.equity.iloc[-1]-10000)))
            for e in episodes:rows.append(dict(scope=scope,period=period,strategy=name,**e))
            for symbol in prices:
                part=[e for e in episodes if e['symbol']==symbol]
                attribution.append(dict(scope=scope,period=period,strategy=name,symbol=symbol,total_net_pnl_usd=sum(e['net_pnl_usd'] for e in part),closed_episode_net_pnl_usd=sum(e['net_pnl_usd'] for e in part if e['status']=='closed'),open_episode_net_pnl_usd=sum(e['net_pnl_usd'] for e in part if e['status']=='open'),combined_fees_usd=sum(e['combined_fees_usd'] for e in part)))
    table('round_trip_episodes',rows);table('round_trip_summary',metrics);table('total_pnl_attribution',attribution)
    # Summaries support interpretation without treating partial profit-taking as whole trades.
    summary=json.loads((OUT/'diagnostics/summary.json').read_text())
    summary['portfolio']['round_trips']=metrics
    summary['supplement']=dict(registration=registration,completed_at=datetime.now(timezone.utc).isoformat(),source_sha256=digest(__file__))
    for name in ['round_trip_episodes','round_trip_summary','total_pnl_attribution']:
        p=OUT/'tables'/f'diag_{name}.csv';summary['artifacts'][str(p.relative_to(ROOT))]=digest(p)
    save_json(OUT/'diagnostics/summary.json',summary)
    print('Episode accounting reconciles all20 historical and fresh accounts',flush=True)


def write_diagnostics_report():
    """Generate human-readable interpretation and claim selectors from table rows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((OUT/'diagnostics/summary.json').read_text())
    def read(n):return pd.read_csv(OUT/'tables'/f'diag_{n}.csv')
    forecast=read('forecast_summary');inf=read('paired_inference');econ=read('portfolio');episodes=read('round_trip_summary');stress_data=read('stress');cal=read('calibration_contrast');cover=read('coverage')
    def row(frame,**filters):
        for key,value in filters.items():frame=frame.loc[frame[key]==value]
        if len(frame)!=1:raise ValueError(f'Claim selector resolved {len(frame)} rows: {filters}')
        return frame.iloc[0]
    h=row(forecast,period='audit',model='hybrid');prior=row(forecast,period='audit',model='prior');effect=row(inf,period='audit',model='hybrid',endpoint='log_loss',block_length=10)
    fh=row(econ,scope='matched_fresh_10000',period='audit',strategy='adaptive_hybrid');fe=row(econ,scope='matched_fresh_10000',period='audit',strategy='equal_weight')
    ch=row(econ,scope='continuous_account_period',period='audit',strategy='adaptive_hybrid');ce=row(econ,scope='continuous_account_period',period='audit',strategy='equal_weight')
    full=row(econ,scope='continuous_2024_start',period='full',strategy='adaptive_hybrid');ep=row(episodes,scope='continuous_2024_start',period='full',strategy='adaptive_hybrid');audit_ep=row(episodes,scope='matched_fresh_10000',period='audit',strategy='adaptive_hybrid')
    gap=row(stress_data,scenario='gap_down_10pct',strategy='adaptive_hybrid');delay=row(stress_data,scenario='delay_1',strategy='adaptive_hybrid');cost4=row(stress_data,scenario='cost_4x',strategy='adaptive_hybrid');base=row(stress_data,scenario='base',strategy='adaptive_hybrid');calh=row(cal,period='audit');high=row(cover,period='audit',model='hybrid',threshold=.6)
    claims=[]
    for cid,table_name,selector,fields in [
        ('D01','forecast_summary',dict(period='audit',model='hybrid'),['n','sessions','accuracy','log_loss','brier']),
        ('D02','forecast_summary',dict(period='audit',model='prior'),['n','sessions','accuracy','log_loss','brier']),
        ('D03','paired_inference',dict(period='audit',model='hybrid',endpoint='log_loss',block_length=10),['effect','ci_low','ci_high','holm_p','family_size','mde80_approx']),
        ('D04','portfolio',dict(scope='matched_fresh_10000',period='audit',strategy='adaptive_hybrid'),['net_return','gross_fixed_fill_return','combined_fees_usd','max_drawdown']),
        ('D05','portfolio',dict(scope='matched_fresh_10000',period='audit',strategy='equal_weight'),['net_return','gross_fixed_fill_return','combined_fees_usd','max_drawdown']),
        ('D06','round_trip_summary',dict(scope='continuous_2024_start',period='full',strategy='adaptive_hybrid'),['closed_episodes','open_episodes','closed_episode_win_rate','closed_episode_net_pnl_usd','terminal_open_episode_pnl_usd']),
        ('D07','round_trip_summary',dict(scope='matched_fresh_10000',period='audit',strategy='adaptive_hybrid'),['closed_episodes','open_episodes','closed_episode_win_rate','closed_episode_net_pnl_usd','terminal_open_episode_pnl_usd']),
        ('D08','calibration_contrast',dict(period='audit'),['effect','ci_low','ci_high','same_argmax']),
        ('D09','stress',dict(scenario='delay_1',strategy='adaptive_hybrid'),['net_return','max_drawdown','rejections']),
        ('D10','stress',dict(scenario='cost_4x',strategy='adaptive_hybrid'),['net_return','max_drawdown','rejections']),
        ('D11','stress',dict(scenario='gap_down_10pct',strategy='adaptive_hybrid'),['net_return','max_drawdown','rejections']),
        ('D12','coverage',dict(period='audit',model='hybrid',threshold=.6),['n','sessions','coverage','accuracy'])]:
        r=row(read(table_name),**selector);path=OUT/'tables'/f'diag_{table_name}.csv'
        claims.append(dict(id=cid,artifact=str(path.relative_to(ROOT)),sha256=digest(path),selector=selector,values={k:r[k] for k in fields},interpretation='exploratory consumed history'))
    save_json(OUT/'diagnostics/claims.json',claims)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8))
    axes[0].bar(['Partial sell fills','Closed episodes'],[full.sell_fill_win_rate*100,ep.closed_episode_win_rate*100],color=['#859fb5','#14786e']);axes[0].set_ylim(0,100);axes[0].set_ylabel('Profitable fraction (%)');axes[0].set_title('Hybrid: changing the denominator changes the story')
    for i,(v,n) in enumerate([(full.sell_fill_win_rate,full.sell_fills),(ep.closed_episode_win_rate,ep.closed_episodes)]):axes[0].text(i,v*100+2,f'{v*100:.1f}% | n={int(n)}',ha='center',fontsize=9)
    vals=[audit_ep.closed_episode_net_pnl_usd,audit_ep.terminal_open_episode_pnl_usd,audit_ep.total_pnl_usd];axes[1].bar(['Closed episodes','Open episodes','Net account'],vals,color=['#a8584b','#4776a6','#14786e']);axes[1].axhline(0,color='gray',lw=.8);axes[1].set_ylabel('Net P&L (USD)');axes[1].set_title('Fresh audit account: exact P&L reconciliation')
    fig.tight_layout();fig.savefig(OUT/'figures/diag_error_accounting.png',dpi=200,bbox_inches='tight');fig.savefig(OUT/'figures/diag_error_accounting.pdf',bbox_inches='tight');plt.close(fig)
    ds=summary['dataset'];ver=summary['evidence_verification'];data_cov=read('data_coverage');warmup=int(data_cov.feature_warmup_rows.iloc[0]);complete=int(data_cov.feature_complete_rows.iloc[0])
    report=f'''# Saved-evidence diagnostics — {summary['run_id']}

This is an exploratory follow-up of history already inspected before registration. It supplies dataset, predictive, accounting and robustness evidence; it does not approve a historical trading edge or manufacture prospective observations. The source protocol is `data/research-v2/protocol.json`; the diagnostic registration precedes evaluation and is retained in `data/research-v2/diagnostics/registration.json`.

## Verification and data accounting

All {ver['sealed_files_checked']} baseline sealed-file hashes were checked and preserved. {ver['reproduced_files_checked']} experiment outputs were compared again against the archived reproduction with exact numeric or JSON-object equality, without retraining. The saved JUnit contains {ver['saved_junit']['total']} historical cases and the historical provider exclusion record reports {ver['provider_integrations_unrun_saved']} unrun integrations. These saved tests do not certify changed current source.

There are {ds['raw_dates']} common market sessions × {ds['instruments']} instruments = {ds['raw_symbol_sessions']} raw instrument-session observations. Each symbol loses {warmup} feature warmup rows, leaving {complete:,} complete feature rows. The saved experiment has {ds['decision_dates']} decisions per symbol, {ds['decision_rows']} rows, and {ds['folds']} rolling refits. Of these, {ds['mature_rows']} rows across {ds['mature_dates']} dates have mature labels; {ds['unmatured_rows']} endpoint-unobserved rows are withheld. No missing exchange sessions, duplicate keys or nonfinite adjusted OHLCV rows were found. Tables `diag_data_coverage.csv`, `diag_dataset_flow.csv` and `diag_class_counts.csv` reconcile these quantities.

This is a fixed survivor universe with overlapping ETFs, not historical constituent reconstruction. The archived raw columns contain current-vintage provider prices and adjusted closes; no split/dividend event tape is saved. Adjustment-factor ranges can be checked, but event-level corporate-action accounting cannot be certified. These are total-return proxy units, not broker-share records. Current-vintage adjustments and survivor membership limit external generalization.

## Probability quality and dependence

The consumed audit has {int(h.n)} labeled rows on {int(h.sessions)} dates. Hybrid accuracy is {h.accuracy:.8f}, compared with {prior.accuracy:.8f} for the training-frequency prior. Hybrid log loss is {h.log_loss:.8f}, versus {prior.log_loss:.8f}. The paired daily mean difference is {effect.effect:+.8f}, with block-10 percentile interval [{effect.ci_low:+.8f}, {effect.ci_high:+.8f}], where a negative difference would favor the hybrid. Holm-adjusted p is {effect.holm_p:.6f} across the actual registered {int(effect.family_size)} predictive comparisons. Block lengths 6 and 21 are sensitivity checks. The approximate 80%-power detectable difference is {effect.mde80_approx:.6f} log-loss units, larger than the protocol's practical 0.01 gain. This approximation does not turn the {int(h.sessions)} strongly dependent dates into independent trials.

The predictive family contains four candidates against the prior, two endpoints and three consumed periods. Accuracy comparisons for raw mixture and temperature-calibrated mixture are algebraically identical but retained as registered, making this part of Holm conservative. Directional classification is reported alongside log loss and multiclass Brier; the almost-deterministic always-UP control uses explicit 1e-12 probability clipping, so its extreme log loss is a scoring consequence of certainty, not an equivalent learned probability model.

Calibration lowers audit log loss relative to the uncalibrated arithmetic mixture by {calh.effect:.8f}, interval [{calh.ci_low:.8f}, {calh.ci_high:.8f}]. Temperature leaves all class argmax decisions unchanged. This can support a probability-calibration claim on consumed history, but not an independent gate-benefit claim: the legacy gate and temperature both used the same trailing block. Reliability bins retain support and date-block ratio intervals. Confidence ≥0.60 retains only {int(high.n)} rows on {int(high.sessions)} dates ({high.coverage:.3%} coverage), with accuracy {high.accuracy:.3%}; this is too sparse to advertise as a robust high-confidence trading threshold. No threshold was selected.

Per-class reports and confusion counts, instrument/quarter/regime/fold summaries, model-age support and gate/disagreement slices are exported as separate CSV tables. Regimes use only contemporaneously available SPY trailing return and volatility with a fixed 20% annual volatility cutoff. The slices are descriptive, not an extra family of discovered significance tests. The single-symbol models do not by themselves establish cross-instrument transfer.

## Capital-matched economics and failure accounting

The original continuous audit account period returns remain hybrid {ch.net_return:.5%} and equal-weight {ce.net_return:.5%}. New matched accounts both start the audit with $10,000 in cash: hybrid {fh.net_return:.5%}, equal-weight {fe.net_return:.5%}. Different opening inventory and commissioning costs explain why fresh-period returns are not copied from a segment of the continuous account. All new economic comparisons use these common starting conditions and the same tested release-pinned ledger; all dates and targets are frozen.

Costs comprise 10 basis points commission and 5 basis points additive slippage on filled notional on each side. Fixed-fill gross return adds actual debits to terminal equity; this is distinct from a zero-cost rerun, which can change feasible quantities and rejections. Turnover is two-way absolute filled notional divided by initial capital. Sharpe uses 252 sessions/year and zero risk-free rate; Sortino uses zero minimum acceptable return over all sessions. Cash risk ratios are undefined instead of artificial numerical values.

A high win rate from partial reductions is misleading. The full hybrid record has profitable sell-fill fraction {full.sell_fill_win_rate:.3%}, but only {int(ep.closed_episodes)} fully closed flat-to-flat episodes, with win rate {ep.closed_episode_win_rate:.3%}; {int(ep.open_episodes)} episodes remain open. The fresh audit has {int(audit_ep.closed_episodes)} closed episodes with win rate {audit_ep.closed_episode_win_rate:.3%}, net closed P&L ${audit_ep.closed_episode_net_pnl_usd:.2f}, and {int(audit_ep.open_episodes)} open episodes with terminal marked P&L ${audit_ep.terminal_open_episode_pnl_usd:.2f}. Their sum exactly reconciles the net account loss ${audit_ep.total_pnl_usd:.2f}. Entry and exit costs are included. Equal-weight has no fully closed episodes and therefore no defined closed-trade win rate. Tables `diag_round_trip_episodes.csv`, `diag_round_trip_summary.csv` and `diag_total_pnl_attribution.csv` expose these denominators.

Worst-loss and high-confidence error cases follow registered ranking rules; a representative error is selected nearest the within-class error-loss median, with a chronological tie-break. They explain how outcomes occurred without implying a causal market mechanism or choosing anecdotes to imply profitability.

## Frozen-decision robustness

The fresh-audit hybrid baseline return is {base.net_return:.5%}; a one-session submission delay changes it to {delay.net_return:.5%}, and fourfold costs to {cost4.net_return:.5%}. A synthetic one-session 10% mark shock and next-session rebound leaves terminal wealth unchanged in this realized target path but worsens maximum drawdown from {base.max_drawdown:.4%} to {gap.max_drawdown:.4%}; unchanged final return is not proof of gap resilience. Partial-fill, gross-cap and reservation-buffer tests also change account execution while preserving model forecasts. Removing the reservation buffer can improve a realized path by rejecting more buys; that contingent selection effect is not a new preferred policy. Every perturbation is listed, including unfavorable outcomes. Injecting one missing SPY bar halts all four replays before execution.

These synthetic scenarios do not model a quote book or estimate event probabilities. No stress result changes the frozen prospective candidate. Report `diag_stress.csv` with its capital scope and include both final wealth and drawdown.

## Reproduction and scope

Run from the repository root with `/Users/a/miniconda3/bin/python -m research_v2.diagnostics`, followed by `/Users/a/miniconda3/bin/python -c 'from research_v2.diagnostics import episode_supplement, write_diagnostics_report; episode_supplement(); write_diagnostics_report()'`. Run `/Users/a/miniconda3/bin/python -m pytest tests/test_research_v2_diagnostics.py -q`. The source and archived inputs, every table and plot, the diagnostic claims map, run log, JUnit, and runtime metadata are retained in `data/research-v2/diagnostics/` and the referenced tables/figures directories. The baseline verification intentionally fails if any sealed input changes. Reproduction writes only the unsealed research-v2 output workspace; preserve a dated copy before intentionally regenerating it.

Historical diagnostics are complete within their registered scope. Missing corporate-action events and historical membership are explicit limitations; independent instrument transfer, refitted ablations, prospective replication and operational readiness belong to their separate trials. Prospective outcomes require actual future sessions and six-session label maturity. No evidence here approves external broker orders.
'''
    (ROOT/'docs/research-v2/DIAGNOSTICS.md').write_text(report)
    summary['report_source_sha256']=digest(__file__)
    for p in [OUT/'diagnostics/claims.json',OUT/'figures/diag_error_accounting.png',OUT/'figures/diag_error_accounting.pdf',ROOT/'docs/research-v2/DIAGNOSTICS.md']:
        summary['artifacts'][str(p.relative_to(ROOT))]=digest(p)
    save_json(OUT/'diagnostics/summary.json',summary)
