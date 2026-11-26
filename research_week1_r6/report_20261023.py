"""Validate and summarize observed pilot evidence; refuse premature final freezing.

No data requests, orders, collection events or changes to the account workspace.
The registered comparator is replayed from the exact observed decision snapshots.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RELEASE=ROOT/'data/releases/research-v2-runtime-20261022-week1-r6'
PROTOCOL=ROOT/'data/research-week1-r6/protocol.json'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=str))


def validate_output_location(release,workspace,output):
    release,workspace,output=map(lambda p:Path(p).resolve(),[release,workspace,output])
    protected_paths=[release,workspace]
    if PROTOCOL.exists():
        for item in read(PROTOCOL).get('continuation',{}).get('prior_segments',[]):
            protected_paths.extend([Path(item['prior_release']).resolve(),Path(item['prior_workspace']).resolve()])
    for protected in protected_paths:
        if output==protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Report output must be separate from the pilot workspace and sealed release')
    if output.exists():raise ValueError('Output must be a new versioned directory')


@contextmanager
def readonly_snapshot_lock(workspace):
    # Never create a lock or initialize state. The operational writer uses an
    # exclusive flock on this same persistent file; a reader blocks mutations.
    with (Path(workspace)/'.writer.lock').open('rb') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Collector is writing; retry the report after its receipt completes') from exc
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def require_complete(status,tail_count,halted):
    if (status['valid_observed_sessions']!=status['required_sessions']
        or status['label_complete_decisions']!=status['required_sessions']
        or status['missing_sessions'] or not status['ledger_consistent']
        or tail_count!=6 or halted):
        raise ValueError('Final freeze requires every registered decision, all mature outcomes, six terminal receipts, reconciled ledgers and no unresolved halt')


def predictive_summary(rows,final=False):
    from research_v2.diagnostics import probability_metrics,block_effect
    if not rows:return dict(status='unobserved',labeled_rows=0,decision_dates=0,metrics=[],paired_inference=[])
    df=pd.DataFrame(rows); y=df.label.to_numpy(int)
    metrics=[];loss={}
    for model in ['hybrid','prior']:
        p=np.stack(df[model]);metrics.append(dict(model=model,**probability_metrics(y,p)))
        loss[model]=-np.log(np.maximum(p[np.arange(len(y)),y],1e-12))
    df['delta']=loss['hybrid']-loss['prior']
    daily=df.groupby('session').delta.mean()
    # Predeclared interim display is descriptive; inference is retained for final analysis.
    inference=[block_effect(daily.to_numpy(),b) for b in [6,10,21]] if final and len(daily)>=21 else []
    return dict(status='observed_final' if final else 'observed_interim',labeled_rows=len(df),decision_dates=len(daily),metrics=metrics,
                paired_inference=inference,multiplicity='One predeclared hybrid-minus-prior endpoint; block lengths are sensitivity analyses, primary=10')


def require_week_complete(state):
    if (state['required_sessions'] != 1 or state['valid_observed_sessions'] != 1
        or state['missing_sessions'] or not state['ledger_consistent'] or state['halted']):
        raise ValueError('Week-close report requires one valid new decision, reconciled ledgers and no unresolved halt')


def analyze(release,workspace,output,freeze=False,week_close=False):
    release,workspace,output=map(lambda p:Path(p).resolve(),[release,workspace,output])
    validate_output_location(release,workspace,output)
    sys.dont_write_bytecode=True
    with readonly_snapshot_lock(workspace):
        return _analyze_locked(release,workspace,output,freeze,week_close)


def _analyze_locked(release,workspace,output,freeze,week_close=False):
    # Import the sealed runtime itself. It verifies its own files and exact dependencies.
    if any(k.startswith('runtime_v2') for k in sys.modules):
        raise RuntimeError('Use a fresh process so the selected sealed runtime owns imports')
    sys.path.insert(0,str(release))
    from runtime_v2 import operations as op
    from runtime_v2.model import AdaptiveModel
    from runtime_v2.account import Account,ExecutionSession
    from runtime_v2.risk import limits,constrain_weights
    config=op.verify(release);state=op.status(release,workspace,validated_config=config)
    if config['trial_id'] != 'PILOT-TREND-WEEK1-R6-20261022' or config['sessions'] != 1:
        raise ValueError('This report requires the registered single-decision continuation')
    if digest(PROTOCOL) != config['protocol_sha256']:
        raise ValueError('Continuation protocol differs from sealed configuration')
    if week_close: require_week_complete(state)
    decisions=op._valid_results(workspace,config)
    decisions=[r for r in decisions if r['mode']=='prospective_shadow']
    outcomes=op.valid_outcomes(workspace,config,decisions)
    # r2 includes validated terminal mark/fill receipts, distinct from pilot decisions.
    tail=op.valid_tail_results(workspace,config,decisions) if hasattr(op,'valid_tail_results') else []
    if freeze:require_complete(state,len(tail),(workspace/'halt.json').exists())
    prior=prior_segments(config,decisions+tail)
    failures=failed_continuations()
    if freeze and prior['mature_decisions'] != 4:
        raise ValueError('Combined final report requires all four archived decision outcomes too')
    by_session={r['session']:r for r in decisions};models={};rows=[]
    for outcome in outcomes:
        decision=by_session[outcome['decision_session']]
        signals={s['symbol']:s for s in decision['signals']}
        for actual in outcome['outcomes']:
            symbol=actual['symbol'];location=decision['models'][symbol]
            if location not in models:models[location]=AdaptiveModel.load(location)
            rows.append(active_prediction_row(decision,actual,models[location].prior.tolist()))
    rows.extend(prior['prediction_rows'])
    predictive=predictive_summary(rows,final=False)
    predictive['multiplicity']='No inferential testing in this interrupted five-observation study'
    if rows and freeze: predictive['status']='observed_final_descriptive'
    # Benchmark uses identical capital, observed proxy marks, risk limits, weekly
    # target timing, next-close fills and recorded-vintage covariance inputs.
    benchmark=Account(config['symbols'],config['capital'],limits())
    executor=ExecutionSession(benchmark);history=[]
    for receipt in decisions+tail:
        weights=None
        if receipt in decisions and receipt['rebalance']:
            snapshot=Path(receipt['snapshot'])
            prices=pd.DataFrame({s:pd.read_parquet(snapshot/f'{s}.parquet').Close for s in config['symbols']})
            prices=prices.loc[:pd.Timestamp(receipt['session'],tz='UTC')]
            weights=constrain_weights(pd.Series(.8/len(config['symbols']),index=config['symbols']),prices.pct_change(fill_method=None)).to_dict()
        bench=executor.on_close(receipt['information_cutoff'],receipt['marks'],weights)
        history.append(dict(session=receipt['session'],mode=receipt['mode'],shadow=receipt['shadow']['equity'],
                            cash=receipt['deployment']['equity'],equal_weight=bench['equity'],
                            equal_weight_fees=bench['fees'],equal_weight_reconciliation=bench['reconciliation_error']))
    economics=dict(status='unobserved',observed_sessions=0,comparisons=[])
    if history:
        from research_v2.diagnostics import block_effect
        h=pd.DataFrame(history).set_index('session');returns=h[['shadow','cash','equal_weight']].pct_change()
        returns.iloc[0]=h[['shadow','cash','equal_weight']].iloc[0]/config['capital']-1
        delta=(returns.shadow-returns.equal_weight).to_numpy()
        economics=dict(status='observed_final' if freeze else 'observed_interim',observed_sessions=len(h),decision_sessions=len(decisions),terminal_mark_sessions=len(tail),
            net_returns=(h[['shadow','cash','equal_weight']].iloc[-1]/config['capital']-1).to_dict(),
            comparisons=[],
            benchmark_contract='0.8/N target, same per-side costs and portfolio limits, same first/weekly decision timing, same observed snapshot covariance scaling and adjusted proxy marks; terminal sessions hold without new targets',
            multiplicity='Descriptive trend-shadow-minus-costed-equal-weight endpoint; no p-values or automatic promotion')
    output.mkdir(parents=True)
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),status='frozen_descriptive_observation' if freeze else ('week_complete_outcomes_pending' if week_close else 'interim'),
                release=str(release),workspace=str(workspace),release_sha256=digest(release/'manifest.json'),
                protocol_sha256=digest(PROTOCOL),status_snapshot=state,predictive=predictive,economic=economics,
                prior_segments=prior['segments'],failed_continuations=failures,earlier_failed_trials=read(PROTOCOL)['continuation'].get('earlier_failed_trials',[]),total_valid_observed_decisions=prior['valid_decisions']+len(decisions),
                interrupted_design=True,continuous_five_session_success=False,
                return_series_policy='Separate segment accounts; never compound or splice old and new account returns',
                readiness=dict(engineering='See source-bound release qualification',historical_edge='not_approved',
                               prospective='observations_complete_not_economic_approval' if freeze else 'pending',broker='disabled'),
                source={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'research_v2/diagnostics.py']},
                operational_complete=state['valid_observed_sessions']==1 and not state['missing_sessions'] and state['ledger_consistent'] and not state['halted'],
                inference_scope='Descriptive only; five dates are insufficient for the registered dependence-aware inference; no p-values or economic approval',
                limitations=['No outcome imputation or backfill','Current-vintage adjusted-return proxy units','Five observations in three segments cannot establish durable returns or sustained operational reliability'])
    write(output/'report.json',report);write(output/'prediction_rows.json',rows)
    pd.DataFrame(history).to_csv(output/'account_comparison.csv',index=False)
    write(output/'benchmark_events.json',benchmark.events)
    if freeze:
        # Preserve copies as evidence; absolute paths inside receipts retain their
        # original identity. This is an analysis archive, not a relocated live account.
        shutil.copytree(release,output/'sealed-release',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(workspace,output/'observed-state',ignore=shutil.ignore_patterns('backups','commissioning','*.lock','__pycache__'))
        for item in read(PROTOCOL)['continuation']['prior_segments']:
            target=output/'archived-segments'/item['segment']
            shutil.copytree(item['prior_release'],target/'sealed-release',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            shutil.copytree(item['prior_workspace'],target/'observed-state',ignore=shutil.ignore_patterns('backups','commissioning','*.lock','__pycache__'))
    write(output/'manifest.json',dict(files={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
                                    final=freeze,restore_note='Archive retains original receipt paths; operational restoration uses the tested exact-original-path procedure'))
    print(json.dumps(dict(output=str(output),status=report['status'],valid_sessions=state['valid_observed_sessions'],mature_decisions=state['label_complete_decisions'])))
    return report


def active_prediction_row(decision,actual,prior):
    if decision['trial_id'] != 'PILOT-TREND-WEEK1-R6-20261022':
        raise ValueError('Active outcome belongs to a different trial')
    signals={s['symbol']:s for s in decision['signals']}
    return dict(session=decision['session'],segment='r6',symbol=actual['symbol'],
                label=actual['label'],realized_return=actual['realized_return'],
                hybrid=signals[actual['symbol']]['probabilities'],prior=prior)


def prior_segment(config,receipts,previous):
    # Validate the parent in a separate process: its sealed runtime owns imports.
    import subprocess
    release=Path(previous['prior_release']); workspace=Path(previous['prior_workspace'])
    if digest(release/'manifest.json') != previous['prior_release_sha256']:
        raise ValueError('Parent release changed')
    result=subprocess.run([sys.executable,'-B',str(release/'scripts/collector_v2.py'),'status','--release',str(release),'--workspace',str(workspace)],check=True,capture_output=True,text=True)
    state=json.loads(result.stdout)
    if state['valid_observed_sessions']!=2 or not state['ledger_consistent'] or not state['halted']:
        raise ValueError('Archived prior segment differs from registered evidence')
    previous_rows=[]; outcomes=[]; mature_dates=[]; prediction_rows=[]; models={}
    from runtime_v2.calendar import calendar
    from runtime_v2 import market
    from runtime_v2.model import AdaptiveModel
    cal=calendar()
    for date,sha in previous['prior_receipt_sha256'].items():
        path=workspace/'days'/date/'result.json'
        if digest(path)!=sha:raise ValueError('Archived decision receipt changed')
        r=read(path); previous_rows.append(dict(session=date,receipt=str(path),sha256=sha,shadow=r['shadow'],deployment=r['deployment']))
        pos=cal.sessions.get_loc(pd.Timestamp(date)); entry=cal.sessions[pos+1]; end=cal.sessions[pos+6]
        eligible=sorted([x for x in receipts if pd.Timestamp(x['session'])>=end],key=lambda x:x['session'])
        if not eligible:continue
        source=eligible[0]; data,_=market.load(Path(source['snapshot']),config['symbols'],source['session'])
        if digest(Path(source['snapshot'])/'manifest.json')!=source['snapshot_sha256']:raise ValueError('Label source changed')
        signals={item['symbol']:item for item in r['signals']}
        for symbol,frame in data.items():
            ret=float(frame.loc[end.tz_localize('UTC'),'Close']/frame.loc[entry.tz_localize('UTC'),'Close']-1)
            outcomes.append(dict(decision_session=date,symbol=symbol,realized_return=ret,label=0 if ret < -.01 else (2 if ret > .01 else 1),endpoint=str(end.date()),source_snapshot=source['snapshot'],source_snapshot_sha256=source['snapshot_sha256']))
            location=r['models'][symbol]
            if location not in models:models[location]=AdaptiveModel.load(location)
            prediction_rows.append(dict(session=date,segment=previous['segment'],symbol=symbol,label=outcomes[-1]['label'],realized_return=ret,hybrid=signals[symbol]['probabilities'],prior=models[location].prior.tolist()))
        mature_dates.append(date)
    return dict(trial_id=state['trial_id'],status='preserved_failed_segment',valid_decisions=2,mature_decisions=len(mature_dates),missing_sessions=state['missing_sessions'],halt_preserved=True,receipts=previous_rows,outcomes=outcomes,prediction_rows=prediction_rows,segment=previous['segment'],
                net_returns_at_last_actual_receipt={name:previous_rows[-1][name]['equity']/config['capital']-1 for name in ['shadow','deployment']},
                limitation='Two earlier forecasts remain prospective; later label derivation is analysis only. Missed trades are never fabricated; old portfolio accounting stops at its last actual receipt.')


def prior_segments(config,receipts):
    previous=read(PROTOCOL)['continuation']['prior_segments']
    dates=[d for segment in previous for d in segment['prior_receipt_sha256']]
    if len(previous)!=2 or len(dates)!=4 or len(set(dates))!=4:
        raise ValueError('Exactly four distinct archived decision dates across two segments required')
    segments=[prior_segment(config,receipts,item) for item in previous]
    if sum(s['valid_decisions'] for s in segments)!=4:
        raise ValueError('Archived observation count changed')
    return dict(segments=segments,valid_decisions=4,mature_decisions=sum(s['mature_decisions'] for s in segments),
                prediction_rows=[row for s in segments for row in s['prediction_rows']])


def failed_continuations():
    import subprocess
    rows=[]
    for item in read(PROTOCOL)['continuation']['failed_continuations']:
        release=Path(item['release']); workspace=Path(item['workspace'])
        if digest(release/'manifest.json')!=item['release_sha256']:
            raise ValueError('Superseded release changed')
        result=subprocess.run([sys.executable,'-B',str(release/'scripts/collector_v2.py'),'status','--release',str(release),'--workspace',str(workspace)],check=True,capture_output=True,text=True)
        state=json.loads(result.stdout)
        if state['valid_observed_sessions']!=0 or not state['halted'] or not state['ledger_consistent']:
            raise ValueError('Failed continuation differs from registered evidence')
        rows.append(dict(item,verification_status=state,counts_as_prospective=0,
            note='Missing at supersession: use status_at_supersession; later planned dates were cancelled before their decision windows.'))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,default=DEFAULT_RELEASE)
    p.add_argument('--workspace',type=Path,default=ROOT/'data/paper/research-week1-r6')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',action='store_true');p.add_argument('--week-close',action='store_true');a=p.parse_args()
    analyze(a.release,a.workspace,a.output,a.freeze,a.week_close)
if __name__=='__main__':main()
