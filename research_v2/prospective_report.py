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
DEFAULT_RELEASE=ROOT/'data/releases/research-v2-runtime-20260928-r2'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=str))


def validate_output_location(release,workspace,output):
    release,workspace,output=map(lambda p:Path(p).resolve(),[release,workspace,output])
    for protected in [release,workspace]:
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


def analyze(release,workspace,output,freeze=False):
    release,workspace,output=map(lambda p:Path(p).resolve(),[release,workspace,output])
    validate_output_location(release,workspace,output)
    sys.dont_write_bytecode=True
    with readonly_snapshot_lock(workspace):
        return _analyze_locked(release,workspace,output,freeze)


def _analyze_locked(release,workspace,output,freeze):
    # Import the sealed runtime itself. It verifies its own files and exact dependencies.
    if any(k.startswith('runtime_v2') for k in sys.modules):
        raise RuntimeError('Use a fresh process so the selected sealed runtime owns imports')
    sys.path.insert(0,str(release))
    from runtime_v2 import operations as op
    from runtime_v2.model import AdaptiveModel
    from runtime_v2.account import Account,ExecutionSession
    from runtime_v2.risk import limits,constrain_weights
    config=op.verify(release);state=op.status(release,workspace,validated_config=config)
    decisions=op._valid_results(workspace,config)
    decisions=[r for r in decisions if r['mode']=='prospective_shadow']
    outcomes=op.valid_outcomes(workspace,config,decisions)
    # r2 includes validated terminal mark/fill receipts, distinct from pilot decisions.
    tail=op.valid_tail_results(workspace,config,decisions) if hasattr(op,'valid_tail_results') else []
    if freeze:require_complete(state,len(tail),(workspace/'halt.json').exists())
    by_session={r['session']:r for r in decisions};models={};rows=[]
    for outcome in outcomes:
        decision=by_session[outcome['decision_session']]
        signals={s['symbol']:s for s in decision['signals']}
        for actual in outcome['outcomes']:
            symbol=actual['symbol'];location=decision['models'][symbol]
            if location not in models:models[location]=AdaptiveModel.load(location)
            rows.append(dict(session=decision['session'],symbol=symbol,label=actual['label'],
                             realized_return=actual['realized_return'],hybrid=signals[symbol]['probabilities'],
                             prior=models[location].prior.tolist()))
    predictive=predictive_summary(rows,final=freeze)
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
            comparisons=[block_effect(delta,b) for b in [6,10,21]] if freeze and len(delta)>=21 else [],
            benchmark_contract='0.8/N target, same per-side costs and portfolio limits, same first/weekly decision timing, same observed snapshot covariance scaling and adjusted proxy marks; terminal sessions hold without new targets',
            multiplicity='One predeclared trend-shadow-minus-costed-equal-weight endpoint, primary block=10; no automatic promotion')
    output.mkdir(parents=True)
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),status='frozen_complete_observation' if freeze else 'interim',
                release=str(release),workspace=str(workspace),release_sha256=digest(release/'manifest.json'),
                protocol_sha256=digest(ROOT/'data/research-v2/protocol.json'),status_snapshot=state,predictive=predictive,economic=economics,
                readiness=dict(engineering='See source-bound release qualification',historical_edge='not_approved',
                               prospective='observations_complete_not_economic_approval' if freeze else 'pending',broker='disabled'),
                source={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'research_v2/diagnostics.py']},
                limitations=['No outcome imputation or backfill','Current-vintage adjusted-return proxy units','Date-block inference is conditional on the realized pilot period; 63 decisions do not guarantee power'])
    write(output/'report.json',report);write(output/'prediction_rows.json',rows)
    pd.DataFrame(history).to_csv(output/'account_comparison.csv',index=False)
    write(output/'benchmark_events.json',benchmark.events)
    if freeze:
        # Preserve copies as evidence; absolute paths inside receipts retain their
        # original identity. This is an analysis archive, not a relocated live account.
        shutil.copytree(release,output/'sealed-release',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(workspace,output/'observed-state',ignore=shutil.ignore_patterns('backups','commissioning','*.lock','__pycache__'))
    write(output/'manifest.json',dict(files={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
                                    final=freeze,restore_note='Archive retains original receipt paths; operational restoration uses the tested exact-original-path procedure'))
    print(json.dumps(dict(output=str(output),status=report['status'],valid_sessions=state['valid_observed_sessions'],mature_decisions=state['label_complete_decisions'])))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,default=DEFAULT_RELEASE)
    p.add_argument('--workspace',type=Path,default=ROOT/'data/paper/research-v2')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    analyze(a.release,a.workspace,a.output,a.freeze)
if __name__=='__main__':main()
