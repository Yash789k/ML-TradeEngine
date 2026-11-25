"""Sealed, outcome-only observer. It has no order, account or fitting entry point."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
import numpy as np
import pandas as pd

BASE = Path('/Users/a/Downloads/ml-trade-engine/data/releases/research-v2-runtime-20261022-week1-r6')
sys.path.insert(0, str(BASE))
from runtime_v2 import market, operations as op
from runtime_v2.calendar import calendar
from runtime_v2.storage import digest, read, save, lock

SERVICE = 'local.ml-trade-engine.collector-v2'


def utc(value=None):
    result = pd.Timestamp.now(tz='UTC') if value is None else pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError('Timestamp requires a time zone')
    return result.tz_convert('UTC')


def schedule(first, last):
    cal = calendar()
    return [dict(session=str(d.date()), open=cal.session_open(d).isoformat(),
                 close=cal.session_close(d).isoformat(),
                 collection=(cal.session_close(d)+pd.Timedelta(minutes=30)).isoformat(),
                 collection_hkt=(cal.session_close(d)+pd.Timedelta(minutes=30)).tz_convert('Asia/Hong_Kong').isoformat(),
                 watchdog=(cal.session_close(d)+pd.Timedelta(minutes=90)).isoformat(),
                 observer_open=(cal.session_open(d)+pd.Timedelta(minutes=5)).isoformat(),
                 observer_close=(cal.session_close(d)+pd.Timedelta(minutes=95)).isoformat())
            for d in cal.sessions_in_range(first, last)]


def attempt_slot(row, now):
    minutes = (utc(now)-utc(row['close'])).total_seconds()/60
    for slot, lower, upper in [('initial',30,40),('retry45',45,55),('retry60',60,65)]:
        if lower <= minutes <= upper:
            return slot
    return None


def boundary_window(protocol, now):
    now = utc(now)
    for row in protocol['schedule']:
        if utc(row['open']) <= now <= utc(row['open'])+pd.Timedelta(minutes=15):
            return dict(session=row['session'], kind='open')
        if utc(row['watchdog']) <= now <= utc(row['watchdog'])+pd.Timedelta(minutes=15):
            return dict(session=row['session'], kind='post_close')
    return None


def verify(release):
    release = Path(release).resolve()
    manifest = read(release/'manifest.json')
    if manifest['status'] != 'sealed' or manifest['external_orders'] is not False:
        raise ValueError('Unsealed or order-enabled outcome collector')
    for name, expected in manifest['files'].items():
        if digest(release/name) != expected:
            raise ValueError('Sealed outcome source changed: '+name)
    if digest(Path(__file__).resolve()) != manifest['files']['collector.py']:
        raise ValueError('Executing source differs from sealed outcome collector')
    protocol = read(release/'protocol.json')
    if (protocol['external_orders'] is not False or protocol['create_targets'] is not False
        or protocol['modify_archived_accounts'] is not False or protocol['refit'] is not False):
        raise ValueError('Outcome-only authority exceeded')
    if str(BASE) != protocol['parent_release'] or digest(BASE/'manifest.json') != protocol['parent_manifest_sha256']:
        raise ValueError('Pinned runtime changed')
    if digest(Path(protocol['parent_protocol'])) != protocol['parent_protocol_sha256']:
        raise ValueError('Earlier trial disclosures changed')
    op.verify(BASE)
    rows = protocol['schedule']
    if len(rows) != 6 or rows != schedule(rows[0]['session'], rows[-1]['session']):
        raise ValueError('Six-session calendar schedule changed')
    if len(protocol['decisions']) != 5 or len({x['session'] for x in protocol['decisions']}) != 5:
        raise ValueError('Expected five unique archived decisions')
    cal = calendar()
    for d in protocol['decisions']:
        idx = cal.sessions.get_loc(pd.Timestamp(d['session']))
        if d['entry'] != str(cal.sessions[idx+1].date()) or d['endpoint'] != str(cal.sessions[idx+6].date()):
            raise ValueError('Original forecast horizon changed')
    return protocol


def validate_archives(protocol):
    """Verify old receipts/accounts read-only; never clear their preserved halts."""
    all_results = {}
    summaries = []
    for segment in protocol['segments']:
        rel, ws = Path(segment['release']), Path(segment['workspace'])
        if digest(rel/'manifest.json') != segment['manifest_sha256']:
            raise ValueError('Archived release manifest changed')
        manifest = read(rel/'manifest.json')
        for kind in ['source','artifacts']:
            for name, expected in manifest[kind].items():
                if digest(rel/name) != expected:
                    raise ValueError('Archived release source/artifact changed')
        config = read(rel/'config.json')
        results = op._valid_results(ws, config)
        errors = op.state_errors(ws, results, config)
        if errors:
            raise ValueError('Archived ledger mismatch: '+repr(errors))
        observed = [r for r in results if r['mode']=='prospective_shadow']
        registered = [d for d in protocol['decisions'] if d['segment']==segment['segment']]
        if {r['session'] for r in observed} != {d['session'] for d in registered}:
            raise ValueError('Archived observed dates changed')
        for d in registered:
            if digest(Path(d['receipt'])) != d['receipt_sha256']:
                raise ValueError('Archived decision changed')
            r = next(r for r in observed if r['session']==d['session'])
            if r['content_sha256'] != d['content_sha256'] or r['signals'] != d['signals']:
                raise ValueError('Prediction is not the original sealed forecast')
            all_results[(d['segment'], d['session'])] = r
        latest = observed[-1]
        if {n:latest[n] for n in ['deployment','shadow']} != segment['accounts_at_last_actual_receipt']:
            raise ValueError('Archived account mark changed')
        summaries.append(dict(segment=segment['segment'], decisions=len(observed),
            last_mark=latest['session'], ledger_consistent_at_last_mark=True,
            net_returns_at_last_actual_receipt={n:latest[n]['equity']/config['capital']-1 for n in ['deployment','shadow']},
            full_account_tail_complete=False))
    return summaries


def derive_outcomes(protocol, frames, source_session):
    """Original-date returns only; later retrieval never moves entry or endpoint."""
    rows = []
    for d in protocol['decisions']:
        if pd.Timestamp(d['endpoint']) > pd.Timestamp(source_session):
            continue
        if len(d['signals']) != len(protocol['symbols']):
            raise ValueError('Forecast universe incomplete')
        signals = {s['symbol']:s for s in d['signals']}
        for symbol in protocol['symbols']:
            frame = frames[symbol]
            entry = float(frame.loc[pd.Timestamp(d['entry'],tz='UTC'),'Close'])
            endpoint = float(frame.loc[pd.Timestamp(d['endpoint'],tz='UTC'),'Close'])
            if not np.isfinite([entry,endpoint]).all() or min(entry,endpoint)<=0:
                raise ValueError('Invalid outcome prices')
            value = endpoint/entry-1
            rows.append(dict(segment=d['segment'], decision_session=d['session'], symbol=symbol,
                entry=d['entry'], endpoint=d['endpoint'], entry_price=entry, endpoint_price=endpoint,
                realized_return=value, label=0 if value<-.01 else (2 if value>.01 else 1),
                hybrid=signals[symbol]['probabilities'], prior=d['priors'][symbol],
                decision_sha256=d['content_sha256'], source_session=source_session,
                measurement='original fixed horizon, later retrieved adjusted prices'))
    return rows


def validated_reads(protocol, workspace):
    receipts = []
    allowed = {r['session']:r for r in protocol['schedule']}
    for path in sorted((Path(workspace)/'reads').glob('*/result.json')):
        r = op.read_record(path)
        date = r['session']
        if date not in allowed or path.parent.name != date or r['trial_id'] != protocol['id']:
            raise ValueError('Unregistered outcome receipt')
        if r['mode'] != 'scheduled_outcome_read' or r['new_decisions'] != 0 or r['account_mutations'] != 0:
            raise ValueError('Outcome read acquired decision/account authority')
        if r['parent_pid'] != 1 or r['service'] != SERVICE:
            raise ValueError('Outcome read lacks launchd provenance')
        if attempt_slot(allowed[date],r['started_at']) != r['slot']:
            raise ValueError('Receipt outside the registered collection window')
        start, end = utc(r['started_at']), utc(r['completed_at'])
        if not start <= end <= utc(allowed[date]['watchdog']) or end > utc():
            raise ValueError('Invalid completion time')
        attempt_path = Path(workspace)/'attempts'/f'{date}-{r["slot"]}.json'
        attempt = op.read_record(attempt_path)
        if any(attempt.get(k) != r.get(k) for k in ['session','slot','started_at','parent_pid','service']):
            raise ValueError('Receipt differs from durable scheduled attempt')
        snapshot = Path(r['snapshot'])
        if not snapshot.resolve().is_relative_to((Path(workspace)/'snapshots').resolve()):
            raise ValueError('Snapshot outside the outcome workspace')
        if digest(snapshot/'manifest.json') != r['snapshot_sha256']:
            raise ValueError('Outcome snapshot manifest changed')
        frames, meta = market.load(snapshot, protocol['symbols'], date, now=end)
        if utc(meta['retrieved_at']) != start or meta['expected_session'] != date:
            raise ValueError('Snapshot date or acquisition time differs from scheduled attempt')
        expected = derive_outcomes(protocol,frames,date)
        if r['outcomes'] != expected:
            raise ValueError('Outcome labels differ from original dates and saved prices')
        receipts.append(r)
    return receipts


def summary(protocol, workspace, receipts, now=None):
    now = utc(now); workspace = Path(workspace)
    observed = {r['session'] for r in receipts}
    mature = {r['decision_session'] for receipt in receipts for r in receipt['outcomes']}
    missing = [r['session'] for r in protocol['schedule'] if utc(r['watchdog'])<=now and r['session'] not in observed]
    future = [r for r in protocol['schedule'] if utc(r['collection'])>now]
    halted = (workspace/'halt.json').exists()
    return dict(trial_id=protocol['id'], decision_dates_preserved=5, new_decisions=0,
        completed_outcome_reads=len(observed), required_outcome_reads=6,
        mature_decision_dates=len(mature), required_mature_decision_dates=5,
        missing_new_sessions=missing, original_missed_outcome_sessions=protocol['original_missed_outcome_sessions'],
        halted=halted, external_orders=False, original_account_tail_complete=False,
        original_operational_gate='incomplete; missed execution and marks remain disclosed',
        amended_outcome_collection_complete=len(observed)==6 and len(mature)==5 and not missing and not halted,
        next_event=future[0] if future else None)


def status(release, workspace):
    protocol = verify(release)
    archives = validate_archives(protocol)
    receipts = validated_reads(protocol, workspace)
    result = summary(protocol, workspace, receipts)
    result['archived_accounts'] = archives
    p = Path(workspace)/'heartbeat.json'
    result['heartbeat'] = read(p) if p.exists() else None
    result['boundary_window'] = boundary_window(protocol,utc())
    return result


def run_cycle(protocol, workspace, now, downloader=market.download):
    """Only the launchd CLI invokes this in production; tests use isolated fixtures."""
    workspace = Path(workspace); now = utc(now)
    existing = validated_reads(protocol,workspace)
    good = {r['session'] for r in existing}
    for row in protocol['schedule']:
        date = row['session']; event = workspace/'events'/f'{date}-watchdog.json'
        if now >= utc(row['watchdog']) and not event.exists():
            op.record(event,dict(session=date,checked_at=now.isoformat(),
                status='valid_outcome_read' if date in good else 'missing_outcome_read',
                action_required=date not in good))
        slot = attempt_slot(row,now)
        attempt = workspace/'attempts'/f'{date}-{slot}.json'
        if not slot or date in good or attempt.exists() or (workspace/'halt.json').exists():
            continue
        started = dict(session=date,slot=slot,started_at=now.isoformat(),parent_pid=os.getppid(),service=os.environ.get('XPC_SERVICE_NAME'),status='started')
        op.record(attempt,started)
        snapshot = workspace/'snapshots'/f'{date}-{slot}-{uuid.uuid4().hex[:8]}'
        try:
            downloader(snapshot,protocol['symbols'],date,now=now)
            completed = utc()
            if completed > utc(row['watchdog']):
                raise ValueError('Download completed after the registered watchdog')
            frames, _ = market.load(snapshot,protocol['symbols'],date,now=completed)
            outcomes = derive_outcomes(protocol,frames,date)
            result = dict(trial_id=protocol['id'],mode='scheduled_outcome_read',**{k:v for k,v in started.items() if k!='status'},
                completed_at=completed.isoformat(),snapshot=str(snapshot.resolve()),snapshot_sha256=digest(snapshot/'manifest.json'),
                outcomes=outcomes,new_decisions=0,account_mutations=0,external_orders=False)
            op.record(workspace/'reads'/date/'result.json',result)
            good.add(date)
        except Exception as exc:
            transient = isinstance(exc,market.TransientDataError)
            op.record(workspace/'failures'/f'{date}-{slot}.json',dict(**started,failed_at=utc().isoformat(),type=type(exc).__name__,error=str(exc),transient=transient))
            if not transient:
                op.record(workspace/'halt.json',dict(at=utc().isoformat(),session=date,error=str(exc),reason='deterministic collection/validation failure'))
    return sorted(good)


def tick(release, workspace):
    if os.getppid()!=1 or os.environ.get('XPC_SERVICE_NAME')!=SERVICE:
        raise ValueError('Only the registered launchd service may collect')
    protocol = verify(release)
    validate_archives(protocol)
    workspace = Path(workspace)
    with lock(workspace):
        now = utc()
        save(workspace/'heartbeat.json',dict(at=now.isoformat(),parent_pid=os.getppid(),pid=os.getpid(),
            service=SERVICE,trial_id=protocol['id'],release=str(Path(release).resolve()),phase='checking'))
        completed = run_cycle(protocol,workspace,now)
        save(workspace/'heartbeat.json',dict(at=utc().isoformat(),parent_pid=os.getppid(),pid=os.getpid(),
            service=SERVICE,trial_id=protocol['id'],release=str(Path(release).resolve()),phase='idle',completed_sessions=completed))
    return dict(completed_sessions=completed)


def metrics(rows, model):
    y = np.array([r['label'] for r in rows],dtype=int)
    p = np.array([r[model] for r in rows],dtype=float)
    if p.shape!=(len(rows),3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Invalid frozen forecast probabilities')
    return dict(model=model,rows=len(y),accuracy=float((p.argmax(axis=1)==y).mean()),
        log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
        brier=float(((p-np.eye(3)[y])**2).sum(axis=1).mean()))


def require_complete(state):
    if not (state['amended_outcome_collection_complete'] and state['completed_outcome_reads']==6
            and state['mature_decision_dates']==5 and not state['halted'] and not state['missing_new_sessions']):
        raise ValueError('Final outcome package requires all six new reads, five mature original-date labels and no halt')


def report(release, workspace, output, final=False):
    release, workspace, output = [Path(p).resolve() for p in [release,workspace,output]]
    protocol = verify(release)
    protected = [release,workspace,BASE] + [Path(x[k]) for x in protocol['segments'] for k in ['release','workspace']]
    if output.exists() or any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Report needs a new output directory outside preserved runtime/evidence')
    archives = validate_archives(protocol)
    receipts = validated_reads(protocol,workspace)
    state = summary(protocol,workspace,receipts)
    if final:
        require_complete(state)
    rows = {}
    for receipt in receipts:
        for row in receipt['outcomes']:
            key = (row['segment'],row['decision_session'],row['symbol'])
            rows.setdefault(key,dict(row,observed_at=receipt['started_at'],
                source_snapshot=receipt['snapshot'],source_snapshot_sha256=receipt['snapshot_sha256']))
    values = list(rows.values())
    result = dict(created_at=utc().isoformat(),status='final_amended_outcome_dataset' if final else 'interim_outcome_recovery',
        status_snapshot=state,protocol_sha256=digest(release/'protocol.json'),manifest_sha256=digest(release/'manifest.json'),
        archived_accounts=archives,metrics=[metrics(values,m) for m in ['hybrid','prior']] if values else [],
        prediction_rows=len(values),decision_dates=len({r['decision_session'] for r in values}),
        original_pilot_operational_complete=False,profitability_approved=False,external_orders=False,
        failed_trials_reference=protocol['parent_protocol'],failed_trials_reference_sha256=protocol['parent_protocol_sha256'],
        original_missed_outcome_sessions=protocol['original_missed_outcome_sessions'],
        limitations=['Five interrupted forecast dates across three accounts; descriptive evidence only',
                    'Forecasts remain prospective; outcomes use explicitly later retrieved adjusted-price vintages',
                    'Old missed executions and terminal account marks cannot be recreated; original operational gate remains incomplete',
                    'Separate archived account returns only; no stitched series, no profitability approval'],
        inputs={str(Path(workspace)/'reads'/r['session']/'result.json'):digest(Path(workspace)/'reads'/r['session']/'result.json') for r in receipts})
    output.mkdir(parents=True)
    save(output/'report.json',result,immutable=True)
    save(output/'prediction_rows.json',values,immutable=True)
    save(output/'protocol.json',protocol,immutable=True)
    save(output/'manifest.json',dict(final_outcomes=final,files={p.name:digest(p) for p in output.iterdir() if p.is_file()}),immutable=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['verify','status','tick','report','window'])
    parser.add_argument('--release',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--workspace',type=Path,default=Path('/Users/a/Downloads/ml-trade-engine/data/paper/research-outcomes-r7'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--final-outcomes',action='store_true')
    parser.add_argument('--at')
    args = parser.parse_args()
    if args.command=='verify': result = verify(args.release)
    elif args.command=='status': result = status(args.release,args.workspace)
    elif args.command=='window': result = boundary_window(verify(args.release),utc(args.at))
    elif args.command=='tick': result = tick(args.release,args.workspace)
    else:
        if args.output is None: parser.error('report requires --output')
        result = report(args.release,args.workspace,args.output,args.final_outcomes)
    print(json.dumps(result,indent=2,allow_nan=False,default=str))


if __name__=='__main__':
    main()
