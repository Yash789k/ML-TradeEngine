"""Durable local-only collector, calendar runner and evidence-derived status."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
import uuid
import pandas as pd
from runtime_v2 import market
from runtime_v2.account import Account
from runtime_v2.calendar import calendar
from runtime_v2.policy import utc, session, plan, commit, weekly
from runtime_v2.risk import limits
from runtime_v2.storage import save, read, digest, lock

CRITICAL_TESTS = ['test_calendar_schedule_dst_holiday', 'test_policy_refits_holiday_future_invariance',
                  'test_restart_account_boundary', 'test_lock_and_storage_failure',
                  'test_scheduler_retries_and_outage', 'test_receipt_tampering', 'test_backup_restore',
                  'test_outcome_provenance_maturity_and_uniqueness','test_underlying_snapshot_and_model_bytes',
                  'test_readonly_status_preserves_truncated_state','test_restore_original_path_and_continue','test_direct_daily_respects_halt',
                  'test_interrupted_refit_freezes_input_before_models','test_readonly_missing_events_table',
                  'test_restore_tail_outcomes_halt_and_terminal_accounting','test_halted_tail_still_attaches_mature_outcomes']


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def record(path, value):
    value = dict(value)
    value['content_sha256'] = content_hash(value)
    save(path, value, immutable=True)
    return value


def read_record(path):
    value = read(path)
    signature = value.pop('content_sha256')
    if content_hash(value) != signature:
        raise ValueError(f'Receipt/plan hash mismatch: {path}')
    value['content_sha256'] = signature
    return value


def verify(release):
    release = Path(release).resolve()
    manifest = read(release / 'manifest.json')
    if manifest['status'] != 'sealed' or manifest['external_orders'] is not False:
        raise ValueError('Unsealed or order-enabled runtime')
    source_root = Path(__file__).resolve().parents[1]
    for relative, expected in manifest['source'].items():
        if digest(release / relative) != expected or digest(source_root / relative) != expected:
            raise ValueError(f'Runtime source differs from sealed manifest: {relative}')
    for relative, expected in manifest['artifacts'].items():
        if digest(release / relative) != expected:
            raise ValueError(f'Sealed artifact changed: {relative}')
    for name, expected in manifest['dependencies'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'Dependency drift: {name}')
    evidence = read(release / 'test-evidence.json')
    if evidence['source'] != manifest['source'] or evidence['failures'] or evidence['errors'] or not set(CRITICAL_TESTS) <= set(evidence['passing_test_names']):
        raise ValueError('Missing critical source-bound test evidence')
    if digest(release / 'tests.xml') != evidence['junit_sha256']:
        raise ValueError('Test receipt mismatch')
    config = read(release / 'config.json')
    if config['external_orders'] is not False or config['deployment'] != 'cash' or config['shadow'] != 'trend':
        raise ValueError('Unsupported pilot selection/order authorization')
    return config


def bind(workspace, release, config):
    path = Path(workspace) / 'binding.json'
    value = dict(release=str(Path(release).resolve()), release_sha256=digest(Path(release) / 'manifest.json'),
                 trial_id=config['trial_id'], config_sha256=digest(Path(release) / 'config.json'))
    save(path, value, immutable=True)
    return value


def completed(now):
    dates = calendar().sessions_in_range((utc(now)-pd.Timedelta(days=10)).date(), utc(now).date())
    return [d for d in dates if calendar().session_close(d) <= utc(now)][-1]


def _valid_results(workspace, config=None):
    results=[]
    workspace=Path(workspace)
    binding=read(workspace/'binding.json') if (workspace/'binding.json').exists() else None
    for path in sorted([*Path(workspace).glob('days/*/result.json'),*Path(workspace).glob('terminal/*/result.json')],key=lambda p:p.parent.name):
        result=read_record(path)
        if binding is None or result.get('binding') != binding or result.get('trial_id') != binding['trial_id']:
            raise ValueError('Receipt release/trial differs from workspace binding')
        if result.get('mode') not in ['prospective_shadow','commissioning_preview','outcome_tail']:
            raise ValueError('Invalid receipt mode')
        if config:
            if result['trial_id'] != config['trial_id'] or set(result['marks']) != set(config['symbols']):
                raise ValueError('Receipt universe/trial mismatch')
            if result['mode']=='prospective_shadow' and session(result['session']) not in calendar().sessions_in_range(config['start'],config['end']):
                raise ValueError('Receipt outside expected pilot sessions')
            if result['mode']=='outcome_tail':
                last=calendar().sessions.get_loc(session(config['end']))
                if session(result['session']) not in calendar().sessions[last+1:last+7] or result['rebalance']:
                    raise ValueError('Invalid terminal mark session or new strategy target')
        for account in ['deployment','shadow']:
            if abs(result[account]['reconciliation_error'])>1e-6 or result[account]['cash'] < -1e-6:
                raise ValueError('Invalid account reconciliation in receipt')
        decision=read_record(path.with_name('plan.json'))
        if result['plan_sha256'] != decision['content_sha256'] or result['binding'] != decision['binding']:
            raise ValueError('Result is not bound to immutable decision plan')
        if any(result.get(k)!=v for k,v in decision.items() if k not in ['content_sha256','timings_seconds']):
            raise ValueError('Result decision fields differ from original immutable plan')
        if result['session'] != path.parent.name:
            raise ValueError('Result session differs from receipt directory')
        if result['snapshot_sha256'] != digest(Path(result['snapshot'])/'manifest.json'):
            raise ValueError('Result input snapshot changed')
        market_meta=read(Path(result['snapshot'])/'manifest.json')
        symbols=config['symbols'] if config else list(result['marks'])
        if market_meta['status']!='complete' or market_meta.get('errors') or set(market_meta['data'])!=set(symbols):
            raise ValueError('Receipt snapshot universe/status invalid')
        for symbol, item in market_meta['data'].items():
            for suffix,key in [('.parquet','sha256'),('-raw.parquet','raw_sha256')]:
                if digest(Path(result['snapshot'])/(symbol+suffix)) != item[key]:
                    raise ValueError('Receipt snapshot bytes changed')
            if item.get('response_sha256') and digest(Path(result['snapshot'])/'responses'/f'{symbol}.json') != item['response_sha256']:
                raise ValueError('Receipt raw provider response changed')
        if len(result['signals']) != len(symbols) or {x['symbol'] for x in result['signals']} != set(symbols):
            raise ValueError('Receipt signal universe mismatch')
        for signal in result['signals']:
            model_path=Path(result['models'][signal['symbol']])
            model_meta=read(model_path/'manifest.json')
            if digest(model_path/'model.joblib') != model_meta['model_sha256']:
                raise ValueError('Receipt model bytes changed')
            if digest(Path(result['models'][signal['symbol']])/'manifest.json') != signal['model_manifest_sha256']:
                raise ValueError('Receipt model artifact changed')
        results.append(result)
    return results


def state_errors(workspace, results, config):
    if not results: return []
    latest=results[-1]; errors=[]
    for name in ['deployment','shadow']:
        path=Path(workspace)/f'{name}.sqlite'
        if not path.exists():
            errors.append(f'{name}: state database missing'); continue
        try:
            # Inspect without creating tables or replacing an empty/truncated state.
            with sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True) as db:
                if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('SQLite integrity check failed')
                state_rows=db.execute('SELECT revision,payload FROM state WHERE id=1').fetchall()
                if len(state_rows)!=1: raise ValueError('Missing persisted account state')
                journal=[json.loads(r[0]) for r in db.execute('SELECT payload FROM events ORDER BY id')]
                payload=json.loads(state_rows[0][1])
            account=Account(config['symbols'],config['capital'],limits())
            account._restore(payload)
            account.events=journal
            marks=[e for e in journal if e['kind']=='mark']
            if len(marks)!=len(account.decisions) or not marks or marks[-1]['asof']!=latest['information_cutoff']:
                raise ValueError('Missing or inconsistent persisted mark journal')
            by_close={e['asof']:e for e in marks}
            for receipt in results:
                stored=by_close.get(receipt['information_cutoff'])
                if stored is None or any(stored.get(k)!=v for k,v in receipt[name].items()):
                    raise ValueError('Historical account receipt differs from persisted mark journal')
            if {e['fill_id'] for e in journal if e['kind']=='fill'} != set(account.fill_ids):
                raise ValueError('Missing persisted fill journal')
            actual=account.snapshot(latest['marks'])
            if account.last_event != latest['information_cutoff'] or actual != latest[name]:
                errors.append(f'{name}: ledger differs from last complete receipt; interrupted plan or state corruption')
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    return errors


def daily(release, workspace, date=None, mode='manual', snapshot=None, now=None, fault=None, validated_config=None):
    """Public entry locks all dual-account and receipt operations together."""
    with lock(workspace):
        return _daily(release, workspace, date, mode, snapshot, now, fault, validated_config)


def _daily(release, workspace, date=None, mode='manual', snapshot=None, now=None, fault=None, validated_config=None):
    t0 = time.perf_counter()
    provider_requests = 0
    release, workspace = Path(release), Path(workspace)
    now = utc(now)
    config = validated_config or verify(release)
    binding = bind(workspace, release, config)
    date = completed(now) if date is None else session(date)
    if calendar().session_close(date) > now:
        raise ValueError('Cannot observe an uncompleted session')
    commissioning = mode == 'commissioning'
    if not commissioning and date not in calendar().sessions_in_range(config['start'],config['end']):
        raise ValueError('Session outside fixed pilot')
    if not commissioning and not (calendar().session_close(date)+pd.Timedelta(minutes=30) <= now <= calendar().session_close(date)+pd.Timedelta(minutes=65)):
        raise ValueError('Decision outside registered close+30..65 operational window; outage halt')
    day = workspace / 'days' / str(date.date())
    if not commissioning and (workspace/'halt.json').exists() and not (day/'result.json').exists():
        raise ValueError('Collector is halted; register resolution before any new daily decision')
    day.mkdir(parents=True, exist_ok=True)
    attempt_id = f'{now.strftime("%Y%m%dT%H%M%S%f")}-{uuid.uuid4().hex[:8]}'
    attempt_path = workspace / 'attempts' / f'{date.date()}-{attempt_id}.json'
    receipt = dict(session=str(date.date()), mode=mode, started_at=now.isoformat(), trial_id=config['trial_id'],
                   host=platform.node(), pid=os.getpid(), parent_pid=os.getppid(),
                   service=os.environ.get('XPC_SERVICE_NAME'), binding=binding, status='started')
    save(attempt_path, receipt)
    try:
        if (day / 'result.json').exists():
            results=_valid_results(workspace,config)
            result=next(r for r in results if r['session']==str(date.date()))
            errors=state_errors(workspace,results,config)
            if errors: raise ValueError('; '.join(errors))
            data,_=market.load(Path(result['snapshot']),config['symbols'],date,now)
            mature(workspace,data,date,config,snapshot=result['snapshot'])
            save(workspace/'latest.json',result)
            receipt.update(status='idempotent', result_sha256=result['content_sha256'])
            save(attempt_path, receipt)
            return result
        prior_results = [r for r in _valid_results(workspace,config) if r['session'] < str(date.date())]
        prior = prior_results[-1] if prior_results else None
        if not (day/'plan.json').exists():
            errors=state_errors(workspace,prior_results,config)
            if errors: raise ValueError('; '.join(errors))
        if not commissioning:
            if prior is None and date != session(config['start']):
                raise ValueError('First expected session absent; cannot start a backfilled pilot')
            if prior is not None and calendar().next_session(session(prior['session'])) != date:
                raise ValueError('Missed expected session; explicit outage halt')
        if (day / 'plan.json').exists():
            decision = read_record(day / 'plan.json')
            if decision['binding'] != binding:
                raise ValueError('Plan has different release binding')
            snapshot = Path(decision['snapshot'])
            data, _ = market.load(snapshot, config['symbols'], date, now)
            if digest(snapshot / 'manifest.json') != decision['snapshot_sha256']:
                raise ValueError('Plan input snapshot changed')
        else:
            t = time.perf_counter()
            input_path=day/'input.json'
            if input_path.exists():
                inputs=read_record(input_path)
                if inputs['binding']!=binding or inputs['session']!=str(date.date()):
                    raise ValueError('Frozen input binding mismatch')
                if snapshot is not None and str(Path(snapshot).resolve())!=inputs['snapshot']:
                    raise ValueError('Interrupted refit cannot resume with different input snapshot')
                snapshot=Path(inputs['snapshot'])
                if digest(snapshot/'manifest.json')!=inputs['snapshot_sha256']:
                    raise ValueError('Frozen input snapshot manifest changed')
            if snapshot is None:
                snapshot = day / ('market-' + attempt_id)
                provider_requests = len(config['symbols'])
                market.download(snapshot, config['symbols'], date, now)
            data, _ = market.load(snapshot, config['symbols'], date, now)
            download_time = time.perf_counter() - t
            if not input_path.exists():
                record(input_path,dict(binding=binding,session=str(date.date()),snapshot=str(Path(snapshot).resolve()),
                                       snapshot_sha256=digest(Path(snapshot)/'manifest.json')))
            if fault: fault('after_input')
            models = prior['models'] if prior else {s:str(release / 'models' / s) for s in config['symbols']}
            decision = plan(data, models, date, config, day / 'models', prior)
            decision.update(binding=binding, prepared_at=now.isoformat(), snapshot=str(Path(snapshot).resolve()),
                            snapshot_sha256=digest(Path(snapshot) / 'manifest.json'),
                            mode='commissioning_preview' if commissioning else 'prospective_shadow',
                            trigger_mode=mode, trial_id=config['trial_id'])
            decision['timings_seconds']['data_read'] = download_time
            if fault:
                fault('before_plan')
            decision = record(day / 'plan.json', decision)
        if fault:
            fault('after_plan')
        decision_for_commit = {k:v for k,v in decision.items() if k != 'content_sha256'}
        t = time.perf_counter()
        if commissioning:
            # Separate commissioning state proves full ledger path without adding pilot observations.
            result = commit(decision_for_commit, workspace, config, fault)
        else:
            result = commit(decision_for_commit, workspace, config, fault)
        result['timings_seconds'] = dict(result['timings_seconds'], ledger=time.perf_counter()-t,
                                         whole_batch=time.perf_counter()-t0)
        result['completed_at'] = utc().isoformat()
        result['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
        result['hardware'] = dict(platform=platform.platform(), machine=platform.machine(), python=sys.version)
        result['provider_request_count'] = provider_requests
        result['plan_sha256'] = decision['content_sha256']
        result['external_orders'] = False
        if fault:
            fault('before_receipt')
        result = record(day / 'result.json', result)
        save(workspace / 'latest.json', result)
        if fault:
            fault('after_receipt')
        mature(workspace, data, date, config, snapshot=decision['snapshot'])
        receipt.update(status='complete', result_sha256=result['content_sha256'], elapsed_seconds=time.perf_counter()-t0)
        save(attempt_path, receipt)
        return result
    except Exception as exc:
        receipt.update(status='failed', error=type(exc).__name__, message=str(exc), transient=isinstance(exc,market.TransientDataError),
                       elapsed_seconds=time.perf_counter()-t0, finished_at=utc().isoformat(),provider_request_count=provider_requests)
        save(attempt_path, receipt)
        raise


def valid_outcomes(workspace, config, results=None):
    """Each outcome is a unique matured decision joined to actual immutable inputs."""
    workspace=Path(workspace)
    results=_valid_results(workspace,config) if results is None else results
    decisions={r['session']:r for r in results if r['mode']=='prospective_shadow'}
    binding=read(workspace/'binding.json') if (workspace/'binding.json').exists() else None
    output=[]; seen=set(); snapshots={}
    for path in sorted((workspace/'outcomes').glob('*.json')):
        value=read_record(path);d=value.get('decision_session')
        if d not in decisions or d in seen or path.stem!=d:
            raise ValueError('Outcome is not a unique expected prospective decision')
        seen.add(d);decision=decisions[d]
        if value.get('binding')!=binding or value.get('trial_id')!=config['trial_id'] or value.get('decision_sha256')!=decision['content_sha256']:
            raise ValueError('Outcome trial/release/decision linkage invalid')
        idx=calendar().sessions.get_loc(session(d))
        entry,endpoint=calendar().sessions[idx+1],calendar().sessions[idx+6]
        if value.get('entry')!=str(entry.date()) or value.get('endpoint')!=str(endpoint.date()):
            raise ValueError('Outcome maturity endpoint differs from contract')
        source=session(value['source_session'])
        last=calendar().sessions.get_loc(session(config['end']))
        if source<endpoint or source>calendar().sessions[last+6] or utc(value['observed_at'])<calendar().session_close(source) or utc(value['observed_at'])>utc():
            raise ValueError('Outcome was observed before maturity or outside allowed source period')
        location=Path(value['source_snapshot'])
        if digest(location/'manifest.json')!=value['source_snapshot_sha256']:
            raise ValueError('Outcome source snapshot manifest changed')
        key=str(location)
        if key not in snapshots:
            meta=read(location/'manifest.json')
            if meta['status']!='complete' or meta.get('errors') or set(meta['data'])!=set(config['symbols']):
                raise ValueError('Outcome source snapshot incomplete')
            frames={}
            for symbol,item in meta['data'].items():
                for suffix,hashkey in [('.parquet','sha256'),('-raw.parquet','raw_sha256')]:
                    if digest(location/(symbol+suffix))!=item[hashkey]:
                        raise ValueError('Outcome source data bytes changed')
                if item.get('response_sha256') and digest(location/'responses'/f'{symbol}.json')!=item['response_sha256']:
                    raise ValueError('Outcome provider response changed')
                frame=pd.read_parquet(location/f'{symbol}.parquet',columns=['Close'])
                if not frame.index.is_unique or not frame.index.is_monotonic_increasing or session(frame.index[-1])!=source:
                    raise ValueError('Outcome source sessions invalid')
                frames[symbol]=frame
            snapshots[key]=frames
        frames=snapshots[key]
        rows=value['outcomes']
        if len(rows)!=len(config['symbols']) or {r['symbol'] for r in rows}!=set(config['symbols']):
            raise ValueError('Outcome universe contains missing or duplicate symbols')
        for row in rows:
            f=frames[row['symbol']]
            realized=float(f.loc[endpoint.tz_localize('UTC'),'Close']/f.loc[entry.tz_localize('UTC'),'Close']-1)
            label=2 if realized>.01 else 0 if realized<-.01 else 1
            if abs(float(row['realized_return'])-realized)>1e-12 or row['label']!=label:
                raise ValueError('Outcome label/return differs from source prices')
        output.append(value)
    return output


def mature(workspace, data, current, config, snapshot):
    results=_valid_results(workspace,config)
    # Existing self-hashes do not excuse an invalid maturity or source linkage.
    existing={r['decision_session'] for r in valid_outcomes(workspace,config,results)}
    for r in results:
        if r['mode'] != 'prospective_shadow': continue
        index=calendar().sessions.get_loc(session(r['session']))
        entry,endpoint=calendar().sessions[index+1],calendar().sessions[index+6]
        if endpoint>session(current) or r['session'] in existing: continue
        rows=[]
        for symbol in config['symbols']:
            ret=float(data[symbol].loc[endpoint.tz_localize('UTC'),'Close']/data[symbol].loc[entry.tz_localize('UTC'),'Close']-1)
            rows.append(dict(symbol=symbol,realized_return=ret,label=2 if ret>.01 else 0 if ret<-.01 else 1))
        record(Path(workspace)/'outcomes'/f'{r["session"]}.json',dict(decision_session=r['session'],entry=str(entry.date()),
               endpoint=str(endpoint.date()),observed_at=utc().isoformat(),source_session=str(session(current).date()),
               source_snapshot=str(Path(snapshot).resolve()),source_snapshot_sha256=digest(Path(snapshot)/'manifest.json'),
               binding=r['binding'],trial_id=config['trial_id'],decision_sha256=r['content_sha256'],outcomes=rows))


def valid_tail_results(workspace, config, decisions):
    """Validated terminal receipts for the prospective reporting/freeze contract."""
    tail=[r for r in _valid_results(workspace,config) if r['mode']=='outcome_tail']
    if not tail: return []
    if not any(r['mode']=='prospective_shadow' and r['session']==config['end'] for r in decisions):
        raise ValueError('Terminal evidence lacks final registered decision')
    idx=calendar().sessions.get_loc(session(config['end']))
    expected=[str(d.date()) for d in calendar().sessions[idx+1:idx+1+len(tail)]]
    if [r['session'] for r in tail]!=expected:
        raise ValueError('Terminal evidence has missing or duplicate sessions')
    return tail


def terminal_close(workspace, data, date, snapshot, config, now):
    """Six post-cutoff closes fill remaining orders and mark holdings, without alpha targets."""
    workspace=Path(workspace);date=session(date);day=workspace/'terminal'/str(date.date())
    if (workspace/'halt.json').exists(): raise ValueError('Halted pilot cannot process terminal fills')
    existing=_valid_results(workspace,config)
    if not existing or not any(r['session']==config['end'] and r['mode']=='prospective_shadow' for r in existing):
        raise ValueError('Missing final pilot decision; terminal accounting unavailable')
    completed_path=day/'result.json'
    if completed_path.exists():
        result=read_record(completed_path)
        errors=state_errors(workspace,existing,config)
        if errors: raise ValueError('; '.join(errors))
        if result['session']==existing[-1]['session']: save(workspace/'latest.json',result)
        return result
    prior=existing[-1]
    if calendar().next_session(session(prior['session']))!=date:
        raise ValueError('Missing terminal close; no backfilled fills')
    plan_path=day/'plan.json'
    if plan_path.exists():
        decision=read_record(plan_path)
        data,_=market.load(decision['snapshot'],config['symbols'],date,now)
        if digest(Path(decision['snapshot'])/'manifest.json')!=decision['snapshot_sha256']:
            raise ValueError('Terminal input snapshot changed')
    else:
        errors=state_errors(workspace,existing,config)
        if errors: raise ValueError('; '.join(errors))
        marks={s:float(prior['marks'][s]*f.loc[date.tz_localize('UTC'),'Close']/f.loc[session(prior['session']).tz_localize('UTC'),'Close']) for s,f in data.items()}
        decision=dict(session=str(date.date()),information_cutoff=calendar().session_close(date).isoformat(),mode='outcome_tail',
              trial_id=config['trial_id'],binding=prior['binding'],snapshot=str(Path(snapshot).resolve()),snapshot_sha256=digest(Path(snapshot)/'manifest.json'),
              refit=False,rebalance=False,marks=marks,shadow_weights=prior['shadow_weights'],models=prior['models'],signals=prior['signals'],
              carried_forecast_session=config['end'],prepared_at=utc(now).isoformat(),timings_seconds={})
        decision=record(plan_path,decision)
    payload={k:v for k,v in decision.items() if k!='content_sha256'}
    result=commit(payload,workspace,config)
    result.update(completed_at=utc().isoformat(),plan_sha256=decision['content_sha256'],external_orders=False)
    result=record(completed_path,result)
    save(workspace/'latest.json',result)
    return result


def events(config):
    cal=calendar()
    dates = cal.sessions_in_range(config['start'],config['end'])
    output=[]
    for n,date in enumerate(dates,1):
        close=cal.session_close(date)
        for minutes,kind in [(30,'daily'),(45,'retry45'),(60,'retry60'),(90,'watchdog')]:
            output.append(dict(id=f'{date.date()}:{kind}', session=str(date.date()), kind=kind,
                               due=close+pd.Timedelta(minutes=minutes), ordinal=n))
        nxt=cal.next_session(date)
        output.append(dict(id=f'{date.date()}:preopen',session=str(date.date()),kind='preopen',ordinal=n,
                           due=cal.session_open(nxt)-pd.Timedelta(minutes=30)))
    # After the decision cutoff retain six mature-outcome collections, with no strategy decisions.
    last=dates[-1]
    for k in range(1,7):
        d=cal.sessions[cal.sessions.get_loc(last)+k]
        for minutes,kind in [(30,'outcomes'),(45,'outcome_retry45'),(60,'outcome_retry60'),(90,'outcome_watchdog')]:
            output.append(dict(id=f'{d.date()}:{kind}',session=str(d.date()),kind=kind,ordinal=len(dates)+k,
                               due=cal.session_close(d)+pd.Timedelta(minutes=minutes)))
    return sorted(output,key=lambda x:x['due'])


def status(release, workspace, now=None, validated_config=None):
    config=validated_config or verify(release)
    now=utc(now)
    if (Path(workspace)/'binding.json').exists():
        binding=read(Path(workspace)/'binding.json')
        if binding['release_sha256'] != digest(Path(release)/'manifest.json') or binding['trial_id'] != config['trial_id']:
            raise ValueError('Workspace binding differs from current release/trial')
    valid=_valid_results(workspace,config)
    consistency_errors=state_errors(workspace,valid,config)
    actual={r['session'] for r in valid if r['mode']=='prospective_shadow'}
    terminal={r['session'] for r in valid_tail_results(workspace,config,valid)}
    expected=[str(d.date()) for d in calendar().sessions_in_range(config['start'],config['end']) if calendar().session_close(d)+pd.Timedelta(minutes=65)<now]
    outcomes=valid_outcomes(workspace,config,valid)
    missing=sorted(set(expected)-actual)
    upcoming=[e for e in events(config) if e['due']>now]
    next_event=upcoming[0] if upcoming else None
    if next_event:
        next_event=dict(next_event, due=next_event['due'].isoformat(), hong_kong=next_event['due'].tz_convert('Asia/Hong_Kong').isoformat())
    return dict(trial_id=config['trial_id'], external_orders=False, deployment='cash', shadow='trend',
                state_consistency_errors=consistency_errors, ledger_consistent=not consistency_errors,
                valid_observed_sessions=len(actual), expected_elapsed_sessions=len(expected), missing_sessions=missing,
                label_complete_decisions=len(outcomes), required_sessions=config['sessions'], next_event=next_event,
                terminal_observed_sessions=len(terminal),terminal_required_sessions=6,
                halted=(Path(workspace)/'halt.json').exists(),
                prospective_status='incomplete' if len(actual)<config['sessions'] or len(outcomes)<config['sessions'] or len(terminal)<6 else 'collected_not_economic_approval',
                limitations=['Local launch agent requires logged-in awake host; no wake guarantee', 'No external broker execution',
                             'Commissioning/replays never count as prospective', 'Missed sessions halt; no fabricated backfill'])


def backup(workspace, stamp):
    workspace=Path(workspace)
    out=workspace / 'backups' / stamp
    if out.exists():
        return read_record(out / 'receipt.json')
    temp=Path(tempfile.mkdtemp(prefix='.backup-',dir=workspace))
    try:
        for name in ['deployment','shadow']:
            source=workspace / (name+'.sqlite')
            if source.exists():
                with sqlite3.connect(source) as a, sqlite3.connect(temp/(name+'.sqlite')) as b:
                    a.backup(b)
                with sqlite3.connect(temp/(name+'.sqlite')) as db:
                    if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
                        raise ValueError('Backup restore SQLite integrity failure')
        for name in ['days','attempts','outcomes','events','terminal','outcome-snapshots','outcome-runs','outcome-attempts','milestones','outages']:
            if (workspace/name).exists():
                shutil.copytree(workspace/name,temp/name)
        for source in workspace.glob('*.json'):
            shutil.copy2(source,temp/source.name)
        hashes={str(p.relative_to(temp)):digest(p) for p in temp.rglob('*') if p.is_file()}
        # Independently restore copied state into a second directory and compare every content hash.
        with tempfile.TemporaryDirectory(prefix='collector-restore-') as restored:
            shutil.copytree(temp,Path(restored)/'state')
            for relative, sha in hashes.items():
                if digest(Path(restored)/'state'/relative)!=sha: raise ValueError('Restore hash mismatch')
        receipt=record(temp/'receipt.json',dict(created_at=utc().isoformat(),status='copied_and_hash_verified',files=hashes,origin_workspace=str(workspace.resolve()),restore_contract='exact_original_path_after_scheduler_stop; preserve original state archive'))
        out.parent.mkdir(exist_ok=True)
        os.replace(temp,out)
        return receipt
    finally:
        if temp.exists(): shutil.rmtree(temp)


def restore_original_path(backup_path, workspace, archive):
    """Restore to the exact registered path; scheduler must be stopped first.

    Receipt text is immutable and contains absolute model/snapshot paths. A copy
    at a different path is therefore an archive, not a usable relocated account.
    Validate and stage first; preserve the previous directory before replacement.
    """
    backup_path,workspace,archive=map(lambda p:Path(p).resolve(),[backup_path,workspace,archive])
    receipt=read_record(backup_path/'receipt.json')
    if receipt['origin_workspace']!=str(workspace):
        raise ValueError('Restore requires exact original workspace path')
    if archive.exists() or archive==workspace or workspace in archive.parents:
        raise ValueError('Preservation archive must be a new separate sibling path')
    stage=Path(tempfile.mkdtemp(prefix='.restore-',dir=workspace.parent))
    try:
        for relative,expected in receipt['files'].items():
            source=backup_path/relative
            if digest(source)!=expected: raise ValueError('Backup content checksum failed')
            target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        for name in ['deployment','shadow']:
            path=stage/f'{name}.sqlite'
            if path.exists():
                with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
                    if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok': raise ValueError('Restore SQLite integrity failed')
        if workspace.exists():
            with lock(workspace):
                os.rename(workspace,archive)
                os.rename(stage,workspace)
        else:
            os.rename(stage,workspace)
        return dict(status='restored_at_original_path',workspace=str(workspace),preserved_original=str(archive),
                    backup_receipt_sha256=receipt['content_sha256'],at=utc().isoformat())
    finally:
        if stage.exists(): shutil.rmtree(stage)


def tick(release,workspace,now=None,validated_config=None,run=None):
    """A 60-second OS wakeup applies XNYS guards; it never invents missed runs."""
    now=utc(now)
    workspace=Path(workspace)
    config=validated_config or verify(release)
    run=run or _daily
    with lock(workspace):
        bind(workspace,release,config)
        previous=read(workspace/'heartbeat.json') if (workspace/'heartbeat.json').exists() else None
        heartbeat=dict(at=now.isoformat(),pid=os.getpid(),parent_pid=os.getppid(),service=os.environ.get('XPC_SERVICE_NAME'),
                       scheduler='launchd' if os.environ.get('XPC_SERVICE_NAME','').startswith('local.ml-trade-engine') else 'manual_tick',
                       source_sha256=digest(Path(__file__)), status='awake')
        if previous and now-utc(previous['at'])>pd.Timedelta(minutes=5):
            gap=dict(previous=previous['at'],resumed=now.isoformat(),gap_seconds=(now-utc(previous['at'])).total_seconds(),
                     reason='Observed scheduler gap; sleep, logout or outage indistinguishable locally')
            record(workspace/'outages'/f'{now.strftime("%Y%m%dT%H%M%S")}.json',gap)
            heartbeat['gap']=gap
        save(workspace/'heartbeat.json',heartbeat)
        if not (workspace/'scheduler-trigger.json').exists():
            record(workspace/'scheduler-trigger.json',heartbeat)
        # Actual OS trigger runs a real provider commissioning smoke once, in an isolated subworkspace.
        if now < calendar().session_close(session(config['start'])) and not (workspace/'commissioning-attempted.json').exists():
            try:
                result=run(release,workspace/'commissioning',mode='commissioning',now=now,validated_config=config)
                record(workspace/'commissioning-complete.json',dict(trigger=heartbeat,result_sha256=result['content_sha256'],
                      mode='commissioning_preview',completed_at=utc().isoformat()))
                backup(workspace/'commissioning','commissioning')
            except Exception as exc:
                save(workspace/'commissioning-failure.json',dict(at=utc().isoformat(),type=type(exc).__name__,message=str(exc)))
            # No automatic sub-minute repeated public commissioning reads.
            if not (workspace/'commissioning-attempted.json').exists():
                save(workspace/'commissioning-attempted.json',dict(at=now.isoformat()))
        for event in events(config):
            due=event['due']
            if due>now: break
            event_path=workspace/'events'/(event['id'].replace(':','-')+'.json')
            if event_path.exists(): continue
            event=dict(event,due=due.isoformat(),checked_at=now.isoformat())
            d=session(event['session']); close=calendar().session_close(d)
            result_path=workspace/'days'/event['session']/'result.json'
            exists=result_path.exists()
            kind=event['kind']
            try:
                if kind in ['daily','retry45','retry60']:
                    prior_attempts=[read(p) for p in sorted((workspace/'attempts').glob(f'{d.date()}-*.json'))]
                    retry_allowed=bool(prior_attempts and (prior_attempts[-1]['status']=='started' or (prior_attempts[-1]['status']=='failed' and prior_attempts[-1].get('transient'))))
                    if (workspace/'halt.json').exists() and not exists:
                        event.update(status='halted',action_required=True,reason=read(workspace/'halt.json'))
                    elif exists:
                        read_record(result_path)
                        event.update(status='covered_by_valid_receipt')
                    elif now>due+pd.Timedelta(minutes=5):
                        event.update(status='missed_deadline',action_required=True)
                        if kind=='daily':
                            save(workspace/'halt.json',dict(reason='Missed daily decision window; no backfill',session=event['session'],at=now.isoformat()))
                    elif kind!='daily' and not retry_allowed:
                        event.update(status='ineligible_retry',action_required=bool(prior_attempts))
                    else:
                        run(release,workspace,date=d,mode='scheduled',now=now,validated_config=config)
                        event.update(status='complete')
                elif kind in ['watchdog','preopen']:
                    if exists:
                        r=read_record(result_path)
                        if abs(r['deployment']['reconciliation_error'])>1e-6 or abs(r['shadow']['reconciliation_error'])>1e-6:
                            raise ValueError('Receipt reconciliation failed')
                        event.update(status='valid_receipt')
                        if kind=='watchdog' and weekly(d):
                            backup(workspace,str(d.date()))
                        if kind=='watchdog' and event['ordinal'] in config.get('operational_milestones', [5,21,42,63]):
                            record(workspace/'milestones'/f'{event["ordinal"]}.json',status(release,workspace,now,config))
                    else:
                        event.update(status='missing_expected_receipt',action_required=True)
                        save(workspace/'halt.json',dict(reason='Missing expected reconciled receipt',session=event['session'],at=now.isoformat()))
                elif kind in ['outcomes','outcome_retry45','outcome_retry60','outcome_watchdog']:
                    marker=workspace/'outcome-runs'/f'{d.date()}.json'
                    attempts=[read(p) for p in sorted((workspace/'outcome-attempts').glob(f'{d.date()}-*.json'))]
                    eligible=bool(attempts and (attempts[-1]['status']=='started' or attempts[-1].get('transient')))
                    if marker.exists():
                        outcome_run=read_record(marker)
                        if outcome_run.get('terminal_status')=='failed':
                            event.update(status='labels_attached_terminal_failed',action_required=True,terminal_error=outcome_run['terminal_error'])
                        else: event.update(status='outcomes_attached')
                    elif kind=='outcome_watchdog':
                        event.update(status='missing_outcome_receipt',action_required=True)
                    elif now>due+pd.Timedelta(minutes=5):
                        event.update(status='missed_outcome_read',action_required=True)
                    elif kind!='outcomes' and not eligible:
                        event.update(status='ineligible_retry',action_required=True)
                    else:
                        attempt=workspace/'outcome-attempts'/f'{d.date()}-{now.strftime("%H%M%S")}.json'
                        save(attempt,dict(status='started',at=now.isoformat()))
                        location=workspace/'outcome-snapshots'/f'{d.date()}-{now.strftime("%H%M%S")}'
                        try:
                            market.download(location,config['symbols'],d,now)
                            data,_=market.load(location,config['symbols'],d,now)
                            terminal_error=None
                            try: terminal_close(workspace,data,d,location,config,now)
                            except Exception as exc:
                                terminal_error=dict(type=type(exc).__name__,message=str(exc))
                            mature(workspace,data,d,config,snapshot=location)
                            record(marker,dict(session=str(d.date()),snapshot=str(location),snapshot_sha256=digest(location/'manifest.json'),
                                   status='complete',terminal_status='failed' if terminal_error else 'complete',terminal_error=terminal_error,at=utc().isoformat()))
                            save(attempt,dict(status='complete',at=utc().isoformat()))
                            event.update(status='labels_attached_terminal_failed' if terminal_error else 'outcomes_attached',
                                         action_required=bool(terminal_error),terminal_error=terminal_error)
                            if event['ordinal']==config['sessions']+6: backup(workspace,'final-label-maturity')
                        except Exception as exc:
                            save(attempt,dict(status='failed',transient=isinstance(exc,market.TransientDataError),error=type(exc).__name__,message=str(exc),at=utc().isoformat()))
                            raise
            except Exception as exc:
                event.update(status='failed',error=type(exc).__name__,message=str(exc),action_required=True,
                             retryable=isinstance(exc,market.TransientDataError))
                if kind in ['daily','retry45','retry60'] and not isinstance(exc,market.TransientDataError):
                    save(workspace/'halt.json',dict(reason='Deterministic policy/integrity failure',session=event['session'],error=type(exc).__name__,message=str(exc),at=now.isoformat()))
            record(event_path,event)
        result=status(release,workspace,now,config)
        save(workspace/'status.json',result)
        return result
