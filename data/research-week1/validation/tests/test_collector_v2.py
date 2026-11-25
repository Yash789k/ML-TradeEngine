"""Registered R11 behavioral tests; replay state never touches active pilot."""
from pathlib import Path
import json
import multiprocessing
import os
import shutil
import sqlite3
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits
from runtime_v2 import operations as op
from runtime_v2 import market
from runtime_v2.calendar import calendar
from runtime_v2.model import AdaptiveModel, features
from runtime_v2.policy import plan, commit, replay, weekly, refit_due, utc, session
from runtime_v2.storage import save, read, digest, lock

ROOT=Path(__file__).resolve().parents[1]


def configuration(start='2026-09-29', count=63):
    cal=calendar(); first=cal.sessions.get_loc(pd.Timestamp(start)); dates=cal.sessions[first:first+count]
    return dict(trial_id='R11-ISOLATED-TEST',symbols=['AAPL','SPY'],capital=10000,anchor=start,start=start,
                end=str(dates[-1].date()),sessions=count,external_orders=False,deployment='cash',shadow='trend')


def fake_release(tmp_path, config):
    release=tmp_path/'release'; release.mkdir()
    save(release/'manifest.json',dict(test_fixture=True))
    save(release/'config.json',config)
    return release


def snapshot(path,data,date):
    path.mkdir(parents=True)
    meta=dict(status='complete',errors={},data={})
    for s,f in data.items():
        f=f.loc[:session(date).tz_localize('UTC')]
        f.to_parquet(path/f'{s}.parquet');f.to_parquet(path/f'{s}-raw.parquet')
        meta['data'][s]=dict(sha256=digest(path/f'{s}.parquet'),raw_sha256=digest(path/f'{s}-raw.parquet'))
    save(path/'manifest.json',meta)
    return path


@pytest.fixture(scope='module')
def history():
    return {s:pd.read_parquet(ROOT/'data/releases/2026-09-28-adaptive/market/adjusted'/f'{s}.parquet') for s in ['AAPL','SPY']}


@pytest.fixture(scope='module')
def trained(tmp_path_factory,history):
    tmp_path=tmp_path_factory.mktemp('trained-models')
    cfg=configuration('2026-05-20',45)
    release=fake_release(tmp_path,cfg)
    paths={}
    with threadpool_limits(limits=1):
        for s,f in history.items():
            m=AdaptiveModel().fit_at(features(f,history['SPY']),f.Close,pd.Timestamp('2026-05-19',tz='UTC'))
            m.save(release/'models'/s); paths[s]=str(release/'models'/s)
    return cfg,release,paths


def test_calendar_schedule_dst_holiday():
    cfg=configuration()
    events=op.events(cfg)
    daily={e['session']:e for e in events if e['kind']=='daily'}
    assert daily['2026-09-29']['due'].tz_convert('Asia/Hong_Kong').strftime('%H:%M')=='04:30'
    assert daily['2026-11-02']['due'].tz_convert('Asia/Hong_Kong').strftime('%H:%M')=='05:30'
    assert '2026-11-26' not in daily
    assert daily['2026-11-27']['due'].tz_convert('America/New_York').strftime('%H:%M')=='13:30'
    assert weekly('2026-07-02') and not weekly('2026-07-01')
    assert cfg['end']=='2026-12-28'
    assert events[-1]['session']=='2027-01-06'
    assert refit_due('2026-10-28',cfg['anchor'])
    assert refit_due('2026-11-27',cfg['anchor'])
    assert not refit_due('2026-10-27',cfg['anchor'])


def test_policy_refits_holiday_future_invariance(tmp_path,history,trained):
    cfg,release,models=trained
    dates=calendar().sessions_in_range(cfg['start'],cfg['end'])
    expected=replay(history,models,dates,cfg,tmp_path/'research')
    actual=[]
    for d in dates:
        snap=snapshot(tmp_path/'inputs'/str(d.date()),history,d)
        actual.append(op.daily(release,tmp_path/'daily',d,mode='replay_test',snapshot=snap,
                               now=calendar().session_close(d)+pd.Timedelta(minutes=30),validated_config=cfg))
    assert sum(r['refit'] for r in actual)==2
    for a,b in zip(actual,expected):
        for key in ['deployment','shadow','shadow_weights','rebalance','refit','marks']:
            assert a[key]==b[key]
        assert [s['probabilities'] for s in a['signals']]==[s['probabilities'] for s in b['signals']]
        assert [s['model_decision'] for s in a['signals']]==[s['model_decision'] for s in b['signals']]
    mutated={s:f.copy() for s,f in history.items()}
    for s in mutated: mutated[s].loc[mutated[s].index>pd.Timestamp(cfg['start'],tz='UTC'),'Close']*=3
    a=plan(history,models,cfg['start'],cfg,tmp_path/'before')
    b=plan(mutated,models,cfg['start'],cfg,tmp_path/'after')
    assert a['signals']==b['signals'] and a['shadow_weights']==b['shadow_weights']
    assert sum(x['rebalance'] for x in actual)>5
    assert any(x['session']=='2026-07-02' and x['rebalance'] for x in actual)
    save(ROOT/'data/research-v2/operations/parity-replay.json',dict(status='passed',sessions=len(actual),
         symbols=cfg['symbols'],refit_sessions=[x['session'] for x in actual if x['refit']],
         holiday_rebalance='2026-07-02',contract='same full policy, carried seed models, daily persisted restart',
         max_reconciliation_error=max(abs(x['shadow']['reconciliation_error']) for x in actual),
         evidence_mode='historical_controlled_replay_not_prospective'))


@pytest.mark.parametrize('point',['before_plan','after_input','after_plan','after_deployment','after_shadow','before_receipt','after_receipt'])
def test_restart_account_boundary(tmp_path,history,trained,point):
    cfg,release,models=trained; date=cfg['start']; snap=snapshot(tmp_path/'input',history,date)
    now=calendar().session_close(date)+pd.Timedelta(minutes=30)
    def crash(stage):
        if stage==point: raise RuntimeError('injected crash '+point)
    with pytest.raises(RuntimeError,match='injected crash'):
        op.daily(release,tmp_path/'crash',date,snapshot=snap,now=now,fault=crash,validated_config=cfg)
    restored=op.daily(release,tmp_path/'crash',date,snapshot=snap,now=now,validated_config=cfg)
    clean=op.daily(release,tmp_path/'clean',date,snapshot=snap,now=now,validated_config=cfg)
    for key in ['shadow','deployment']:
        assert restored[key]==clean[key]
        with sqlite3.connect(tmp_path/'crash'/f'{key}.sqlite') as db:
            counts=db.execute('SELECT COUNT(*) FROM events').fetchone()
        with sqlite3.connect(tmp_path/'clean'/f'{key}.sqlite') as db:
            assert counts==db.execute('SELECT COUNT(*) FROM events').fetchone()


def lock_worker(workspace,queue):
    try:
        with lock(workspace): queue.put('incorrectly acquired')
    except RuntimeError: queue.put('blocked')


def test_lock_and_storage_failure(tmp_path,monkeypatch):
    ctx=multiprocessing.get_context('spawn'); q=ctx.Queue()
    with lock(tmp_path):
        p=ctx.Process(target=lock_worker,args=(str(tmp_path),q));p.start();p.join(20)
        assert q.get(timeout=5)=='blocked' and p.exitcode==0
    save(tmp_path/'valid.json',{'v':1})
    import runtime_v2.storage as storage
    def full(*args): raise OSError('disk full')
    monkeypatch.setattr(storage.os,'replace',full)
    with pytest.raises(OSError): save(tmp_path/'valid.json',{'v':2})
    assert read(tmp_path/'valid.json')=={'v':1}


def test_scheduler_retries_and_outage(tmp_path):
    cfg=configuration();release=fake_release(tmp_path,cfg);ws=tmp_path/'state'
    d=session(cfg['start']);close=calendar().session_close(d); calls=[]
    def failed(release,workspace,**kwargs):
        calls.append(kwargs['now'])
        save(Path(workspace)/'attempts'/f'{d.date()}-{len(calls)}.json',dict(status='failed',transient=True))
        raise market.TransientDataError('offline')
    for n in [30,45,60,90]: op.tick(release,ws,close+pd.Timedelta(minutes=n),cfg,failed)
    assert len(calls)==3
    assert read(ws/'events'/f'{d.date()}-watchdog.json')['status']=='missing_expected_receipt'
    assert list((ws/'outages').glob('*.json'))
    other=tmp_path/'missed'
    op.tick(release,other,close+pd.Timedelta(minutes=90),cfg,failed)
    assert read(other/'events'/f'{d.date()}-daily.json')['status']=='missed_deadline'
    assert len(calls)==3
    # Deterministic failures must never authorize provider retries.
    def deterministic(release,workspace,**kwargs):
        save(Path(workspace)/'attempts'/f'{d.date()}-1.json',dict(status='failed',transient=False))
        calls.append(kwargs['now']);raise ValueError('corrupt model')
    third=tmp_path/'deterministic'
    for n in [30,45,60]: op.tick(release,third,close+pd.Timedelta(minutes=n),cfg,deterministic)
    assert len(calls)==4


def test_receipt_tampering(tmp_path):
    path=tmp_path/'r.json';op.record(path,dict(status='valid',count=0)); assert op.read_record(path)['count']==0
    value=read(path);value['count']=99;save(path,value)
    with pytest.raises(ValueError,match='hash mismatch'): op.read_record(path)
    with pytest.raises(ValueError,match='Immutable'): op.record(path,dict(status='changed'))


def test_backup_restore(tmp_path,history,trained):
    cfg,release,models=trained;date=cfg['start'];snap=snapshot(tmp_path/'input',history,date)
    op.daily(release,tmp_path/'state',date,snapshot=snap,now=calendar().session_close(date)+pd.Timedelta(minutes=30),validated_config=cfg)
    evidence=op.backup(tmp_path/'state','test')
    assert evidence['status']=='copied_and_hash_verified'
    assert 'shadow.sqlite' in evidence['files']
    copied=tmp_path/'state/backups/test'
    assert op.read_record(copied/'days'/date/'result.json')['session']==date


def test_stale_missing_mutated_data(tmp_path,history):
    snap=snapshot(tmp_path/'input',history,'2026-05-20')
    with pytest.raises(ValueError,match='Stale'): market.load(snap,list(history),'2026-05-21')
    path=snap/'AAPL.parquet';frame=pd.read_parquet(path).iloc[:-1];frame.to_parquet(path)
    with pytest.raises(ValueError,match='hash mismatch'): market.load(snap,list(history),'2026-05-20')


def test_progress_requires_unique_valid_receipts(tmp_path):
    cfg=configuration();release=fake_release(tmp_path,cfg)
    current=calendar().session_close(session(cfg['start']))+pd.Timedelta(minutes=90)
    result=op.status(release,tmp_path/'ws',current,cfg)
    assert result['valid_observed_sessions']==0 and result['missing_sessions']==[cfg['start']]
    op.record(tmp_path/'ws/days'/cfg['start']/'result.json',dict(mode='commissioning_preview',session=cfg['start']))
    with pytest.raises(ValueError,match='release/trial'): op.status(release,tmp_path/'ws',current,cfg)


def test_outcome_tail_retries_watchdog(tmp_path,monkeypatch):
    cfg=configuration();release=fake_release(tmp_path,cfg);ws=tmp_path/'state'
    tail=[e for e in op.events(cfg) if e['kind']=='outcomes'][0]
    calls=[]
    def offline(*args,**kwargs):
        calls.append(1);raise market.TransientDataError('transient tail read failure')
    monkeypatch.setattr(market,'download',offline)
    for n in [0,15,30,60]:
        op.tick(release,ws,tail['due']+pd.Timedelta(minutes=n),cfg)
    assert len(calls)==3
    assert read(ws/'events'/f'{tail["session"]}-outcome_watchdog.json')['status']=='missing_outcome_receipt'


def test_db_corruption_and_unbound_receipts_are_not_readiness(tmp_path,trained,history):
    cfg,release,models=trained;date=cfg['start'];snap=snapshot(tmp_path/'input',history,date)
    now=calendar().session_close(date)+pd.Timedelta(minutes=30);ws=tmp_path/'state'
    result=op.daily(release,ws,date,snapshot=snap,now=now,validated_config=cfg)
    assert op.status(release,ws,now,cfg)['ledger_consistent']
    with sqlite3.connect(ws/'shadow.sqlite') as db:
        row=db.execute('SELECT payload FROM state WHERE id=1').fetchone();payload=json.loads(row[0]);payload['cash']-=1
        db.execute('UPDATE state SET payload=? WHERE id=1',(json.dumps(payload),))
    assert not op.status(release,ws,now,cfg)['ledger_consistent']
    with pytest.raises(ValueError): op.daily(release,ws,date,snapshot=snap,now=now,validated_config=cfg)
    save(release/'manifest.json',dict(test_fixture='mutated'))
    with pytest.raises(ValueError,match='binding'): op.status(release,ws,now,cfg)
    save(release/'manifest.json',dict(test_fixture=True))


def test_model_and_partial_snapshot_failure_before_orders(tmp_path,trained,history):
    cfg,release,models=trained;date=cfg['start'];snap=snapshot(tmp_path/'input',history,date)
    # No provider padding: remove an interior bar and update its transport checksum.
    path=snap/'AAPL.parquet';f=pd.read_parquet(path);f.drop(f.index[-10]).to_parquet(path)
    meta=read(snap/'manifest.json');meta['data']['AAPL']['sha256']=digest(path);save(snap/'manifest.json',meta)
    with pytest.raises(ValueError,match='Missing exchange sessions'):
        op.daily(release,tmp_path/'state',date,snapshot=snap,now=calendar().session_close(date)+pd.Timedelta(minutes=30),validated_config=cfg)
    assert not (tmp_path/'state/shadow.sqlite').exists()


def test_outcome_provenance_maturity_and_uniqueness(tmp_path,trained,history):
    cfg,release,models=trained;ws=tmp_path/'state'
    dates=calendar().sessions_in_range(cfg['start'],cfg['end'])[:7]
    for d in dates:
        snap=snapshot(ws/'days'/str(d.date())/'market',history,d)
        op.daily(release,ws,d,snapshot=snap,now=calendar().session_close(d)+pd.Timedelta(minutes=30),validated_config=cfg)
    values=op.valid_outcomes(ws,cfg);assert len(values)==1
    path=ws/'outcomes'/f'{cfg["start"]}.json';original=read(path)
    def rewrite(value):
        value=dict(value);value.pop('content_sha256',None);value['content_sha256']=op.content_hash(value);save(path,value)
    mutations=[dict(trial_id='foreign'),dict(binding={'foreign':'release'}),dict(decision_sha256='0'*64),
               dict(endpoint=cfg['start']),dict(observed_at='2000-01-01T00:00:00Z'),
               dict(outcomes=original['outcomes'][:1]),dict(outcomes=[original['outcomes'][0]]*2),
               dict(outcomes=[dict(r,realized_return=999) for r in original['outcomes']])]
    for changes in mutations:
        rewrite(dict(original,**changes))
        with pytest.raises(ValueError): op.status(release,ws,validated_config=cfg)
        with pytest.raises(ValueError): op.mature(ws,history,dates[-1],cfg,snapshot=ws/'days'/str(dates[-1].date())/'market')
        save(path,original)
    duplicate=ws/'outcomes/duplicate.json';shutil.copy2(path,duplicate)
    with pytest.raises(ValueError,match='unique'): op.status(release,ws,validated_config=cfg)
    duplicate.unlink()
    orphan=read(path);orphan['decision_session']='2026-05-19';rewrite(orphan)
    with pytest.raises(ValueError,match='unique'):op.valid_outcomes(ws,cfg)


def test_underlying_snapshot_and_model_bytes(tmp_path,trained,history):
    cfg,release,models=trained;ws=tmp_path/'state';date=cfg['start'];snap=snapshot(ws/'days'/date/'market',history,date)
    now=calendar().session_close(date)+pd.Timedelta(minutes=30)
    op.daily(release,ws,date,snapshot=snap,now=now,validated_config=cfg)
    for path in [snap/'AAPL.parquet',snap/'AAPL-raw.parquet',release/'models/AAPL/model.joblib']:
        original=path.read_bytes()
        try:
            path.write_bytes(original+b'corruption')
            with pytest.raises(ValueError,match='bytes changed'):op.status(release,ws,now,cfg)
        finally:path.write_bytes(original)


def test_readonly_status_preserves_truncated_state(tmp_path,trained,history):
    cfg,release,models=trained;date=cfg['start'];snap=snapshot(tmp_path/'input',history,date);ws=tmp_path/'state'
    now=calendar().session_close(date)+pd.Timedelta(minutes=30)
    op.daily(release,ws,date,snapshot=snap,now=now,validated_config=cfg)
    path=ws/'shadow.sqlite';path.write_bytes(b'')
    assert not op.status(release,ws,now,cfg)['ledger_consistent']
    assert path.read_bytes()==b''


def test_restore_original_path_and_continue(tmp_path,trained,history):
    cfg,release,models=trained;date=cfg['start'];ws=tmp_path/'state';snap=snapshot(ws/'days'/date/'market',history,date)
    now=calendar().session_close(date)+pd.Timedelta(minutes=30)
    first=op.daily(release,ws,date,snapshot=snap,now=now,validated_config=cfg)
    op.backup(ws,'test')
    lost=tmp_path/'lost-preserved';ws.rename(lost)
    assert not Path(first['snapshot']).exists()
    recovered=op.restore_original_path(lost/'backups/test',ws,tmp_path/'unused-prior')
    assert recovered['status']=='restored_at_original_path' and lost.exists()
    assert op.status(release,ws,now,cfg)['ledger_consistent']
    nxt=calendar().next_session(session(date));nextsnap=snapshot(ws/'days'/str(nxt.date())/'market',history,nxt)
    actual=op.daily(release,ws,nxt,snapshot=nextsnap,now=calendar().session_close(nxt)+pd.Timedelta(minutes=30),validated_config=cfg)
    reference=tmp_path/'reference';op.daily(release,reference,date,snapshot=snap,now=now,validated_config=cfg)
    expected=op.daily(release,reference,nxt,snapshot=nextsnap,now=calendar().session_close(nxt)+pd.Timedelta(minutes=30),validated_config=cfg)
    assert actual['shadow']==expected['shadow'] and actual['deployment']==expected['deployment']


def test_direct_daily_respects_halt(tmp_path,trained,history):
    cfg,release,models=trained;date=cfg['start'];ws=tmp_path/'state';save(ws/'halt.json',dict(reason='integrity'))
    snap=snapshot(tmp_path/'input',history,date)
    with pytest.raises(ValueError,match='halted'):
        op.daily(release,ws,date,snapshot=snap,now=calendar().session_close(date)+pd.Timedelta(minutes=30),validated_config=cfg)
    assert not (ws/'shadow.sqlite').exists()


def test_interrupted_refit_freezes_input_before_models(tmp_path,trained,history,monkeypatch):
    import runtime_v2.operations as ops
    cfg,release,models=trained;cfg=dict(cfg);date=cfg['start']
    idx=calendar().sessions.get_loc(session(date));cfg['anchor']=str(calendar().sessions[idx-21].date())
    ws=tmp_path/'state';snap=snapshot(tmp_path/'old',history,date);new=snapshot(tmp_path/'new',history,date)
    original=ops.plan
    def crash(*args,**kwargs):
        original(*args,**kwargs);raise RuntimeError('crash after fitted models before plan')
    monkeypatch.setattr(ops,'plan',crash)
    now=calendar().session_close(date)+pd.Timedelta(minutes=30)
    with pytest.raises(RuntimeError):op.daily(release,ws,date,snapshot=snap,now=now,validated_config=cfg)
    assert (ws/'days'/date/'input.json').exists() and (ws/'days'/date/'models/AAPL/manifest.json').exists()
    monkeypatch.setattr(ops,'plan',original)
    with pytest.raises(ValueError,match='different input'):
        op.daily(release,ws,date,snapshot=new,now=now,validated_config=cfg)
    result=op.daily(release,ws,date,now=now,validated_config=cfg)
    assert result['snapshot']==str(snap.resolve())


def test_readonly_missing_events_table(tmp_path,trained,history):
    cfg,release,models=trained;date=cfg['start'];ws=tmp_path/'state';snap=snapshot(tmp_path/'input',history,date)
    now=calendar().session_close(date)+pd.Timedelta(minutes=30)
    op.daily(release,ws,date,snapshot=snap,now=now,validated_config=cfg)
    dbpath=ws/'shadow.sqlite'
    with sqlite3.connect(dbpath) as db:db.execute('DROP TABLE events')
    before=dbpath.read_bytes()
    assert not op.status(release,ws,now,cfg)['ledger_consistent']
    assert dbpath.read_bytes()==before
    with sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name='events'").fetchall()==[]


def test_restore_tail_outcomes_halt_and_terminal_accounting(tmp_path,trained,history):
    cfg,release,models=trained;cfg=dict(cfg);cfg['end']=cfg['start'];cfg['sessions']=1
    ws=tmp_path/'state';date=session(cfg['start']);snap=snapshot(ws/'days'/str(date.date())/'market',history,date)
    first=op.daily(release,ws,date,snapshot=snap,now=calendar().session_close(date)+pd.Timedelta(minutes=30),validated_config=cfg)
    decisions={s:len(first[s]['positions']) for s in ['shadow','deployment']}
    index=calendar().sessions.get_loc(date)
    for d in calendar().sessions[index+1:index+7]:
        location=snapshot(ws/'outcome-snapshots'/str(d.date()),history,d)
        data,_=market.load(location,cfg['symbols'],d)
        result=op.terminal_close(ws,data,d,location,cfg,calendar().session_close(d)+pd.Timedelta(minutes=30))
        op.mature(ws,data,d,cfg,snapshot=location)
        op.record(ws/'outcome-runs'/f'{d.date()}.json',dict(status='complete',session=str(d.date())))
        save(ws/'outcome-attempts'/f'{d.date()}-test.json',dict(status='complete'))
    assert len(op.valid_outcomes(ws,cfg))==1
    assert result['mode']=='outcome_tail' and not result['rebalance']
    save(ws/'latest.json',first)
    op.terminal_close(ws,data,d,location,cfg,calendar().session_close(d)+pd.Timedelta(minutes=30))
    assert read(ws/'latest.json')['session']==str(d.date())
    assert op.status(release,ws,validated_config=cfg)['valid_observed_sessions']==1
    assert op.status(release,ws,validated_config=cfg)['ledger_consistent']
    save(ws/'halt.json',dict(reason='test_preserved_halt'))
    op.backup(ws,'tail-test')
    lost=tmp_path/'lost-preserved';ws.rename(lost)
    op.restore_original_path(lost/'backups/tail-test',ws,tmp_path/'unused-prior')
    assert read(ws/'halt.json')['reason']=='test_preserved_halt'
    assert len(op.valid_outcomes(ws,cfg))==1
    assert op.status(release,ws,validated_config=cfg)['ledger_consistent']
    assert len(list((ws/'terminal').glob('*/result.json')))==6
    assert len(list((ws/'outcome-attempts').glob('*.json')))==6


def test_halted_tail_still_attaches_mature_outcomes(tmp_path,trained,history,monkeypatch):
    cfg,release,models=trained;cfg=dict(cfg);cfg['end']=cfg['start'];cfg['sessions']=1
    ws=tmp_path/'state';date=session(cfg['start']);snap=snapshot(ws/'days'/str(date.date())/'market',history,date)
    op.daily(release,ws,date,snapshot=snap,now=calendar().session_close(date)+pd.Timedelta(minutes=30),validated_config=cfg)
    save(ws/'halt.json',dict(reason='outage preserves labels'))
    index=calendar().sessions.get_loc(date);endpoint=calendar().sessions[index+6]
    def provider(destination,symbols,d,now):
        snapshot(Path(destination),history,d)
    monkeypatch.setattr(market,'download',provider)
    op.tick(release,ws,calendar().session_close(endpoint)+pd.Timedelta(minutes=30),cfg)
    assert len(op.valid_outcomes(ws,cfg))==1
    event=read(ws/'events'/f'{endpoint.date()}-outcomes.json')
    assert event['status']=='labels_attached_terminal_failed' and event['action_required']
    assert not list((ws/'terminal').glob('*/result.json'))
