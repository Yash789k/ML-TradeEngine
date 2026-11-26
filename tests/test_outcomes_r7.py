"""Isolated time-reveal/fault tests; synthetic data never enters pilot evidence."""
import copy
import json
from pathlib import Path
import pandas as pd
import pytest
from scripts.outcomes_r7 import collector as c


@pytest.fixture
def fixture(monkeypatch):
    schedule=c.schedule('2026-10-27','2026-11-03')
    days=c.calendar().sessions_in_range('2026-10-01','2026-11-03')
    data=pd.DataFrame({'Close':[100+i for i in range(len(days))]},index=days.tz_localize('UTC'))
    dates=['2026-10-01','2026-10-02','2026-10-14','2026-10-15','2026-10-22']
    ds=[]
    for date in dates:
        idx=c.calendar().sessions.get_loc(pd.Timestamp(date))
        ds.append(dict(segment='test',session=date,entry=str(c.calendar().sessions[idx+1].date()),endpoint=str(c.calendar().sessions[idx+6].date()),signals=[dict(symbol='A',probabilities=[.2,.3,.5])],priors={'A':[.3,.4,.3]},content_sha256='decision-'+date))
    protocol=dict(id='fixture-only',symbols=['A'],decisions=ds,schedule=schedule,original_missed_outcome_sessions=['2026-10-23','2026-10-26'])
    current={'now':pd.Timestamp('2026-10-27T20:30:05Z')}; real=c.utc
    monkeypatch.setattr(c,'utc',lambda v=None: real(current['now'] if v is None else v))
    monkeypatch.setattr(c.os,'getppid',lambda:1)
    monkeypatch.setenv('XPC_SERVICE_NAME',c.SERVICE)
    def download(path,symbols,date,now=None):
        path.mkdir(parents=True)
        c.save(path/'manifest.json',dict(expected_session=date,retrieved_at=c.utc(now).isoformat(),status='complete'))
    def load(path,symbols,date,now=None):
        return {'A':data.loc[:date].copy()},c.read(Path(path)/'manifest.json')
    monkeypatch.setattr(c.market,'load',load)
    return protocol,current,download,data


def test_calendar_dst_and_boundaries():
    rows=c.schedule('2026-10-27','2026-11-03')
    assert [r['session'] for r in rows]==['2026-10-27','2026-10-28','2026-10-29','2026-10-30','2026-11-02','2026-11-03']
    assert rows[0]['collection_hkt']=='2026-10-28T04:30:00+08:00'
    assert rows[-1]['collection_hkt']=='2026-11-04T05:30:00+08:00'
    p={'schedule':rows}
    assert c.boundary_window(p,'2026-11-02T14:35:00Z')['kind']=='open'
    assert c.boundary_window(p,'2026-11-02T13:35:00Z') is None
    assert c.boundary_window(p,'2026-10-31T13:35:00Z') is None


@pytest.mark.parametrize('minute,slot',[(29,None),(30,'initial'),(40,'initial'),(41,None),(45,'retry45'),(55,'retry45'),(56,None),(60,'retry60'),(65,'retry60'),(66,None),(90,None)])
def test_bounded_attempt_windows(minute,slot):
    row=c.schedule('2026-10-27','2026-11-03')[0]
    assert c.attempt_slot(row,pd.Timestamp(row['close'])+pd.Timedelta(minutes=minute))==slot


def test_original_dates_maturity_and_price_oracle(fixture):
    p,clock,download,data=fixture
    early=c.derive_outcomes(p,{'A':data.loc[:'2026-10-27']},'2026-10-27')
    assert len(early)==4
    assert all(x['decision_session']!='2026-10-22' for x in early)
    final=c.derive_outcomes(p,{'A':data},'2026-11-03')
    r=next(x for x in final if x['decision_session']=='2026-10-22')
    assert r['entry']=='2026-10-23' and r['endpoint']=='2026-10-30'
    assert r['realized_return']==pytest.approx(data.loc['2026-10-30','Close']/data.loc['2026-10-23','Close']-1)
    changed=data.copy();changed.loc['2026-11-02':,'Close']=1000000
    assert c.derive_outcomes(p,{'A':changed},'2026-11-03')==final


def test_restart_idempotence_and_unchanged_accounts(tmp_path,fixture):
    p,clock,download,data=fixture;ws=tmp_path/'outcomes';archive=tmp_path/'old-account.sqlite'
    archive.write_bytes(b'preserved-account')
    c.run_cycle(p,ws,clock['now'],download)
    before={str(x):c.digest(x) for x in tmp_path.rglob('*') if x.is_file()}
    c.run_cycle(p,ws,clock['now'],lambda *a,**k:pytest.fail('duplicate download'))
    assert before=={str(x):c.digest(x) for x in tmp_path.rglob('*') if x.is_file()}
    r=c.validated_reads(p,ws)
    assert len(r)==1 and r[0]['new_decisions']==0 and r[0]['account_mutations']==0
    assert not list(ws.rglob('*.sqlite'))


def test_transient_retry_and_persistent_failure(tmp_path,fixture):
    p,clock,download,data=fixture;calls=[]
    def failed(*args,**kwargs):calls.append(kwargs['now']);raise c.market.TransientDataError('provider pending')
    ws=tmp_path/'persistent'
    for m in [30,30,45,45,60,60,90]:
        clock['now']=pd.Timestamp('2026-10-27T20:00Z')+pd.Timedelta(minutes=m,seconds=5)
        c.run_cycle(p,ws,clock['now'],failed)
    assert len(calls)==3 and not (ws/'halt.json').exists()
    assert c.op.read_record(ws/'events/2026-10-27-watchdog.json')['action_required']
    recovered=tmp_path/'recovered'
    clock['now']=pd.Timestamp('2026-10-27T20:30:05Z');c.run_cycle(p,recovered,clock['now'],failed)
    clock['now']=pd.Timestamp('2026-10-27T20:45:05Z');c.run_cycle(p,recovered,clock['now'],download)
    assert len(c.validated_reads(p,recovered))==1


def test_missed_read_cannot_be_backfilled_or_promoted(tmp_path,fixture):
    p,clock,download,data=fixture;ws=tmp_path/'state'
    clock['now']=pd.Timestamp('2026-10-28T20:30:05Z')
    c.run_cycle(p,ws,clock['now'],download)
    assert not (ws/'reads/2026-10-27/result.json').exists()
    state=c.summary(p,ws,c.validated_reads(p,ws))
    assert state['missing_new_sessions']==['2026-10-27']
    with pytest.raises(ValueError,match='six new reads'):c.require_complete(state)


def test_deterministic_error_halts_new_requests(tmp_path,fixture):
    p,clock,download,data=fixture
    def invalid(*a,**k):raise ValueError('invalid old provider price')
    c.run_cycle(p,tmp_path,clock['now'],invalid)
    assert (tmp_path/'halt.json').exists()
    clock['now']=pd.Timestamp('2026-10-27T20:45:05Z')
    c.run_cycle(p,tmp_path,clock['now'],lambda *a,**k:pytest.fail('halt bypassed'))


@pytest.mark.parametrize('fault',['hash','price','time','source','owner','duplicate'])
def test_integrity_rejection(tmp_path,fixture,fault):
    p,clock,download,data=fixture;c.run_cycle(p,tmp_path,clock['now'],download)
    path=tmp_path/'reads/2026-10-27/result.json';r=c.op.read_record(path)
    if fault=='hash':
        Path(r['snapshot'],'manifest.json').write_text('{}')
    else:
        if fault=='price':r['outcomes'][0]['realized_return']+=.1
        if fault=='time':r['started_at']='2026-10-27T19:00:00Z'
        if fault=='source':r['outcomes'][0]['endpoint']='2026-10-27'
        if fault=='owner':r['parent_pid']=999
        if fault=='duplicate':path=tmp_path/'reads/2026-10-28/result.json'
        r.pop('content_sha256');r['content_sha256']=c.op.content_hash(r);c.save(path,r)
    with pytest.raises((ValueError,KeyError)):c.validated_reads(p,tmp_path)


def test_full_six_read_replay_with_five_original_outcomes(tmp_path,fixture):
    p,clock,download,data=fixture
    for row in p['schedule']:
        clock['now']=pd.Timestamp(row['collection'])+pd.Timedelta(seconds=5)
        c.run_cycle(p,tmp_path,clock['now'],download)
    receipts=c.validated_reads(p,tmp_path);state=c.summary(p,tmp_path,receipts)
    c.require_complete(state)
    assert state['completed_outcome_reads']==6 and state['mature_decision_dates']==5
    assert state['original_account_tail_complete'] is False
    assert state['new_decisions']==0 and state['external_orders'] is False


def test_manual_production_tick_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(c.os,'getppid',lambda:42)
    with pytest.raises(ValueError,match='Only the registered launchd'):c.tick(tmp_path,tmp_path/'state')
    assert not (tmp_path/'state').exists()


def test_metrics_oracle():
    rows=[{'label':0,'hybrid':[1,0,0]},{'label':2,'hybrid':[0,0,1]}]
    m=c.metrics(rows,'hybrid')
    assert m['accuracy']==1 and m['log_loss']==0 and m['brier']==0
