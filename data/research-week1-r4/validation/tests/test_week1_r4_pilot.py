"""Exact amended calendar plus a complete isolated historical pipeline replay."""
import json
from pathlib import Path
import pandas as pd
import pytest
from runtime_v2 import operations as op, market
from runtime_v2.calendar import calendar
from runtime_v2.policy import session, refit_due
from tests.test_collector_v2 import history, trained, snapshot


def test_week1_r4_exact_calendar_and_no_extra_decisions():
    root=Path(op.__file__).resolve().parents[1]
    config_path=root/'config.json'
    if not config_path.exists():
        config_path=root/'data/releases/research-v2-runtime-20261014-week1-r4/config.json'
    cfg=json.loads(config_path.read_text())
    ev=op.events(cfg)
    daily=[e for e in ev if e['kind']=='daily']
    tail=[e for e in ev if e['kind']=='outcomes']
    assert [e['session'] for e in daily]==['2026-10-14','2026-10-15','2026-10-16']
    assert cfg['sessions']==len(daily)==3
    assert cfg['trial_id']=='PILOT-TREND-WEEK1-R4-20261014'
    assert cfg['anchor']=='2026-09-29'
    assert pd.Timestamp(cfg['registered_at']) < calendar().session_close(daily[0]['session'])
    assert len(tail)==6 and tail[-1]['session']=='2026-10-26'
    assert tail[-1]['due'].tz_convert('Asia/Hong_Kong').isoformat()=='2026-10-27T04:30:00+08:00'
    assert not any(refit_due(e['session'],cfg['anchor']) for e in daily)
    assert cfg['external_orders'] is False and cfg['deployment']=='cash'


def test_week1_r4_real_replay_reaches_three_decisions_and_six_tail_receipts(tmp_path,trained,history):
    cfg,release,models=trained
    cfg=dict(cfg)
    cal=calendar(); first=cal.sessions.get_loc(session(cfg['start']))
    dates=cal.sessions[first:first+3]
    cfg.update(sessions=3,end=str(dates[-1].date()))
    ws=tmp_path/'isolated-three-session-replay'
    results=[]
    for d in dates:
        loc=snapshot(ws/'days'/str(d.date())/'market',history,d)
        results.append(op.daily(release,ws,d,snapshot=loc,now=cal.session_close(d)+pd.Timedelta(minutes=30),validated_config=cfg))
    assert not any(r['refit'] for r in results)
    state=op.status(release,ws,cal.session_close(dates[-1])+pd.Timedelta(minutes=90),cfg)
    assert state['valid_observed_sessions']==3 and state['label_complete_decisions']==0
    assert state['prospective_status']=='incomplete'
    with pytest.raises(ValueError,match='outside fixed pilot'):
        op.daily(release,ws,cal.sessions[first+3],now=cal.session_close(cal.sessions[first+3])+pd.Timedelta(minutes=30),validated_config=cfg)
    for d in cal.sessions[first+3:first+9]:
        loc=snapshot(ws/'outcome-snapshots'/str(d.date()),history,d)
        data,_=market.load(loc,cfg['symbols'],d)
        receipt=op.terminal_close(ws,data,d,loc,cfg,cal.session_close(d)+pd.Timedelta(minutes=30))
        op.mature(ws,data,d,cfg,snapshot=loc)
        assert receipt['mode']=='outcome_tail' and not receipt['rebalance'] and not receipt['refit']
    state=op.status(release,ws,cal.session_close(d)+pd.Timedelta(minutes=90),cfg)
    assert state['valid_observed_sessions']==state['label_complete_decisions']==3
    assert state['terminal_observed_sessions']==6 and state['ledger_consistent']
    assert not state['missing_sessions'] and not state['halted']
    assert state['prospective_status']=='collected_not_economic_approval'


def test_continuation_milestone_uses_registered_three_sessions(tmp_path, trained, history):
    cfg, release, models = trained
    dates = calendar().sessions_in_range(cfg['start'], cfg['end'])[:3]
    cfg = dict(cfg, sessions=3, end=str(dates[-1].date()), operational_milestones=[3])
    ws = tmp_path/'milestone-replay'
    for d in dates:
        snap = snapshot(ws/'days'/str(d.date())/'market', history, d)
        op.daily(release, ws, d, snapshot=snap, now=calendar().session_close(d)+pd.Timedelta(minutes=30), validated_config=cfg)
    op.tick(release, ws, now=calendar().session_close(dates[-1])+pd.Timedelta(minutes=90), validated_config=cfg)
    milestone = op.read_record(ws/'milestones/3.json')
    assert milestone['valid_observed_sessions'] == 3
    assert milestone['ledger_consistent'] and not milestone['halted']
    assert not (ws/'milestones/5.json').exists()


def test_power_guard_expiry_and_relogin():
    import importlib.util
    from datetime import datetime, timezone
    root = Path(op.__file__).resolve().parents[1]
    path = root/'scripts/power_guard.py'
    spec = importlib.util.spec_from_file_location('qualified_power_guard', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    until = '2026-10-22T22:00:00+00:00'
    assert module.remaining_seconds(until, datetime(2026,10,22,21,0,tzinfo=timezone.utc)) == 3600
    assert module.remaining_seconds(until, datetime(2026,10,23,0,0,tzinfo=timezone.utc)) == 0
    with pytest.raises(ValueError): module.remaining_seconds('2026-10-22T22:00:00')


def test_same_day_commissioning_once_before_first_close(tmp_path, monkeypatch):
    from tests.test_collector_v2 import configuration, fake_release
    cfg=configuration('2026-10-14',3); release=fake_release(tmp_path,cfg)
    ws=tmp_path/'same-day'; calls=[]
    def preview(release,workspace,**kwargs):
        assert kwargs['mode']=='commissioning'
        assert workspace==ws/'commissioning'
        calls.append(op.completed(kwargs['now']).date().isoformat())
        return {'content_sha256':'test-preview'}
    monkeypatch.setattr(op,'backup',lambda *args:None)
    now=calendar().session_open(cfg['start'])
    first=op.tick(release,ws,now=now,validated_config=cfg,run=preview)
    op.tick(release,ws,now=now+pd.Timedelta(minutes=1),validated_config=cfg,run=preview)
    assert calls==['2026-10-13']
    assert first['valid_observed_sessions']==0
    assert (ws/'commissioning-complete.json').exists()
    assert not list((ws/'days').glob('*/result.json'))


def test_no_late_commissioning_at_first_close(tmp_path):
    from tests.test_collector_v2 import configuration, fake_release
    cfg=configuration('2026-10-14',3); release=fake_release(tmp_path,cfg)
    def forbidden(*args,**kwargs):raise AssertionError('No commissioning after first close')
    ws=tmp_path/'late'
    state=op.tick(release,ws,now=calendar().session_close(cfg['start']),validated_config=cfg,run=forbidden)
    assert state['valid_observed_sessions']==0
    assert not (ws/'commissioning-attempted.json').exists()
