"""Exact amended calendar plus a complete isolated historical pipeline replay."""
import json
from pathlib import Path
import pandas as pd
import pytest
from runtime_v2 import operations as op, market
from runtime_v2.calendar import calendar
from runtime_v2.policy import session, refit_due
from test_collector_v2 import history, trained, snapshot


def test_week1_exact_calendar_and_no_extra_decisions():
    root=Path(op.__file__).resolve().parents[1]
    cfg=json.loads((root/'config.json').read_text())
    ev=op.events(cfg)
    daily=[e for e in ev if e['kind']=='daily']
    tail=[e for e in ev if e['kind']=='outcomes']
    assert [e['session'] for e in daily]==['2026-09-29','2026-09-30','2026-10-01','2026-10-02','2026-10-05']
    assert cfg['sessions']==len(daily)==5
    assert len(tail)==6 and tail[-1]['session']=='2026-10-13'
    assert tail[-1]['due'].tz_convert('Asia/Hong_Kong').isoformat()=='2026-10-14T04:30:00+08:00'
    assert not any(refit_due(e['session'],cfg['anchor']) for e in daily)
    assert cfg['external_orders'] is False and cfg['deployment']=='cash'


def test_week1_real_replay_reaches_five_decisions_and_six_tail_receipts(tmp_path,trained,history):
    cfg,release,models=trained
    cfg=dict(cfg)
    cal=calendar(); first=cal.sessions.get_loc(session(cfg['start']))
    dates=cal.sessions[first:first+5]
    cfg.update(sessions=5,end=str(dates[-1].date()))
    ws=tmp_path/'isolated-five-session-replay'
    results=[]
    for d in dates:
        loc=snapshot(ws/'days'/str(d.date())/'market',history,d)
        results.append(op.daily(release,ws,d,snapshot=loc,now=cal.session_close(d)+pd.Timedelta(minutes=30),validated_config=cfg))
    assert not any(r['refit'] for r in results)
    state=op.status(release,ws,cal.session_close(dates[-1])+pd.Timedelta(minutes=90),cfg)
    assert state['valid_observed_sessions']==5 and state['label_complete_decisions']==0
    assert state['prospective_status']=='incomplete'
    with pytest.raises(ValueError,match='outside fixed pilot'):
        op.daily(release,ws,cal.sessions[first+5],now=cal.session_close(cal.sessions[first+5])+pd.Timedelta(minutes=30),validated_config=cfg)
    for d in cal.sessions[first+5:first+11]:
        loc=snapshot(ws/'outcome-snapshots'/str(d.date()),history,d)
        data,_=market.load(loc,cfg['symbols'],d)
        receipt=op.terminal_close(ws,data,d,loc,cfg,cal.session_close(d)+pd.Timedelta(minutes=30))
        op.mature(ws,data,d,cfg,snapshot=loc)
        assert receipt['mode']=='outcome_tail' and not receipt['rebalance'] and not receipt['refit']
    state=op.status(release,ws,cal.session_close(d)+pd.Timedelta(minutes=90),cfg)
    assert state['valid_observed_sessions']==state['label_complete_decisions']==5
    assert state['terminal_observed_sessions']==6 and state['ledger_consistent']
    assert not state['missing_sessions'] and not state['halted']
    assert state['prospective_status']=='collected_not_economic_approval'
