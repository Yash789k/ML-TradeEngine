import pytest
from research_week1_r6.report_20261023 import require_week_complete, require_complete, predictive_summary


def state():
    return dict(required_sessions=1,valid_observed_sessions=1,label_complete_decisions=0,
                missing_sessions=[],ledger_consistent=True,halted=False)


def test_week_close_can_report_operations_with_labels_pending():
    s=state()
    require_week_complete(s)
    with pytest.raises(ValueError): require_complete(s,0,False)
    s['label_complete_decisions']=1
    with pytest.raises(ValueError): require_complete(s,5,False)
    require_complete(s,6,False)


@pytest.mark.parametrize('key,value',[('valid_observed_sessions',0),('missing_sessions',['2026-10-01']),('ledger_consistent',False),('halted',True),('required_sessions',63)])
def test_week_close_rejects_missing_or_invalid_operations(key,value):
    s=state();s[key]=value
    with pytest.raises(ValueError):require_week_complete(s)


def test_five_dates_cannot_masquerade_as_many_independent_stock_observations():
    rows=[dict(session=str(d),symbol=str(s),label=2,hybrid=[.05,.05,.9],prior=[.3,.3,.4]) for d in range(5) for s in range(22)]
    summary=predictive_summary(rows,final=True)
    assert summary['decision_dates']==5 and summary['labeled_rows']==110
    assert summary['paired_inference']==[]


def test_archived_labels_require_their_endpoints_and_leave_accounts_untouched(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import pandas as pd
    import research_week1_r6.report_20261023 as report
    from runtime_v2 import market
    release=tmp_path/'old-release'; release.mkdir()
    (release/'manifest.json').write_text('{}')
    workspace=tmp_path/'old-state'; hashes={}
    for date in ['2026-10-01','2026-10-02']:
        path=workspace/'days'/date/'result.json'; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'shadow':{'equity':10000},'deployment':{'equity':10000},'signals':[{'symbol':'AAPL','probabilities':[.1,.2,.7]}],'models':{'AAPL':'fake-model'}}))
        hashes[date]=report.digest(path)
    proto=tmp_path/'protocol.json'
    proto.write_text(json.dumps({'amendment':{'parent_release_sha256':report.digest(release/'manifest.json')},'continuation':{'prior_release':str(release),'prior_release_sha256':report.digest(release/'manifest.json'),'prior_workspace':str(workspace),'prior_receipt_sha256':hashes}}))
    monkeypatch.setattr(report,'PROTOCOL',proto)
    from runtime_v2.model import AdaptiveModel
    import numpy as np
    monkeypatch.setattr(AdaptiveModel,'load',lambda path:SimpleNamespace(prior=np.array([.3,.3,.4])))
    previous=json.loads(proto.read_text())['continuation'];previous['segment']='r2'
    monkeypatch.setattr('subprocess.run',lambda *a,**k:SimpleNamespace(stdout=json.dumps({'valid_observed_sessions':2,'ledger_consistent':True,'halted':True,'trial_id':'prior','missing_sessions':['2026-10-05']})))
    frame=pd.DataFrame({'Close':[100.,100.,102.,98.]},index=pd.to_datetime(['2026-10-02','2026-10-05','2026-10-09','2026-10-12'],utc=True))
    monkeypatch.setattr(market,'load',lambda path,symbols,date:({'AAPL':frame.loc[:pd.Timestamp(date,tz='UTC')]},{}))
    receipts=[]
    for date in ['2026-10-08','2026-10-09','2026-10-12']:
        snap=tmp_path/date; snap.mkdir(); (snap/'manifest.json').write_text('{}')
        receipts.append({'session':date,'snapshot':str(snap),'snapshot_sha256':report.digest(snap/'manifest.json')})
    assert report.prior_segment({'symbols':['AAPL'],'capital':10000},receipts[:1],previous)['mature_decisions']==0
    one=report.prior_segment({'symbols':['AAPL'],'capital':10000},receipts[:2],previous)
    assert one['mature_decisions']==1 and one['outcomes'][0]['label']==2
    both=report.prior_segment({'symbols':['AAPL'],'capital':10000},receipts,previous)
    assert both['mature_decisions']==2 and both['outcomes'][1]['label']==0
    assert [x['realized_return'] for x in both['outcomes']]==pytest.approx([.02,-.02])
    assert hashes=={date:report.digest(workspace/'days'/date/'result.json') for date in hashes}


@pytest.mark.parametrize('halted,count',[(False,0),(True,1),(True,0)])
def test_failed_continuation_preserves_zero_observations_and_halt(tmp_path,monkeypatch,halted,count):
    import json
    from types import SimpleNamespace
    import research_week1_r6.report_20261023 as report
    release=tmp_path/'r3';release.mkdir();(release/'manifest.json').write_text('{}')
    item={'release':str(release),'workspace':str(tmp_path/'old-state'),'release_sha256':report.digest(release/'manifest.json'),
          'status_at_supersession':{'missing_sessions':['2026-10-12','2026-10-13']},'unattempted_after_supersession':['2026-10-14']}
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps({'continuation':{'failed_continuations':[item]}}))
    monkeypatch.setattr(report,'PROTOCOL',protocol)
    monkeypatch.setattr('subprocess.run',lambda *a,**k:SimpleNamespace(stdout=json.dumps({'valid_observed_sessions':count,'halted':halted,'ledger_consistent':True})))
    if not halted or count:
        with pytest.raises(ValueError):report.failed_continuations()
    else:
        result=report.failed_continuations()[0]
        assert result['counts_as_prospective']==0
        assert result['status_at_supersession']['missing_sessions']==['2026-10-12','2026-10-13']
        assert result['unattempted_after_supersession']==['2026-10-14']


def test_prior_segments_reject_duplicate_observations(tmp_path,monkeypatch):
    import json
    import research_week1_r6.report_20261023 as report
    path=tmp_path/'protocol.json'
    path.write_text(json.dumps({'continuation':{'prior_segments':[{'prior_receipt_sha256':{'2026-10-01':'a','2026-10-02':'b'}},{'prior_receipt_sha256':{'2026-10-01':'a','2026-10-02':'b'}}]}}))
    monkeypatch.setattr(report,'PROTOCOL',path)
    with pytest.raises(ValueError,match='distinct archived'):report.prior_segments({},[])


def test_four_archived_dates_aggregate_without_splicing_returns(tmp_path,monkeypatch):
    import json
    import research_week1_r6.report_20261023 as report
    path=tmp_path/'protocol.json'
    segments=[{'segment':'r2','prior_receipt_sha256':{'2026-10-01':'a','2026-10-02':'b'}},{'segment':'r4','prior_receipt_sha256':{'2026-10-14':'c','2026-10-15':'d'}}]
    path.write_text(json.dumps({'continuation':{'prior_segments':segments}}));monkeypatch.setattr(report,'PROTOCOL',path)
    def prior(config,receipts,segment):
        return {'segment':segment['segment'],'valid_decisions':2,'mature_decisions':2,'prediction_rows':[{'session':d} for d in segment['prior_receipt_sha256']], 'net_returns_at_last_actual_receipt':{'shadow':-.001}}
    monkeypatch.setattr(report,'prior_segment',prior)
    result=report.prior_segments({},[])
    assert result['valid_decisions']==result['mature_decisions']==4
    assert len(result['segments'])==2 and len(result['prediction_rows'])==4
    assert 'net_returns_at_last_actual_receipt' not in result


def test_active_outcome_keeps_r6_identity_and_rejects_another_trial():
    from research_week1_r6.report_20261023 import active_prediction_row
    d={'trial_id':'PILOT-TREND-WEEK1-R6-20261022','session':'2026-10-22','signals':[{'symbol':'SPY','probabilities':[.2,.3,.5]}]}
    actual={'symbol':'SPY','label':1,'realized_return':.001}
    row=active_prediction_row(d,actual,[.3,.3,.4])
    assert row==dict(session='2026-10-22',segment='r6',symbol='SPY',label=1,realized_return=.001,hybrid=[.2,.3,.5],prior=[.3,.3,.4])
    with pytest.raises(ValueError,match='different trial'):
        active_prediction_row(dict(d,trial_id='PILOT-TREND-WEEK1-R5-20261021'),actual,[.3,.3,.4])
