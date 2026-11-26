import pytest
from research_week1_r3.report import require_week_complete, require_complete, predictive_summary


def state():
    return dict(required_sessions=3,valid_observed_sessions=3,label_complete_decisions=0,
                missing_sessions=[],ledger_consistent=True,halted=False)


def test_week_close_can_report_operations_with_labels_pending():
    s=state()
    require_week_complete(s)
    with pytest.raises(ValueError): require_complete(s,0,False)
    s['label_complete_decisions']=3
    with pytest.raises(ValueError): require_complete(s,5,False)
    require_complete(s,6,False)


@pytest.mark.parametrize('key,value',[('valid_observed_sessions',2),('missing_sessions',['2026-10-01']),('ledger_consistent',False),('halted',True),('required_sessions',63)])
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
    import research_week1_r3.report as report
    from runtime_v2 import market
    release=tmp_path/'old-release'; release.mkdir()
    (release/'manifest.json').write_text('{}')
    workspace=tmp_path/'old-state'; hashes={}
    for date in ['2026-10-01','2026-10-02']:
        path=workspace/'days'/date/'result.json'; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'shadow':{'equity':10000},'deployment':{'equity':10000}}))
        hashes[date]=report.digest(path)
    proto=tmp_path/'protocol.json'
    proto.write_text(json.dumps({'amendment':{'parent_release_sha256':report.digest(release/'manifest.json')},'continuation':{'prior_release':str(release),'prior_workspace':str(workspace),'prior_receipt_sha256':hashes}}))
    monkeypatch.setattr(report,'PROTOCOL',proto)
    monkeypatch.setattr('subprocess.run',lambda *a,**k:SimpleNamespace(stdout=json.dumps({'valid_observed_sessions':2,'ledger_consistent':True,'halted':True,'trial_id':'prior','missing_sessions':['2026-10-05']})))
    frame=pd.DataFrame({'Close':[100.,100.,102.,98.]},index=pd.to_datetime(['2026-10-02','2026-10-05','2026-10-09','2026-10-12'],utc=True))
    monkeypatch.setattr(market,'load',lambda path,symbols,date:({'AAPL':frame.loc[:pd.Timestamp(date,tz='UTC')]},{}))
    receipts=[]
    for date in ['2026-10-08','2026-10-09','2026-10-12']:
        snap=tmp_path/date; snap.mkdir(); (snap/'manifest.json').write_text('{}')
        receipts.append({'session':date,'snapshot':str(snap),'snapshot_sha256':report.digest(snap/'manifest.json')})
    assert report.prior_segment({'symbols':['AAPL']},receipts[:1])['mature_decisions']==0
    one=report.prior_segment({'symbols':['AAPL']},receipts[:2])
    assert one['mature_decisions']==1 and one['outcomes'][0]['label']==2
    both=report.prior_segment({'symbols':['AAPL']},receipts)
    assert both['mature_decisions']==2 and both['outcomes'][1]['label']==0
    assert [x['realized_return'] for x in both['outcomes']]==pytest.approx([.02,-.02])
    assert hashes=={date:report.digest(workspace/'days'/date/'result.json') for date in hashes}
