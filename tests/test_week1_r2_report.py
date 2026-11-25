import pytest
from research_week1_r2.report import require_week_complete, require_complete, predictive_summary


def state():
    return dict(required_sessions=5,valid_observed_sessions=5,label_complete_decisions=0,
                missing_sessions=[],ledger_consistent=True,halted=False)


def test_week_close_can_report_operations_with_labels_pending():
    s=state()
    require_week_complete(s)
    with pytest.raises(ValueError): require_complete(s,0,False)
    s['label_complete_decisions']=5
    with pytest.raises(ValueError): require_complete(s,5,False)
    require_complete(s,6,False)


@pytest.mark.parametrize('key,value',[('valid_observed_sessions',4),('missing_sessions',['2026-10-01']),('ledger_consistent',False),('halted',True),('required_sessions',63)])
def test_week_close_rejects_missing_or_invalid_operations(key,value):
    s=state();s[key]=value
    with pytest.raises(ValueError):require_week_complete(s)


def test_five_dates_cannot_masquerade_as_many_independent_stock_observations():
    rows=[dict(session=str(d),symbol=str(s),label=2,hybrid=[.05,.05,.9],prior=[.3,.3,.4]) for d in range(5) for s in range(22)]
    summary=predictive_summary(rows,final=True)
    assert summary['decision_dates']==5 and summary['labeled_rows']==110
    assert summary['paired_inference']==[]
