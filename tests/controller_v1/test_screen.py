"""Causal delayed-feedback and integrity oracles for the isolated controller."""
import numpy as np
import pandas as pd
import pytest
from research_controller_v1.screen import DelayedHedge, bootstrap, checked, digest, expert_probabilities, forecast, holm, stack_features, valid_probabilities


def make_panel():
    dates=pd.bdate_range('2024-12-01',periods=60,tz='UTC')
    rows=[]
    for i,date in enumerate(dates[:-6]):
        for symbol in ['A','B']:
            row=dict(date=date,endpoint=dates[i+6],symbol=symbol,label=i%3)
            for name,p in [('prior',[.3,.4,.3]),('linear',[.2,.3,.5]),('tree',[.5,.3,.2])]:
                row.update({f'{name}_{c}':v for c,v in enumerate(p)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_endpoint_t_plus_six_and_strict_delay():
    dates=pd.bdate_range('2025-01-01',periods=8,tz='UTC');h=DelayedHedge()
    p=np.array([[[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]])
    assert not h.feedback('x',dates[6],dates[5],p,[0])
    assert not h.feedback('x',dates[6],dates[6],p,[0])
    np.testing.assert_allclose(h.weights,[1/3]*3)
    assert h.feedback('x',dates[6],dates[7],p,[0])
    assert h.weights[0]>h.weights[1]


def test_cross_section_is_one_average_update():
    p=np.array([[[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]],[[.2,.4,.4],[.7,.2,.1],[.2,.2,.6]]])
    a=DelayedHedge();b=DelayedHedge()
    a.feedback('x','2025-01-01','2025-01-02',p,[0,2])
    b.feedback('x','2025-01-01','2025-01-02',np.tile(p,(10,1,1)),[0,2]*10)
    np.testing.assert_allclose(a.weights,b.weights)
    loss=-np.log(np.array([[.8,.1,.1],[.4,.1,.6]])).mean(0)
    expected=np.exp(-.1*loss);expected/=expected.sum()
    np.testing.assert_allclose(a.weights,expected)


def test_duplicate_feedback_is_idempotent_and_deterministic():
    p=np.array([[[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]]);a=DelayedHedge();b=DelayedHedge()
    for h in [a,b]:assert h.feedback('x','2025-01-01','2025-01-02',p,[0])
    before=a.weights.copy()
    assert not a.feedback('x','2025-01-01','2025-01-03',p,[1])
    np.testing.assert_array_equal(a.weights,before)
    np.testing.assert_array_equal(a.weights,b.weights)


def test_future_label_mutation_does_not_change_earlier_forecasts():
    panel=make_panel();cutoff=pd.Timestamp('2025-01-15',tz='UTC')
    changed=panel.copy();mask=changed.endpoint>=cutoff
    changed.loc[mask,'label']=(changed.loc[mask,'label']+1)%3
    a,wa,_=forecast(panel,use_stack=False);b,wb,_=forecast(changed,use_stack=False)
    pd.testing.assert_frame_equal(a[a.date<=cutoff][['date','symbol','model','p0','p1','p2']],b[b.date<=cutoff][['date','symbol','model','p0','p1','p2']])
    pd.testing.assert_frame_equal(wa[pd.to_datetime(wa.date)<=cutoff],wb[pd.to_datetime(wb.date)<=cutoff])


@pytest.mark.parametrize('p',[[[.2,.3,.6]],[[np.nan,.3,.7]],[[-.1,.5,.6]],[[.5,.5]]])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError,match='Invalid probability'):valid_probabilities(p)


def test_checksum_tampering_fails(tmp_path):
    path=tmp_path/'input';path.write_bytes(b'original');expected=digest(path)
    checked(path,expected);path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Checksum mismatch'):checked(path,expected)


def test_bootstrap_and_holm_oracles():
    blocks=np.repeat(['H1','H2'],30)
    null=bootstrap(np.zeros(60),blocks,10)
    assert null['p_value']==1 and null['ci_low']==null['ci_high']==0
    positive=bootstrap(np.ones(60),blocks,10)
    assert positive['effect']==positive['ci_low']==positive['ci_high']==1
    np.testing.assert_allclose(holm([.01,.03,.1]),[.03,.06,.1])


def test_stack_two_mature_refits_and_training_only_scaler():
    dates=pd.bdate_range('2024-01-02','2025-07-15',tz='UTC')
    rng=np.random.default_rng(9);rows=[]
    for i,date in enumerate(dates[:-6]):
        for symbol in ['A','B']:
            row=dict(date=date,endpoint=dates[i+6],symbol=symbol,label=i%3)
            for name in ['prior','linear','tree']:
                p=rng.dirichlet([3,3,3]);row.update({f'{name}_{c}':v for c,v in enumerate(p)})
            rows.append(row)
    panel=pd.DataFrame(rows);_,_,fits=forecast(panel)
    assert len(fits)==2
    for fit in fits:
        decision=pd.Timestamp(fit['decision'])
        training=panel[(panel.date<decision)&(panel.endpoint<decision)]
        assert pd.Timestamp(fit['latest_endpoint'])<decision
        assert len(training)==fit['n_rows']
        np.testing.assert_allclose(fit['scaler_mean'],stack_features(expert_probabilities(training)).mean(axis=0))
