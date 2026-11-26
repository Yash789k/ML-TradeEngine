"""Mathematical/causal contracts for controlled research, not profitability gates."""
import numpy as np
import pandas as pd
import pytest
from research_v2.experiments import Config, ControlledModel, bootstrap, holm, make_panel, partition, temp


@pytest.fixture(scope='module')
def panel():
    rng=np.random.default_rng(126)
    idx=pd.bdate_range('2015-01-01',periods=1700,tz='UTC')
    data={}
    for s in ['SPY','TEST','HELD']:
        close=100*np.exp(np.cumsum(rng.normal(.0003,.015,len(idx))))
        data[s]=pd.DataFrame(dict(Open=close*(1+rng.normal(0,.001,len(idx))),High=close*1.01,Low=close*.99,Close=close,Volume=rng.uniform(1000,10000,len(idx))),index=idx)
    return make_panel(data)


def test_endpoint_purged_disjoint_roles(panel):
    x,y,r,end=panel['TEST'];decision=x.index[-63]
    f,c,g=partition(x,y,end,decision,Config())
    assert end.iloc[f].max()<x.index[c[0]]
    assert end.iloc[c].max()<x.index[g[0]]
    assert end.iloc[g].max()<decision
    assert not (set(f)&set(c) or set(f)&set(g) or set(c)&set(g))
    assert len(f)==870 and len(c)==len(g)==63


def test_future_mutation_and_training_only_scaler(panel):
    x,y,r,end=panel['TEST'];decision=x.index[-63]
    model=ControlledModel().fit(panel,['TEST'],decision)
    changed=dict(panel)
    xx=x.copy();yy=y.copy();rr=r.copy()
    ix=xx.index>=decision;xx.loc[ix]=xx.loc[ix]*5;yy.loc[ix]=(yy.loc[ix]+1)%3;rr.loc[ix]=.9
    changed['TEST']=(xx,yy,rr,end)
    again=ControlledModel().fit(changed,['TEST'],decision)
    np.testing.assert_array_equal(model.linear[0].mean_,again.linear[0].mean_)
    np.testing.assert_array_equal(model.prior,again.prior)
    assert model.metadata['training_hash']==again.metadata['training_hash']
    f,_,_=partition(x,y,end,decision,Config())
    np.testing.assert_allclose(model.linear[0].mean_,x.iloc[f].mean(),rtol=1e-12)
    a=model.predict(x.loc[decision:].iloc[:2]);b=again.predict(x.loc[decision:].iloc[:2])
    for name in a:np.testing.assert_array_equal(a[name],b[name])


def test_transfer_excludes_held_target(panel):
    decision=panel['TEST'][0].index[-63]
    m=ControlledModel().fit(panel,['TEST','SPY'],decision)
    altered=dict(panel);x,y,r,e=panel['HELD'];altered['HELD']=(x*99,(y+1)%3,r+1,e)
    n=ControlledModel().fit(altered,['TEST','SPY'],decision)
    assert 'HELD' not in m.symbols
    assert m.metadata['training_hash']==n.metadata['training_hash']
    np.testing.assert_array_equal(m.linear[0].mean_,n.linear[0].mean_)


def test_probability_schema_serialization_and_tamper(panel,tmp_path):
    decision=panel['TEST'][0].index[-63]
    m=ControlledModel().fit(panel,['TEST'],decision)
    x=panel['TEST'][0].loc[decision:].iloc[:3]
    a=m.predict(x)
    for p in a.values():
        assert np.isfinite(p).all() and (p>=0).all()
        np.testing.assert_allclose(p.sum(1),1)
    with pytest.raises(ValueError,match='schema'):
        m.predict(x[x.columns[::-1]])
    m.save(tmp_path/'model');n=ControlledModel.load(tmp_path/'model')
    for k,v in n.predict(x).items():np.testing.assert_array_equal(v,a[k])
    model_path=tmp_path/'model/model.joblib'
    model_path.write_bytes(model_path.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='checksum'):ControlledModel.load(tmp_path/'model')


def test_mature_labels_required(panel):
    x,y,r,end=panel['TEST'];end=end.copy();end.iloc[-100:]=pd.NaT
    f,c,g=partition(x,y,end,x.index[-1],Config())
    assert max(g)<len(x)-100
    assert y.iloc[f].ge(0).all()


def test_block_inference_null_positive_negative_and_holm():
    folds=np.repeat(np.arange(4),63)
    assert bootstrap(np.zeros(252),folds,10)['p_value']==1
    for sign in [-1,1]:
        r=bootstrap(np.ones(252)*sign,folds,10)
        assert r['effect']==sign and r['ci_low']==r['ci_high']==sign and r['p_value']<.001
    np.testing.assert_allclose(holm([.01,.03,.1]),[.03,.06,.1])
    np.testing.assert_allclose(temp(np.array([[.2,.3,.5]]),1),[[.2,.3,.5]])


def test_actual_cli_checkpoint_portable_loader():
    """Checks the actual -m artifact rather than only imported test models."""
    from pathlib import Path
    from research_v2.models import load_checkpoint
    from research_v2.experiments import ROOT, load_data
    checkpoint=ROOT/'data/research-v2/experiments/checkpoints/base-AAPL-F0-R0-S42'
    if not (checkpoint/'complete.json').exists():
        pytest.skip('Versioned real-data experiment checkpoint not available')
    model=load_checkpoint(checkpoint/'model')
    panel=make_panel(load_data())
    saved=pd.read_parquet(checkpoint/'predictions.parquet')
    idx=pd.DatetimeIndex(sorted(saved.date.unique()))
    actual=model.predict(panel['AAPL'][0].loc[idx,model.columns])
    for name,p in actual.items():
        expected=saved[saved.model.eq(name)].sort_values('date')[['p0','p1','p2']].to_numpy()
        np.testing.assert_allclose(p,expected,rtol=1e-12,atol=1e-14)
