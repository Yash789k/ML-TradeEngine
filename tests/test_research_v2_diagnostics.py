"""Oracles for new scientific diagnostics; no network or active pilot state."""
import numpy as np
import pandas as pd
import pytest
from research_v2.diagnostics import (
    account_module, block_effect, block_indices, drawdown_duration, holm,
    portfolio_metrics, probability_metrics, replay,
)


def test_multiclass_probability_oracle_and_maturity_guard():
    p=np.tile([.2,.3,.5],(3,1));m=probability_metrics([0,1,2],p)
    assert m['accuracy']==pytest.approx(1/3)
    assert m['log_loss']==pytest.approx(-np.log([.2,.3,.5]).mean())
    assert m['brier']==pytest.approx(np.square(p-np.eye(3)).sum(1).mean())
    assert m['predicted_up']==3
    with pytest.raises(ValueError,match='Unmatured'):probability_metrics([-1,1,2],p)
    with pytest.raises(ValueError,match='Invalid'):probability_metrics([0,1,2],p*.9)


def test_block_resampling_retains_temporal_runs_and_is_reproducible():
    a=block_indices(12,3,20);b=block_indices(12,3,20)
    np.testing.assert_array_equal(a,b)
    blocks=a.reshape(20,4,3)
    assert np.all((np.diff(blocks,axis=2)%12)==1)
    assert a.min()>=0 and a.max()<12


@pytest.mark.parametrize('effect',[.1,0,-.1])
def test_block_inference_known_constant_positive_null_negative(effect):
    got=block_effect(np.full(60,effect))
    assert got['effect']==pytest.approx(effect)
    assert got['ci_low']==pytest.approx(effect)
    assert got['ci_high']==pytest.approx(effect)
    assert got['p_two_sided']==pytest.approx(1 if effect==0 else 1/2001)
    assert got['mde80_approx']==pytest.approx(0,abs=1e-14)


def test_holm_actual_family_hand_calculation():
    np.testing.assert_allclose(holm([.03,.001,.04,.02]),[.06,.004,.06,.06])
    np.testing.assert_allclose(holm([.01]),[.01])


def test_drawdown_includes_initial_capital_and_counts_sessions():
    dd,longest,terminal=drawdown_duration([9900,10000,11000,10000,10100,11000,10900])
    assert dd==pytest.approx(-1/11)
    assert longest==2 and terminal==1


def test_flat_price_roundtrip_gross_net_and_both_side_fee_oracle():
    dates=pd.to_datetime(['2024-01-04','2024-01-05','2024-01-08'],utc=True)
    prices=pd.DataFrame({'AAA':[100.,100.,100.]},index=dates)
    target=pd.DataFrame({'AAA':[.08,0.,0.]},index=dates)
    history,account=replay(prices,target,account_module())
    metrics=portfolio_metrics(history,account.events)
    assert metrics['fills']==2
    assert metrics['terminal_equity']==pytest.approx(9997.6)
    assert metrics['combined_fees_usd']==pytest.approx(2.4)
    assert metrics['commission_usd']==pytest.approx(1.6)
    assert metrics['slippage_proxy_usd']==pytest.approx(.8)
    assert metrics['net_realized_sell_pnl_usd']==pytest.approx(-2.4)
    assert metrics['gross_fixed_fill_return']==pytest.approx(0)
    assert metrics['turnover_initial_capital']==pytest.approx(.16)
    assert metrics['terminal_positions']==0


def test_missing_price_halts_without_account_or_fill():
    dates=pd.date_range('2024-01-04',periods=3,tz='UTC')
    prices=pd.DataFrame({'AAA':[100.,np.nan,100.]},index=dates)
    with pytest.raises(ValueError,match='Incomplete'):
        replay(prices,pd.DataFrame({'AAA':[.08,0.,0.]},index=dates),account_module())


def test_cash_has_undefined_risk_ratios_instead_of_spurious_values():
    dates=pd.bdate_range('2024-01-04',periods=5,tz='UTC')
    prices=pd.DataFrame({'AAA':100.},index=dates)
    history,account=replay(prices,prices*0,account_module())
    metrics=portfolio_metrics(history,account.events)
    assert metrics['net_return']==0
    assert np.isnan(metrics['sharpe_zero_rf'])
    assert np.isnan(metrics['sortino_zero_mar'])


def test_benchmark_mismatch_and_nonfinite_values_are_rejected():
    from research_v2.diagnostics import paired_difference
    a=pd.Series([.1,.2],index=['a','b'])
    with pytest.raises(ValueError,match='identical'):
        paired_difference(a,pd.Series([.1,.2],index=['b','c']))
    with pytest.raises(ValueError,match='finite'):
        paired_difference(a,pd.Series([np.nan,.2],index=['a','b']))
    np.testing.assert_array_equal(paired_difference(a,a),[0,0])


def test_trade_episodes_separate_partial_sales_and_open_inventory():
    from research_v2.diagnostics import trade_episodes
    dates=pd.bdate_range('2024-01-02',periods=3,tz='UTC')
    events=[dict(kind='fill',symbol='AAA',side=side,qty=qty,price=price,fee=fee,asof=dates[i].isoformat())
            for i,side,qty,price,fee in [(0,'buy',2.,100.,.3),(1,'sell',1.,110.,.165),(2,'sell',1.,90.,.135)]]
    full=trade_episodes(events,{'AAA':90.},dates)
    assert len(full)==1 and full[0]['status']=='closed'
    assert full[0]['sell_fills']==2
    assert full[0]['net_pnl_usd']==pytest.approx(-.6)
    assert full[0]['combined_fees_usd']==pytest.approx(.6)
    partial=trade_episodes(events[:2],{'AAA':90.},dates)
    assert len(partial)==1 and partial[0]['status']=='open'
    assert partial[0]['net_pnl_usd']==pytest.approx(-.465)
    assert partial[0]['terminal_inventory_value']==90.
