"""Real rejected/corrected JPM bodies and isolated recovery fault injection."""
from copy import deepcopy
from pathlib import Path
import json
import pandas as pd
import pytest
from runtime_v2 import market, operations as op
from runtime_v2.calendar import calendar, InvalidOHLCError
from runtime_v2.storage import read, digest, save
from tests.test_collector_v2 import history, trained


def response(body):
    class Response:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return deepcopy(body)
    return Response()


def body(frame):
    return {'chart':{'result':[{'timestamp':[int(t.timestamp()) for t in frame.index],
        'indicators':{'quote':[{k.lower():frame[k].tolist() for k in ['Open','High','Low','Close','Volume']}],
                      'adjclose':[{'adjclose':frame.Close.tolist()}]}}]}}


def test_saved_jpm_bad_response_rejected_then_corrected_body_accepted(tmp_path,monkeypatch):
    fixtures=Path(__file__).parent/'fixtures'
    invalid=json.loads((fixtures/'JPM-invalid.json').read_text())
    corrected=json.loads((fixtures/'JPM-corrected.json').read_text())
    close=calendar().session_close('2026-10-21')
    monkeypatch.setattr(market.requests,'get',lambda *a,**k:response(invalid))
    failed=tmp_path/'failed'
    with pytest.raises(market.TransientDataError,match='latest provider OHLC'):
        market.download(failed,['JPM'],'2026-10-21',close+pd.Timedelta(minutes=30))
    meta=read(failed/'manifest.json');error=meta['errors']['JPM']
    assert meta['status']=='failed' and not meta['data'] and error['transient']
    assert error['rejected_bars'][0]['Open']>error['rejected_bars'][0]['High']
    assert error['response_sha256']==digest(failed/'responses/JPM.json')
    assert error['raw_sha256']==digest(failed/'JPM-raw.parquet')
    assert not (failed/'JPM.parquet').exists()
    original={str(p.relative_to(failed)):digest(p) for p in failed.rglob('*') if p.is_file()}
    monkeypatch.setattr(market.requests,'get',lambda *a,**k:response(corrected))
    retry=tmp_path/'retry'
    market.download(retry,['JPM'],'2026-10-21',close+pd.Timedelta(minutes=45))
    prices,_=market.load(retry,['JPM'],'2026-10-21',close+pd.Timedelta(minutes=45))
    last=prices['JPM'].iloc[-1]
    assert last.Open==last.High==pytest.approx(337.1199951171875)
    assert original=={str(p.relative_to(failed)):digest(p) for p in failed.rglob('*') if p.is_file()}
    # Timing here is an injected replay, never evidence Yahoo corrected by +45.


@pytest.mark.parametrize('fault',['historical_range','range_and_missing_history','nonfinite','nonpositive','duplicate','expired_window'])
def test_other_integrity_errors_never_become_provider_retries(tmp_path,monkeypatch,history,fault):
    f=history['AAPL'].loc[:'2026-05-20'].copy()
    minutes=30
    if fault=='historical_range': f.iloc[-2,f.columns.get_loc('Open')]=f.High.iloc[-2]+1
    else: f.iloc[-1,f.columns.get_loc('Open')]=f.High.iloc[-1]+1
    if fault=='range_and_missing_history': f=f.drop(f.index[-10])
    if fault=='nonfinite': f.iloc[-1,f.columns.get_loc('Close')]=float('nan')
    if fault=='nonpositive': f.iloc[-1,f.columns.get_loc('Low')]=-1
    if fault=='duplicate': f=pd.concat([f,f.iloc[-1:]])
    if fault=='expired_window': minutes=66
    monkeypatch.setattr(market.requests,'get',lambda *a,**k:response(body(f)))
    with pytest.raises(ValueError):
        market.download(tmp_path/'snapshot',['AAPL'],'2026-05-20',calendar().session_close('2026-05-20')+pd.Timedelta(minutes=minutes))
    assert read(tmp_path/'snapshot/manifest.json')['errors']['AAPL']['transient'] is False


def test_stored_invalid_bar_is_deterministic_even_with_updated_transport_hash(tmp_path,monkeypatch,history):
    f=history['AAPL'].loc[:'2026-05-20'].copy();now=calendar().session_close('2026-05-20')+pd.Timedelta(minutes=30)
    monkeypatch.setattr(market.requests,'get',lambda *a,**k:response(body(f)))
    loc=tmp_path/'snapshot';market.download(loc,['AAPL'],'2026-05-20',now)
    path=loc/'AAPL.parquet';saved=pd.read_parquet(path);saved.iloc[-1,saved.columns.get_loc('Open')]=saved.High.iloc[-1]+1;saved.to_parquet(path)
    meta=read(loc/'manifest.json');meta['data']['AAPL']['sha256']=digest(path);save(loc/'manifest.json',meta)
    with pytest.raises(InvalidOHLCError):market.load(loc,['AAPL'],'2026-05-20',now)


@pytest.mark.parametrize('recovers',[True,False])
def test_scheduler_provider_retry_is_bounded_and_never_commits_bad_input(tmp_path,monkeypatch,history,trained,recovers):
    config,release,_=trained
    cfg=dict(config,sessions=1,end=config['start'],operational_milestones=[1])
    date=cfg['start'];close=calendar().session_close(date);ws=tmp_path/'workspace';calls=[]
    current={'attempt':0}
    def fetch(url,**kwargs):
        symbol=url.rsplit('/',1)[-1];calls.append(symbol)
        f=history[symbol].loc[:date].copy()
        if symbol=='AAPL' and (not recovers or current['attempt']==0):
            f.iloc[-1,f.columns.get_loc('Open')]=f.High.iloc[-1]+1
        return response(body(f))
    monkeypatch.setattr(market.requests,'get',fetch)
    op.tick(release,ws,close+pd.Timedelta(minutes=30),cfg)
    assert not (ws/'halt.json').exists() and not list(ws.glob('*.sqlite'))
    assert not list((ws/'days').glob('*/input.json')) and not list((ws/'days').glob('*/result.json'))
    attempt=read(next((ws/'attempts').glob('*.json')))
    assert attempt['status']=='failed' and attempt['transient']
    failed=next((ws/'days'/date).glob('market-*'))
    before={str(p.relative_to(failed)):digest(p) for p in failed.rglob('*') if p.is_file()}
    for minute in [45,60,90]:
        current['attempt']+=1
        op.tick(release,ws,close+pd.Timedelta(minutes=minute),cfg)
    assert before=={str(p.relative_to(failed)):digest(p) for p in failed.rglob('*') if p.is_file()}
    state=op.status(release,ws,close+pd.Timedelta(minutes=90),cfg)
    if recovers:
        assert len(calls)==4 and state['valid_observed_sessions']==1 and state['ledger_consistent'] and not state['halted']
        assert read(ws/'events'/f'{date}-retry45.json')['status']=='complete'
        assert read(ws/'events'/f'{date}-retry60.json')['status']=='covered_by_valid_receipt'
        assert op.read_record(ws/'milestones/1.json')['valid_observed_sessions']==1
    else:
        assert len(calls)==6 and state['valid_observed_sessions']==0 and state['halted']
        assert len(list((ws/'attempts').glob('*.json')))==3 and not list(ws.glob('*.sqlite'))
        assert read(ws/'events'/f'{date}-watchdog.json')['status']=='missing_expected_receipt'
