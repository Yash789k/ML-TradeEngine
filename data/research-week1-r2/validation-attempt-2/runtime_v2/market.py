"""Public completed bars plus raw corporate-action response provenance."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import pandas as pd
import requests
from runtime_v2.calendar import observed_equity_bars
from runtime_v2.policy import utc, session
from runtime_v2.storage import read, save, digest


class TransientDataError(RuntimeError):
    pass


def download(destination, symbols, date, now=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    now = utc(now)
    meta = dict(status='fetching', provider='Yahoo Finance public chart v8', retrieved_at=now.isoformat(),
                data={}, errors={}, expected_session=str(session(date).date()),
                convention='Current-vintage dividend/split-adjusted OHLC proxy; volume unadjusted; never broker shares')
    def fetch(symbol):
        start = int((now - pd.Timedelta(days=3650)).timestamp())
        response = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',
                                params=dict(interval='1d', period1=start, period2=int(now.timestamp()), events='splits,dividends'),
                                headers={'User-Agent':'Mozilla/5.0'}, timeout=25)
        if 400 <= response.status_code < 500 and response.status_code != 429:
            raise ValueError(f'Deterministic provider HTTP {response.status_code}: {symbol}')
        response.raise_for_status()
        body = response.json()
        save(destination / 'responses' / f'{symbol}.json', body, immutable=True)
        chart = body['chart']['result'][0]
        q = chart['indicators']['quote'][0]
        adj = chart['indicators']['adjclose'][0]['adjclose']
        raw = pd.DataFrame({k.title():q[k] for k in ['open','high','low','close','volume']},
                           index=pd.to_datetime(chart['timestamp'], unit='s', utc=True).normalize())
        raw['Adj_Close'] = adj
        raw.index.name = 'Date'
        raw.to_parquet(destination / f'{symbol}-raw.parquet')
        # Partial current sessions may be returned: retain raw and validate only completed.
        raw = raw.loc[:session(date).tz_localize('UTC')]
        adjusted = raw.copy()
        ratio = raw.Adj_Close / raw.Close
        adjusted[['Open','High','Low','Close']] = raw[['Open','High','Low','Close']].mul(ratio, axis=0)
        adjusted = observed_equity_bars(adjusted, now=now)
        if session(adjusted.index[-1]) != session(date):
            raise TransientDataError(f'{symbol}: stale provider last bar {adjusted.index[-1]}')
        adjusted.to_parquet(destination / f'{symbol}.parquet')
        return dict(rows=len(adjusted), start=str(adjusted.index[0]), end=str(adjusted.index[-1]),
                    sha256=digest(destination / f'{symbol}.parquet'),
                    raw_sha256=digest(destination / f'{symbol}-raw.parquet'),
                    response_sha256=digest(destination / 'responses' / f'{symbol}.json'),
                    corporate_actions=chart.get('events', {}))
    transient = True
    meta["attempted_requests"] = len(symbols)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch,s):s for s in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                meta['data'][symbol] = future.result()
            except Exception as exc:
                is_transient = isinstance(exc,(requests.RequestException,TransientDataError))
                transient = transient and is_transient
                meta['errors'][symbol] = dict(type=type(exc).__name__, message=str(exc), transient=is_transient)
            save(destination / 'manifest.json', meta)
    meta['status'] = 'failed' if meta['errors'] else 'complete'
    meta['finished_at'] = utc().isoformat()
    save(destination / 'manifest.json', meta)
    if meta['errors']:
        raise (TransientDataError if transient else ValueError)(f'Incomplete provider batch: {meta["errors"]}')
    return meta


def load(path, symbols, date, now=None):
    path = Path(path)
    meta = read(path / 'manifest.json')
    if meta['status'] != 'complete' or meta.get('errors') or set(meta['data']) != set(symbols):
        raise ValueError('Incomplete snapshot/universe')
    result = {}
    for symbol in symbols:
        item = meta['data'][symbol]
        for suffix, key in [('.parquet','sha256'),('-raw.parquet','raw_sha256')]:
            if digest(path / (symbol + suffix)) != item[key]:
                raise ValueError(f'Data hash mismatch: {symbol}')
        if item.get('response_sha256') and digest(path / 'responses' / f'{symbol}.json') != item['response_sha256']:
            raise ValueError('Provider response changed')
        raw = pd.read_parquet(path / f'{symbol}.parquet')
        frame = observed_equity_bars(raw, now=utc(now))
        if not frame.index.equals(raw.index) or session(frame.index[-1]) != session(date):
            raise ValueError('Stale/partial/invalid snapshot')
        result[symbol] = frame
    return result, meta
