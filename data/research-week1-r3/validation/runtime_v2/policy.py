"""One versioned causal policy for historical replay, CLI, UI and scheduler.

The carried startup model was fit on 25 September. With anchor at pilot index
zero, refreshes occur at +21,+42,+63 XNYS sessions. This explicit v2 contract
is different from v1 historical walk-forward anchoring; no equivalence is claimed.
"""
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from runtime_v2.calendar import calendar
from runtime_v2.model import AdaptiveModel, features
from runtime_v2.risk import constrain_weights, limits
from runtime_v2.account import Account, ExecutionSession
from runtime_v2.storage import digest, read, save


def session(value):
    return pd.Timestamp(value).tz_localize(None).normalize()


def utc(value=None):
    value = pd.Timestamp.now(tz='UTC') if value is None else pd.Timestamp(value)
    return value.tz_localize('UTC') if value.tzinfo is None else value.tz_convert('UTC')


def ordinal(date, anchor):
    return calendar().sessions.get_loc(session(date)) - calendar().sessions.get_loc(session(anchor))


def refit_due(date, anchor):
    n = ordinal(date, anchor)
    return n > 0 and n % 21 == 0


def weekly(date):
    date = session(date)
    following = calendar().next_session(date)
    return date.isocalendar()[:2] != following.isocalendar()[:2]


def targets(data, model_paths, date, anchor, destination):
    """All callers pass the revealed prefix; future provider rows cannot enter."""
    date = session(date)
    frames = {s: f.loc[:date.tz_localize('UTC')] for s, f in data.items()}
    if any(f.empty or session(f.index[-1]) != date for f in frames.values()):
        raise ValueError('Stale/incomplete decision bars')
    paths = dict(model_paths)
    weights, signals = {}, []
    timings = dict(features=0., training=0., inference=0.)
    with threadpool_limits(limits=1):
        for symbol in sorted(frames):
            t = time.perf_counter()
            raw = frames[symbol]
            x = features(raw, frames['SPY'])
            timings['features'] += time.perf_counter() - t
            model = AdaptiveModel.load(paths[symbol])
            if refit_due(date, anchor):
                target = Path(destination) / symbol
                t = time.perf_counter()
                if target.exists():
                    model = AdaptiveModel.load(target)
                    if session(model.metadata['decision']) != date:
                        raise ValueError('Resumed model has wrong fit date')
                else:
                    model = AdaptiveModel(model.config).fit_at(x, raw.Close, raw.index[-1])
                    temp = target.with_name(target.name + '.partial')
                    if temp.exists():
                        shutil.rmtree(temp)
                    model.save(temp)
                    os.replace(temp, target)
                paths[symbol] = str(target)
                timings['training'] += time.perf_counter() - t
            if session(model.metadata['latest_label_end']) >= session(model.metadata['decision']):
                raise ValueError('Model was fitted with unavailable labels')
            if session(model.metadata['decision']) > date:
                raise ValueError('Model fitted after decision')
            t = time.perf_counter()
            probabilities = model.predict(x.iloc[[-1]])['hybrid'][0]
            timings['inference'] += time.perf_counter() - t
            if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1):
                raise ValueError('Invalid probabilities')
            trend = bool(x.average_200.iloc[-1] > 0 and x.momentum_12_1.iloc[-1] > 0)
            weights[symbol] = float(trend) * .8 / len(frames)
            signals.append(dict(symbol=symbol, probabilities=probabilities.tolist(),
                                expected_return=float(probabilities @ model.class_returns),
                                trend=trend, gate=bool(model.gate),
                                model_decision=model.metadata['decision'],
                                latest_label_end=model.metadata['latest_label_end'],
                                model_manifest_sha256=digest(Path(paths[symbol]) / 'manifest.json')))
    prices = pd.DataFrame({s: f.Close for s, f in frames.items()})
    weights = constrain_weights(pd.Series(weights), prices.pct_change(fill_method=None)).to_dict()
    return weights, signals, paths, timings


def plan(data, models, date, config, destination, prior=None):
    date = session(date)
    weights, signals, models, timings = targets(data, models, date, config['anchor'], destination)
    if prior:
        if calendar().next_session(session(prior['session'])) != date:
            raise ValueError('Missed prospective session: halt; no backfilled decisions or fills')
        marks = {s: float(prior['marks'][s] * f.loc[date.tz_localize('UTC'), 'Close'] /
                           f.loc[session(prior['session']).tz_localize('UTC'), 'Close']) for s, f in data.items()}
    else:
        marks = {s: float(f.loc[date.tz_localize('UTC'), 'Close']) for s, f in data.items()}
    return dict(session=str(date.date()), information_cutoff=calendar().session_close(date).isoformat(),
                refit=refit_due(date, config['anchor']), rebalance=prior is None or weekly(date),
                marks=marks, shadow_weights=weights, signals=signals, models=models,
                timings_seconds=timings, anchor=config['anchor'])


def commit(plan, workspace, config, fault=None):
    """Plan persists first. Individual account transactions are idempotent on retry."""
    result = dict(plan)
    for name, weights in [('deployment', {s: 0. for s in config['symbols']}),
                          ('shadow', plan['shadow_weights'])]:
        account = Account(config['symbols'], config['capital'], limits(), Path(workspace) / f'{name}.sqlite')
        result[name] = ExecutionSession(account).on_close(plan['information_cutoff'], plan['marks'],
                                                        weights if plan['rebalance'] else None)
        if fault:
            fault('after_' + name)
    return result


def replay(data, models, dates, config, workspace):
    """Reference sequential reveal replay. State and models remain separate from pilot."""
    previous, output = None, []
    for date in dates:
        day = Path(workspace) / str(session(date).date())
        p = plan(data, models, date, config, day / 'models', previous)
        previous = commit(p, workspace, config)
        models = previous['models']
        output.append(previous)
    return output
