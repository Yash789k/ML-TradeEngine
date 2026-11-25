"""Full, separately budgeted deterministic validation of the frozen experiment."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import sys
import time
import types
import traceback

ROOT=Path(__file__).resolve().parents[3]
ORIGINAL=ROOT/'data/research-v2/experiments'
DEST=ORIGINAL/'full-reproduction'
ARCHIVE=ORIGINAL/'source'
PROTOCOL=ORIGINAL/'full_reproduction_protocol.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,default=str)+'\n')
def now():return datetime.now(timezone.utc).isoformat()


def register():
    if PROTOCOL.exists():raise FileExistsError(PROTOCOL)
    design=json.loads((ORIGINAL/'design.json').read_text())
    record=dict(registered_at=now(),role='Full deterministic clean-environment validation; no new candidate, selection, split, endpoint or hypothesis',parent_design_sha256=sha(ORIGINAL/'design.json'),execution_source_sha256=sha(__file__),source_hashes=design['source_hashes'],budget_amendment='The original1800-estimator research-selection budget is unchanged. A separate1696-estimator validation replay budget is added before replay, requested by the lead task to verify every registered trial in the clean environment.',maximum_estimator_fits=1696,maximum_cumulative_seconds=1800,expected_trials=len(design['trials']),expected_probability_view_rows=352800,expected_portfolio_policies=10,expected_portfolio_blocks=4,rtol=1e-10,atol=1e-12,comparison='Every forecast field, complete non-runtime fit metadata, and40actual cash-ledger/target/event histories',output_directory=str(DEST.relative_to(ROOT)),clean_environment=str((ROOT/'data/research-v2/clean-env').relative_to(ROOT)),adapter='Namespace-only packages; exact archived experiment, feature/label, risk and account implementations; output paths isolated',old_artifacts='Read only; original source/manifests/results retained')
    write(PROTOCOL,record)
    print('Registered full non-selection validation budget',flush=True)


def bootstrap():
    design=json.loads((ORIGINAL/'design.json').read_text())
    for rel,expected in design['source_hashes'].items():
        for path in [ROOT/rel,ARCHIVE/rel]:
            if sha(path)!=expected:raise RuntimeError('Source mismatch: '+str(path))
    for name,path in [('src',ARCHIVE/'src'),('src.models',ARCHIVE/'src/models'),('research_v2',ARCHIVE/'research_v2')]:
        m=types.ModuleType(name);m.__path__=[str(path)];m.__package__=name;sys.modules[name]=m
    spec=importlib.util.spec_from_file_location('research_v2.experiments',ARCHIVE/'research_v2/experiments.py')
    module=importlib.util.module_from_spec(spec);module.__file__=str(ROOT/'research_v2/experiments.py');sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module,design


def run():
    protocol=json.loads(PROTOCOL.read_text())
    if sha(__file__)!=protocol['execution_source_sha256']:raise RuntimeError('Validation runner changed after registration')
    completed_path=ORIGINAL/'full_reproduction.json'
    if completed_path.exists():
        report=json.loads(completed_path.read_text())
        if report['status']=='passed':
            print('Full validation already completed; no refits',flush=True);return
        raise RuntimeError('Existing final report requires investigation')
    started=time.perf_counter()
    module,design=bootstrap()
    import numpy as np
    import pandas as pd
    from pandas.testing import assert_frame_equal
    DEST.mkdir(exist_ok=True)
    design_copy=DEST/'design.json'
    if design_copy.exists():
        if sha(design_copy)!=sha(ORIGINAL/'design.json'):raise RuntimeError('Validation design copy changed')
    else:design_copy.write_bytes((ORIGINAL/'design.json').read_bytes())
    # Only output root changes; the archived trial fitting implementation is exact.
    module.OUT=DEST
    data=module.load_data();panel=module.make_panel(data)
    rows=[]
    previous_seconds=sum(json.loads(p.read_text())['seconds'] for p in (DEST/'checkpoints').glob('*/complete.json')) if (DEST/'checkpoints').exists() else 0.
    try:
        for i,trial in enumerate(design['trials']):
            if time.perf_counter()-started+previous_seconds>protocol['maximum_cumulative_seconds']:raise TimeoutError('Registered cumulative validation budget reached')
            done=module.trial_run(trial,panel,design)
            before=ORIGINAL/'checkpoints'/trial['id'];after=DEST/'checkpoints'/trial['id']
            old=json.loads((before/'complete.json').read_text())
            expected=pd.read_parquet(before/'predictions.parquet');actual=pd.read_parquet(after/'predictions.parquet')
            assert_frame_equal(actual,expected,check_exact=False,rtol=protocol['rtol'],atol=protocol['atol'])
            old_meta={k:v for k,v in old['model_metadata'].items() if k!='fit_seconds'}
            new_meta={k:v for k,v in done['model_metadata'].items() if k!='fit_seconds'}
            if new_meta!=old_meta:raise AssertionError('Non-runtime fit metadata differs: '+trial['id'])
            max_error=float(np.max(np.abs(actual[['p0','p1','p2']].to_numpy()-expected[['p0','p1','p2']].to_numpy())))
            row=dict(trial=trial['id'],probability_view_rows=len(actual),maximum_probability_error=max_error,all_forecast_columns_equal=True,all_nonruntime_fit_metadata_equal=True,actual_prediction_sha256=sha(after/'predictions.parquet'),expected_prediction_sha256=sha(before/'predictions.parquet'),model_sha256=sha(after/'model/model.joblib'))
            write(after/'comparison.json',row);rows.append(row)
            if (i+1)%25==0:print(f'Numerically reproduced {i+1}/{len(design["trials"])} trials in {time.perf_counter()-started:.1f}s',flush=True)
        # Replay policies from the actual newly generated forecasts; isolate all outputs.
        forecasts=pd.concat([pd.read_parquet(DEST/'checkpoints'/r['trial']/'predictions.parquet') for r in rows],ignore_index=True)
        forecasts.to_parquet(DEST/'predictions.parquet',index=False)
        stage=DEST/'report-root';(stage/'tables').mkdir(parents=True,exist_ok=True)
        module.ROOT=stage
        economic=module.portfolio_experiments(design,data,forecasts[forecasts.seed.eq(42)])
        module.ROOT=ROOT
        portfolios=[]
        for expected_path in sorted((ORIGINAL/'portfolios').glob('fold-*/*-account.parquet')):
            rel=expected_path.relative_to(ORIGINAL);actual_path=DEST/rel
            assert_frame_equal(pd.read_parquet(actual_path),pd.read_parquet(expected_path),check_exact=False,rtol=protocol['rtol'],atol=protocol['atol'])
            policy=expected_path.name.removesuffix('-account.parquet');folder=expected_path.parent.name
            original_targets=expected_path.parent/(policy+'-targets.parquet');actual_targets=actual_path.parent/(policy+'-targets.parquet')
            assert_frame_equal(pd.read_parquet(actual_targets),pd.read_parquet(original_targets),check_exact=False,rtol=protocol['rtol'],atol=protocol['atol'])
            original_events=expected_path.parent/(policy+'-events.json');actual_events=actual_path.parent/(policy+'-events.json')
            if json.loads(actual_events.read_text())!=json.loads(original_events.read_text()):raise AssertionError('Account events differ: '+str(rel))
            portfolios.append(dict(block=folder,policy=policy,n_sessions=len(pd.read_parquet(actual_path)),account_equal=True,targets_equal=True,events_equal=True,account_sha256=sha(actual_path),targets_sha256=sha(actual_targets),events_sha256=sha(actual_events)))
        for filename in ['exp_portfolios.csv','exp_economic_contrasts.csv']:
            assert_frame_equal(pd.read_csv(stage/'tables'/filename),pd.read_csv(ROOT/'data/research-v2/tables'/filename),check_exact=False,rtol=protocol['rtol'],atol=protocol['atol'])
        if len(portfolios)!=40:raise AssertionError('Incomplete portfolio family')
        elapsed=time.perf_counter()-started
        if elapsed+previous_seconds>protocol['maximum_cumulative_seconds']:raise TimeoutError('Validation exceeded registered cumulative runtime')
        loaded={name:str(Path(sys.modules[name].__file__).relative_to(ROOT)) for name in ['research_v2.experiments','src.models.adaptive','src.execution.account','src.backtest.portfolio','src.research.adaptive_portfolio']}
        report=dict(status='passed',completed_at=now(),protocol_sha256=sha(PROTOCOL),runner_sha256=sha(__file__),original_design_sha256=sha(ORIGINAL/'design.json'),runtime=sys.executable,python=sys.version,elapsed_seconds=elapsed,cumulative_seconds=elapsed+previous_seconds,actual_trials=len(rows),actual_estimator_fits=len(rows)*2,probability_view_rows=sum(r['probability_view_rows'] for r in rows),maximum_probability_error=max(r['maximum_probability_error'] for r in rows),rtol=protocol['rtol'],atol=protocol['atol'],all_forecast_fields_compared=True,all_nonruntime_fit_metadata_compared=True,actual_portfolio_histories=len(portfolios),actual_portfolio_sessions=sum(p['n_sessions'] for p in portfolios),all_ledger_targets_events_compared=True,economic_metrics_and_inference_equal=True,executed_archive_sources=loaded,trials=rows,portfolios=portfolios,limitations=['Deterministic reproduction validates these artifacts, not predictive or economic superiority.','Current-vintage source-data limitations and historical evaluation exposure are unchanged.','Original source, checkpoints, account artifacts and manifests were not modified.'])
        write(completed_path,report)
        print(json.dumps({k:report[k] for k in ['status','elapsed_seconds','actual_trials','actual_estimator_fits','probability_view_rows','maximum_probability_error','actual_portfolio_histories','actual_portfolio_sessions']}),flush=True)
    except Exception as e:
        failure=dict(status='failed',recorded_at=now(),error=repr(e),traceback=traceback.format_exc(),completed_comparison_trials=len(rows),elapsed_seconds=time.perf_counter()-started,protocol_sha256=sha(PROTOCOL))
        name=DEST/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json');write(name,failure);raise


if __name__=='__main__':
    if sys.argv[1:] == ['register']:register()
    elif sys.argv[1:] == ['run']:run()
    else:raise SystemExit('Use register or run')
