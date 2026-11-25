import pytest
from research_v2.prospective_report import require_complete,predictive_summary

def test_final_freeze_refuses_missing_maturity_or_terminal_marks():
    state=dict(valid_observed_sessions=63,required_sessions=63,label_complete_decisions=62,missing_sessions=[],ledger_consistent=True)
    with pytest.raises(ValueError,match='Final freeze'):require_complete(state,6,False)
    state['label_complete_decisions']=63
    with pytest.raises(ValueError):require_complete(state,5,False)
    with pytest.raises(ValueError):require_complete(state,6,True)
    state['ledger_consistent']=False
    with pytest.raises(ValueError):require_complete(state,6,False)
    state['ledger_consistent']=True;require_complete(state,6,False)

def test_prospective_unobserved_stays_unobserved():
    r=predictive_summary([])
    assert r['status']=='unobserved' and r['metrics']==[] and r['decision_dates']==0

def test_prospective_predictive_oracle_preserves_date_dependence():
    rows=[dict(session=f'2026-01-{d:02d}',symbol=s,label=2,hybrid=[.05,.05,.9],prior=[.3,.3,.4]) for d in range(1,22) for s in ['A','B']]
    interim=predictive_summary(rows)
    assert interim['paired_inference']==[]
    r=predictive_summary(rows,final=True)
    assert r['labeled_rows']==42 and r['decision_dates']==21
    assert len(r['paired_inference'])==3 and r['status']=='observed_final'
    assert all(c['effect']<0 for c in r['paired_inference'])
    assert all(c['sessions']==21 for c in r['paired_inference'])


def test_prospective_report_output_separation_and_symlink(tmp_path):
    from research_v2.prospective_report import validate_output_location
    release=tmp_path/'release';workspace=tmp_path/'pilot'
    release.mkdir();workspace.mkdir()
    alias=tmp_path/'alias';alias.symlink_to(workspace,target_is_directory=True)
    for output in [release,workspace,release/'report',workspace/'report',alias/'report',tmp_path]:
        with pytest.raises(ValueError,match='separate'):
            validate_output_location(release,workspace,output)
        assert not (workspace/'report').exists() and not (release/'report').exists()
    validate_output_location(release,workspace,tmp_path/'safe-report')
    assert not (tmp_path/'safe-report').exists()


def test_prospective_report_read_lock_never_initializes_state(tmp_path):
    from research_v2.prospective_report import readonly_snapshot_lock
    missing=tmp_path/'absent'
    with pytest.raises(FileNotFoundError):
        with readonly_snapshot_lock(missing):pass
    assert not missing.exists()
    workspace=tmp_path/'pilot';workspace.mkdir()
    with pytest.raises(FileNotFoundError):
        with readonly_snapshot_lock(workspace):pass
    assert list(workspace.iterdir())==[]


def test_prospective_report_read_lock_blocks_writers_and_releases(tmp_path):
    import fcntl
    from research_v2.prospective_report import readonly_snapshot_lock
    path=tmp_path/'.writer.lock';path.write_bytes(b'existing-lock-preserved')
    with path.open('rb') as writer:
        fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='Collector is writing'):
            with readonly_snapshot_lock(tmp_path):pass
        fcntl.flock(writer,fcntl.LOCK_UN)
    with pytest.raises(ValueError,match='analysis failure'):
        with readonly_snapshot_lock(tmp_path):
            with path.open('rb') as writer:
                with pytest.raises(BlockingIOError):fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
            raise ValueError('analysis failure')
    with path.open('rb') as writer:
        fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(writer,fcntl.LOCK_UN)
    assert path.read_bytes()==b'existing-lock-preserved'


def _prospective_report_fixture(tmp_path):
    """Synthetic validated-receipt boundary, real account/risk arithmetic.

    The stub replaces release/receipt authentication only. Those contracts are
    independently tested by test_collector_v2; these tests exercise the report's
    joins, benchmark, maturity decisions and read-only archive behavior.
    """
    import json
    from pathlib import Path
    import shutil
    import pandas as pd
    root=Path(__file__).resolve().parents[1]
    release=tmp_path/'release';package=release/'runtime_v2';package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    for file in ['account.py','risk.py']:
        shutil.copy2(root/'runtime_v2'/file,package/file)
    (package/'model.py').write_text('''from pathlib import Path
import json
import numpy as np
from types import SimpleNamespace
def features(*args):raise AssertionError("Features must not be refitted by report")
class AdaptiveModel:
    @classmethod
    def load(cls,path):return SimpleNamespace(prior=np.array(json.loads((Path(path)/"prior.json").read_text())))
''')
    (package/'operations.py').write_text('''import json
from pathlib import Path
def fixture(workspace):return json.loads((Path(workspace)/"fixture.json").read_text())
def verify(release):return json.loads((Path(release)/"config.json").read_text())
def status(release,workspace,validated_config):return fixture(workspace)["state"]
def _valid_results(workspace,config):return fixture(workspace)["decisions"]
def valid_outcomes(workspace,config,decisions):return fixture(workspace)["outcomes"]
def valid_tail_results(workspace,config,decisions):return fixture(workspace)["tail"]
''')
    config=dict(symbols=['SPY'],capital=10000,sessions=1)
    (release/'config.json').write_text(json.dumps(config));(release/'manifest.json').write_text(json.dumps({'scope':'isolated synthetic test'}))
    model=release/'models/SPY';model.mkdir(parents=True);(model/'prior.json').write_text('[0.2,0.6,0.2]')
    workspace=tmp_path/'pilot';workspace.mkdir();(workspace/'.writer.lock').write_bytes(b'preserve')
    snapshot=workspace/'snapshot';snapshot.mkdir()
    index=pd.bdate_range('2025-01-02',periods=80,tz='UTC');decision=index[70]
    # Deliberate future mutation would distort risk scaling if report omitted as-of slicing.
    close=pd.Series(100.,index=index);close.loc[index>decision]=1e9
    pd.DataFrame({'Close':close}).to_parquet(snapshot/'SPY.parquet')
    def receipt(date,mode,rebalance):
        return dict(session=str(date.date()),information_cutoff=date.isoformat(),mode=mode,rebalance=rebalance,snapshot=str(snapshot),
                    marks={'SPY':100.},shadow={'equity':10000.},deployment={'equity':10000.},
                    models={'SPY':str(model)},signals=[dict(symbol='SPY',probabilities=[.1,.8,.1])])
    decisions=[receipt(decision,'prospective_shadow',True)]
    tail=[receipt(date,'outcome_tail',False) for date in index[71:77]]
    state=dict(valid_observed_sessions=1,required_sessions=1,label_complete_decisions=1,missing_sessions=[],ledger_consistent=True)
    payload=dict(state=state,decisions=decisions,tail=tail,outcomes=[dict(decision_session=str(decision.date()),outcomes=[dict(symbol='SPY',label=1,realized_return=0.)])])
    (workspace/'fixture.json').write_text(json.dumps(payload))
    return root,release,workspace,payload


def _run_prospective_report(root,release,workspace,output,freeze=False):
    import os
    import subprocess
    import sys
    command=[sys.executable,'-m','research_v2.prospective_report','--release',str(release),'--workspace',str(workspace),'--output',str(output)]
    if freeze:command.append('--freeze')
    return subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=30,
                          env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})


def test_prospective_report_model_prior_next_close_tail_and_readonly_archive(tmp_path):
    import hashlib
    import json
    import math
    import numpy as np
    import pandas as pd
    root,release,workspace,payload=_prospective_report_fixture(tmp_path)
    before={str(p.relative_to(tmp_path)):hashlib.sha256(p.read_bytes()).hexdigest()
            for parent in [release,workspace] for p in parent.rglob('*') if p.is_file()}
    output=tmp_path/'report'
    result=_run_prospective_report(root,release,workspace,output,freeze=True)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((output/'report.json').read_text())
    assert report['status']=='frozen_complete_observation'
    assert report['readiness']['historical_edge']=='not_approved' and report['readiness']['broker']=='disabled'
    assert report['predictive']['labeled_rows']==1 and report['predictive']['decision_dates']==1
    metrics={row['model']:row for row in report['predictive']['metrics']}
    assert metrics['prior']['log_loss']==pytest.approx(-math.log(.6))
    assert metrics['hybrid']['log_loss']==pytest.approx(-math.log(.8))
    history=pd.read_csv(output/'account_comparison.csv')
    assert len(history)==7 and (history['mode']=='outcome_tail').sum()==6
    assert history.equal_weight.iloc[0]==10000. # no same-close execution
    quantity=math.floor((1000/(100*(1+.1*.0015)))/1e-6)*1e-6
    fee=quantity*100*.0015
    np.testing.assert_allclose(history.equal_weight.iloc[1:],10000-fee,rtol=0,atol=1e-9)
    np.testing.assert_allclose(history.equal_weight_fees.iloc[1:],fee,rtol=0,atol=1e-9)
    events=json.loads((output/'benchmark_events.json').read_text())
    fills=[e for e in events if e['kind']=='fill']
    assert len(fills)==1 and fills[0]['asof']==payload['tail'][0]['information_cutoff']
    assert report['economic']['decision_sessions']==1 and report['economic']['terminal_mark_sessions']==6
    assert (output/'observed-state/fixture.json').exists()
    after={str(p.relative_to(tmp_path)):hashlib.sha256(p.read_bytes()).hexdigest()
           for parent in [release,workspace] for p in parent.rglob('*') if p.is_file()}
    assert after==before # includes refusing new bytecode files in the sealed package


def test_prospective_report_freeze_refuses_missing_terminal_receipt(tmp_path):
    import json
    root,release,workspace,payload=_prospective_report_fixture(tmp_path)
    payload['tail']=payload['tail'][:-1]
    (workspace/'fixture.json').write_text(json.dumps(payload))
    output=tmp_path/'report'
    result=_run_prospective_report(root,release,workspace,output,freeze=True)
    assert result.returncode!=0 and 'Final freeze' in result.stderr
    assert not output.exists()
