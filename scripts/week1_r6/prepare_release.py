"""Register a distinct one-decision continuation restart after a failed trial; preserve its seal and halt."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'data/releases/research-v2-runtime-20261021-week1-r5'
PRIOR = ROOT/'data/releases/research-v2-runtime-20261001-week1-r2'
PRIOR_R4 = ROOT/'data/releases/research-v2-runtime-20261014-week1-r4'
DEST = ROOT/'data/releases/research-v2-runtime-20261022-week1-r6'
DATA = ROOT/'data/research-week1-r6'
STAGE = DATA/'validation'
sys.path.insert(0, str(OLD))
from runtime_v2.operations import verify, status, CRITICAL_TESTS, events
from runtime_v2.calendar import calendar

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str)+'\n')

def main():
    if DEST.exists() or (STAGE/'runtime_v2').exists():
        raise SystemExit('Use a new version for any retry; prepared or sealed source already exists')
    config = verify(OLD)
    before = {str(p.relative_to(OLD)):digest(p) for p in OLD.rglob('*') if p.is_file()}
    def checked_state(release, workspace):
        return json.loads(subprocess.run([sys.executable,'-B',str(release/'scripts/collector_v2.py'),'status','--release',str(release),'--workspace',str(workspace)],check=True,capture_output=True,text=True).stdout)
    superseded_status = checked_state(OLD, ROOT/'data/paper/research-week1-r5')
    prior_status = checked_state(PRIOR, ROOT/'data/paper/research-week1-r2')
    prior_r4_status = checked_state(PRIOR_R4, ROOT/'data/paper/research-week1-r4')
    assert superseded_status['valid_observed_sessions']==0 and superseded_status['halted'] and superseded_status['ledger_consistent']
    for state in [prior_r4_status,prior_status]:
        assert state['valid_observed_sessions']==2 and state['halted'] and state['ledger_consistent']
    now = datetime.now(timezone.utc).isoformat()
    cal=calendar(); dates=cal.sessions_in_range('2026-10-22','2026-10-22')
    assert len(dates)==1
    if datetime.now(timezone.utc) >= cal.session_close(dates[0]).to_pydatetime():
        raise SystemExit('Registration must finish before the new session closes')
    endpoint=cal.sessions[cal.sessions.get_loc(dates[-1])+6]
    original=json.loads((OLD/'protocol.json').read_text())
    protocol=dict(original)
    protocol['prior_amendment']=original.get('amendment')
    protocol.update(version='research-week1-r6-20261022',registered_at=now,
        amendment={'reason':'Resolve the r5 provider-response failure; bounded retries for a rejected newly completed OHLC bar',
          'authorization':'Resolve the response, what happened why did it fail; continuing the requested fifth collection',
          'timing':'Before US 22 October market open; r5 rejected data and later corrected provider response already inspected',
          'repair_registration':str(DATA/'repair-registration.json'),'repair_registration_sha256':digest(DATA/'repair-registration.json'),
          'parent_protocol_sha256':digest(OLD/'protocol.json'),'parent_release_sha256':digest(OLD/'manifest.json'),
          'parent_trial':superseded_status['trial_id'],'superseded_status':superseded_status,
          'supersedes':'Future collection ownership only; preserve prior receipts, snapshots, accounts, releases and halts',
          'selection':'Same frozen models, symbols, costs, cash deployment and trend shadow; no performance-based reselection'})
    protocol['pilot']=dict(original['pilot'],trial_id='PILOT-TREND-WEEK1-R6-20261022',sessions=1,
        first_session=str(dates[0].date()),last_session=str(dates[-1].date()),
        cutoff_rule='One fixed future decision in fresh accounts; four archived observations retained as two separate failed segments; no backfill or return splicing',
        operational_review_sessions=[1],final_outcome_session=str(endpoint.date()),
        refit_scope='No scheduled refit within the single-decision continuation; carried models remain fixed',
        ship_scope='Interrupted five-observation report across three separate account segments; labels may remain pending',
        schedule='XNYS close+30, transient retries +45/+60, watchdog +90; segment milestone 1; six hold-only tail sessions')
    protocol['inference']=dict(original['inference'],prospective='Descriptive only; five interrupted dates do not support dependence-aware inference or profitability approval')
    protocol['promotion']=dict(original['promotion'],prospective='Separate one-decision operational completion from six-session outcome maturity; no economic qualification')
    archived=[]
    for tag,release,state,ds in [('r2',PRIOR,prior_status,['2026-10-01','2026-10-02']),('r4',PRIOR_R4,prior_r4_status,['2026-10-14','2026-10-15'])]:
        workspace=ROOT/f'data/paper/research-week1-{tag}'
        archived.append({'segment':tag,'prior_release':str(release),'prior_release_sha256':digest(release/'manifest.json'),
            'prior_workspace':str(workspace),'prior_decision_dates':ds,'status_at_supersession':state,
            'prior_receipt_sha256':{d:digest(workspace/'days'/d/'result.json') for d in ds},
            'missed_dates':state['missing_sessions'],'halt_preserved':True})
    protocol['continuation']={'design':'Five interrupted observed dates across three separate account segments',
        'prior_segments':archived,'failed_continuations':original['continuation']['failed_continuations']+[
            dict(release=str(OLD),workspace=str(ROOT/'data/paper/research-week1-r5'),release_sha256=digest(OLD/'manifest.json'),
                 status_at_supersession=superseded_status,reason='Invalid JPM OHLC rejected on US 21 October; no valid decision; original halt retained',
                 cancelled_tail_dates=['2026-10-22','2026-10-23','2026-10-26','2026-10-27','2026-10-28','2026-10-29'])],
        'earlier_failed_trials':[{'release':str(ROOT/'data/releases/research-v2-runtime-20260928-week1'), 'workspace':str(ROOT/'data/paper/research-week1'),'note':'Original five-session trial failed; superseded before r2; retain original evidence'}],
        'new_account':'Fresh 10000 cash per account; no positions, pending orders or fills cross the gap',
        'analysis':'Retain date and segment identifiers; separate account returns and reliability; every failed trial remains failed',
        'total_observed_target':5,'archived_observed_target':4,'future_observed_target':1,
        'prior_label_source':'Derive labels from validated new snapshots in report artifacts only; never mutate archived accounts',
        'cancelled_r4_future_tail_dates':original['continuation']['cancelled_r4_future_tail_dates']}
    protocol['provider_repair'] = json.loads((DATA/'repair-registration.json').read_text())
    save(DATA/'protocol.json',protocol)
    save(DATA/'parent-status-before-restart.json',superseded_status)
    save(DATA/'archived-observed-status.json',prior_status)
    config=dict(config,trial_id=protocol['pilot']['trial_id'],registered_at=now,start=str(dates[0].date()),end=str(dates[-1].date()),sessions=1,
        protocol_sha256=digest(DATA/'protocol.json'),protocol_path=str(DATA/'protocol.json'),parent_release_sha256=digest(OLD/'manifest.json'),
        refit_rule='Carried 2026-09-25 models; existing 21-session cadence has no event on 22 October; tail never refits',
        first_refit=None,operational_milestones=[1],reporting_scope='descriptive interrupted operational study')
    shutil.copytree(OLD/'runtime_v2', STAGE/'runtime_v2', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Two reviewed provider-validation modules change; all policy, execution,
    # scheduling and frozen models remain byte-identical to r5.
    for name in ['market.py','calendar.py']:
        shutil.copy2(ROOT/'scripts/week1_r6/runtime'/name,STAGE/'runtime_v2'/name)
    (STAGE/'scripts').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'scripts/week1_r6/power_guard.py',STAGE/'scripts/power_guard.py')
    shutil.copy2(OLD/'scripts/collector_v2.py', STAGE/'scripts/collector_v2.py')
    shutil.copytree(OLD/'models', STAGE/'models')
    shutil.copy2(OLD/'requirements.lock', STAGE/'requirements.lock')
    (STAGE/'tests').mkdir(exist_ok=True)
    (STAGE/'tests/__init__.py').write_text('')
    shutil.copy2(OLD/'tests/test_collector_v2.py', STAGE/'tests/test_collector_v2.py')
    shutil.copy2(ROOT/'tests/test_week1_r6_pilot.py', STAGE/'tests/test_week1_r6_pilot.py')
    shutil.copy2(ROOT/'tests/test_week1_r6_provider.py',STAGE/'tests/test_week1_r6_provider.py')
    (STAGE/'tests/fixtures').mkdir()
    shutil.copy2(ROOT/'data/paper/research-week1-r5/days/2026-10-21/market-20261021T203041615851-ea629ee7/responses/JPM.json',STAGE/'tests/fixtures/JPM-invalid.json')
    shutil.copy2(ROOT/'data/diagnostics/provider-ohlc-20261022/JPM-response.json',STAGE/'tests/fixtures/JPM-corrected.json')
    # The unchanged legacy tests derive ROOT from their location. Give that
    # isolated root read-only historical inputs and its own replay output paths.
    (STAGE/'data/releases').mkdir(parents=True)
    (STAGE/'data/releases/2026-09-28-adaptive').symlink_to(ROOT/'data/releases/2026-09-28-adaptive', target_is_directory=True)
    save(STAGE/'config.json', config)
    save(STAGE/'protocol.json', protocol)
    source = {str(p.relative_to(STAGE)):digest(p) for p in sorted((STAGE/'runtime_v2').glob('*.py'))+list(sorted((STAGE/'scripts').glob('*.py')))}
    test_source = {str(p.relative_to(STAGE)):digest(p) for p in sorted((STAGE/'tests').rglob('*')) if p.is_file()}
    config_hash = digest(STAGE/'config.json')
    command = [sys.executable,'-B','-m','pytest',str(STAGE/'tests'),'-q',f'--junitxml={DATA}/qualification.xml']
    env = dict(os.environ, PYTHONPATH=str(STAGE)+os.pathsep+str(ROOT), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=STAGE, env=env, text=True, capture_output=True)
    (DATA/'qualification.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise SystemExit(result.stdout+result.stderr)
    assert source == {rel:digest(STAGE/rel) for rel in source}
    assert test_source == {rel:digest(STAGE/rel) for rel in test_source}
    assert config_hash == digest(STAGE/'config.json')
    tree = ET.parse(DATA/'qualification.xml')
    cases = tree.findall('.//testcase')
    assert not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
    names = [c.attrib['name'].split('[')[0] for c in cases]
    assert set(CRITICAL_TESTS) <= set(names)
    assert 'test_week1_r6_real_replay_reaches_one_decision_and_six_tail_receipts' in names
    DEST.mkdir(parents=True)
    for rel in source:
        target = DEST/rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(STAGE/rel,target)
    shutil.copytree(STAGE/'models', DEST/'models')
    for name in ['config.json','protocol.json','requirements.lock']: shutil.copy2(STAGE/name, DEST/name)
    shutil.copy2(DATA/'qualification.xml', DEST/'tests.xml')
    shutil.copytree(STAGE/'tests', DEST/'tests', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    save(DEST/'test-evidence.json', dict(command=command, started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
        source=source, config_sha256=config_hash, protocol_sha256=digest(DATA/'protocol.json'), test_source=test_source,
        passing_test_names=names, failures=0, errors=0, junit_sha256=digest(DEST/'tests.xml'),
        scope='Source-bound critical regression suite, typed latest-provider-bar rejection and bounded recovery, stored JPM reproduction, exact r6 calendar and isolated one-decision/six-tail replay'))
    parent = json.loads((OLD/'manifest.json').read_text())
    artifacts = {str(p.relative_to(DEST)):digest(p) for p in DEST.rglob('*') if p.is_file() and str(p.relative_to(DEST)) not in source}
    save(DEST/'manifest.json', dict(version=2,status='sealed',created_at=now,source=source,artifacts=artifacts,
        dependencies=parent['dependencies'],external_orders=False,parent_release_sha256=digest(OLD/'manifest.json'),
        readiness='Interrupted five-observation study, fresh one-decision segment only; prospective unobserved; economics unapproved'))
    subprocess.run([sys.executable,'-B',str(DEST/'scripts/collector_v2.py'),'verify','--release',str(DEST)], check=True, capture_output=True)
    assert before == {str(p.relative_to(OLD)):digest(p) for p in OLD.rglob('*') if p.is_file()}
    save(DATA/'schedule.json', {'config':config, 'events':[dict(e, due=e['due'].isoformat(), hong_kong=e['due'].tz_convert('Asia/Hong_Kong').isoformat()) for e in events(config)]})
    save(DATA/'release-verification.json', {'release':str(DEST),'manifest_sha256':digest(DEST/'manifest.json'),'parent_files_unchanged':len(before),'passing_cases':len(cases),'external_orders':False})
    save(DATA/'registry.json', dict(id='R12-WEEK1-R5', registered_protocol=str(DATA/'protocol.json'), active_trial=config['trial_id'], superseded_trial=superseded_status['trial_id'], status='registered_unobserved', decision_sessions=1, outcome_tail_sessions=6, decision_dates=[str(d.date()) for d in dates], final_outcome_session=str(endpoint.date()), analysis='descriptive only', preserved_parent_halt=True, historical_experiments_unchanged=True, readiness_approved=False))
    print(json.dumps({'release':str(DEST),'cases':len(cases),'decision_end':config['end'],'final_outcome_session':str(endpoint.date())}))

if __name__ == '__main__': main()
