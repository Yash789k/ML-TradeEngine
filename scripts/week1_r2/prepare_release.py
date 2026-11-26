"""Register a distinct five-session restart after a failed trial; preserve its seal and halt."""
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
OLD = ROOT/'data/releases/research-v2-runtime-20260928-week1'
DEST = ROOT/'data/releases/research-v2-runtime-20261001-week1-r2'
DATA = ROOT/'data/research-week1-r2'
STAGE = DATA/'validation-attempt-2'
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
    prior_status = status(OLD, ROOT/'data/paper/research-week1')
    if prior_status['valid_observed_sessions'] or not prior_status['halted']:
        raise SystemExit('Expected the explicitly superseded halted trial with zero observed decisions')
    now = datetime.now(timezone.utc).isoformat()
    cal = calendar()
    dates = cal.sessions_in_range('2026-10-01', '2027-02-01')[:5]
    if datetime.now(timezone.utc) >= cal.session_close(dates[0]).to_pydatetime():
        raise SystemExit('Registration must finish before the first new session closes')
    endpoint = cal.sessions[cal.sessions.get_loc(dates[-1])+6]
    original = json.loads((OLD/'protocol.json').read_text())
    protocol = dict(original)
    protocol['prior_amendment'] = original.get('amendment')
    protocol.update(version='research-week1-r2-20261001', registered_at=now,
        amendment={'reason':'User authorized a ready replacement tonight after being told the original pilot is halted',
                   'timing':'Registered after US 1 October market open but before close, before any replacement decision or outcome inspection',
                   'parent_protocol_sha256':digest(OLD/'protocol.json'),
                   'parent_release_sha256':digest(OLD/'manifest.json'),
                   'parent_observations':prior_status['valid_observed_sessions'],
                   'parent_trial':prior_status['trial_id'],
                   'parent_status':'halted; missing 29 and 30 September; retained as failed operational evidence',
                   'supersedes':'Only future collection ownership; preserve all original dates, halt, receipts and releases',
                   'selection':'Same frozen models, symbols, costs, cash deployment and trend shadow; no performance-based reselection',
                   'authorization':'What was blocked? i will leave the laptop open tnite better make sure whatever needs to be done is ready'})
    protocol['pilot'] = dict(original['pilot'], trial_id='PILOT-TREND-WEEK1-R2-20261001', sessions=5,
        first_session=str(dates[0].date()), last_session=str(dates[-1].date()),
        cutoff_rule='Five fixed expected XNYS sessions; do not extend for outcomes or backfill missing sessions',
        operational_review_sessions=[5], final_outcome_session=str(endpoint.date()),
        refit_scope='No scheduled refit within the five-session window; carried models remain fixed',
        ship_scope='Historical study plus five-session operational report; labels may remain pending at week close')
    protocol['inference'] = dict(original['inference'], prospective='Descriptive only: five dates cannot support the registered block-bootstrap inference; no p-values or profitability approval')
    protocol['promotion'] = dict(original['promotion'], prospective='Separate five-session operational completion from six-session outcome tail; no economic qualification')
    save(DATA/'protocol.json', protocol)
    save(DATA/'parent-status-before-restart.json', prior_status)
    config = dict(config, trial_id=protocol['pilot']['trial_id'], registered_at=now,
        start=str(dates[0].date()), end=str(dates[-1].date()), sessions=5, protocol_sha256=digest(DATA/'protocol.json'),
        protocol_path=str(DATA/'protocol.json'), parent_release_sha256=digest(OLD/'manifest.json'),
        refit_rule='Carried 2026-09-25 models; existing 21-session cadence has no event inside this five-session pilot',
        first_refit=None, operational_milestones=[5], reporting_scope='descriptive operational pilot')
    shutil.copytree(OLD/'runtime_v2', STAGE/'runtime_v2', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (STAGE/'scripts').mkdir(exist_ok=True)
    shutil.copy2(OLD/'scripts/collector_v2.py', STAGE/'scripts/collector_v2.py')
    shutil.copytree(OLD/'models', STAGE/'models')
    shutil.copy2(OLD/'requirements.lock', STAGE/'requirements.lock')
    (STAGE/'tests').mkdir(exist_ok=True)
    shutil.copy2(OLD/'tests/test_collector_v2.py', STAGE/'tests/test_collector_v2.py')
    shutil.copy2(ROOT/'tests/test_week1_r2_pilot.py', STAGE/'tests/test_week1_r2_pilot.py')
    # The unchanged legacy tests derive ROOT from their location. Give that
    # isolated root read-only historical inputs and its own replay output paths.
    (STAGE/'data/releases').mkdir(parents=True)
    (STAGE/'data/releases/2026-09-28-adaptive').symlink_to(ROOT/'data/releases/2026-09-28-adaptive', target_is_directory=True)
    save(STAGE/'config.json', config)
    save(STAGE/'protocol.json', protocol)
    source = {str(p.relative_to(STAGE)):digest(p) for p in sorted((STAGE/'runtime_v2').glob('*.py'))+[STAGE/'scripts/collector_v2.py']}
    test_source = {str(p.relative_to(STAGE)):digest(p) for p in sorted((STAGE/'tests').glob('test_*.py'))}
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
    assert 'test_week1_r2_real_replay_reaches_five_decisions_and_six_tail_receipts' in names
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
        scope='Unchanged parent runtime regression suite plus exact restarted five-session schedule and full isolated historical five-decision/six-tail replay'))
    parent = json.loads((OLD/'manifest.json').read_text())
    artifacts = {str(p.relative_to(DEST)):digest(p) for p in DEST.rglob('*') if p.is_file() and str(p.relative_to(DEST)) not in source}
    save(DEST/'manifest.json', dict(version=2,status='sealed',created_at=now,source=source,artifacts=artifacts,
        dependencies=parent['dependencies'],external_orders=False,parent_release_sha256=digest(OLD/'manifest.json'),
        readiness='Five-session local operational study only; prospective unobserved; economics unapproved'))
    verify(DEST)
    assert before == {str(p.relative_to(OLD)):digest(p) for p in OLD.rglob('*') if p.is_file()}
    save(DATA/'schedule.json', {'config':config, 'events':[dict(e, due=e['due'].isoformat(), hong_kong=e['due'].tz_convert('Asia/Hong_Kong').isoformat()) for e in events(config)]})
    save(DATA/'release-verification.json', {'release':str(DEST),'manifest_sha256':digest(DEST/'manifest.json'),'parent_files_unchanged':len(before),'passing_cases':len(cases),'external_orders':False})
    save(DATA/'registry.json', dict(id='R12-WEEK1-R2', registered_protocol=str(DATA/'protocol.json'), active_trial=config['trial_id'], superseded_trial=prior_status['trial_id'], status='registered_unobserved', decision_sessions=5, outcome_tail_sessions=6, decision_dates=[str(d.date()) for d in dates], final_outcome_session=str(endpoint.date()), analysis='descriptive only', preserved_parent_halt=True, historical_experiments_unchanged=True, readiness_approved=False))
    print(json.dumps({'release':str(DEST),'cases':len(cases),'decision_end':config['end'],'final_outcome_session':str(endpoint.date())}))

if __name__ == '__main__': main()
