"""Register a distinct three-session continuation restart after a failed trial; preserve its seal and halt."""
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
OLD = ROOT/'data/releases/research-v2-runtime-20261001-week1-r2'
DEST = ROOT/'data/releases/research-v2-runtime-20261011-week1-r3'
DATA = ROOT/'data/research-week1-r3'
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
    prior_status = status(OLD, ROOT/'data/paper/research-week1-r2')
    if prior_status['valid_observed_sessions'] != 2 or not prior_status['halted']:
        raise SystemExit('Expected the explicitly superseded halted trial with two observed decisions')
    now = datetime.now(timezone.utc).isoformat()
    cal = calendar()
    dates = cal.sessions_in_range('2026-10-12', '2027-02-01')[:3]
    if datetime.now(timezone.utc) >= cal.session_close(dates[0]).to_pydatetime():
        raise SystemExit('Registration must finish before the first new session closes')
    endpoint = cal.sessions[cal.sessions.get_loc(dates[-1])+6]
    original = json.loads((OLD/'protocol.json').read_text())
    protocol = dict(original)
    protocol['prior_amendment'] = original.get('amendment')
    protocol.update(version='research-week1-r3-20261011', registered_at=now,
        amendment={'reason':'User instructed: So make it work, after discussion of retaining two decisions and collecting three more',
                   'timing':'Registered before US 12 October session; existing two observations and failed sessions already inspected',
                   'parent_protocol_sha256':digest(OLD/'protocol.json'),
                   'parent_release_sha256':digest(OLD/'manifest.json'),
                   'parent_observations':prior_status['valid_observed_sessions'],
                   'parent_trial':prior_status['trial_id'],
                   'parent_status':'Two valid decisions on 1 and 2 October; missed 5, 6 and 7 October; halted account preserved',
                   'supersedes':'Only future collection ownership; preserve all original dates, halt, receipts and releases',
                   'selection':'Same frozen models, symbols, costs, cash deployment and trend shadow; no performance-based reselection',
                   'authorization':'So make it work'})
    protocol['pilot'] = dict(original['pilot'], trial_id='PILOT-TREND-WEEK1-R3-20261012', sessions=3,
        first_session=str(dates[0].date()), last_session=str(dates[-1].date()),
        cutoff_rule='Three fixed future XNYS sessions in a fresh account; two prior observations retained as a separate failed segment; never backfill or splice portfolio returns',
        operational_review_sessions=[3], final_outcome_session=str(endpoint.date()),
        refit_scope='No scheduled refit within the three-session continuation window; carried models remain fixed',
        ship_scope='Historical study plus three-session continuation operational report; labels may remain pending at week close')
    protocol['inference'] = dict(original['inference'], prospective='Descriptive only: five dates cannot support the registered block-bootstrap inference; no p-values or profitability approval')
    protocol['promotion'] = dict(original['promotion'], prospective='Separate three-session continuation operational completion from six-session outcome tail; no economic qualification')
    protocol['continuation'] = {'design':'Five observed decision dates across two explicitly separate segments',
        'prior_release':str(OLD),'prior_workspace':str(ROOT/'data/paper/research-week1-r2'),
        'prior_decision_dates':['2026-10-01','2026-10-02'], 'missed_dates':['2026-10-05','2026-10-06','2026-10-07'],
        'prior_receipt_sha256':{d:digest(ROOT/'data/paper/research-week1-r2/days'/d/'result.json') for d in ['2026-10-01','2026-10-02']},
        'new_account':'Fresh 10000 cash per account; no positions, pending orders or fills carried across the gap',
        'analysis':'Descriptive forecasts may be pooled with date/segment identifiers; returns and reliability reported per segment; original failed trial never marked complete',
        'total_observed_target':5, 'future_observed_target':3,
        'prior_label_source':'Derive later labels from new segment recorded snapshots, in report artifacts only; do not mutate archived prior state'}
    protocol['pilot']['schedule']='XNYS close+30, retries +45/+60, watchdog +90; segment milestone 3; six hold-only tail sessions'
    protocol['pilot']['ship_scope']='Interrupted five-observation report: two archived plus three new, separately disclosed; labels may remain pending'
    save(DATA/'protocol.json', protocol)
    save(DATA/'parent-status-before-restart.json', prior_status)
    config = dict(config, trial_id=protocol['pilot']['trial_id'], registered_at=now,
        start=str(dates[0].date()), end=str(dates[-1].date()), sessions=3, protocol_sha256=digest(DATA/'protocol.json'),
        protocol_path=str(DATA/'protocol.json'), parent_release_sha256=digest(OLD/'manifest.json'),
        refit_rule='Carried 2026-09-25 models; existing 21-session cadence has no event inside this three-session continuation pilot',
        first_refit=None, operational_milestones=[3], reporting_scope='descriptive operational pilot')
    shutil.copytree(OLD/'runtime_v2', STAGE/'runtime_v2', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Small versioned scheduling fix: milestone count comes from sealed config.
    operations=STAGE/'runtime_v2/operations.py'
    text=operations.read_text()
    old_line="event['ordinal'] in [5,21,42,63]"
    assert old_line in text
    operations.write_text(text.replace(old_line,"event['ordinal'] in config.get('operational_milestones', [5,21,42,63])"))
    (STAGE/'scripts').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'scripts/week1_r3/power_guard.py',STAGE/'scripts/power_guard.py')
    shutil.copy2(OLD/'scripts/collector_v2.py', STAGE/'scripts/collector_v2.py')
    shutil.copytree(OLD/'models', STAGE/'models')
    shutil.copy2(OLD/'requirements.lock', STAGE/'requirements.lock')
    (STAGE/'tests').mkdir(exist_ok=True)
    (STAGE/'tests/__init__.py').write_text('')
    shutil.copy2(OLD/'tests/test_collector_v2.py', STAGE/'tests/test_collector_v2.py')
    shutil.copy2(ROOT/'tests/test_week1_r3_pilot.py', STAGE/'tests/test_week1_r3_pilot.py')
    # The unchanged legacy tests derive ROOT from their location. Give that
    # isolated root read-only historical inputs and its own replay output paths.
    (STAGE/'data/releases').mkdir(parents=True)
    (STAGE/'data/releases/2026-09-28-adaptive').symlink_to(ROOT/'data/releases/2026-09-28-adaptive', target_is_directory=True)
    save(STAGE/'config.json', config)
    save(STAGE/'protocol.json', protocol)
    source = {str(p.relative_to(STAGE)):digest(p) for p in sorted((STAGE/'runtime_v2').glob('*.py'))+list(sorted((STAGE/'scripts').glob('*.py')))}
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
    assert 'test_week1_r3_real_replay_reaches_three_decisions_and_six_tail_receipts' in names
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
        scope='Unchanged parent runtime regression suite plus exact restarted three-session continuation schedule and full isolated historical three-decision/six-tail replay'))
    parent = json.loads((OLD/'manifest.json').read_text())
    artifacts = {str(p.relative_to(DEST)):digest(p) for p in DEST.rglob('*') if p.is_file() and str(p.relative_to(DEST)) not in source}
    save(DEST/'manifest.json', dict(version=2,status='sealed',created_at=now,source=source,artifacts=artifacts,
        dependencies=parent['dependencies'],external_orders=False,parent_release_sha256=digest(OLD/'manifest.json'),
        readiness='Interrupted five-observation study, fresh three-session segment only; prospective unobserved; economics unapproved'))
    subprocess.run([sys.executable,'-B',str(DEST/'scripts/collector_v2.py'),'verify','--release',str(DEST)], check=True, capture_output=True)
    assert before == {str(p.relative_to(OLD)):digest(p) for p in OLD.rglob('*') if p.is_file()}
    save(DATA/'schedule.json', {'config':config, 'events':[dict(e, due=e['due'].isoformat(), hong_kong=e['due'].tz_convert('Asia/Hong_Kong').isoformat()) for e in events(config)]})
    save(DATA/'release-verification.json', {'release':str(DEST),'manifest_sha256':digest(DEST/'manifest.json'),'parent_files_unchanged':len(before),'passing_cases':len(cases),'external_orders':False})
    save(DATA/'registry.json', dict(id='R12-WEEK1-R3', registered_protocol=str(DATA/'protocol.json'), active_trial=config['trial_id'], superseded_trial=prior_status['trial_id'], status='registered_unobserved', decision_sessions=3, outcome_tail_sessions=6, decision_dates=[str(d.date()) for d in dates], final_outcome_session=str(endpoint.date()), analysis='descriptive only', preserved_parent_halt=True, historical_experiments_unchanged=True, readiness_approved=False))
    print(json.dumps({'release':str(DEST),'cases':len(cases),'decision_end':config['end'],'final_outcome_session':str(endpoint.date())}))

if __name__ == '__main__': main()
