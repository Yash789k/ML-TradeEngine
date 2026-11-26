"""Replace only the collection owner after qualifying a distinct restart."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import plistlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'data/releases/research-v2-runtime-20261014-week1-r4'
NEW = ROOT / 'data/releases/research-v2-runtime-20261021-week1-r5'
OLD_STATE = ROOT / 'data/paper/research-week1-r4'
STATE = ROOT / 'data/paper/research-week1-r5'
DATA = ROOT / 'data/research-week1-r5'
PYTHON = ROOT / 'data/research-v2/clean-env/bin/python'
LABEL = 'local.ml-trade-engine.collector-v2'
PLIST = Path.home() / 'Library/LaunchAgents' / f'{LABEL}.plist'
DOMAIN = f'gui/{os.getuid()}'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def cli(release, state, command):
    result = subprocess.run([str(PYTHON), '-B', str(release / 'scripts/collector_v2.py'),
        command, '--release', str(release), '--workspace', str(state)],
        cwd=release, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main():
    config = cli(NEW, STATE, 'verify')
    assert config['trial_id'] == 'PILOT-TREND-WEEK1-R5-20261021'
    assert config['sessions'] == 1 and config['external_orders'] is False
    assert config['start'] == '2026-10-21' and config['end'] == '2026-10-21'
    if datetime.now(timezone.utc) >= datetime.fromisoformat('2026-10-21T20:00:00+00:00'):
        raise SystemExit('Installation must finish before the registered first session close')
    prior = cli(OLD, OLD_STATE, 'status')
    assert prior['halted'] and prior['valid_observed_sessions'] == 2
    if STATE.exists():
        raise SystemExit('Replacement workspace exists; inspect before retrying')
    guard_label='local.ml-trade-engine.awake-week1-r5'
    guard_path=PLIST.with_name(guard_label+'.plist')
    if guard_path.exists(): raise SystemExit('Power guard already exists; inspect before retrying')
    guard={'Label':guard_label,'ProgramArguments':[str(PYTHON),'-B',str(NEW/'scripts/power_guard.py'),'--until','2026-10-30T06:00:00+08:00'],
           'RunAtLoad':True,'KeepAlive':{'SuccessfulExit':False},'ThrottleInterval':30,
           'StandardOutPath':str(DATA/'power-guard.stdout.log'),'StandardErrorPath':str(DATA/'power-guard.stderr.log')}
    guard_path.write_bytes(plistlib.dumps(guard))
    subprocess.run(['plutil','-lint',str(guard_path)],check=True)
    subprocess.run(['launchctl','bootstrap',DOMAIN,str(guard_path)],check=True)
    save(DATA/'power-guard-installation.json', {'label':guard_label,'until':'2026-10-30T06:00:00+08:00','plist_sha256':digest(guard_path),'scope':'Prevent idle sleep through final watchdog; physical lid closure, logout, shutdown and depleted battery remain external limits'})
    original = PLIST.read_bytes()
    old = plistlib.loads(original)
    assert old['Label'] == LABEL and str(OLD) in old['ProgramArguments']
    # A sleep assertion is not a collection owner. Reject a different live collector.
    listing = subprocess.run(['launchctl', 'list'], check=True, capture_output=True, text=True).stdout
    competing = [line for line in listing.splitlines()
                 if ('ml-trade-engine' in line and 'collector' in line and LABEL not in line)]
    if competing:
        raise SystemExit('Competing collection owner: ' + repr(competing))
    with (DATA / 'superseded-schedule.plist').open('xb') as f:
        f.write(original)
    subprocess.run(['launchctl', 'bootout', DOMAIN, str(PLIST)], check=True)
    preserved = inventory(OLD_STATE)
    save(DATA / 'failed-trial-preservation.json', {
        'at': datetime.now(timezone.utc).isoformat(), 'workspace': str(OLD_STATE),
        'status': prior, 'files': preserved, 'halt_preserved': True,
        'release_manifest_sha256': digest(OLD / 'manifest.json')})
    try:
        (STATE / 'scheduler').mkdir(parents=True)
        new = dict(old, ProgramArguments=[str(PYTHON), '-B', str(NEW / 'scripts/collector_v2.py'),
            'tick', '--release', str(NEW), '--workspace', str(STATE)],
            WorkingDirectory=str(NEW), StandardOutPath=str(STATE / 'scheduler/stdout.log'),
            StandardErrorPath=str(STATE / 'scheduler/stderr.log'))
        payload = plistlib.dumps(new)
        (DATA / 'collector.plist').write_bytes(payload)
        PLIST.write_bytes(payload)
        subprocess.run(['plutil', '-lint', str(PLIST)], check=True)
        subprocess.run(['launchctl', 'bootstrap', DOMAIN, str(PLIST)], check=True)
    except Exception:
        PLIST.write_bytes(original)
        subprocess.run(['launchctl', 'bootstrap', DOMAIN, str(PLIST)], check=True)
        raise
    assert inventory(OLD_STATE) == preserved
    previous_guard=PLIST.with_name('local.ml-trade-engine.awake-week1-r4.plist')
    if previous_guard.exists():
        (DATA/'superseded-power-guard.plist').write_bytes(previous_guard.read_bytes())
        subprocess.run(['launchctl','bootout',DOMAIN,str(previous_guard)],check=True)
        previous_guard.unlink()
    receipt = dict(at=datetime.now(timezone.utc).isoformat(),
        reason='User-authorized fifth decision tonight; preserve four archived observations in r2 and r4, r3 zero-observation failure and every original halt; no account carryover',
        sole_owner=LABEL, release=str(NEW), workspace=str(STATE),
        preserved_release=str(OLD), preserved_workspace=str(OLD_STATE),
        prior_status=prior, old_workspace_files_unchanged=len(preserved),
        new_manifest_sha256=digest(NEW / 'manifest.json'),
        installed_plist_sha256=digest(PLIST), external_orders=False)
    save(DATA / 'installation.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
