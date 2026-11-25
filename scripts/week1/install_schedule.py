"""Switch the sole local owner to the qualified five-session trial."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import plistlib
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'data/releases/research-v2-runtime-20260928-r2'
NEW=ROOT/'data/releases/research-v2-runtime-20260928-week1'
STATE=ROOT/'data/paper/research-week1'
DATA=ROOT/'data/research-week1'
LABEL='local.ml-trade-engine.collector-v2'
PLIST=Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
DOMAIN=f'gui/{os.getuid()}'
PYTHON='/Users/a/miniconda3/bin/python3.13'

def cli(release,workspace,action):
    result=subprocess.run([PYTHON,'-B',str(release/'scripts/collector_v2.py'),action,'--release',str(release),'--workspace',str(workspace)],cwd=release,capture_output=True,text=True,check=True)
    return json.loads(result.stdout)

def main():
    config=cli(NEW,STATE,'verify')
    assert config['sessions']==5 and config['external_orders'] is False
    old_status=cli(OLD,ROOT/'data/paper/research-v2','status')
    assert old_status['valid_observed_sessions']==0 and not old_status['halted']
    original=PLIST.read_bytes();old=plistlib.loads(original)
    if str(OLD) not in old['ProgramArguments'] or old['Label']!=LABEL:
        raise SystemExit('Unexpected current owner; refusing to replace it')
    listing=subprocess.run(['launchctl','list'],text=True,capture_output=True,check=True).stdout
    candidates=[line for line in listing.splitlines() if any(s in line.lower() for s in ['ml-trade-engine','collector-v2','trader.py'])]
    if any(LABEL not in line for line in candidates): raise SystemExit('Competing owner: '+str(candidates))
    if STATE.exists(): raise SystemExit('New workspace already exists; inspect before retry')
    archive=DATA/'superseded-schedule.plist'
    with archive.open('xb') as f:f.write(original)
    subprocess.run(['launchctl','bootout',DOMAIN,str(PLIST)],check=True)
    try:
        (STATE/'scheduler').mkdir(parents=True)
        new=dict(old,ProgramArguments=[PYTHON,'-B',str(NEW/'scripts/collector_v2.py'),'tick','--release',str(NEW),'--workspace',str(STATE)],
            WorkingDirectory=str(NEW),StandardOutPath=str(STATE/'scheduler/stdout.log'),StandardErrorPath=str(STATE/'scheduler/stderr.log'))
        content=plistlib.dumps(new)
        (ROOT/'scripts/week1/collector.plist').write_bytes(content)
        PLIST.write_bytes(content)
        subprocess.run(['plutil','-lint',str(PLIST)],check=True)
        subprocess.run(['launchctl','bootstrap',DOMAIN,str(PLIST)],check=True)
    except Exception:
        PLIST.write_bytes(original)
        subprocess.run(['launchctl','bootstrap',DOMAIN,str(PLIST)],check=True)
        raise
    receipt=dict(at=datetime.now(timezone.utc).isoformat(),reason='User-requested five-session amendment before first observation',
        sole_owner=LABEL,plist=str(PLIST),release=str(NEW),workspace=str(STATE),
        preserved_release=str(OLD),preserved_workspace=str(ROOT/'data/paper/research-v2'),old_status=old_status,
        new_manifest_sha256=hashlib.sha256((NEW/'manifest.json').read_bytes()).hexdigest(),
        new_plist_sha256=hashlib.sha256(PLIST.read_bytes()).hexdigest(),external_orders=False)
    (DATA/'installation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
