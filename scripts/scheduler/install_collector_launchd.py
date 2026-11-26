#!/usr/bin/env python
"""Install the single requested local schedule after source-bound release verification."""
from pathlib import Path
import json
import os
import plistlib
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
RELEASE=ROOT/'data/releases/research-v2-runtime-20260928-r2'
WORKSPACE=ROOT/'data/paper/research-v2'
LABEL='local.ml-trade-engine.collector-v2'
PYTHON=Path(sys.executable).resolve()
command=[str(PYTHON),'-B',str(RELEASE/'scripts/collector_v2.py')]
subprocess.run(command+['verify','--release',str(RELEASE)],check=True,stdout=subprocess.DEVNULL,cwd=RELEASE)
# A listing proves local discovery; it does not establish remote workflow state.
listing=subprocess.run(['launchctl','list'],capture_output=True,text=True,check=True).stdout
existing=[line for line in listing.splitlines() if any(key in line.lower() for key in ['trade-engine','trader','collector-v2','daily-signals'])]
if any(LABEL not in line for line in existing):
    raise SystemExit('Potential competing local owner: '+str(existing))
cron=subprocess.run(['crontab','-l'],capture_output=True,text=True)
if cron.returncode==0 and any(key in cron.stdout.lower() for key in ['ml-trade-engine','collector','trader.py','live.py']):
    raise SystemExit('Existing crontab collector must be reconciled before installation')
migration=None
if '--migrate-r1' in sys.argv:
    archive=ROOT/'data/paper/research-v2-r1-commissioning'
    if archive.exists(): raise SystemExit('r1 archive exists; do not repeat migration')
    binding=json.loads((WORKSPACE/'binding.json').read_text())
    if binding['release']!=str(ROOT/'data/releases/research-v2-runtime-20260928'):
        raise SystemExit('Only exact commissioning r1 workspace may be migrated by this command')
    if list((WORKSPACE/'days').glob('*/result.json')):
        raise SystemExit('Prospective observations present; a new protocol is required')
    old_plist=Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
    if any(LABEL in line for line in existing):
        subprocess.run(['launchctl','bootout',f'gui/{os.getuid()}',str(old_plist)],check=True)
    stored=ROOT/'scripts/scheduler/archive'/f'{LABEL}-r1.plist'
    stored.parent.mkdir(parents=True,exist_ok=True);stored.write_bytes(old_plist.read_bytes())
    os.rename(WORKSPACE,archive)
    migration=dict(old_workspace=str(WORKSPACE),preserved_archive=str(archive),old_binding=binding,
                   new_release=str(RELEASE),reason='Independent review fixes before first prospective session',
                   prospective_rows_before_migration=0,archive_contract='Preserved evidence; restore to original canonical path before operational use')
    existing=[line for line in existing if LABEL not in line]
WORKSPACE.mkdir(parents=True,exist_ok=True)
if migration: (WORKSPACE/'migration-r1-r2.json').write_text(json.dumps(migration,indent=2))
logs=WORKSPACE/'scheduler';logs.mkdir(exist_ok=True)
plist=dict(Label=LABEL,ProgramArguments=command+['tick','--release',str(RELEASE),'--workspace',str(WORKSPACE)],
           WorkingDirectory=str(RELEASE),StartInterval=60,RunAtLoad=True,ProcessType='Background',
           StandardOutPath=str(logs/'stdout.log'),StandardErrorPath=str(logs/'stderr.log'),
           EnvironmentVariables=dict(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                     XPC_SERVICE_NAME=LABEL),ThrottleInterval=30)
path=Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
if path.exists() and plistlib.loads(path.read_bytes()) != plist and migration is None:
    raise SystemExit('Existing collector definition differs; review before replacement')
path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(plistlib.dumps(plist))
(ROOT/'scripts/scheduler'/f'{LABEL}.plist').write_bytes(plistlib.dumps(plist))
subprocess.run(['plutil','-lint',str(path)],check=True)
domain=f'gui/{os.getuid()}'
if not any(LABEL in line for line in existing):
    subprocess.run(['launchctl','bootstrap',domain,str(path)],check=True)
info=subprocess.run(['launchctl','print',f'{domain}/{LABEL}'],capture_output=True,text=True,check=True)
(logs/'launchctl-installed.txt').write_text(info.stdout)
(logs/'schedule-discovery.json').write_text(json.dumps(dict(existing_local_candidates=existing,
 cron_returncode=cron.returncode,cron_output=cron.stdout+cron.stderr,owner=LABEL,plist=str(path),
 command=plist['ProgramArguments'],interval_seconds=60,market_guard='XNYS close +30,+45,+60 watchdog+90,preopen,final outcomes',
 remote_schedule='legacy workflow schedule removed locally; remote repository publishing was not authorized'),indent=2))
print(str(path))
