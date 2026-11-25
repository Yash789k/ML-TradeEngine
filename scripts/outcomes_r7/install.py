"""User-authorized schedule replacement, preserving old trial and service bytes."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,plistlib,subprocess,sys,shutil,hashlib

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/research-outcomes-r7'
NEW=ROOT/'data/releases/outcomes-r7-20261027'
OLD=ROOT/'data/releases/research-v2-runtime-20261022-week1-r6'
STATE=ROOT/'data/paper/research-outcomes-r7'
PYTHON=ROOT/'data/research-v2/clean-env/bin/python'
AGENTS=Path.home()/'Library/LaunchAgents'
DOMAIN=f'gui/{os.getuid()}'
LABEL='local.ml-trade-engine.collector-v2'
GUARD='local.ml-trade-engine.awake-outcomes-r7'
DASH='local.ml-trade-engine.dashboard-outcomes-r7'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory(p):return {str(x.relative_to(p)):digest(x) for x in sorted(p.rglob('*')) if x.is_file()}
def command(args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')

def main():
 if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-10-27T20:00:00+00:00'):
  raise SystemExit('Register a new future first session: installation is too late for this one')
 status=json.loads(command([str(PYTHON),'-B',str(NEW/'collector.py'),'status']))
 assert status['decision_dates_preserved']==5 and status['completed_outcome_reads']==0 and not status['halted']
 if STATE.exists():raise SystemExit('New workspace exists; inspect instead of reinstalling')
 old_paths=[AGENTS/(x+'.plist') for x in [LABEL,'local.ml-trade-engine.awake-week1-r6','local.ml-trade-engine.dashboard-week1-r6']]
 old_bytes={p:p.read_bytes() for p in old_paths}
 assert str(OLD) in plistlib.loads(old_bytes[old_paths[0]])['ProgramArguments']
 listing=command(['launchctl','list'])
 assert [line.split()[-1] for line in listing.splitlines() if 'ml-trade-engine' in line and 'collector' in line]==[LABEL]
 assert not (AGENTS/(GUARD+'.plist')).exists() and not (AGENTS/(DASH+'.plist')).exists()
 saved=DATA/'superseded';saved.mkdir()
 for p,body in old_bytes.items():(saved/p.name).write_bytes(body)
 # Only stop the old collector between sessions. No tick or collection is invoked manually.
 command(['launchctl','bootout',DOMAIN,str(old_paths[0])])
 ws=ROOT/'data/paper/research-week1-r6'
 preserved=inventory(ws)
 save(DATA/'r6-preservation.json',dict(at=datetime.now(timezone.utc).isoformat(),workspace=str(ws),files=preserved,
      manifest_sha256=digest(OLD/'manifest.json'),missed_outcome_sessions=['2026-10-23','2026-10-26'],future_tail_superseded=['2026-10-27','2026-10-28','2026-10-29','2026-10-30'],halt_preserved=(ws/'halt.json').exists()))
 (STATE/'scheduler').mkdir(parents=True)
 old_plist=plistlib.loads(old_bytes[old_paths[0]])
 collector=dict(old_plist,ProgramArguments=[str(PYTHON),'-B',str(NEW/'collector.py'),'tick','--release',str(NEW),'--workspace',str(STATE)],
      WorkingDirectory=str(NEW),StandardOutPath=str(STATE/'scheduler/stdout.log'),StandardErrorPath=str(STATE/'scheduler/stderr.log'))
 guard=dict(Label=GUARD,ProgramArguments=[str(PYTHON),'-B',str(NEW/'power_guard.py'),'--until','2026-11-04T07:00:00+08:00'],
      RunAtLoad=True,KeepAlive={'SuccessfulExit':False},ThrottleInterval=30,StandardOutPath=str(DATA/'power-guard.stdout.log'),StandardErrorPath=str(DATA/'power-guard.stderr.log'))
 dashboard=plistlib.loads(old_bytes[old_paths[2]])
 dashboard['Label']=DASH
 dashboard['ProgramArguments']=[str(NEW/'dashboard.py') if x==str(ROOT/'app_week1_r6.py') else x for x in dashboard['ProgramArguments']]
 dashboard['StandardOutPath']=str(DATA/'dashboard.stdout.log');dashboard['StandardErrorPath']=str(DATA/'dashboard.stderr.log')
 installed=[]
 try:
  for label,config in [(GUARD,guard),(LABEL,collector)]:
   path=AGENTS/(label+'.plist');payload=plistlib.dumps(config);path.write_bytes(payload)
   (DATA/(label+'.plist')).write_bytes(payload)
   command(['plutil','-lint',str(path)]);command(['launchctl','bootstrap',DOMAIN,str(path)]);installed.append(path)
  # Move superseded guard/dashboard plists outside LaunchAgents; collector keeps its sole-owner label.
  for p in old_paths[1:]:
   command(['launchctl','bootout',DOMAIN,str(p)])
   assert p.read_bytes()==old_bytes[p]
   p.unlink()
  path=AGENTS/(DASH+'.plist');payload=plistlib.dumps(dashboard);path.write_bytes(payload)
  (DATA/(DASH+'.plist')).write_bytes(payload)
  command(['plutil','-lint',str(path)]);command(['launchctl','bootstrap',DOMAIN,str(path)]);installed.append(path)
 except Exception:
  for path in reversed(installed):
   subprocess.run(['launchctl','bootout',DOMAIN,str(path)],capture_output=True)
   if path.exists():path.unlink()
  for p,body in old_bytes.items():
   subprocess.run(['launchctl','bootout',DOMAIN,str(p)],capture_output=True)
   p.write_bytes(body);subprocess.run(['launchctl','bootstrap',DOMAIN,str(p)],check=True,capture_output=True)
  raise
 assert inventory(ws)==preserved
 receipt=dict(at=datetime.now(timezone.utc).isoformat(),authorization='User requested outcome collection dates from tonight onwards on 27 October HKT',
      release=str(NEW),workspace=str(STATE),release_manifest_sha256=digest(NEW/'manifest.json'),sole_collector=LABEL,
      guard=GUARD,guard_until='2026-11-04T07:00:00+08:00',dashboard=DASH,
      old_workspace_files_preserved=len(preserved),superseded_plists={str(p):digest(saved/p.name) for p in old_paths},
      installed_plists={str(p):digest(p) for p in installed},external_orders=False,new_targets=False,old_accounts_unchanged=True)
 save(DATA/'installation.json',receipt)
 print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
