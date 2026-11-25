#!/usr/bin/env python
"""Actual process-death and artifact-gate proofs in isolated temporary state."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'data/releases/research-v2-runtime-20260928-r2'
sys.path.insert(0,str(RELEASE))
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from runtime_v2.operations import daily,verify,read_record
from runtime_v2.policy import utc
from runtime_v2.storage import read,save,digest

if len(sys.argv)>1 and sys.argv[1]=='child':
 stage,workspace,snapshot=sys.argv[2:]
 def fault(point):
  if point==stage: os.kill(os.getpid(),signal.SIGKILL)
 config=verify(RELEASE)
 if stage.startswith('model_'):
  from runtime_v2.calendar import calendar
  from runtime_v2.model import AdaptiveModel
  import pandas as pd
  d=pd.Timestamp('2026-09-25');config['anchor']=str(calendar().sessions[calendar().sessions.get_loc(d)-21].date())
  if stage=='model_partial':
   old=AdaptiveModel.save
   def interrupted(self,path):
    old(self,path);os.kill(os.getpid(),signal.SIGKILL)
   AdaptiveModel.save=interrupted
  else:
   old=os.replace
   def interrupted(src,dst):
    old(src,dst)
    if str(src).endswith('.partial'): os.kill(os.getpid(),signal.SIGKILL)
   os.replace=interrupted
 daily(RELEASE,workspace,'2026-09-25',mode='commissioning',snapshot=snapshot,fault=fault,validated_config=config)
 raise SystemExit('Fault was not triggered')

start=utc().isoformat()
commission=read_record(ROOT/'data/paper/research-v2/commissioning/days/2026-09-25/result.json')
config=verify(RELEASE)
rows=[]
with tempfile.TemporaryDirectory(prefix='collector-process-proof-') as temp:
 temp=Path(temp)
 clean=daily(RELEASE,temp/'clean','2026-09-25',mode='commissioning',snapshot=commission['snapshot'])
 for point in ['before_plan','after_plan','after_deployment','after_shadow','before_receipt','after_receipt','model_partial','model_committed']:
  workspace=temp/point
  t=time.perf_counter()
  child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'child',point,str(workspace),commission['snapshot']],capture_output=True,text=True)
  if child.returncode != -signal.SIGKILL: raise RuntimeError(f'Expected SIGKILL {point}: {child.returncode} {child.stderr}')
  cfg=dict(config)
  reference=clean
  if point.startswith('model_'):
   from runtime_v2.calendar import calendar
   import pandas as pd
   d=pd.Timestamp('2026-09-25');cfg['anchor']=str(calendar().sessions[calendar().sessions.get_loc(d)-21].date())
   reference=daily(RELEASE,temp/(point+'-reference'),'2026-09-25',mode='commissioning',snapshot=commission['snapshot'],validated_config=cfg)
  recovered=daily(RELEASE,workspace,'2026-09-25',mode='commissioning',snapshot=commission['snapshot'],validated_config=cfg)
  for key in ['shadow','deployment','shadow_weights','rebalance','marks']:
   if recovered[key] != reference[key]: raise AssertionError(f'{point}:{key} mismatch')
  for name in ['shadow','deployment']:
   import sqlite3
   with sqlite3.connect(workspace/f'{name}.sqlite') as db: count=db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
   refpath=temp/(point+'-reference') if point.startswith('model_') else temp/'clean'
   with sqlite3.connect(refpath/f'{name}.sqlite') as db: expected=db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
   if count!=expected: raise AssertionError('Duplicate persisted events')
  rows.append(dict(point=point,exit_code=child.returncode,status='recovered_exactly',elapsed_seconds=time.perf_counter()-t))
  save(ROOT/'data/research-v2/operations/process-crash-checkpoint-r2.json',rows)
 gates=[]
 for kind in ['changed_source','changed_config','changed_model','unrelated_junit','missing_critical_tests']:
  copy=temp/kind;shutil.copytree(RELEASE,copy)
  manifest=read(copy/'manifest.json')
  if kind=='changed_source':
   with (copy/'runtime_v2/policy.py').open('a') as f:f.write('\n# mutation\n')
  elif kind=='changed_config':
   value=read(copy/'config.json');value['capital']=20000;save(copy/'config.json',value)
  elif kind=='changed_model':
   with (copy/'models/SPY/model.joblib').open('ab') as f:f.write(b'mutation')
  elif kind=='unrelated_junit':
   (copy/'tests.xml').write_text('<testsuite tests="1" failures="0" errors="0"/>')
  else:
   value=read(copy/'test-evidence.json');value['passing_test_names']=[];save(copy/'test-evidence.json',value)
   manifest['artifacts']['test-evidence.json']=digest(copy/'test-evidence.json');save(copy/'manifest.json',manifest)
  try: verify(copy)
  except ValueError as exc: gates.append(dict(case=kind,status='refused',reason=str(exc)))
  else: raise AssertionError(f'Unsupported gate passed: {kind}')
result=dict(trial_id='R11-PROCESS-CRASH-R2',started_at=start,finished_at=utc().isoformat(),release_sha256=digest(RELEASE/'manifest.json'),
 source=read(RELEASE/'manifest.json')['source'],snapshot_sha256=commission['snapshot_sha256'],process_kills=rows,artifact_gates=gates,
 scope='isolated local processes; SIGKILL on actual process, not physical power loss; no pilot observations')
save(ROOT/'data/research-v2/operations/process-recovery-r2.json',result)
print(json.dumps(result,indent=2))
