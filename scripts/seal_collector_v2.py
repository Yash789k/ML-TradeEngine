#!/usr/bin/env python
"""Seal a NEW isolated collector only after running the required source-bound suite."""
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_v2.storage import digest, save, read
from runtime_v2.policy import utc
from runtime_v2.model import AdaptiveModel, AdaptiveConfig
from runtime_v2.operations import CRITICAL_TESTS
from runtime_v2.calendar import calendar

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/releases/research-v2-runtime-20260928-r2'
if DEST.exists(): raise SystemExit('Destination exists; sealed releases must never be overwritten')
files=sorted((ROOT/'runtime_v2').glob('*.py'))+[ROOT/'scripts/collector_v2.py']
source={str(p.relative_to(ROOT)):digest(p) for p in files}
test_hash=digest(ROOT/'tests/test_collector_v2.py')
report=ROOT/'data/research-v2/operations/r2-final-tests.xml'
command=[sys.executable,'-m','pytest','tests/test_collector_v2.py','-q',f'--junitxml={report}']
started=utc().isoformat()
subprocess.run(command,cwd=ROOT,check=True)
if source!={str(p.relative_to(ROOT)):digest(p) for p in files} or test_hash!=digest(ROOT/'tests/test_collector_v2.py'):
    raise SystemExit('Source/tests changed during evaluation; seal refused')
tree=ET.parse(report)
suites=tree.findall('.//testsuite'); failures=sum(int(t.attrib['failures']) for t in suites);errors=sum(int(t.attrib['errors']) for t in suites)
names=[t.attrib['name'].split('[')[0] for t in tree.findall('.//testcase') if not list(t)]
if failures or errors or not set(CRITICAL_TESTS)<=set(names): raise SystemExit('Critical suite missing or failed')
DEST.mkdir(parents=True)
for p in files:
 target=DEST/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
shutil.copy2(report,DEST/'tests.xml')
shutil.copy2(ROOT/'tests/test_collector_v2.py',DEST/'test_collector_v2.py')
save(DEST/'test-evidence.json',dict(command=command,started_at=started,finished_at=utc().isoformat(),source=source,
     passing_test_names=names,failures=failures,errors=errors,junit_sha256=digest(report),test_source_sha256=test_hash,
     environment=dict(python=sys.version,platform=platform.platform())))
old=ROOT/'data/releases/2026-09-28-adaptive'; readiness=read(old/'readiness.json')
# Trusted locally verified bundles: migrate both Python class paths, without refitting or changing metadata.
from src.live.adaptive import verify_release
verify_release(old)
for symbol in readiness['symbols']:
 model=AdaptiveModel.load(old/'models'/symbol)
 model.__class__=AdaptiveModel
 model.config=AdaptiveConfig(**asdict(model.config))
 model.save(DEST/'models'/symbol)
start='2026-09-29';dates=calendar().sessions_in_range(start,'2027-02-01')[:63]
config=dict(version=2,trial_id='PILOT-TREND-V2-20260929',registered_at=utc().isoformat(),symbols=readiness['symbols'],
 capital=10000,start=start,end=str(dates[-1].date()),sessions=63,anchor=start,
 refit_rule='carried 2026-09-25 model; refit at anchor+21,+42,+63 XNYS sessions',
 first_refit=str(calendar().sessions[calendar().sessions.get_loc(dates[0])+21].date()),
 weekly_rule='first decision and last XNYS session of ISO week',shadow='trend',deployment='cash',external_orders=False,
 price_units='chained adjusted total-return proxy units; not broker shares',
 missed_session='visible halt; no backfill; fixed end date',
 protocol_sha256=digest(ROOT/'data/research-v2/protocol.json'),
 parent_release_sha256=digest(old/'manifest.json'))
save(DEST/'config.json',config)
packages=['numpy','pandas','scipy','scikit-learn','joblib','pyarrow','exchange-calendars','requests','threadpoolctl']
deps={p:importlib.metadata.version(p) for p in packages}
(DEST/'requirements.lock').write_text('\n'.join(f'{p}=={v}' for p,v in deps.items())+'\n')
artifacts={str(p.relative_to(DEST)):digest(p) for p in DEST.rglob('*') if p.is_file() and str(p.relative_to(DEST)) not in source}
save(DEST/'manifest.json',dict(version=2,status='sealed',created_at=utc().isoformat(),source=source,artifacts=artifacts,
 dependencies=deps,external_orders=False,readiness='local simulation only; prospective unobserved; historical edge unapproved'))
print(str(DEST))
