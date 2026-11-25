#!/usr/bin/env python
"""Capture actual OS-trigger, read-only validation and canonical-path restore proof."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'data/releases/research-v2-runtime-20260928-r2'
sys.path.insert(0,str(RELEASE))
import json
import os
import subprocess
from runtime_v2.operations import verify,read_record,status,restore_original_path
from runtime_v2.storage import read,save,digest
from runtime_v2.policy import utc

workspace=ROOT/'data/paper/research-v2';out=ROOT/'data/research-v2/operations'
config=verify(RELEASE)
if not (workspace/'commissioning-complete.json').exists():raise SystemExit('Actual scheduled commissioning not complete')
commission=workspace/'commissioning';result=read_record(commission/'days/2026-09-25/result.json')
check=status(RELEASE,commission)
if not check['ledger_consistent']:raise SystemExit(check['state_consistency_errors'])
archive=workspace/'commissioning-before-restore-r2'
if not archive.exists():
 restore=restore_original_path(commission/'backups/commissioning',commission,archive)
 restored=status(RELEASE,commission)
 if not restored['ledger_consistent']:raise AssertionError(restored)
 if read_record(commission/'days/2026-09-25/result.json')['content_sha256']!=result['content_sha256']:raise AssertionError('Receipt changed on restore')
 save(out/'commissioning-restore-r2.json',dict(**restore,ledger_consistent=True,receipt_sha256=result['content_sha256'],symbols=len(result['signals']),release_sha256=digest(RELEASE/'manifest.json')))
info=subprocess.run(['launchctl','print',f'gui/{os.getuid()}/local.ml-trade-engine.collector-v2'],capture_output=True,text=True,check=True).stdout
(out/'launchctl-verified-r2.txt').write_text(info)
verification=dict(verified_at=utc().isoformat(),release=str(RELEASE),release_sha256=digest(RELEASE/'manifest.json'),
 trial_id=config['trial_id'],scheduler_label='local.ml-trade-engine.collector-v2',os_report=[x.strip() for x in info.splitlines() if any(w in x for w in ['state =','runs =','pid =','last exit code','run interval'])],
 original_trigger=read_record(workspace/'scheduler-trigger.json'),latest_heartbeat=read(workspace/'heartbeat.json'),
 commissioning=dict(mode=result['mode'],session=result['session'],symbols=len(result['signals']),result_sha256=result['content_sha256'],snapshot_sha256=result['snapshot_sha256'],
 timings_seconds=result['timings_seconds'],peak_rss_bytes=result['peak_rss_bytes'],provider_request_count=result['provider_request_count'],
 deployment_equity=result['deployment']['equity'],shadow_equity=result['shadow']['equity'],deployment_reconciliation=result['deployment']['reconciliation_error'],shadow_reconciliation=result['shadow']['reconciliation_error']),
 restore=read(out/'commissioning-restore-r2.json'),status=status(RELEASE,workspace),remote_workflow_tested=False,physical_sleep_restart_tested=False,host_must_remain_awake=True,external_orders=False)
save(out/'scheduler-verification-r2.json',verification)
print(json.dumps(verification,indent=2))
