"""Evidence manifests bound to commands, collected test IDs and exact local source."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hashes():
    paths=set()
    for pat in ['src/**/*.py','runtime_v2/**/*.py','research_v2/**/*.py','tests/*.py','scripts/*.py','*.py','requirements.txt','pytest.ini']:
        paths.update(ROOT.glob(pat))
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths) if p.is_file()}

def run_tests(destination, arguments):
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    before=source_hashes(); started=time.time()
    env=os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    # Override repository addopts for this read-only collection, then repeat the
    # explicit marker selection. Double -q otherwise collapses IDs to file counts.
    collect_args=[a for a in arguments if a not in ['-q','--quiet']]
    if not any(a in ['-m','--markexpr'] or a.startswith('--markexpr=') for a in collect_args):
        collect_args+=['-m','not integration']
    collect=[sys.executable,'-m','pytest','-o','addopts=',*collect_args,'--collect-only','-q']
    c=subprocess.run(collect,cwd=ROOT,env=env,capture_output=True,text=True)
    (out/'collected.txt').write_text(c.stdout+c.stderr)
    expected=[s.strip() for s in c.stdout.splitlines() if '::test_' in s and not s.startswith(' ')]
    report=out/'results.xml'
    cmd=[sys.executable,'-m','pytest',*arguments,'--junitxml',str(report)]
    with (out/'output.log').open('w') as f:
        run=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    tree=ET.parse(report).getroot() if report.exists() else ET.Element('empty')
    cases=tree.findall('.//testcase');failed=tree.findall('.//failure')+tree.findall('.//error');skipped=tree.findall('.//skipped')
    after=source_hashes()
    manifest={'started_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(started)),
              'finished_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'elapsed_seconds':time.time()-started,'command':cmd,'collection_command':collect,
              'source_before':before,'source_after':after,'source_unchanged':before==after,
              'collected_ids':expected,'test_cases':[{'classname':e.get('classname'),'name':e.get('name'),'seconds':e.get('time'),'status':'failed' if list(e.findall('failure'))+list(e.findall('error')) else 'skipped' if e.find('skipped') is not None else 'passed'} for e in cases],
              'collected_count':len(expected),'executed_count':len(cases),'passed':len(cases)-len(failed)-len(skipped),'failed':len(failed),'skipped':len(skipped),
              'exit_code':run.returncode,'collection_exit_code':c.returncode,'environment':{'python':sys.version,'platform':platform.platform(),'executable':sys.executable},
              'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
              'data_manifest_sha256':digest(ROOT/'data/releases/2026-09-28-adaptive/market/manifest.json'),
              'protocol_sha256':digest(ROOT/'data/research-v2/protocol.json'),
              'reports':{p.name:digest(p) for p in out.iterdir() if p.is_file()}}
    manifest['status']='passed' if before==after and c.returncode==0 and run.returncode==0 and len(cases)==len(expected) and len(cases)>0 and not failed and not skipped else 'not_qualified'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:manifest[k] for k in ['status','passed','failed','skipped','source_unchanged','elapsed_seconds']},indent=2))
    return 0 if manifest['status']=='passed' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('pytest_args',nargs=argparse.REMAINDER);a=p.parse_args()
    args=a.pytest_args[1:] if a.pytest_args and a.pytest_args[0]=='--' else a.pytest_args
    raise SystemExit(run_tests(a.output,args))
