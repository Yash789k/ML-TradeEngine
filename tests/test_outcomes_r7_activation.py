"""Verify the actual sealed adapter and report gates without mutating its release."""
import json
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'data/releases/outcomes-r7-20261027'
PYTHON=ROOT/'data/research-v2/clean-env/bin/python'


@pytest.mark.parametrize('name',['collector.py','protocol.json','power_guard.py'])
def test_sealed_source_or_config_tampering_rejected(tmp_path,name):
    local=tmp_path/'release';shutil.copytree(RELEASE,local)
    with (local/name).open('a') as f:f.write('\n# changed\n')
    # Execute the original sealed adapter against the corrupt isolated copy.
    r=subprocess.run([str(PYTHON),'-B',str(RELEASE/'collector.py'),'verify','--release',str(local)],capture_output=True,text=True)
    assert r.returncode!=0 and 'Sealed outcome source changed' in r.stderr


def test_premature_final_report_rejected_before_output_created(tmp_path):
    output=tmp_path/'report'
    r=subprocess.run([str(PYTHON),'-B',str(RELEASE/'collector.py'),'report','--final-outcomes','--workspace',str(tmp_path/'empty'),'--output',str(output)],capture_output=True,text=True)
    assert r.returncode!=0 and 'all six new reads' in r.stderr
    assert not output.exists()


def test_actual_readonly_interim_report_preserves_account_bytes(tmp_path):
    import hashlib
    protected=[ROOT/'data/paper'/f'research-week1-{s}'/f'{a}.sqlite' for s in ['r2','r4','r6'] for a in ['shadow','deployment']]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    output=tmp_path/'report'
    r=subprocess.run([str(PYTHON),'-B',str(RELEASE/'collector.py'),'report','--workspace',str(tmp_path/'empty'),'--output',str(output)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    report=json.loads((output/'report.json').read_text())
    assert report['prediction_rows']==0 and report['original_pilot_operational_complete'] is False
    assert report['profitability_approved'] is False
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
