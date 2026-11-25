"""Create a new sealed outcome runtime without changing any existing release."""
from pathlib import Path
import json,hashlib,shutil,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'scripts/outcomes_r7';DATA=ROOT/'data/research-outcomes-r7'
RELEASE=ROOT/'data/releases/outcomes-r7-20261027'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
tree=ET.parse(DATA/'qualification-initial.xml')
assert not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
assert len(tree.findall('.//testcase'))>=26
RELEASE.mkdir()
for name in ['collector.py','dashboard.py']:shutil.copy2(SOURCE/name,RELEASE/name)
shutil.copy2(ROOT/'scripts/week1_r6/power_guard.py',RELEASE/'power_guard.py')
shutil.copy2(DATA/'protocol.json',RELEASE/'protocol.json')
shutil.copy2(DATA/'qualification-initial.xml',RELEASE/'qualification.xml')
shutil.copy2(ROOT/'tests/test_outcomes_r7.py',RELEASE/'test_outcomes_r7.py')
write(RELEASE/'test-evidence.json',dict(command='data/research-v2/clean-env/bin/python -B -m pytest tests/test_outcomes_r7.py -q --junitxml=data/research-outcomes-r7/qualification-initial.xml',passing=len(tree.findall('.//testcase')),failures=0,source_sha256=sha(SOURCE/'collector.py'),tests_sha256=sha(ROOT/'tests/test_outcomes_r7.py'),junit_sha256=sha(DATA/'qualification-initial.xml'),reused_qualification='Parent r6 strict provider validation and archived account contract; source remains unchanged'))
write(RELEASE/'manifest.json',dict(status='sealed',id='OUTCOME-RECOVERY-R7-20261027',external_orders=False,files={p.name:sha(p) for p in RELEASE.iterdir() if p.is_file()}))
print(json.dumps({'release':str(RELEASE),'manifest_sha256':sha(RELEASE/'manifest.json')}))
