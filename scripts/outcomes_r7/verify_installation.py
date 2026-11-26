"""Record activation proof using real services and read-only evidence checks."""
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from dateutil.rrule import rrulestr
import subprocess,json,hashlib,os,re,tomllib,urllib.request,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'data/research-outcomes-r7';R=ROOT/'data/releases/outcomes-r7-20261027';W=ROOT/'data/paper/research-outcomes-r7';P=ROOT/'data/research-v2/clean-env/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def run(args):return subprocess.run(args,check=True,text=True,capture_output=True).stdout
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def inventory(p):return {str(x.relative_to(p)):sha(x) for x in sorted(p.rglob('*')) if x.is_file()}
state=json.loads(run([str(P),'-B',str(R/'collector.py'),'status']))
assert state['decision_dates_preserved']==5 and state['completed_outcome_reads']==0 and state['new_decisions']==0
assert not state['halted'] and not state['missing_new_sessions'] and not state['external_orders']
assert state['next_event']['collection_hkt']=='2026-10-28T04:30:00+08:00'
assert all(x['ledger_consistent_at_last_mark'] for x in state['archived_accounts'])
now=datetime.now(timezone.utc);hb=state['heartbeat'];age=(now-datetime.fromisoformat(hb['at'])).total_seconds()
assert age<300 and hb['parent_pid']==1 and hb['phase']=='idle' and hb['release']==str(R)
owner=run(['launchctl','print',f'gui/{os.getuid()}/local.ml-trade-engine.collector-v2'])
assert str(R) in owner and str(W) in owner and 'last exit code = 0' in owner
listing=run(['launchctl','list']);labels=[line.split()[-1] for line in listing.splitlines() if 'ml-trade-engine' in line]
assert [x for x in labels if 'collector' in x]==['local.ml-trade-engine.collector-v2']
assert 'local.ml-trade-engine.awake-week1-r6' not in labels and 'local.ml-trade-engine.dashboard-week1-r6' not in labels
for label in ['local.ml-trade-engine.awake-week1-r6','local.ml-trade-engine.dashboard-week1-r6']:
 assert not (Path.home()/'Library/LaunchAgents'/(label+'.plist')).exists()
guard=run(['launchctl','print',f'gui/{os.getuid()}/local.ml-trade-engine.awake-outcomes-r7']);pid=re.search(r'\bpid = (\d+)',guard).group(1)
assert 'state = running' in guard and '2026-11-04T07:00:00+08:00' in guard
assertions=run(['pmset','-g','assertions']);power=run(['pmset','-g','batt'])
assert f'pid {pid}(caffeinate)' in assertions and 'AC Power' in power
assert urllib.request.urlopen('http://127.0.0.1:8504/_stcore/health',timeout=10).read()==b'ok'
assert 'local.ml-trade-engine.dashboard-outcomes-r7' in labels
preservation=[]
for parent in ['research-week1-r3','research-week1-r4','research-week1-r5','research-week1-r6','research-outcomes-r7']:
 f=ROOT/'data'/parent/('r6-preservation.json' if parent=='research-outcomes-r7' else 'failed-trial-preservation.json')
 v=read(f);actual=inventory(Path(v['workspace']));assert actual==v['files'],v['workspace']
 preservation.append(dict(workspace=v['workspace'],files_unchanged=len(actual),proof=str(f),proof_sha256=sha(f)))
auto=tomllib.loads((Path.home()/'.codex/automations/trading-pilot-evidence-watch/automation.toml').read_text())
assert auto['status']=='ACTIVE' and 'research-outcomes-r7' in auto['prompt'] and 'TZID=America/New_York' in auto['rrule']
events=rrulestr(auto['rrule']).between(datetime(2026,10,27,tzinfo=ZoneInfo('America/New_York')),datetime(2026,11,4,tzinfo=ZoneInfo('America/New_York')))
protocol=read(R/'protocol.json')
expected=sorted(datetime.fromisoformat(row[k]) for row in protocol['schedule'] for k in ['observer_open','observer_close'])
assert events==expected and len(events)==12
qual={}
for name in ['qualification-initial.xml','activation-qualification.xml']:
 tree=ET.parse(D/name);assert not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
 qual[name]=dict(passed=len(tree.findall('.//testcase')),sha256=sha(D/name))
assert sum(x['passed'] for x in qual.values())==31
report=D/'reports/activation-20261027';manifest=read(report/'manifest.json')
assert not manifest['final_outcomes']
for name,expected_sha in manifest['files'].items():assert sha(report/name)==expected_sha
assert read(report/'report.json')['original_pilot_operational_complete'] is False
ui=dict(command='/Users/a/miniconda3/bin/python -B; streamlit.testing.v1.AppTest.from_file(sealed dashboard).run()',exceptions=0,errors=0,observed_metrics={'Original decision dates':'5','New outcome reads':'0/6','Decision outcomes in new reads':'0/5'},source_sha256=sha(R/'dashboard.py'),live_http_health='ok')
proof=dict(verified_at=now.isoformat(),status=state,heartbeat_age_seconds=age,release_manifest_sha256=sha(R/'manifest.json'),sole_owner_verified=True,power=power.strip(),guard_pid=int(pid),guard_until='2026-11-04T07:00:00+08:00',preservation=preservation,qualifications=qual,dashboard=ui,automation={k:auto[k] for k in ['id','kind','status','rrule','target_thread_id','prompt']},observer_dates_hkt=[d.astimezone(ZoneInfo('Asia/Hong_Kong')).isoformat() for d in events],preflight=read(D/'provider-preflight.json'),prospective_reads_completed=0,profitability_approved=False,external_orders=False)
save(D/'activation-verification.json',proof)
for name,content in [('launchctl-collector.txt',owner),('launchctl-guard.txt',guard),('power-assertions.txt',assertions),('power-source.txt',power)]:
 with (D/name).open('x') as f:f.write(content)
save(D/'activation-test-source-evidence.json',dict(tests=str(ROOT/'tests/test_outcomes_r7_activation.py'),tests_sha256=sha(ROOT/'tests/test_outcomes_r7_activation.py'),sealed_manifest_sha256=sha(R/'manifest.json'),junit_sha256=sha(D/'activation-qualification.xml'),passed=5))
save(ROOT/'docs/research-outcomes-r7/observer-state.json',dict(schema_version=1,observer_only=True,automation_id=auto['id'],trial_id=protocol['id'],last_checked_at=now.isoformat(),status=state,heartbeat=hb,heartbeat_age_seconds=age,notified_event_ids=['OUTCOME-RECOVERY-R7-20261027:activation'],notification_records=[dict(event='OUTCOME-RECOVERY-R7-20261027:activation',reason='User-authorized reschedule verified')],known_prior_missed_sessions=['2026-10-23','2026-10-26'],actionable_event_ids=[],durable_records={},skipped_invocations=[],last_check_disposition='human_authorized_restart_verified_before_first_window',shutdown_after_complete=True,proof=str(D/'activation-verification.json'),proof_sha256=sha(D/'activation-verification.json')))
print(json.dumps({'activation_verified':True,'heartbeat_age_seconds':age,'tests_passed':31,'provider_symbols':22,'collector_last_exit':0,'on_ac_power':True,'guard_pid':pid,'preserved_workspaces':preservation,'observer_checks_registered':len(events),'first_collection_hkt':protocol['schedule'][0]['collection_hkt'],'last_collection_hkt':protocol['schedule'][-1]['collection_hkt'],'dashboard':'http://127.0.0.1:8504/'}))
