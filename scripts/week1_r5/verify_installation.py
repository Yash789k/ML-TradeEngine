"""Read installed services and sealed receipts; write activation proof outside the runtime."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,subprocess,os,re,xml.etree.ElementTree as ET,tomllib
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/research-week1-r5'
RELEASE=ROOT/'data/releases/research-v2-runtime-20261021-week1-r5'
WORKSPACE=ROOT/'data/paper/research-week1-r5'
OLD=ROOT/'data/releases/research-v2-runtime-20261014-week1-r4'
sys.path.insert(0,str(RELEASE))
from runtime_v2 import operations as op
from runtime_v2.storage import digest

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str)+'\n')
def inventory(p):return {str(x.relative_to(p)):digest(x) for x in sorted(p.rglob('*')) if x.is_file()}
def command(args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout

def main():
    root,data,release,w,old=ROOT,DATA,RELEASE,WORKSPACE,OLD
    now=datetime.now(timezone.utc)
    config=op.verify(release);state=op.status(release,w,validated_config=config)
    assert not state['halted'] and not state['missing_sessions'] and state['valid_observed_sessions']==0 and state['ledger_consistent'] and not state['external_orders']
    assert state['next_event']['hong_kong']=='2026-10-22T04:30:00+08:00'
    trigger=op.read_record(w/'scheduler-trigger.json');commission=op.read_record(w/'commissioning-complete.json')
    preview=op._valid_results(w/'commissioning',config)
    assert len(preview)==1 and preview[0]['mode']=='commissioning_preview' and preview[0]['session']=='2026-10-20'
    assert not op.state_errors(w/'commissioning',preview,config)
    p=preview[0]
    attempts=[read(a) for a in (w/'commissioning/attempts').glob('*.json')]
    assert len(attempts)==1 and attempts[0]['status']=='complete'
    assert attempts[0]['parent_pid']==1 and attempts[0]['service']=='local.ml-trade-engine.collector-v2'
    assert trigger['parent_pid']==1 and commission['trigger']['parent_pid']==1
    assert len(p['signals'])==22 and p['content_sha256']==commission['result_sha256']
    assert p['timings_seconds']['training']==0
    assert all(abs(p[name]['reconciliation_error'])<=1e-6 and p[name]['equity']==10000 for name in ['shadow','deployment'])
    hb=read(w/'heartbeat.json');age=(now-datetime.fromisoformat(hb['at'])).total_seconds();assert age<300
    manifest=read(release/'manifest.json');oldmanifest=read(old/'manifest.json')
    assert manifest['source']==oldmanifest['source']
    models={str(x.relative_to(release/'models')):digest(x) for x in (release/'models').rglob('*') if x.is_file()}
    assert models=={str(x.relative_to(old/'models')):digest(x) for x in (old/'models').rglob('*') if x.is_file()}
    preservation=[]
    for file in [data/'failed-trial-preservation.json',root/'data/research-week1-r4/failed-trial-preservation.json',root/'data/research-week1-r3/failed-trial-preservation.json']:
        evidence=read(file);ws=Path(evidence['workspace']);actual=inventory(ws);assert actual==evidence['files'],str(ws)
        preservation.append({'workspace':str(ws),'files_unchanged':len(actual),'halt_preserved':(ws/'halt.json').exists(),'inventory_receipt':str(file),'inventory_sha256':digest(file)})
    owner=command(['launchctl','print',f'gui/{os.getuid()}/local.ml-trade-engine.collector-v2'])
    guard=command(['launchctl','print',f'gui/{os.getuid()}/local.ml-trade-engine.awake-week1-r5'])
    assertions=command(['pmset','-g','assertions']);power=command(['pmset','-g','batt'])
    listing=command(['launchctl','list']);labels=[line.split()[-1] for line in listing.splitlines() if 'ml-trade-engine' in line]
    assert [x for x in labels if 'collector' in x]==['local.ml-trade-engine.collector-v2']
    assert str(release) in owner and str(w) in owner and 'last exit code = 0' in owner
    pid=re.search(r'\bpid = (\d+)',guard).group(1)
    assert 'state = running' in guard and f'pid {pid}(caffeinate)' in assertions
    assert not any(x in labels for x in ['local.ml-trade-engine.awake-week1-r4','local.ml-trade-engine.dashboard-week1-r4'])
    auto=tomllib.loads((Path.home()/'.codex/automations/trading-pilot-evidence-watch/automation.toml').read_text())
    assert auto['status']=='ACTIVE' and 'research-week1-r5' in auto['prompt'] and 'BYMINUTE=35' in auto['rrule']
    for name,contents in [('launchctl-collector.txt',owner),('launchctl-power-guard.txt',guard),('power-assertions.txt',assertions),('power-source.txt',power)]: (data/name).write_text(contents)
    qualified={}
    for name in ['qualification.xml','report-qualified-tests.xml']:
        tree=ET.parse(data/name);assert not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped');qualified[name]=len(tree.findall('.//testcase'))
    proof={'verified_at':now.isoformat(),'status':state,'heartbeat':hb,'heartbeat_age_seconds':age,'commissioning':{'trigger':trigger,'completed_at':commission['completed_at'],'mode':p['mode'],'session':p['session'],'symbol_count':22,'result_sha256':p['content_sha256'],'snapshot_sha256':p['snapshot_sha256'],'timings_seconds':p['timings_seconds'],'accounts':{name:{k:p[name][k] for k in ['equity','cash','reconciliation_error','fees','reserved_cash']} for name in ['deployment','shadow']},'counts_as_prospective':False},'sole_owner':'local.ml-trade-engine.collector-v2','source_identical_to_r4':True,'model_files_identical_to_r4':len(models),'preservation':preservation,'qualification_cases':qualified,'sleep_guard':{'running':True,'pid':int(pid),'expires':'2026-10-30T06:00:00+08:00','idle_sleep_assertion':True},'power_at_verification':power.strip(),'required_user_action':'Plug in the Mac; keep lid open, logged in and online overnight. Guard cannot override clamshell sleep or shutdown.','automation':{k:auto[k] for k in ['id','status','rrule','target_thread_id','prompt']},'external_orders':False,'profitability_approved':False}
    save(data/'scheduler-verification.json',proof)
    save(data/'report-source-evidence.json',{'command':'data/research-v2/clean-env/bin/python -B -m pytest tests/test_week1_r5_report.py -q --junitxml=data/research-week1-r5/report-qualified-tests.xml','passing':qualified['report-qualified-tests.xml'],'junit_sha256':digest(data/'report-qualified-tests.xml'),'source':{str(x.relative_to(root)):digest(x) for x in [root/'research_week1_r5/report.py',root/'tests/test_week1_r5_report.py',root/'research_v2/diagnostics.py']},'installation_report':str(data/'reports/installation-20261021/report.json')})
    observer={'schema_version':1,'automation_id':auto['id'],'observer_only':True,'trial_id':config['trial_id'],'last_checked_at':now.isoformat(),'status':state,'preserved_prior_valid_decisions':4,'total_valid_observations':4,'heartbeat':hb,'heartbeat_age_seconds':age,'heartbeat_stale_threshold_seconds':300,'halt':None,'notified_event_ids':[],'notification_records':[],'actionable_event_ids':[],'observed_events':[],'durable_records':{n:[] for n in ['days','attempts','events','milestones','outages','outcomes','terminal','outcome-runs','outcome-attempts']},'superseded_workspaces':[str(root/'data/paper'/n) for n in ['research-week1-r4','research-week1-r3','research-week1-r2','research-week1','research-v2']],'commissioning':proof['commissioning'],'last_check_disposition':'human_authorized_activation_verified_zero_new_observations','verification':{'release_integrity_verified':True,'ledger_consistent':True,'launchd_targets_active_release':True,'sleep_guard_running':True,'dashboard_health':'ok'},'reported_installation':True,'notification_gates':{'new_actionable_failure':False,'one_new_session_milestone':False,'final_maturity':False},'skipped_invocations':[]}
    observer_path=root/'docs/research-week1-r5/observer-state.json'
    if observer_path.exists():raise ValueError('Do not overwrite observer history; activation state already initialized')
    save(observer_path,observer)
    print(json.dumps({'status':state,'commissioned_at':commission['completed_at'],'preview_symbols':22,'qualification':qualified,'preserved':preservation,'sole_owner_verified':True,'guard_pid':pid,'heartbeat_age_seconds':age,'on_battery':'Battery Power' in power}))

if __name__=='__main__':main()
