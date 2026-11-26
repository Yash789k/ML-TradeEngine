"""Register the outcome-only restart before testing, downloads or activation."""
from pathlib import Path
from datetime import datetime, timezone
import json, sys, hashlib, subprocess
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/releases/research-v2-runtime-20261022-week1-r6'
sys.path.insert(0,str(BASE))
from runtime_v2 import operations as op
from runtime_v2.calendar import calendar
from runtime_v2.model import AdaptiveModel
import pandas as pd
D=ROOT/'data/research-outcomes-r7'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 with p.open('x') as f: json.dump(v,f,indent=2,allow_nan=False,default=str);f.write('\n')
config=op.verify(BASE);cal=calendar()
segments=[];decisions=[]
for tag,release,dates in [('r2','research-v2-runtime-20261001-week1-r2',['2026-10-01','2026-10-02']),('r4','research-v2-runtime-20261014-week1-r4',['2026-10-14','2026-10-15']),('r6',BASE.name,['2026-10-22'])]:
 rel=ROOT/'data/releases'/release;ws=ROOT/'data/paper'/('research-week1-'+tag)
 state=json.loads(subprocess.run([sys.executable,'-B',str(rel/'scripts/collector_v2.py'),'status','--release',str(rel),'--workspace',str(ws)],check=True,capture_output=True,text=True).stdout)
 assert state['valid_observed_sessions']==len(dates) and state['ledger_consistent'] and not state['external_orders']
 receipts={r['session']:r for r in op._valid_results(ws,json.loads((rel/'config.json').read_text())) if r['mode']=='prospective_shadow'}
 assert sorted(receipts)==dates
 segments.append(dict(segment=tag,release=str(rel),workspace=str(ws),manifest_sha256=sha(rel/'manifest.json'),status_at_registration=state,accounts_at_last_actual_receipt={n:receipts[dates[-1]][n] for n in ['deployment','shadow']}))
 for date in dates:
  r=receipts[date];idx=cal.sessions.get_loc(pd.Timestamp(date));priors={s:AdaptiveModel.load(p).prior.tolist() for s,p in r['models'].items()}
  decisions.append(dict(segment=tag,session=date,receipt=str(ws/'days'/date/'result.json'),receipt_sha256=sha(ws/'days'/date/'result.json'),content_sha256=r['content_sha256'],entry=str(cal.sessions[idx+1].date()),endpoint=str(cal.sessions[idx+6].date()),signals=r['signals'],priors=priors))
sessions=cal.sessions_in_range('2026-10-27','2026-11-03')
assert len(sessions)==6
schedule=[]
for d in sessions:
 c=cal.session_close(d);o=cal.session_open(d)
 schedule.append(dict(session=str(d.date()),open=o.isoformat(),close=c.isoformat(),collection=(c+pd.Timedelta(minutes=30)).isoformat(),collection_hkt=(c+pd.Timedelta(minutes=30)).tz_convert('Asia/Hong_Kong').isoformat(),watchdog=(c+pd.Timedelta(minutes=90)).isoformat(),observer_open=(o+pd.Timedelta(minutes=5)).isoformat(),observer_close=(c+pd.Timedelta(minutes=95)).isoformat()))
protocol=dict(id='OUTCOME-RECOVERY-R7-20261027',registered_at=datetime.now(timezone.utc).isoformat(),authorization='User 27 October 2026 HKT: I wasnt able to keep the Outcome collection working, set the collection dates from tonight onwards.',kind='outcome-only collection restart; no new decision, model, account or strategy',parent_release=str(BASE),parent_manifest_sha256=sha(BASE/'manifest.json'),parent_protocol=str(ROOT/'data/research-week1-r6/protocol.json'),parent_protocol_sha256=sha(ROOT/'data/research-week1-r6/protocol.json'),symbols=config['symbols'],segments=segments,decisions=decisions,schedule=schedule,external_orders=False,create_targets=False,modify_archived_accounts=False,refit=False,return_definition='close[t+6]/close[t+1]-1; original decision dates and endpoints remain fixed',label_rule={'down':'return < -0.01','up':'return > 0.01','flat':'otherwise, only after endpoint'},collection_rule='Six new on-time validated price snapshots; initial close+30..40, bounded retries close+45..55 and +60..65; watchdog +90. No backdated receipts.',outcome_rule='Use earliest successful new snapshot at or after the original endpoint; label retrieval date and current-vintage adjustments. Earlier labels remain preserved. Never extend a forecast horizon or fabricate a fill.',original_missed_outcome_sessions=['2026-10-23','2026-10-26'],cancelled_old_tail_sessions=['2026-10-27','2026-10-28','2026-10-29','2026-10-30'],old_paper_account_tail='incomplete; cannot be repaired by moving dates or filling retrospectively',completion_gate='6/6 new scheduled read receipts, all 5 original-date labels mature and verified, archived receipts/accounts unchanged and reconciling at their last real mark, no active integrity halt; this completes the amended outcome dataset only',original_operational_gate='failed/incomplete and never promoted by the new collection',shutdown_after_amended_completion=True,statistics='Descriptive accuracy, log loss, Brier score against frozen training-frequency priors; five dependent dates, no p-values or profitability approval',qualification_plan=['XNYS calendar including November DST','strict receipt timing, uniqueness and snapshot hash verification','original entry/endpoint and future-label rejection','retry bounded to three slots and idempotent restart','missed windows remain missed, later reads cannot fill old accounts','source drift and incomplete final gate rejected','read-only status preserves archived account and receipt bytes','dashboard smoke and actual launchd heartbeat without prospective count'])
save(D/'protocol.json',protocol)
print(json.dumps({'registered':str(D/'protocol.json'),'schedule':schedule,'decisions':[{k:d[k] for k in ['segment','session','entry','endpoint']} for d in decisions]},indent=2))
