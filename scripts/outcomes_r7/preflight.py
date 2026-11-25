"""One bounded public-data diagnostic; never a scheduled read or decision."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'data/releases/research-v2-runtime-20261022-week1-r6'))
from runtime_v2 import operations as op,market
from runtime_v2.storage import digest,save
from runtime_v2.policy import utc
p=json.loads((ROOT/'data/research-outcomes-r7/protocol.json').read_text())
now=utc();day=op.completed(now);loc=ROOT/'data/research-outcomes-r7/preflight-20261027'
market.download(loc,p['symbols'],day,now=now)
frames,meta=market.load(loc,p['symbols'],day,now=utc())
save(ROOT/'data/research-outcomes-r7/provider-preflight.json',dict(at=utc().isoformat(),snapshot=str(loc),snapshot_sha256=digest(loc/'manifest.json'),session=str(day.date()),symbols=len(frames),mode='manual_provider_preflight',counts_as_scheduled_outcome_read=False,counts_as_decision=False),immutable=True)
print(json.dumps({'valid_symbols':len(frames),'session':str(day.date()),'scheduled_read_count':0,'mode':'manual_provider_preflight'}))
