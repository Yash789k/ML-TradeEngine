#!/usr/bin/env python
"""Registered fixed-budget operational timing, never an economic evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'data/releases/research-v2-runtime-20260928-r2'
sys.path.insert(0,str(RELEASE))
import json
import platform
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from runtime_v2.storage import read,save,digest
from runtime_v2.operations import verify,read_record
from runtime_v2.market import load
from runtime_v2.model import AdaptiveModel,features
from runtime_v2.policy import targets,utc

cfg=verify(RELEASE)
commission=read_record(ROOT/'data/paper/research-v2/commissioning/days/2026-09-25/result.json')
data,manifest=load(commission['snapshot'],cfg['symbols'],'2026-09-25')
paths={s:str(RELEASE/'models'/s) for s in cfg['symbols']}
records=[]
for i in range(20):
 t=time.perf_counter();_,_,_,stages=targets(data,paths,'2026-09-25',cfg['anchor'],ROOT/'data/research-v2/operations/unused-model-destination')
 records.append(dict(repetition=i,whole_batch=time.perf_counter()-t,**stages))
 save(ROOT/'data/research-v2/operations/latency-checkpoint-r2.json',records)
training=[]
with threadpool_limits(limits=1):
 x=features(data['SPY'],data['SPY']);close=data['SPY'].Close
 for i in range(5):
  t=time.perf_counter();model=AdaptiveModel().fit_at(x,close,close.index[-1]);training.append(time.perf_counter()-t)
summary={name:dict(n=len(records),p50=float(np.quantile([r[name] for r in records],.5)),p95=float(np.quantile([r[name] for r in records],.95)),p99=None) for name in ['whole_batch','features','inference']}
result=dict(trial_id='R11-LATENCY-V2-R2',created_at=utc().isoformat(),release_sha256=digest(RELEASE/'manifest.json'),snapshot_sha256=digest(Path(commission['snapshot'])/'manifest.json'),
 hardware=dict(platform=platform.platform(),machine=platform.machine(),python=sys.version),dependencies=read(RELEASE/'manifest.json')['dependencies'],warm_records=records,warm_summary=summary,
 training_spy_seconds=training,training_n=5,training_median=float(np.median(training)),cold_commissioning=commission['timings_seconds'],cold_n=1,
 peak_resident_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)),
 artifact_bytes=sum(p.stat().st_size for p in RELEASE.rglob('*') if p.is_file()),model_bytes=sum(p.stat().st_size for p in (RELEASE/'models').rglob('*') if p.is_file()),
 provider_requests=commission['provider_request_count'],provider_failures=len(manifest['errors']),
 limitations=['timings on one shared laptop under concurrent research workload','cold sample n1 and training n5 insufficient tail estimates','20 warm samples empirical P95 descriptive; no P99 estimate','timing repetitions are not independent financial observations'])
save(ROOT/'data/research-v2/operations/latency-r2.json',result)
print(json.dumps(dict(warm_summary=summary,training_n=5,training_median=result['training_median'],cold=result['cold_commissioning'])))
