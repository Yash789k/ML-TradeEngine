"""Compile evidence once; derive manuscript, figures and claim map from that version."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from datetime import datetime,timezone
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/research-v2'
DOC=ROOT/'docs/research-v2'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,float) and not math.isfinite(v):return None
    return v


def compile_statistics():
    required=['diagnostics/summary.json','experiments/summary.json','experiments/design.json','experiments/full_reproduction.json','protocol.json']
    sources={};summaries={}
    for rel in required:
        p=DATA/rel;summaries[Path(rel).parent.name+'_'+Path(rel).stem if '/' in rel else Path(rel).stem]=read(p);sources[str(p.relative_to(ROOT))]=digest(p)
    tables={}
    for p in sorted((DATA/'tables').glob('*.csv')):
        tables[p.stem]=clean(pd.read_csv(p).to_dict('records'));sources[str(p.relative_to(ROOT))]=digest(p)
    for p in sorted((DATA/'evidence').glob('*.json')):
        summaries[p.stem]=read(p);sources[str(p.relative_to(ROOT))]=digest(p)
    for p in sorted((DATA/'operations').glob('*.json')):
        summaries['operations_'+p.stem]=read(p);sources[str(p.relative_to(ROOT))]=digest(p)
    for p in sorted((DATA/'tests').glob('*.json')):
        summaries['tests_'+p.stem]=read(p);sources[str(p.relative_to(ROOT))]=digest(p)
    prospective=sorted((DATA/'prospective').glob('*/report.json'),key=lambda p:read(p)['created_at'])
    if prospective:
        p=prospective[-1];manifest=p.with_name('manifest.json');meta=read(manifest)
        if meta['files']['report.json']!=digest(p):raise ValueError('Prospective report differs from its evidence manifest')
        summaries['prospective']=read(p)
        summaries['prospective_source']=str(p.relative_to(ROOT))
        sources[str(p.relative_to(ROOT))]=digest(p);sources[str(manifest.relative_to(ROOT))]=digest(manifest)
    tests=[]
    for p in sorted((DATA/'tests').rglob('*.xml')):
        if 'clean-env' in p.parts:continue
        tree=ET.parse(p).getroot();cases=tree.findall('.//testcase')
        tests.append(dict(report=str(p.relative_to(ROOT)),cases=len(cases),failures=len(tree.findall('.//failure')),errors=len(tree.findall('.//error')),skipped=len(tree.findall('.//skipped'))))
        sources[str(p.relative_to(ROOT))]=digest(p)
    tables['verification_reports']=tests
    registry=[]
    for p in sorted((DATA/'registry').glob('R*.json')):
        if re.fullmatch(r'R\d+\.json',p.name):registry.append(read(p))
        sources[str(p.relative_to(ROOT))]=digest(p)
    registry.sort(key=lambda r:int(r['id'][1:]))
    tables['registry']=registry
    ref=DOC/'references.json';sources[str(ref.relative_to(ROOT))]=digest(ref)
    figures={}
    for p in sorted((DATA/'figures').glob('*.png')):
        figures[p.stem]=dict(path=str(p.relative_to(ROOT)),sha256=digest(p))
        sources[str(p.relative_to(ROOT))]=digest(p)
    result=dict(version='research-v2-statistics',compiled_at=datetime.now(timezone.utc).isoformat(),tables=tables,summaries=summaries,sources=sources,figures=figures,references=read(ref),rounding='Source precision retained; percentages two decimals, losses three/four decimals, counts integer. Blank means undefined, never zero.')
    (DATA/'statistics.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False))
    return result


def verify_statistics(stats,root=ROOT):
    for rel,expected in stats['sources'].items():
        p=root/rel
        if not p.exists() or digest(p)!=expected:raise ValueError('Missing or changed evidence; recompile statistics: '+rel)
    if stats['summaries']['experiments_summary']['status']!='complete':raise ValueError('Required controlled experiments incomplete')
    return True


class Manuscript:
    def __init__(self,stats):
        self.s=stats;self.claims=[];self.table_count=0;self.figure_count=0;self.figure_sources=[]
    def table_rows(self,name,**filters):
        return [r for r in self.s['tables'][name] if all(r.get(k)==v for k,v in filters.items())]
    def one(self,name,**filters):
        rows=self.table_rows(name,**filters)
        if len(rows)!=1:raise ValueError(f'Expected one {name}/{filters}, got {len(rows)}')
        return rows[0]
    def value(self,name,row,key,fmt='.3f',scale=1):
        v=row[key]
        self.claims.append(dict(kind='empirical_value',table=name,row=row,column=key,value=v,format=fmt,scale=scale,source='data/research-v2/tables/'+name+'.csv'))
        return 'undefined' if v is None else format(v*scale,fmt) if isinstance(v,(int,float)) else str(v)
    def sv(self,summary,key,fmt=',d'):
        v=self.s['summaries'][summary]
        for part in key.split('.'):v=v[part]
        self.claims.append(dict(kind='summary_value',summary=summary,key=key,value=v,format=fmt))
        return format(v,fmt) if isinstance(v,(float,int)) else str(v)
    def table(self,title,rows,columns,source):
        self.table_count+=1
        out=[f'**Table {self.table_count}. {title}** Source: `{source}`.','', '| '+' | '.join(label for key,label,fmt in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
        for r in rows:
            vals=[]
            for key,label,fmt in columns:
                v=r.get(key)
                if v is None:cell='-'
                elif isinstance(v,(float,int)) and fmt:
                    cell=format(v*100,'.2f')+'%' if fmt=='pct' else format(v,fmt)
                else:cell=str(v)
                cell=cell.replace('|','/').replace('\n',' ')
                vals.append(cell)
                self.claims.append(dict(kind='table_cell',table_number=self.table_count,source=source,row=r,column=key,value=v,display=cell))
            out.append('| '+' | '.join(vals)+' |')
        out+=['']
        return '\n'.join(out)
    def fig(self,key,caption):
        f=self.s['figures'][key]
        self.figure_count+=1
        self.claims.append(dict(kind='figure',number=self.figure_count,**f,caption=caption))
        self.figure_sources.append(key)
        return f'![Figure {self.figure_count}: {caption}]({(ROOT/f["path"]).as_posix()})\n\nFigure {self.figure_count}. {caption}\n'

    def result(self,key):
        v=self.value
        if key in ['abstract','forecasting','discussion','conclusion']:
            h=self.one('diag_forecast_summary',period='audit',model='hybrid');p=self.one('diag_forecast_summary',period='audit',model='prior')
            loss=self.one('diag_paired_inference',period='audit',model='hybrid',endpoint='log_loss',block_length=10)
            eh=self.one('diag_portfolio',period='audit',strategy='adaptive_hybrid',scope='matched_fresh_10000');eb=self.one('diag_portfolio',period='audit',strategy='equal_weight',scope='matched_fresh_10000')
            if key=='abstract':
                return (f"We evaluate causal probabilistic forecasts and capital-constrained local execution using {self.sv('diagnostics_summary','dataset.instruments')} equities and ETFs, preserved historical snapshots, controlled chronological refits and a separately frozen prospective collector. The saved experiment supplies {self.sv('diagnostics_summary','dataset.decision_rows')} instrument-date decisions from {self.sv('diagnostics_summary','dataset.folds')} rolling refits. In its consumed recent audit, the calibrated mixture has {v('diag_forecast_summary',h,'accuracy','.2f',100)}% accuracy and {v('diag_forecast_summary',h,'log_loss','.4f')} log loss, compared with {v('diag_forecast_summary',p,'accuracy','.2f',100)}% and {v('diag_forecast_summary',p,'log_loss','.4f')} for the trailing prior. Its paired log-loss disadvantage is {v('diag_paired_inference',loss,'effect','.4f')}, with a date-block 95% interval [{v('diag_paired_inference',loss,'ci_low','.4f')}, {v('diag_paired_inference',loss,'ci_high','.4f')}]. Fresh-capital audit accounts return {v('diag_portfolio',eh,'net_return','.2f',100)}% for the adaptive hybrid and {v('diag_portfolio',eb,'net_return','.2f',100)}% for the matched equal-weight control after costs. The follow-up executes {self.sv('experiments_summary','complete_trials')} registered training trials, with feature, calibration, adaptation and held-instrument comparisons. The product includes a sealed local collector, persistent ledgers and a session-aware schedule; engineering evidence is scoped separately from trading superiority. These experiments do not establish a defensible deployable edge. Historical analyses are exploratory, prospective sections remain interim, and external broker orders remain disabled.")
            if key=='forecasting':
                return f"The recent audit has {v('diag_forecast_summary',h,'n',',d')} labeled instrument-dates across {v('diag_forecast_summary',h,'sessions',',d')} sessions. Hybrid accuracy is {v('diag_forecast_summary',h,'accuracy','.2f',100)}%, while prior accuracy is {v('diag_forecast_summary',p,'accuracy','.2f',100)}%. Hybrid balanced accuracy is {v('diag_forecast_summary',h,'balanced_accuracy','.2f',100)}% and macro F1 is {v('diag_forecast_summary',h,'macro_f1','.3f')}. The prior's stronger aggregate accuracy does not imply equal recall across classes; the class report makes that tradeoff visible. Log loss also favors the prior in this sample, so there is no probability-quality improvement to hide behind a change in class balance."
            if key=='discussion':
                return f"The audit probability comparison has Holm-adjusted p={v('diag_paired_inference',loss,'holm_p','.3f')} in its registered diagnostic family. The interval crosses zero, and the point estimate favors the prior. This combination supports a bounded statement: the current hybrid has not established better probability forecasts on the consumed audit. It does not prove exact equivalence, nor does it justify selecting a favorable instrument after inspecting the heatmap. The full registry retains the controlled follow-ups so that any narrower improvement is interpreted within its own dates, information set and uncertainty."
            return f"The project now has a reproducible historical investigation and a separately versioned path for collecting prospective local evidence. On the recent audit, the calibrated mixture's {v('diag_forecast_summary',h,'log_loss','.4f')} log loss does not beat the prior's {v('diag_forecast_summary',p,'log_loss','.4f')}; the capital-matched hybrid account also trails its simple investable control. Engineering tests and a real scheduler trigger support the tested collection contract, while sustained unattended sessions and final outcome maturity remain unobserved. The historical manuscript reports the available negative and mixed findings; the complete prospective study remains interim. No profitable-trader or broker-readiness approval follows from this work."
        if key=='data':
            s='diagnostics_summary'
            return f"The verified market panel contains {self.sv(s,'dataset.raw_symbol_sessions')} instrument-session observations on {self.sv(s,'dataset.raw_dates')} common market dates. There are {self.sv(s,'dataset.decision_dates')} recorded decision dates, {self.sv(s,'dataset.mature_dates')} with mature outcomes, and {self.sv(s,'dataset.unmatured_rows')} instrument-date predictions whose outcomes are unavailable at the historical cutoff. The inherited release reports {self.sv(s,'dataset.folds')} refits. The inventory rechecked {self.sv(s,'evidence_verification.sealed_files_checked')} preserved files and directly compared {self.sv(s,'evidence_verification.reproduced_files_checked')} saved reproduction outputs. These are new integrity checks of saved records, not a claim that all old models were trained again."
        if key=='experiments_design':
            return f"The controlled study completed {self.sv('experiments_summary','complete_trials')} registered trials and {self.sv('experiments_summary','estimator_fits')} estimator fits, in addition to {self.sv('experiments_summary','pilot_estimator_fits')} development runtime-pilot fits. Its base evaluation spans {self.sv('experiments_summary','n_outer_dates')} outer dates and {self.sv('experiments_summary','unique_decision_samples')} unique base instrument-date decisions. The registered families contain {self.sv('experiments_summary','predictive_contrast_count')} predictive contrasts and {self.sv('experiments_summary','economic_contrast_count')} economic contrasts. Repeated model views and seed runs are retained, but never counted as additional independent market observations."
        if key=='error_structure':
            rows=self.table_rows('diag_confusion',period='audit',model='hybrid');row=max([r for r in rows if r['true_class']!=r['predicted_class']],key=lambda r:r['count'])
            cal=self.one('diag_calibration_contrast',period='audit')
            return f"The most frequent off-diagonal transition in the audit is {row['true_class']} to {row['predicted_class']}, with {v('diag_confusion',row,'count',',d')} errors, or {v('diag_confusion',row,'row_fraction','.2f',100)}% of that true class. Temperature scaling changes raw-mixture log loss by {v('diag_calibration_contrast',cal,'effect','.4f')} with interval [{v('diag_calibration_contrast',cal,'ci_low','.4f')}, {v('diag_calibration_contrast',cal,'ci_high','.4f')}]. The argmax labels are unchanged, as expected for positive temperature scaling. Better calibrated confidence can coexist with unchanged directional accuracy and still fail to beat the prior."
        if key=='uncertainty':
            row=self.one('diag_paired_inference',period='audit',model='hybrid',endpoint='log_loss',block_length=10)
            return f"For the primary audit hybrid-minus-prior comparison, the estimated loss difference is {v('diag_paired_inference',row,'effect','.4f')}, the bootstrap standard error is {v('diag_paired_inference',row,'bootstrap_se','.4f')}, and the approximate 80%-power detectable magnitude is {v('diag_paired_inference',row,'mde80_approx','.4f')}. With {v('diag_paired_inference',row,'sessions',',d')} dates and primary blocks of {v('diag_paired_inference',row,'block_length',',d')} dates, the effective information is much smaller than the number of stock rows suggests. This approximation ignores additional uncertainty about structural change and is not a prospective power guarantee."
        if key in ['ablations','calibration_gate','learning','transfer']:
            rows=self.table_rows('exp_contrasts',block=10)
            wanted={'ablations':'R5','calibration_gate':'R3','learning':'R6','transfer':'R7'}[key]
            rows=[r for r in rows if r['family']==wanted]
            sentences=[]
            for r in rows:
                direction='lower loss' if r['effect']<0 else 'higher loss'
                sentences.append(f"{r['id']}: candidate-minus-reference log loss {v('exp_contrasts',r,'effect','.4f')} ({direction}), 95% interval [{v('exp_contrasts',r,'ci_low','.4f')}, {v('exp_contrasts',r,'ci_high','.4f')}], Holm p={v('exp_contrasts',r,'p_holm','.3f')}.")
            if key=='calibration_gate':sentences.append('The forecast gate here chooses between calibrated predictions and the prior as a diagnostic view. The portfolio gate separately controls the trend/ML weight. Neither is silently substituted for the frozen trend shadow.')
            if key=='transfer':sentences.append('The held-target comparison reports the aligned target subset only. Training on other instruments increases the number of fitting rows but does not increase the number of independent outer dates.')
            return ' '.join(sentences)
        if key=='economic':
            h=self.one('diag_portfolio',period='audit',strategy='adaptive_hybrid',scope='matched_fresh_10000');b=self.one('diag_portfolio',period='audit',strategy='equal_weight',scope='matched_fresh_10000')
            return f"For the newly matched audit accounts, each policy begins with {v('diag_portfolio',h,'initial_equity',',.2f')} dollars. The adaptive hybrid ends at {v('diag_portfolio',h,'terminal_equity',',.2f')} dollars and the equal-weight control at {v('diag_portfolio',b,'terminal_equity',',.2f')} dollars. Hybrid net return is {v('diag_portfolio',h,'net_return','.2f',100)}%, maximum drawdown {v('diag_portfolio',h,'max_drawdown','.2f',100)}%, and average gross exposure {v('diag_portfolio',h,'mean_gross_exposure','.2f',100)}%. These fresh-capital results differ from the sealed continuous-account audit because startup holdings and entry financing differ. The continuous curves below retain their original account history and are labeled accordingly."
        if key=='attribution':
            h=self.one('diag_portfolio',period='full',strategy='adaptive_hybrid',scope='continuous_2024_start')
            ep=self.one('diag_round_trip_summary',period='full',strategy='adaptive_hybrid',scope='continuous_2024_start')
            return f"Over the full continuous hybrid account, modeled combined costs are {v('diag_portfolio',h,'combined_fees_usd',',.2f')} dollars across {v('diag_portfolio',h,'fills',',d')} fills. There are {v('diag_portfolio',h,'sell_fills',',d')} sell fills, which include partial reductions rather than necessarily completed flat-to-flat episodes. Their net realized profit is {v('diag_portfolio',h,'net_realized_sell_pnl_usd',',.2f')} dollars; open holdings and unclosed costs still matter for total equity. The {v('diag_round_trip_summary',ep,'closed_episodes',',d')} completed position episodes have a {v('diag_round_trip_summary',ep,'closed_episode_win_rate','.2f',100)}% win rate, materially different from the sell-fill statistic. Neither is a strategy's independent trade success probability. The completed-position episode table reconciles closed net profit and terminal open-position profit to total account profit, preventing partial sales from being counted as independent completed trades."
        if key=='controlled_economics':
            rows=self.table_rows('exp_economic_contrasts',block=10)
            low=min(rows,key=lambda r:r['minimum_detectable_effect_80pct']);high=max(rows,key=lambda r:r['minimum_detectable_effect_80pct'])
            return f"The controlled portfolio comparisons do not establish an economic advantage: every primary-block interval below crosses zero after matching the same outer dates and execution rules. Each fold starts a fresh account; the study averages daily active returns across those blocks, not a fictitious continuously invested equity path across the intervening gaps. Approximate detectable daily active effects range from {v('exp_economic_contrasts',low,'minimum_detectable_effect_80pct','.4f',100)}% to {v('exp_economic_contrasts',high,'minimum_detectable_effect_80pct','.4f',100)}%. These are large relative to the registered practical threshold. The bootstrap conditions on the observed outer blocks and cannot establish robustness to an unseen market regime."
        if key=='seeds':
            rows=self.s['tables']['exp_seed_variance'];r=max(rows,key=lambda x:x['maximum_probability_difference'])
            return f"Across the registered seed comparisons, the largest probability difference is {v('exp_seed_variance',r,'maximum_probability_difference','.1e')}. The tested deterministic estimator settings therefore produced identical probabilities across seeds. Zero seed variation is not zero research uncertainty: the calendar path, data vintage, history span and target instrument remain sources of variation. Shorter recent-history fractions change both row count and temporal span, so the learning curves do not identify a pure sample-size effect. Separating the gate also changes its recency as well as its information overlap."
        if key=='sensitivity':
            rows=[r for r in self.table_rows('diag_stress',strategy='adaptive_hybrid') if r['status']=='completed']
            low=min(rows,key=lambda r:r['net_return']);high=max(rows,key=lambda r:r['net_return'])
            return f"Across the registered completed hybrid sensitivity scenarios, the lowest terminal return is {v('diag_stress',low,'net_return','.2f',100)}% in {low['scenario']}; the highest is {v('diag_stress',high,'net_return','.2f',100)}% in {high['scenario']}. These extrema are descriptive stress outcomes, not candidates selected for deployment. A rejected-order scenario can improve a realized sample by avoiding losing exposure, and a temporary gap can worsen drawdown without changing the terminal price. Both effects require examining the full path, fills and rejections, not just ending wealth."
        if key=='runtime':
            name='operations_latency-r2'
            return (f"The actual cold provider-to-ledger commissioning took {self.sv(name,'cold_commissioning.whole_batch','.2f')} seconds, including {self.sv(name,'cold_commissioning.data_read','.2f')} seconds of public-data reads. It made {self.sv(name,'provider_requests',',d')} requests with {self.sv(name,'provider_failures',',d')} observed provider failures. Across {self.sv(name,'warm_summary.whole_batch.n',',d')} warm batches, median batch time was {self.sv(name,'warm_summary.whole_batch.p50','.3f')} seconds and empirical P95 was {self.sv(name,'warm_summary.whole_batch.p95','.3f')} seconds. A separate {self.sv(name,'training_n',',d')}-fit SPY timing sample had median {self.sv(name,'training_median','.3f')} seconds. Cold-run and small training samples do not support tail-latency estimates; warm P95 is descriptive. The collector ran on the recorded arm64 macOS host under concurrent research load.")
        if key=='operations':
            e=self.s['summaries'].get('operations_evidence',{})
            return e.get('manuscript_text','Operational evidence is reported in the verified schedule and test tables. The actual trigger is a commissioning observation, not a prospective session. Future session-close operation and sustained host availability remain time-dependent.')
        if key=='prospective':
            if 'prospective' not in self.s['summaries']:raise ValueError('Run the validated prospective report before compiling the paper')
            r=self.s['summaries']['prospective']
            text=f"The latest validated prospective analysis contains {self.sv('prospective','status_snapshot.valid_observed_sessions')} decision receipts and {self.sv('prospective','status_snapshot.label_complete_decisions')} mature decision dates. It is recorded as {r['status']}. Commissioning, controlled replay and historical predictions are excluded from these counts. "
            if not r['predictive']['metrics']:return text+'No prospective forecast or economic result is available; unobserved quantities are not zero returns or successful sessions.'
            text+='Forecast metrics below use only joined, validated mature labels. Interim reports suppress inferential significance tests; final inference is emitted only after complete registered observations and terminal accounting. '
            if r['economic'].get('net_returns'):
                text+=f"Observed net returns are {self.sv('prospective','economic.net_returns.shadow','.2%')} for the frozen trend shadow and {self.sv('prospective','economic.net_returns.equal_weight','.2%')} for the costed equal-weight replay. Collection completion is separate from economic qualification."
            return text
        if key=='reproducibility':
            tests=self.s['tables']['verification_reports']
            text='The recorded test jobs are listed individually below; counts from separate or overlapping runs are not added into a fictional distinct-test total. '
            repro=self.s['summaries'].get('reproduction',{})
            if repro:text+='Clean-environment reproduction evidence is retained in the source manifest and appendix. '
            if 'experiments_full_reproduction' in self.s['summaries']:
                text+=f"The complete clean replay retrained {self.sv('experiments_full_reproduction','actual_trials')} trials ({self.sv('experiments_full_reproduction','actual_estimator_fits')} estimator fits), compared {self.sv('experiments_full_reproduction','probability_view_rows')} forecast-view rows with maximum probability error {self.sv('experiments_full_reproduction','maximum_probability_error','.1e')}, and reproduced {self.sv('experiments_full_reproduction','actual_portfolio_histories')} portfolio histories covering {self.sv('experiments_full_reproduction','actual_portfolio_sessions')} account sessions. Ledger states, targets, event lists, economic metrics and block-bootstrap/Holm outputs agreed. The comparison tolerances were rtol=1e-10 and atol=1e-12; actual probability error was zero. "
            return text+'The sealed historical artifact comparison is distinct from the new clean-environment refits. Provider checks and UI commissioning are separate integration jobs with actual saved results.'
        if key=='continuation':
            return 'R12 cannot be completed today: prospective decisions, operational coverage and execution-aligned outcome endpoints must occur in real time. The frozen first decision, final decision and final maturity events are listed in the schedule table. Missing sessions create deviations and a visible halt. Rebuilding this manuscript after observation requires refreshing the evidence snapshot, compiling statistics and rerunning the consistency checks.'
        raise KeyError(key)

    def render_table(self,key):
        t=self.table
        if key=='data':return t('Dataset flow; counts are instrument-dates unless the stage explicitly says dates',self.s['tables']['diag_dataset_flow'],[('stage','Stage',''),('n','Count',',d')],'data/research-v2/tables/diag_dataset_flow.csv')
        if key=='features':
            rows=[dict(family='Returns',definition='Close / lagged close - 1',availability='Trailing 1, 5, 21, 63, 126, 252 sessions'),dict(family='Volatility',definition='Sample standard deviation of daily return',availability='Trailing 21 / 63 sessions'),dict(family='Relative return',definition='Instrument return minus SPY return',availability='Trailing 21 / 63 sessions'),dict(family='Trend',definition='Close / moving mean - 1',availability='Trailing 20 / 50 / 200 sessions'),dict(family='Gap / range',definition='Open / previous close - 1; (High-Low)/Close',availability='Completed current session'),dict(family='Volume',definition='Volume / trailing mean - 1',availability='Trailing 20 sessions'),dict(family='Momentum',definition='Close lag21 / Close lag252 - 1',availability='Previous completed observations')]
            return t('Feature definitions and information availability',rows,[('family','Family',''),('definition','Definition',''),('availability','Availability','')],'src/models/adaptive.py::features; research_v2/experiments.py::FAMILIES')
        if key=='models':return t('Fixed model configurations and distinct chronological contracts',self.s['tables']['model_configurations'],[('contract','Contract',''),('component','Component',''),('configuration','Configuration','')],'data/research-v2/tables/model_configurations.csv; evidence/configuration-sources.json')
        if key=='experiment_design':
            d=self.s['summaries']['experiments_design'];groups={}
            for tr in d['trials']:
                groups.setdefault(tr['variant'],[]).append(tr)
            rows=[dict(variant=k,trials=len(vals),symbols=len({r['symbol'] for r in vals}),folds=len({r['fold'] for r in vals}),seeds=','.join(str(x) for x in sorted({r['cfg']['seed'] for r in vals}))) for k,vals in groups.items()]
            return t('Frozen controlled training design',rows,[('variant','Variant',''),('trials','Trials',',d'),('symbols','Targets','d'),('folds','Outer folds','d'),('seeds','Seeds','')],'data/research-v2/experiments/design.json')
        if key=='forecast_summary':return t('Consumed recent-audit forecasts; identical mature observations',self.table_rows('diag_forecast_summary',period='audit'),[('model','Model',''),('n','Rows',',d'),('accuracy','Accuracy','pct'),('balanced_accuracy','Balanced','pct'),('macro_f1','Macro F1','.3f'),('log_loss','Log loss','.4f'),('brier','Brier','.4f')],'data/research-v2/tables/diag_forecast_summary.csv')
        if key=='class_report':return t('Audit class report for calibrated mixture and prior', [r for r in self.table_rows('diag_class_report',period='audit') if r['model'] in ['hybrid','prior']], [('model','Model',''),('class_name','Class',''),('precision','Precision','.3f'),('recall','Recall','.3f'),('f1','F1','.3f'),('support','Support',',d')],'data/research-v2/tables/diag_class_report.csv')
        if key=='calibration':return t('Audit mixture confidence coverage; thresholds fixed for description',self.table_rows('diag_coverage',period='audit',model='hybrid'),[('threshold','Threshold','.2f'),('n','Retained',',d'),('sessions','Dates',',d'),('coverage','Coverage','pct'),('accuracy','Accuracy','pct')],'data/research-v2/tables/diag_coverage.csv')
        if key in ['ablations','learning','transfer']:
            family={'ablations':'R5','learning':'R6','transfer':'R7'}[key]
            return t('Controlled '+{'R5':'feature ablations','R6':'history, fraction and cadence contrasts','R7':'held-instrument transfer contrast'}[family],self.table_rows('exp_contrasts',block=10,family=family),[('id','Contrast',''),('n_rows','Rows',',d'),('effect','Loss delta','.4f'),('ci_low','CI low','.4f'),('ci_high','CI high','.4f'),('p_holm','Holm p','.3f')],'data/research-v2/tables/exp_contrasts.csv')
        if key=='components':return t('Controlled base forecast views on development outer blocks',self.table_rows('exp_predictive',variant='base'),[('model','View',''),('n_rows','Rows',',d'),('accuracy','Accuracy','pct'),('log_loss','Log loss','.4f'),('brier','Brier','.4f'),('gate_separate_rate','Separate gate','pct')],'data/research-v2/tables/exp_predictive.csv')
        if key=='controlled_economics':return t('Controlled daily active returns versus equal weight; primary date-block inference',self.table_rows('exp_economic_contrasts',block=10),[('a','Policy',''),('n_dates','Dates','d'),('effect','Daily effect','pct'),('ci_low','CI low','pct'),('ci_high','CI high','pct'),('p_holm','Holm p','.3f')],'data/research-v2/tables/exp_economic_contrasts.csv')
        if key=='portfolio':return t('Fresh-capital audit accounts; zero-rate Sharpe and modeled per-side costs',self.table_rows('diag_portfolio',period='audit',scope='matched_fresh_10000'),[('strategy','Policy',''),('net_return','Net return','pct'),('sharpe_zero_rf','Sharpe','.3f'),('max_drawdown','Max DD','pct'),('mean_gross_exposure','Exposure','pct'),('combined_fees_usd','Costs ($)','.2f'),('fills','Fills',',d')],'data/research-v2/tables/diag_portfolio.csv')
        if key=='failures':return t('Largest audit hybrid forecast losses, selected by registered loss ranking',self.table_rows('diag_failures',period='audit',selection_rule='worst_log_loss')[:5],[('symbol','Symbol',''),('date','Decision',''),('true_class','Observed',''),('predicted_class','Predicted',''),('confidence','Confidence','.3f'),('log_loss','Loss','.3f')],'data/research-v2/tables/diag_failures.csv')
        if key=='stress':return t('Hybrid execution and risk sensitivities; fresh-capital audit',self.table_rows('diag_stress',strategy='adaptive_hybrid'),[('scenario','Scenario',''),('status','Status',''),('net_return','Net return','pct'),('max_drawdown','Max DD','pct'),('fills','Fills',',.0f'),('rejections','Rejected',',.0f')],'data/research-v2/tables/diag_stress.csv')
        if key=='latency':
            latency=self.s['summaries']['operations_latency-r2']
            rows=[dict(stage=k,**r) for k,r in latency['warm_summary'].items()]
            return t('Warm pipeline stage timings, seconds; P99 intentionally unreported',rows,[('stage','Stage',''),('n','Samples','d'),('p50','P50 (s)','.3f'),('p95','P95 (s)','.3f'),('p99','P99 (s)','.3f')],'data/research-v2/operations/latency-r2.json')
        if key=='prospective':
            report=self.s['summaries']['prospective'];rows=report['predictive']['metrics']
            if not rows:return ''
            return t('Prospective forecasts on mature labels only',rows,[('model','Model',''),('n','Labeled rows','d'),('accuracy','Accuracy','pct'),('balanced_accuracy','Balanced','pct'),('log_loss','Log loss','.4f'),('brier','Brier','.4f')],self.s['summaries']['prospective_source'])
        if key=='episodes':
            return t('Hybrid closed-position episodes versus still-open positions', [r for r in self.s['tables']['diag_round_trip_summary'] if r['strategy']=='adaptive_hybrid' and ((r['scope']=='continuous_2024_start' and r['period']=='full') or (r['scope']=='matched_fresh_10000' and r['period']=='audit'))], [('period','Period',''),('closed_episodes','Closed','d'),('open_episodes','Open','d'),('closed_episode_win_rate','Closed wins','pct'),('closed_episode_net_pnl_usd','Closed P&L ($)','.2f'),('terminal_open_episode_pnl_usd','Open P&L ($)','.2f'),('total_pnl_usd','Total P&L ($)','.2f')],'data/research-v2/tables/diag_round_trip_summary.csv')
        if key=='operations':return t('Actual verification jobs; overlapping suites are separate records',self.s['tables']['verification_reports'],[('report','Saved report',''),('cases','Cases','d'),('failures','Failed','d'),('errors','Errors','d'),('skipped','Skipped','d')],'data/research-v2/tests/*.xml; XML testcase counts')
        if key=='schedule':
            rows=self.s['tables'].get('schedule_milestones',[])
            if not rows:raise ValueError('Missing schedule_milestones table')
            return t('Registered XNYS events and Hong Kong display times',rows,[('event','Event',''),('session','US session',''),('hong_kong','Hong Kong trigger','')],'data/research-v2/tables/schedule_milestones.csv')
        if key=='instruments':return t('Complete audit instrument diagnostic for calibrated mixture',self.table_rows('diag_instrument',period='audit',model='hybrid'),[('symbol','Symbol',''),('n','Rows','d'),('accuracy','Accuracy','pct'),('balanced_accuracy','Balanced','pct'),('log_loss','Log loss','.4f'),('brier','Brier','.4f')],'data/research-v2/tables/diag_instrument.csv')
        if key=='evidence':
            rows=self.s['tables'].get('evidence_matrix',[])
            if not rows:raise ValueError('Missing evidence matrix')
            return t('Question-to-evidence completion map',rows,[('id','ID',''),('status','Status',''),('evidence','Evidence',''),('remaining','Remaining limitation','')],'data/research-v2/tables/evidence_matrix.csv')
        if key=='registry':return t('Research-family registry status',self.s['tables']['registry'],[('id','ID',''),('question','Question',''),('status','Recorded status','')],'data/research-v2/registry/R*.json')
        raise KeyError(key)

    def render_figure(self,key):
        spec={'dataset':('diag_dataset','Coverage, retained observations and class support from the saved panel.'),'architecture':('system_architecture','Architecture of the isolated collection and research paths.'),'timeline':('information_timeline','Information, fill and label availability under the execution-aligned target.'),'confusion':('diag_confusion','Audit confusion matrices with class support; row normalization does not create independent observations.'),'calibration':('diag_calibration','Probability reliability and confidence coverage, including bin support and uncertainty.'),'forecast_inference':('diag_forecast_inference','Paired audit forecast effects and date-block uncertainty. Lower log loss is better.'),'ablations':('exp_contrasts','Registered controlled contrasts on development outer blocks; intervals preserve date dependence.'),'learning':('exp_learning_v2','Fitting versus outer validation log loss under training-design changes.'),'generalization':('diag_generalization','Historical instrument/period heterogeneity. Consumed-period slices are descriptive.'),'portfolio':('diag_portfolio','Continuous accounts from their original common start; these are not fresh audit allocations.'),'costs':('diag_costs','Recorded exposure, turnover and cost diagnostics under the shared ledger contract.'),'stress':('diag_stress','Registered execution sensitivities; terminal return and path risk answer different questions.'),'operations':('exp_runtime','Measured fitting latency by registered variant under differing run order and concurrent host load; these are operational timings, not controlled speedup estimates.')}
        return self.fig(*spec[key])

    def resolve_citations(self,text):
        numbers={r['id']:i+1 for i,r in enumerate(self.s['references'])}
        def cite(m):
            keys=[x.strip() for x in m[1].split(',')]
            if not all(k in numbers for k in keys):raise ValueError('Unresolved bibliography reference: '+m[1])
            return '['+', '.join(str(numbers[k]) for k in keys)+']'
        return re.sub(r'\[([a-z][a-z0-9, ]+)\](?!\()',cite,text)

    def build(self):
        text=(DOC/'manuscript.template.md').read_text()
        text=re.sub(r'^@RESULT (\w+)$',lambda m:self.result(m[1]),text,flags=re.M)
        text=re.sub(r'^@TABLE (\w+)$',lambda m:self.render_table(m[1]),text,flags=re.M)
        text=re.sub(r'^@FIG (\w+)$',lambda m:self.render_figure(m[1]),text,flags=re.M)
        refs=self.s['references']
        text=self.resolve_citations(text)
        text=text.replace('@REFERENCES','\n\n'.join(f"[{i+1}] {r['authors']} ({r['year']}). {r['title']}. {r['venue']}. [{r['id']}]({r['url']})." for i,r in enumerate(refs)))
        commands='''```bash
# From /Users/a/Downloads/ml-trade-engine
# Verify / inspect the frozen collector (read-only)
data/research-v2/clean-env/bin/python data/releases/research-v2-runtime-20260928-r2/scripts/collector_v2.py status --release data/releases/research-v2-runtime-20260928-r2 --workspace data/paper/research-v2
# Launch the product console
/Users/a/miniconda3/bin/python -m streamlit run app_v2.py --server.port 8504
# Recompute saved-data diagnostics under their recorded protocol
/Users/a/miniconda3/bin/python -B -m research_v2.diagnostics
# Resume registered controlled trials; completed checkpoints are verified
/Users/a/miniconda3/bin/python -B -m research_v2.experiments run
# Interim observed evidence (new output directory; never starts collection)
data/research-v2/clean-env/bin/python -B -m research_v2.prospective_report --output data/research-v2/prospective/NEW-DATED-RUN
# After all63 decisions and final six-session maturity only, add --freeze
# Compile checked statistics / editable manuscript
/Users/a/miniconda3/bin/python -B -m research_v2.paper compile
# Render PDF using bundled document runtime
/Users/a/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -m research_v2.paper pdf
```'''
        text=text.replace('@COMMANDS',commands)
        if re.search(r'^@(?:RESULT|TABLE|FIG|REFERENCES|COMMANDS)',text,re.M):raise ValueError('Unresolved manuscript placeholder')
        (DOC/'MANUSCRIPT.md').write_text(text)
        result=dict(statistics_sha256=digest(DATA/'statistics.json'),template_sha256=digest(DOC/'manuscript.template.md'),manuscript_sha256=digest(DOC/'MANUSCRIPT.md'),claims=self.claims,figures=self.figure_count,tables=self.table_count,word_count=len(text.split()),source_hashes=self.s['sources'])
        (DATA/'claim-map.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False))
        print(json.dumps({k:result[k] for k in ['figures','tables','word_count']}))
        return text


def render_pdf():
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from PIL import Image as PILImage
    stats=read(DATA/'statistics.json');verify_statistics(stats)
    claim=read(DATA/'claim-map.json')
    if digest(DOC/'MANUSCRIPT.md')!=claim['manuscript_sha256'] or digest(DATA/'statistics.json')!=claim['statistics_sha256']:raise ValueError('Manuscript or statistics changed after claim mapping')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('BodyV2',fontName='Helvetica',fontSize=10.3,leading=14.3,spaceAfter=7,textColor=colors.HexColor('#1d2d3a')))
    styles.add(ParagraphStyle('H1V2',fontName='Helvetica-Bold',fontSize=22,leading=27,spaceAfter=13,textColor=colors.HexColor('#193e58'),keepWithNext=True))
    styles.add(ParagraphStyle('H2V2',fontName='Helvetica-Bold',fontSize=14,leading=18,spaceBefore=15,spaceAfter=9,textColor=colors.HexColor('#193e58'),keepWithNext=True))
    styles.add(ParagraphStyle('H3V2',fontName='Helvetica-Bold',fontSize=11.5,leading=15,spaceBefore=10,spaceAfter=7,textColor=colors.HexColor('#2b6875'),keepWithNext=True))
    styles.add(ParagraphStyle('CellV2',fontName='Helvetica',fontSize=8,leading=10.5,splitLongWords=True))
    styles.add(ParagraphStyle('SmallV2',fontName='Helvetica',fontSize=8,leading=10.5,spaceAfter=8,textColor=colors.HexColor('#536674')))
    styles.add(ParagraphStyle('TableCaptionV2',parent=styles['SmallV2'],keepWithNext=True))
    styles.add(ParagraphStyle('CodeV2',fontName='Courier',fontSize=7.3,leading=10,spaceAfter=6,splitLongWords=True))
    def fmt(txt):
        txt=html.escape(txt)
        txt=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:f'<link href="{m[2]}" color="#1f6c87">{m[1]}</link>',txt)
        txt=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',txt)
        txt=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',txt)
        return txt
    lines=(DOC/'MANUSCRIPT.md').read_text().splitlines();story=[];i=0;w=A4[0]-92
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('```'):
            chunk=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):chunk.append(lines[i]);i+=1
            for val in chunk:story.append(Paragraph(html.escape(val),styles['CodeV2']))
            i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in cells):rows.append(cells)
                i+=1
            n=len(rows[0]);length=[max(len(r[j]) if j<len(r) else 0 for r in rows) for j in range(n)]
            weights=[min(2.6,max(.7,l/18)) for l in length];widths=[w*x/sum(weights) for x in weights]
            cells=[[Paragraph(fmt(c),styles['CellV2']) for c in r] for r in rows]
            table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9f0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#6b879c'))]))
            story.extend([table,Spacer(1,7)]);continue
        m=re.match(r'!\[([^\]]+)\]\(([^)]+)\)',line)
        if m:
            path=Path(m[2]);im=PILImage.open(path);iw,ih=im.size;scale=min(w/iw,265/ih)
            image=Image(str(path),width=iw*scale,height=ih*scale,hAlign='CENTER')
            j=i+1
            while j<len(lines) and not lines[j].strip():j+=1
            if j<len(lines) and lines[j].startswith('Figure '):
                story.append(KeepTogether([image,Spacer(1,5),Paragraph(fmt(lines[j]),styles['SmallV2'])]));i=j+1
            else:story.append(image);i+=1
            continue
        if line.startswith('# '):style='H1V2';line=line[2:]
        elif line.startswith('## '):style='H2V2';line=line[3:]
        elif line.startswith('### '):style='H3V2';line=line[4:]
        elif line.startswith('Source:'):style='SmallV2'
        elif line.startswith('**Table '):
            style='TableCaptionV2'
        else:style='BodyV2'
        story.append(Paragraph(fmt(line),styles[style]));i+=1
    dest=ROOT/'output/pdf/ML_TradeEngine_Research_v2.pdf';dest.parent.mkdir(parents=True,exist_ok=True)
    def footer(c,doc):
        c.saveState();c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#536674'))
        c.drawString(46,A4[1]-28,'ML-TradeEngine | Research v2 | Historical study / prospective interim')
        c.drawString(46,25,'Evidence cutoff: 25 September 2026  |  Statistics '+claim['statistics_sha256'][:12])
        c.drawRightString(A4[0]-46,25,str(doc.page));c.restoreState()
    doc=SimpleDocTemplate(str(dest),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=48,bottomMargin=44,title='Causal Forecasts, Constrained Portfolios, and Prospective Operation',author='ML-TradeEngine research project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(dest)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['compile','pdf','check']);a=p.parse_args()
    if a.command=='compile':stats=compile_statistics();verify_statistics(stats);Manuscript(stats).build()
    elif a.command=='pdf':render_pdf()
    else:verify_statistics(read(DATA/'statistics.json'));print('All source hashes match compiled statistics')
if __name__=='__main__':main()
