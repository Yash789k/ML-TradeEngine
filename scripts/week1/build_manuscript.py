"""Apply the registered schedule amendment to the preserved native LaTeX paper."""
from pathlib import Path
import hashlib,json,re,zipfile
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/overleaf/week1'
OUT.mkdir(exist_ok=False)
with zipfile.ZipFile(ROOT/'data/research-week1/overleaf-before-amendment.zip') as z:
    for name in z.namelist():
        assert '/' not in name
        (OUT/name).write_bytes(z.read(name))
s=(OUT/'main.tex').read_text()
def replace(old,new):
    global s
    assert s.count(old)==1,(old[:100],s.count(old))
    s=s.replace(old,new)
replace('The active r2 runtime passed 27 source-bound tests.','The predecessor r2 runtime passed 27 source-bound tests.')
needle='The actual cold provider-to-ledger commissioning took'
pos=s.index(needle)
protocol=json.loads((ROOT/'data/research-week1/protocol.json').read_text())
verification=json.loads((ROOT/'data/research-week1/release-verification.json').read_text())
assert verification['passing_cases']==29 and protocol['pilot']['sessions']==5
note=r'''Before the first prospective observation, the study owner requested a shorter observation period to prepare an early manuscript release. The amendment registered on 28 September 2026 supersedes the 63-session commitment with five fixed XNYS decision sessions, from 29 September through 5 October. The old protocol and sealed artifacts are preserved. The strategy, models, capital, costs, forecast horizon and risk rules are unchanged. The active trial is \path{PILOT-TREND-WEEK1-20260929}. Its new release passed 29 source-bound cases, including the exact amended calendar and an isolated historical replay through five decisions and six terminal receipts. These test observations do not count as prospective market evidence.

The fifth-session operational review is due on 6 October at 05:30 Hong Kong time. An interim paper can report that week's observed account activity and failures while forecast outcomes remain pending. No scheduled model refit occurs within this shortened window. Five dates cannot support the original dependence-aware inferential plan: the prospective report is descriptive, with no p-values, claims of durable profitability or approval of sustained operational reliability. The registered amendment and complete calendar are retained in the versioned week-one evidence package.

'''
s=s[:pos]+note+s[pos:]
start=s.index('Observed-session milestone 1 &');end=s.index('\\bottomrule',start)
s=s[:start]+r'''First decision & 2026-09-29 & 2026-09-30 04:30 HKT \\
Fifth and final decision & 2026-10-05 & 2026-10-06 04:30 HKT \\
Five-session watchdog review & 2026-10-05 & 2026-10-06 05:30 HKT \\
Final target-label endpoint & 2026-10-13 & 2026-10-14 04:30 HKT \\
Final outcome watchdog & 2026-10-13 & 2026-10-14 05:30 HKT \\
'''+s[end:]
replace(r'\path{data/research-v2/tables/schedule_milestones.csv}',r'\path{data/research-week1/schedule.json}')
replace('releases inferential results only at the registered final freeze; interim views are descriptive.','retains descriptive results at both week close and final maturity; no inferential test is released for five decision dates.')
replace('The operational observation period is not a promise of adequate power for a small economic effect.','Six outcome sessions follow the fifth decision, with final maturity on 14 October 2026 at 04:30 Hong Kong time. This follow-up does not extend the five-session decision window or prevent an interim manuscript release. The shortened observation period cannot establish adequate power for a small economic effect.')
replace('Registered63 sessions; active launchd; commissioning receipt','Registered five-session amendment; active launchd; commissioning receipt')
replace('Independent sessions and final labels have not yet occurred','Five decision dates and final labels remain unobserved')
s=s.replace('research-v2-runtime-20260928-r2/scripts/collector_v2.py status --release data/releases/research-v2-runtime-20260928-r2 --workspace data/paper/research-v2','research-v2-runtime-20260928-week1/scripts/collector_v2.py status --release data/releases/research-v2-runtime-20260928-week1 --workspace data/paper/research-week1')
s=s.replace('streamlit run app_v2.py','streamlit run app_week1.py')
replace('research_v2.prospective_report --output data/research-v2/prospective/NEW-DATED-RUN','research_week1.report --output data/research-week1/reports/NEW-DATED-RUN')
replace('# After all63 decisions and final six-session maturity only, add --freeze', '# After five valid decisions, use --week-close for the operational report\n# Only after five mature outcomes and six terminal receipts, add --freeze')
start=s.index('# Compile checked statistics / editable manuscript');end=s.index('\\end{Verbatim}',start)
s=s[:start]+'''# Current native LaTeX manuscript source: output/overleaf/week1/main.tex
# Historical builders reproduce the preserved pre-amendment edition.
# Keep future paper updates versioned and validate every empirical claim.
'''+s[end:]
replace('Rebuilding this manuscript after observation requires refreshing the evidence snapshot, compiling statistics and rerunning the consistency checks.','Rebuilding this manuscript after observation requires a new week-one evidence snapshot, a versioned LaTeX update and repeated consistency checks. The original R12 registry remains preserved; the five-session amendment is authoritative for the active pilot.')
replace('The manuscript builder reads a single compiled statistics source and checks its inputs\' hashes;',"The historical manuscript builder reads a single compiled statistics source and checks its inputs' hashes;")
replace('A missing figure source prevents a complete build.','A missing figure source prevents a complete build. The later five-session schedule amendment is generated from its separately versioned protocol and calendar; its provenance is retained alongside the amended native LaTeX source.')
s=s.replace(r'\textit{Source:} \path{data/research-v2/tables/evidence_matrix.csv}.',r'\textit{Source:} \path{data/research-v2/tables/evidence_matrix.csv}.\par\textit{R12 amendment:} \path{data/research-week1/protocol.json}.')
(OUT/'main.tex').write_text(s)
(OUT/'README.md').write_text('''# ML-TradeEngine: five-session protocol amendment

This is the current editable LaTeX paper, amended before prospective observation.
Historical results and native figures are unchanged. The active five-session
schedule and descriptive-only evidence scope supersede the old 63-session pilot.
Compile main.tex with pdfLaTeX and BibTeX. Contents and separate references remain.
See docs/research-week1/STATUS.md for collection, week-close and final reports.
''')
sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/research-week1/protocol.json',ROOT/'data/research-week1/schedule.json',ROOT/'data/research-week1/release-verification.json',ROOT/'data/research-week1/reports/20260928-baseline/report.json',ROOT/'data/research-week1/overleaf-before-amendment.zip',Path(__file__)]}
(ROOT/'data/research-week1/manuscript-amendment.json').write_text(json.dumps({'scope':'Prospective schedule and scope only; historical empirical results unchanged','sources':sources,'main_sha256':hashlib.sha256((OUT/'main.tex').read_bytes()).hexdigest()},indent=2))
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SOURCE_MANIFEST.json'}
(OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(OUT.parent/'ML_TradeEngine_Week1_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):z.write(p,p.name)
print('Wrote versioned amended native LaTeX source.')
