"""Prepare a review-only LaTeX insertion from the validated interim artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'data/research-week1-r6/reports/week-close-20261023T0538HKT'
DEST = ROOT/'output/overleaf/week1-r6-update-20261023'


def read(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    r = read(REPORT/'report.json')
    for path, sha in read(REPORT/'manifest.json')['files'].items():
        assert digest(REPORT/path) == sha
    state = r['status_snapshot']
    assert r['operational_complete'] and r['total_valid_observed_decisions'] == 5
    assert state['valid_observed_sessions'] == 1 and state['ledger_consistent'] and not state['halted']
    assert state['label_complete_decisions'] == 0 and state['terminal_observed_sessions'] == 0
    rows = read(REPORT/'prediction_rows.json')
    dates = sorted({x['session'] for x in rows})
    assert len(rows) == 66 and dates == ['2026-10-01','2026-10-02','2026-10-14']
    assert r['predictive']['paired_inference'] == []
    receipt_path = ROOT/'data/paper/research-week1-r6/days/2026-10-22/result.json'
    receipt = read(receipt_path)
    assert receipt['binding']['release_sha256'] == r['release_sha256']
    original_path = ROOT/'data/research-week1-r2/failed-trial-preservation.json'
    original = read(original_path)['status']
    assert original['valid_observed_sessions'] == 0
    assert original['missing_sessions'] == ['2026-09-29','2026-09-30']
    prior = {s['segment']:s for s in r['prior_segments']}
    assert sum(s['mature_decisions'] for s in prior.values()) == 3
    metrics = {m['model']:m for m in r['predictive']['metrics']}
    h, p = metrics['hybrid'], metrics['prior']
    percent = lambda x: f'{100*x:.4f}'+r'\%'
    completed = datetime.fromisoformat(receipt['completed_at'])
    assert completed.isoformat() == '2026-10-22T20:31:09.651399+00:00'
    text = r'''% Review-only insertion. Generated from a validated report; see evidence.json.
% Requires the existing manuscript's booktabs, array and xurl packages.
\subsection{Interrupted prospective observations: interim update}
\label{sec:prospective-r6-interim}

At the post-close review for the US 22 October 2026 session, the study contained
five valid prospective decision dates: 1, 2, 14, 15 and 22 October. These dates
belong to three separate simulated-account segments (r2, r4 and r6). They are
not five consecutive successful sessions. Commissioning previews, controlled
replays, diagnostic provider reads and follow-up outcome marks contribute no
additional decision dates. Each new segment was registered before its decision
window with fresh capital; no missed decision was reconstructed retrospectively.

\begin{table}[htbp]
\centering\small
\caption{Prospective coverage at the interim cutoff. Missed dates refer to
registered decisions before each supersession; cancelled future dates are excluded.}
\label{tab:interrupted-coverage-r6}
\begin{tabular}{@{}lp{29mm}rp{43mm}@{}}
\toprule
Trial & Valid US dates & Count & Missing or rejected dates \\
\midrule
Original week 1 & None & 0 & 29--30 September \\
r2 & 1--2 October & 2 & 5--7 October \\
r3 & None & 0 & 12--13 October \\
r4 & 14--15 October & 2 & 16 October \\
r5 & None & 0 & 21 October: invalid JPM OHLC \\
r6 & 22 October & 1 & None at this cutoff \\
\bottomrule
\end{tabular}
\end{table}

The r6 scheduler started its scheduled collection at 04:30:48 HKT on
23 October and completed at 04:31:09. All 22 instruments passed validation on
the first attempt. The watchdog confirmed the reconciled receipt at 05:30:48;
the two retry events correctly recognized the existing receipt and performed
no duplicate decision. Both fresh accounts had equity and cash of
\$10,000.00, zero exposure and zero reconciliation residual. The shadow account
reserved \$@RESERVED@ for pending simulated execution. Its first decision is
therefore not evidence of realized investment performance. Deployment remains
cash and external broker orders remain disabled.

The preceding r5 failure remains part of the operational evidence. Yahoo's
saved JPM opening price exceeded its daily high. The validator rejected the
batch, but the prior error classification prevented bounded provider retries.
A later diagnostic received a corrected record; its publication time within
the original retry window is unknown. R6 retains strict validation and permits
only the existing retries for a range error confined to the newly completed
provider bar. Historical corruption, nonfinite inputs and saved-snapshot
integrity errors remain failures. This successful r6 attempt did not exercise
that retry path in production; its recovery evidence remains controlled fault
injection. No frozen models, portfolio policy or forecast horizon were retuned.

Three decision dates (1, 2 and 14 October) have mature labels, comprising
66 instrument-date rows. These outcomes were derived from the validated
22 October snapshot in the report only; archived account states were not
rewritten. The 15 and 22 October decisions remain outcome-pending.

\begin{table}[htbp]
\centering\small
\caption{Descriptive interim forecast scores on 66 labeled rows from three
dates. Lower log loss and Brier score are better. No inferential test is reported.}
\label{tab:r6-interim-prediction}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Predictor & Accuracy & Macro F1 & Log loss & Brier score \\
\midrule
Hybrid & @HACC@ & @HF1@ & @HLOSS@ & @HBRIER@ \\
Training prior & @PACC@ & @PF1@ & @PLOSS@ & @PBRIER@ \\
\bottomrule
\end{tabular}
\end{table}

The hybrid's lower interim probability losses coexist with lower classification
accuracy than the prior. Three dates and correlated instruments cannot establish
statistical power, stable predictive superiority or a trading edge. The partial
label set must not be presented as an evaluation of all five decisions.

\begin{table}[htbp]
\centering\small
\caption{Separate segment accounts at their last actual receipt. Returns are
relative to each segment's own \$10,000 capital; they are never compounded
across restarts or across unobserved gaps.}
\label{tab:r6-interim-accounts}
\begin{tabular}{@{}llrr@{}}
\toprule
Segment & Last receipt & Trend shadow & Cash deployment \\
\midrule
r2 & 2 October & @R2RETURN@ & @R2CASH@ \\
r4 & 15 October & @R4RETURN@ & @R4CASH@ \\
r6 & 22 October & @R6RETURN@ & @R6CASH@ \\
\bottomrule
\end{tabular}
\end{table}

The active continuation has completed its registered decision, but none of its
six terminal receipts has yet occurred. The outcome-only sessions are
23, 26, 27, 28, 29 and 30 October; they may settle pending simulated fills and
mark existing accounts, but may not generate new strategy targets. Final
collection is scheduled for 31 October at 04:30 HKT, followed by the 05:30
watchdog. Final freezing requires all five mature outcomes, six valid new
terminal receipts, reconciled active accounts and no unresolved active halt.
These gates remain pending. The interrupted study supplies descriptive
operational evidence, not approval of sustained reliability, profitability or
broker deployment.

\paragraph{Evidence provenance.}
The report is stored in
\path{data/research-week1-r6/reports/week-close-20261023T0538HKT/}.
The registered release, immutable decision and watchdog receipts, and retained
failure records are linked by SHA256 in the accompanying evidence manifest.
The original r5 failure and all earlier trials remain preserved.
'''
    replacements = {
        'RESERVED':f"{receipt['shadow']['reserved_cash']:,.2f}",
        'HACC':percent(h['accuracy']),'PACC':percent(p['accuracy']),
        'HF1':f"{h['macro_f1']:.4f}",'PF1':f"{p['macro_f1']:.4f}",
        'HLOSS':f"{h['log_loss']:.4f}",'PLOSS':f"{p['log_loss']:.4f}",
        'HBRIER':f"{h['brier']:.4f}",'PBRIER':f"{p['brier']:.4f}",
        'R2RETURN':percent(prior['r2']['net_returns_at_last_actual_receipt']['shadow']),
        'R2CASH':percent(prior['r2']['net_returns_at_last_actual_receipt']['deployment']),
        'R4RETURN':percent(prior['r4']['net_returns_at_last_actual_receipt']['shadow']),
        'R4CASH':percent(prior['r4']['net_returns_at_last_actual_receipt']['deployment']),
        'R6RETURN':percent(r['economic']['net_returns']['shadow']),
        'R6CASH':percent(r['economic']['net_returns']['cash'])}
    for key,value in replacements.items():text=text.replace('@'+key+'@',value)
    assert '@H' not in text and '@R' not in text
    DEST.mkdir(parents=True,exist_ok=False)
    (DEST/'operational-update.tex').write_text(text)
    notes = '''# Proposed operational manuscript update — 23 October 2026

This is a review-only, editable LaTeX insertion. It has not been published or
uploaded to Overleaf. The existing manuscript and its PDF are preserved.
The fragment uses packages already present in the manuscript; it has not yet
been integrated or rendered as a final PDF.

Replace the obsolete zero-observation and original schedule passages in
`output/overleaf/week1/main.tex`, Section 9 (operational evaluation), and the
later prospective-evidence paragraph with `operational-update.tex`. Retain
historical operational tests and their source tables. Update the conclusion
to say five interrupted decision dates have been observed, while final outcome
and terminal-account evidence remain pending. Remove claims that all
prospective observations remain unobserved; retain the absence of economic
approval. Any abstract mention of collection must use the same 5 observed /
3 mature dates / 0 of 6 active terminal receipts cutoff. Keep the contents page
and separate references page. Render and inspect the complete PDF after
integration before treating that manuscript as final.

The 66 labeled rows come from three dates, not 66 independent market sessions.
The active first-close zero return reflects pending execution; it is not a
profitable strategy result. All earlier failures and separate account returns
are disclosed. The final endpoint remains US 30 October, collected on
31 October at 04:30 HKT.
'''
    (DEST/'REVIEW_NOTES.md').write_text(notes)
    sources=[REPORT/'report.json',REPORT/'manifest.json',REPORT/'prediction_rows.json',receipt_path,
             ROOT/'data/paper/research-week1-r6/events/2026-10-22-watchdog.json',
             ROOT/'data/paper/research-week1-r6/milestones/1.json',original_path,
             ROOT/'data/research-week1-r6/protocol.json',Path(__file__)]
    evidence={'created_at':datetime.now(timezone.utc).isoformat(),'review_only':True,
              'report':str(REPORT),'total_decision_dates':5,'mature_dates':dates,'labeled_rows':len(rows),
              'active_terminal_receipts':0,'rounded_values':replacements,
              'sources':{str(p.relative_to(ROOT)):digest(p) for p in sources}}
    (DEST/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (DEST/'manifest.json').write_text(json.dumps({'files':{p.name:digest(p) for p in DEST.iterdir() if p.is_file()}},indent=2)+'\n')
    print(json.dumps({'output':str(DEST),'decision_dates':5,'mature_dates':len(dates),'labeled_rows':len(rows),'review_only':True}))


if __name__=='__main__':main()
