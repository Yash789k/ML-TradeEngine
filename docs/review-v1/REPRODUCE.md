# Reproducing the progress-review demo

Everything below runs from the repository root on the branch `fyp/ceci2-adaptive-ensemble`.
Nothing here sends an order to a broker. The demo and every evaluation use paper accounts only.

## 1. Open the dashboard

```bash
pip install -r requirements.txt
python -B -m streamlit run app_review.py --server.port 8510 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

Then open <http://127.0.0.1:8510>. The dashboard reads the saved result files in `data/review-v1/`, so it opens in a few seconds and needs no network and no credentials.

Port 8510 is used on purpose. The frozen pilot dashboard keeps its own port and is not touched.

### Suggested order for a ten-minute walkthrough

| Step | Link | What to show |
|---|---|---|
| 1 | `/?view=overview` | What the system is, the plain-language reading of the main result, and the audit counts |
| 2 | `/?view=data` | What was collected, and the collection calendar including the missed sessions |
| 3 | `/?view=walkthrough&case=r2-2026-10-01&symbol=NVDA&step=1` | One recorded decision. Step through 1 to 7 with the buttons |
| 4 | `/?view=classifiers` | Classifier statistics against the class-frequency prior |
| 5 | `/?view=regimes` | The hidden Markov model, and the check that it uses no later data |
| 6 | `/?view=sentiment` | What sentiment is connected to (nothing in the pilot) and what the ablation found |
| 7 | `/?view=performance` | Trading results against the benchmark, with costs and intervals |
| 8 | `/?view=live` | Press "Run the pipeline now" (about 30 seconds, needs the network) |
| 9 | `/?view=report` | Progress report, audit table, automated checks, this page |

Every panel in the walkthrough carries one of three labels:

- **RECORDED**: written by the scheduled collector at the time.
- **RECONSTRUCTED**: recomputed now from inputs that were recorded at the time, or from prices downloaded later.
- **SIMULATED**: produced by a replay or a demo run. Never a pilot observation.

Other recorded decisions: `r2-2026-10-02`, `r4-2026-10-14`, `r4-2026-10-15`, `r6-2026-10-22`. Only `r2-2026-10-01` and `r4-2026-10-14` were followed by a recorded fill. The other three show what the system does when no fill was recorded: it says so and shows no position.

## 2. Run the tests

```bash
python -B -m pytest tests/test_review_v1_stats.py tests/test_review_v1_regime.py tests/test_review_v1_sandbox.py tests/test_review_v1_quality.py tests/test_review_v1_sentiment.py tests/test_review_v1_walkthrough.py tests/test_review_v1_app.py -q
```

| Test file | What it establishes |
|---|---|
| `test_review_v1_stats.py` | Classification measures against hand-computed values; Sharpe, probabilistic and deflated Sharpe ratio against the published worked example; short samples are declared uninterpretable |
| `test_review_v1_regime.py` | The HMM recovers a known two-state process; removing later data leaves earlier probabilities unchanged; a deliberately look-ahead version fails that check |
| `test_review_v1_sandbox.py` | Ledger arithmetic by hand (fees, turnover cap, next-close fills); results do not depend on later prices; the registered decision rule; parity with the sealed September account |
| `test_review_v1_quality.py` | The second-source price check passes on agreement and fails on a bad print or a missing day |
| `test_review_v1_sentiment.py` | A headline counts only for the first session that closes after it; the pilot feature set has no news column; the batched FinBERT scorer matches the repository scorer |
| `test_review_v1_walkthrough.py` | Provenance labels; every order, fee and outcome in the saved cases follows by arithmetic; an edited receipt is refused |
| `test_review_v1_app.py` | Every dashboard view and every walkthrough step renders without an error |

Tests that need local evidence (see section 4) are skipped with a stated reason when it is absent. The FinBERT comparison is skipped when the model is not in the local Hugging Face cache.

## 3. Rebuild the results

```bash
python -B -m review_v1.build --list      # the stages, in order
python -B -m review_v1.build all         # rebuild everything (several minutes)
python -B -m review_v1.build report manifest   # or only named stages
```

| Stage | Output | Needs |
|---|---|---|
| `regime` | `data/review-v1/regime/` | sealed September release, fresh price download |
| `classifier` | `data/review-v1/classifier/` | sealed forecasts, recorded outcome reads |
| `replay` | `data/review-v1/replay/` | pilot models, fresh price download |
| `sandbox` | `data/review-v1/sandbox/` | sealed September release, `regime` and `replay` outputs |
| `sentiment` | `data/review-v1/sentiment/` | cached headlines, FinBERT model |
| `walkthrough` | `data/review-v1/walkthrough/` | recorded receipts, ledgers and snapshots |
| `inventory`, `narration`, `report`, `contract` | `data/review-v1/*.json`, `docs/review-v1/PROGRESS_REPORT.md` | the outputs above |
| `manifest` | `data/review-v1/manifest.json` | the outputs above |

The `manifest` stage hashes every source and output file and runs the automated checks. The command exits with a non-zero status if any check fails. The same checks are shown in the dashboard under "Report and evidence", tab "Checks".

The build refuses to run unless `data/review-v1/registration.json` exists. That file fixes the questions, periods, portfolios, costs and decision rules, and was written before any result was computed. Its amendments are in `data/review-v1/registration_amendments.json`.

## 4. What is in the repository and what is local only

**In the repository:** all code, the tests, and the aggregate results under `data/review-v1/` (about 4 MB). These are enough to open the dashboard and to run the tests that do not need raw evidence.

**Local only (ignored by git):**

- `data/releases/` and `data/paper/`: recorded receipts, ledgers, market snapshots, sealed models and forecasts. The rebuild stages and the "Run the pipeline now" button read these. On a machine without them the dashboard still opens from the saved results, but the stages that read them cannot be rebuilt and the live run reports a failure in the page.
- `data/paper/review-v1/`: fresh price downloads, the replay workspace, cached headlines and headline scores. Headline text is never written to a tracked file; only aggregates are.
- `.env`: Alpaca data-API keys. They are needed for two things only, the headline download for the sentiment diagnostic and the second-source price check, and both are cached after the first run. No trading endpoint is called anywhere in this review.

## 5. Run the pipeline from the command line

```bash
python -B -m review_v1.live
```

This is what the dashboard button calls. It downloads the latest completed session, validates it, computes features, forecasts and the trend rule, applies the risk limits and writes pending orders to an isolated demo ledger under `data/paper/review-v1/live/`. It stops there: no fill is simulated and nothing is sent anywhere. The receipt is saved to `data/review-v1/live/last_run.json` and is labelled SIMULATED.

## 6. For the screen recording

`data/review-v1/ui_contract.json` lists the view identifiers, the walkthrough steps, the query parameters and the ready marker (`#review-ready`) that a capture script can wait for. `data/review-v1/narration.json` holds one short script per view and per walkthrough step, with numbers taken from the saved results.
