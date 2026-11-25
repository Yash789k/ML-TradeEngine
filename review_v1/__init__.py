"""Progress-review package (November 2026).

Read-only views over the sealed releases and recorded pilot receipts, plus new
diagnostics that did not exist when those decisions were made: a causal rolling
HMM, consolidated classifier statistics, a sentiment join and a registered
sandbox portfolio comparison. Nothing here writes to a sealed release, a pilot
workspace or an old manifest. New state lives in ``data/review-v1`` (versioned
JSON/CSV) and ``data/paper/review-v1`` (isolated, gitignored).
"""
