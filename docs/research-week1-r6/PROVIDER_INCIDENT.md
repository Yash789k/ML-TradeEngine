# JPM provider incident and bounded-retry repair — 22 October 2026

R5 attempted US 21 October on time: start 22 October 04:30:41 HKT, failure 04:30:58. Yahoo returned JPM Open 337.114990234375 above High 336.8900146484375 (difference 0.2249755859375). Low was 332.6499938964844, Close and Adjusted Close 335.19000244140625, Volume 5,726,436. The inconsistency exists in the saved provider JSON; adjustment ratio is 1. The batch had 21 accepted symbols out of 22. No valid decision, target, account commit or prospective observation was created.

The previous validator raised an undifferentiated ValueError. The scheduler treated this as deterministic integrity failure and preserved a halt; its 04:45 and 05:00 retries were blocked. The 05:30 watchdog recorded the absent reconciled receipt. This was a provider-data incident, not a host sleep or late-trigger incident. The upstream reason for Yahoo's inconsistency is unknown.

A single read-only recheck at 22 October 12:34:17 HKT returned Open = High = 337.1199951171875 and Volume 5,770,600; Low and Close were unchanged. Thus Yahoo had revised its response by that time. We do not know when the correction became available or whether either original retry would have received it. This diagnostic is not a retrospective pilot observation.

Evidence:
- Failed response: `data/paper/research-week1-r5/days/2026-10-21/market-20261021T203041615851-ea629ee7/responses/JPM.json`.
- Failed response SHA256: `08c5c92681630ef9ee687542ac8d8c9bb9c68dffcaeb7147578c1be59d82ae69`.
- Failed snapshot manifest SHA256: `bcb1988b713b9b66cb672bd801fba47c7dc1879ab044843f502030fd4e37ab65`.
- Fresh raw response and request metadata: `data/diagnostics/provider-ohlc-20261022/`.
- Fresh HTTP body SHA256: `c00463d946944e9209d5e9dcc3c45fcf9fa708d82e757d7c7a11088889a48635`.
- Repair registration: `data/research-week1-r6/repair-registration.json`.

The r6 repair changes only calendar.py and market.py in a new sealed runtime. The validator still rejects impossible OHLC bars. A typed range error may be transient only during a fresh download, only when every invalid range is confined to the expected newly completed session, and only within close+30 through close+65. Missing history, duplicate observations, nonfinite/nonpositive prices, older invalid ranges, saved-snapshot corruption and other deterministic failures do not acquire retry permission. Rejected response and raw-bar hashes are now recorded in the failed manifest.

No prices are clamped, inferred or copied from another day. A retry downloads a new complete batch and records a separate snapshot; the rejected snapshot remains evidence. The scheduler still offers at most the initial attempt and two retries, with no extensions, and its watchdog still halts on a missing valid receipt. The latest-only rule also applies to fresh tail downloads; stored snapshots remain strict.

Four valid observations remain in r2 and r4. R5 remains a failed zero-observation trial. R6 registers one future decision for US 22 October in fresh accounts, with the same models, symbols, costs and strategy. It cannot recover the missed US 21 October decision. Every failed trial and separate segment return must remain disclosed; engineering recovery is not profitability approval.
