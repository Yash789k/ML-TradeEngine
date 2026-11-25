# Prospective paper-trading protocol

Status: **not started; external submission not authorized**. The local replay is
historical evidence only. No strategy has earned promotion merely by passing
accounting tests or producing a positive simulated return.

1. Freeze a signed-off model/strategy bundle, data schema, universe, initial
   capital, costs, target policy, risk limits and code hashes before the first
   prospective decision. The earliest reserved holdout is 29 September 2026.
   If setup finishes later, begin then; never backfill the missed decisions.
2. Require completed exchange-session bars and the exact feature calibration
   used by the model bundle. Record feed availability, input snapshot hashes,
   information cutoff, target vector and model version for every decision.
   Missing or stale inputs halt the batch; a failure is not a FLAT signal.
3. Continue local simulation until a candidate passes the predeclared research
   eligibility gates. Cash is the eligible deployment target if none qualifies.
4. Before any broker paper order, obtain separate authorization and validate a
   dedicated adapter against the contract. Research assumes next-close fills;
   an after-close market order ordinarily fills near the next open. Either use
   and verify a supported close-execution mechanism or rerun research with the
   adapter's actual timing. Do not attach untested brackets.
5. Reconcile account cash, positions, fees, pending orders, broker order IDs,
   partial fills, rejections, retries and restarts before and after each event.
   Unknown external positions or orders halt execution pending reconciliation.
   Local SQLite recovery is implemented; remote broker recovery remains a gate.
6. Record every expected daily event, including no-trade days and operational
   failures. Track gross/group exposure, cash reservations, turnover, drawdown,
   stale data, missing signals, slippage and reconciliation residuals.
7. Freeze a 63-completed-session pilot (extend only for previously documented
   operational outages). Do not change thresholds/models after viewing pilot
   outcomes; any changed candidate begins a new registered trial. This is an
   operational sample, not sufficient proof of a durable trading advantage.
8. Publish costs, realized and unrealized outcomes, benchmark coverage and
   dependent-return uncertainty. Distinguish historical exploration, synthetic
   smoke tests, local prospective simulation and broker paper fills. Never
   describe any of them as real-money profitability.

Release gates: documented external authorization; current data and sealed model
bundles; qualified candidate or cash; verified execution timing and adapter;
durable external reconciliation; meaningful prospective evidence. No timer or
scheduled future task was created during these repairs.
