# The 2022 bear-market comparison

The regime gate lost 3.60% over 240 sessions. The classic mixture of experts lost 18.35%, and removing its text inputs increased the loss to 20.11%. Rebalancing the classic mixture every five sessions reduced simulated fees from $386 to $107, with a net return of −12.96%.

The offline reinforcement-learning policy returned −9.25% with 86.45% average exposure. SPY at the same daily exposure returned −8.77%. The policy chose SPY on 176 sessions, the classic mixture's basket on 54, and cash on 10. Holding SPY throughout returned −17.15%.

The regime gate's mean forecast log loss was 1.10592, compared with 1.11024 for the classic mixture. The registered paired interval for the difference was −0.02376 to +0.01741. The sample had 5,280 instrument decisions across 22 stocks and funds. This experiment used a known bear market selected for stress testing. The 2023–25 results were already public within the project.

From a populated five-session research cache, reproduce the fold with the original study dependencies and a new output directory.

```sh
python -m ceci2_demo.periods_v1.extension --output data/raw/periods-replay
```

[Training boundaries and forecast comparison](result.json) · [Portfolio results](comparison.csv) · [Daily portfolios](daily.csv.gz) · [Allocation choices](actions.csv) · [Registered protocol](protocol.json)
