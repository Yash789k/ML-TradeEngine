# Reinforcement learning for model selection

The controller makes one model choice for each instrument at each decision. If it picks the patch transformer for a stock, that model supplies the forecast of down, flat or up for that stock. The shared simulator then applies its existing position-sizing and execution rules.

The fitted Q function estimates the discounted sum of negative forecast log losses, charging 0.01 log-loss units when the expert changes. It sees conditions at the close, sentiment from dated text, and the mean losses of the latest 21 matured forecasts for that symbol. Each annual test freezes the coefficients learned from earlier outcomes. The recorded choices can change as new data arrives. In 2024 the full-pool controller selected OpenAI Decisions most frequently, for 1,702 of 5,302 stock-date decisions. The patch transformer was the most frequent choice in 2025, with 835 of 5,258 decisions.

| Available experts | Net return | Maximum drawdown | Forecast log loss |
| --- | ---: | ---: | ---: |
| All thirteen | 14.55% | 14.39% | 1.1485 |
| Linear and boosted trees | 9.29% | 14.67% | 1.0788 |
| Sequence networks | 18.82% | 17.77% | 1.0772 |

In the full pool the constant learned mixture returned 7.15% and had 16.58% maximum drawdown. The hard model selection increased mean forecast loss by 0.0738 against that mixture. The paired 95% block interval spans 0.0290 to 0.1237. The trading result can differ from the score because trading uses a threshold, ranks stocks and limits positions. All six annual router fits used previously inspected evaluation years.

Training uses complete historical expert forecasts, allowing a hypothetical reward to be calculated for every choice once the five-session outcome has matured. This assumes choosing a model does not change the market and the system has access to all the candidate forecasts. The history contains overlapping label windows. The reported interval resamples 21 sessions within years to account for temporal dependence. Only matured labels enter decision-time feedback.

[Registration](../../../ceci2_demo/periods_v1/rl-router-protocol.json) · [Colab notebook](../../../ceci2_demo/periods_v1/train_rl_router.ipynb) · [Forecasts and selected experts](forecasts.csv.gz) · [Q-function coefficients](checkpoints.json) · [Daily accounts](daily.csv.gz) · [Training receipt](receipt.json)

```sh
python -m ceci2_demo.periods_v1.rl_router --output data/raw/rl-router-reproduction
```
