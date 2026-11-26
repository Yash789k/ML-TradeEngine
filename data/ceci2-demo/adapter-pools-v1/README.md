# Custom gates trained on recorded expert forecasts

Twelve learned gates cover two expanding annual fits, three selections of experts and two ways of assigning weights. The fitted intercepts and coefficients are stored in [checkpoints.json](checkpoints.json). For each fit, the receipt records the last eligible training endpoint and the first evaluation date. The published baseline experts supplied their out-of-time probabilities. Training here updates the gate that combines those predictions.

| Pool | Contextual gate net return | Constant learned weights net return | Contextual drawdown | Constant drawdown |
| --- | ---: | ---: | ---: | ---: |
| All thirteen experts | −1.39% | 7.15% | 16.67% | 16.58% |
| Linear classifier and two boosted trees | 6.34% | 8.99% | 14.03% | 14.82% |
| Four sequence networks | 24.21% | 15.57% | 17.04% | 20.87% |

The portfolios share 480 evaluation dates from 2024 and 2025. Lower log loss means a closer match between predicted probabilities and realized classes. The contextual gate using all experts increased mean daily log loss by 0.01973 relative to the constant learned weights. The paired 95% interval spans 0.00628 to 0.03406, with 21-session blocks sampled within each year. This was the registered primary comparison. Other comparisons are descriptive. These years had already been inspected during earlier studies.

The sequence pool contains the LSTM, GRU, temporal convolutional network and patch transformer. The constant gate learns one weight per expert per annual fit. The contextual gate also observes volatility, price range, drawdown, market return and the sentiment of Federal Reserve statements, company news and annual filings. A newly trained classic mixture has a different source of variation, since its experts train alongside the gate. The present study isolates changes in routing for given forecasts.

[Registration](../../../ceci2_demo/periods_v1/adapter-protocol.json) · [Colab notebook](../../../ceci2_demo/periods_v1/train_adapters.ipynb) · [Run receipt](receipt.json) · [Period comparisons](comparison.csv) · [Daily portfolios](daily.csv.gz)

The repository run used CPU training. Colab uses CUDA when its runtime supplies a GPU. The notebook downloads inputs from the recorded public revision and prints the device actually used. For a local reproduction, run the following command from a checkout containing the same archived data.

```sh
python -m ceci2_demo.periods_v1.adapter_training --output data/raw/adapter-pools-reproduction
```
