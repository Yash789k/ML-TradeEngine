# Recent models and controller research, verified 30 October 2026

This review updates the foundational literature with actual 2024–2026 papers and official releases available or announced by **30 October 2026, Asia/Hong_Kong**. It is a research shortlist, not a benchmark result. None of the new models below was installed, fitted, added to the prototype, or promoted by this work. Released weights, an accepted paper and an announced upcoming service are different evidence levels.

## What best matches the proposal

The strongest near-term additions are **TabICLv2 or TabPFN-3.5 as one new three-class expert**, followed by **one frozen sequence representation plus a small classifier**. Chronos-2, TimesFM-3 and Moirai 2.0 are relevant forecasting advances, but they require an explicit classification adapter before joining this proposal's expert pool. Their internal forecasting capabilities do not replace the learned routing controller.

The proposal's central experiment remains a fixed set of classifiers feeding a context-dependent convex mixture, compared with the same experts under fixed weights and the same downstream policy. A newer architecture can improve the expert pool, but the research contribution must still isolate whether the gate and regime context add value. This is a fit-based priority judgment; no paper establishes which candidate will win on ML-TradeEngine.

Proposal evidence used: the page-marked extraction `data/research-controller-v1/proposal-review-v1/proposal.txt`, SHA-256 `40ac1eb0f7e719af295df89ac0a7d883a6deac43714d503849432e4c23c8b3b3`. Pages 3–5 and 9–10 specify UP/FLAT/DOWN experts, HMM probabilities, a learned expert-weight gate and a fixed policy; pages 24–26 specify next-open-to-open labels and exclude a learned allocator from the committed controller comparison. Proposal descriptions of earlier implementation are evidence to reconcile with current code, not fresh validation claims.

## Release chronology and shortlist

Dates below refer to the stated event, not a claim that paper submission and weight release happened simultaneously. Source links point to primary research or the originating organization.

| Model or method | Verified date/event | Exact role in this FYP | Recommended disposition |
|---|---|---|---|
| TabPFN-3.5 base | Official release **15 September 2026** | Tabular three-class expert; optional separate meta-controller hypothesis | Shortlist one frozen checkpoint; local research evaluation after access/license review. [Release](https://priorlabs.ai/technical-reports/tabpfn-3-5) |
| TabPFN-3.5 Thinking upgrade | Announced **28 October 2026**, rollout described as next week | Higher-compute tabular inference service | Watchlist; not yet generally available under that announcement as of the review date. [Announcement](https://priorlabs.ai/blog-posts/tabpfn-3-5-thinking-upgrade) |
| TabICLv2 | Preprint **11 February 2026**, revision **16 September 2026**; author record identifies ICML 2026 | Tabular three-class expert | Strong first research candidate alongside conventional controls. [Paper](https://arxiv.org/abs/2602.11139) |
| TabM | Preprint **31 October 2024**; ICLR 2025 | Small trainable tabular expert, or explicitly adapted gate | Secondary compact neural challenger. [Paper](https://arxiv.org/abs/2410.24210) |
| Chronos-2 | Preprint **17 October 2025**, official introduction **20 October 2025** | Multivariate/covariate forecast features or an adapted classification expert | One optional frozen forecasting candidate. [Paper](https://arxiv.org/abs/2510.15821), [announcement](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting) |
| TimesFM-3 | Official announcement **31 August 2026** | Multivariate forecast features or an adapted classification expert | Current-generation alternative to Chronos-2, with research-only weight terms. [Announcement](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) |
| Moirai 2.0 | Official introduction **8 August 2025**; paper **12 November 2025**, revised **3 February 2026** | Compact quantile-forecast expert requiring an adapter | Useful resource-constrained alternative. [Announcement](https://www.salesforce.com/blog/moirai-2-0/), [paper](https://arxiv.org/abs/2511.11698) |
| TiRex / frozen forecasting representations | TiRex paper **29 May 2025**, NeurIPS 2025; classification-transfer paper **30 October 2025**, workshop | Frozen sequence features followed by a trained three-class head | Particularly clean optional fit to the proposal's output interface. [TiRex](https://arxiv.org/abs/2505.23719), [classification study](https://arxiv.org/abs/2510.26777) |
| Time-MoE | Preprint **24 September 2024**, ICLR 2025 | Sparse internal forecast architecture, not the external expert gate | Related work; defer an expensive expert trial. [Paper](https://arxiv.org/abs/2409.16040) |
| Timer-S1 | Preprint **5 March 2026**, revised **9 April 2026** | Large sparse forecasting model | Current research context; outside the first small-data/local-compute comparison. [Paper](https://arxiv.org/abs/2603.04791) |
| TSMamba | Preprint **5 November 2024** | State-space forecasting representation/architecture | Research-only alternative; a usable released checkpoint was not verified here. [Paper](https://arxiv.org/abs/2411.02941) |
| Decision-Focused Fine-Tuning (DFF) | AAAI proceedings publication **11 April 2025** | Later decision-loss adaptation under explicit constraints | Related controller innovation, separately registered after the fixed-policy study. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34891) |

### 1. TabPFN-3.5: a genuinely current tabular model

The September release supports the base model plus separate Plus, Thinking and Fast offerings. The official model card identifies `tabpfn-v3.5-20260909.safetensors`; it states that one checkpoint supports classification and regression and that pretraining uses synthetic tabular tasks. An experimental multiclass checkpoint also exists, but adding it would be another trial. The model card specifies restricted research/internal-evaluation terms rather than unrestricted production use. [Official model card](https://huggingface.co/Prior-Labs/tabpfn_3_5).

**Proposed fit:** replace one expert with a classifier on exactly the eligible tabular features and three-class labels already shared by the other experts. This avoids inventing a return-to-class conversion. A separate use as an expert-correctness meta-model would require earlier out-of-time forecasts as its training rows. A TabPFN classifier over market direction is an expert; it does not automatically emit the required convex expert weights.

**Small-data judgment:** tabular in-context learning is a credible challenger for a modest data table, but correlated asset-date rows do not become independent examples because the model accepts many rows. Chronological context selection, past-only labels and untouched later evaluation still apply. Freeze the exact checkpoint and any ensemble configuration. Do not compare a high-compute Thinking service against a tightly limited tree model without counting the difference in budget.

The **28 October upgrade is not the same availability event as the September base release**. Its announcement explicitly says that rollout starts the following week and is API-exclusive initially. It should be listed as announced/upcoming as of 30 October, without claiming access or testing. [Upgrade announcement](https://priorlabs.ai/blog-posts/tabpfn-3-5-thinking-upgrade).

### 2. TabICLv2: an open, direct classification alternative

The 2026 paper introduces both classification and regression with synthetic task pretraining and improved in-context processing. The official implementation identifies the v2 classifier checkpoint as `tabicl-classifier-v2-20260212.ckpt` and also supplies a regressor. Its repository distinguishes v2 from the original classification-only ICML 2025 model. [Paper](https://arxiv.org/abs/2602.11139), [official implementation and available-model table](https://github.com/soda-inria/tabicl).

**Proposed fit:** use the classifier as one eligible expert on the same feature/label schema. Keep class calibration separate from fitting and controller selection. Compare its incremental mixture value, not only standalone score: another expert is useful only if it contributes reliable information beyond the existing pool. A useful diagnostic is paired error/disagreement by date and regime, with support counts.

**Small-data judgment:** more directly aligned than a large general-purpose language model because it consumes the project's actual feature table and returns class probabilities. The authors' generic tabular benchmark results do not establish performance under market dependence, nonstationarity or transaction costs. Pin the implementation and checkpoint before evaluation; model-family names alone are insufficient provenance.

### 3. TabM: a practical neural control for foundation-model claims

TabM studies parameter-efficient ensembling of multilayer perceptrons. Its original paper and implementation provide a modern trainable tabular alternative to transformer-based tabular models. [Paper](https://arxiv.org/abs/2410.24210), [official implementation](https://github.com/yandex-research/tabm).

**Proposed fit:** one small three-class model, or a separately implemented low-capacity gate whose outputs pass through a simplex normalization. The latter is an adaptation, not a reproduction of the paper. Keep a simple MLP or linear gate with the same inputs as the comparison. Neural ensembling may reduce some training variation; it does not remove the need for enough distinct dates or a bounded model-selection budget. Defer if the primary regularized gate already lacks enough data for stable coefficients.

### 4. Chronos-2: multivariate and covariate forecasting

Chronos-2 adds grouped information sharing across series and covariates. The official `amazon/chronos-2` card identifies a 120M-parameter model, quantile outputs, local CPU/GPU inference and Apache-2.0 licensing. It documents real and synthetic pretraining sources; that is useful disclosure, not proof of zero overlap with this project's historical observations. [Official model card](https://huggingface.co/amazon/chronos-2).

**Proposed fit:** use one frozen model to produce eligible features or an execution-aligned return distribution for the three-class adapter described below. Cross-asset context may suit the 22-instrument setting, but all series must stop at the same information cutoff. Future-known covariates mean genuinely known items such as calendar features; future market prices, retrospectively revised macro values and later news are not admissible.

**Scope:** this is a candidate expert. Its grouping mechanism is not the proposal's HMM-conditioned gate over independent direction classifiers. Benchmark forecasting gains, including covariate gains, do not establish incremental costed portfolio value here.

### 5. TimesFM-3: update the version, preserve the role

Google's official announcement dates TimesFM-2.5 to September 2025 and introduces TimesFM-3 on 31 August 2026 with native multivariate and covariate forecasting. Calling 2.5 the newest release would be outdated for this review date. [Official announcement](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/).

The exact available artifact is `google/timesfm-3.0-pytorch`. Its card lists the TimesFM Non-Commercial License v1.0 and identifies pretraining inputs including GIFT-Eval pretraining data, historical page views, Trends and synthetic/augmented data. The card still cites the original TimesFM paper; do not mislabel that older paper as a new 2026 TimesFM-3 paper. [Official model card](https://huggingface.co/google/timesfm-3.0-pytorch).

**Proposed fit:** the same frozen-feature or distribution-to-class path as Chronos-2, with identical target, cutoff and calibration. Select at most one of these large new experts for the first extension. A native multivariate interface makes a cross-asset hypothesis possible, but does not justify unrestricted cross-sectional features or claims that every stock's noise has become predictable.

### 6. Moirai 2.0: a compact quantile alternative

The release changed to decoder-only forecasting with quantile loss and multi-token prediction. The official small checkpoint is `Salesforce/moirai-2.0-R-small`; its card reports 11.4M parameters and CC-BY-NC-4.0. The August 2025 introduction and later November paper are separate chronology facts. [Model card](https://huggingface.co/Salesforce/moirai-2.0-R-small), [paper version history](https://arxiv.org/abs/2511.11698).

**Proposed fit:** an optional compact expert where local runtime matters. The paper's architecture ablations are more informative for this project than a headline leaderboard position. Benchmark rank changes with competing releases and is not presented as a current universal ranking here. Measure local latency/memory before concluding that parameter count alone makes it operationally preferable. Quantile outputs still need a tested mapping to the proposal's classes.

### 7. TiRex: the most direct sequence-to-classification bridge

TiRex is an xLSTM-based forecasting model, presented at NeurIPS 2025. A separate October 2025 workshop paper studies frozen pretrained forecasting models as feature extractors for classification. That is a useful architecture-level alternative to both transformers and Mamba. [TiRex paper](https://arxiv.org/abs/2505.23719), [classification-transfer paper](https://arxiv.org/abs/2510.26777).

**Proposed fit:** freeze one sequence encoder, extract features only from the observed prefix, and train a small regularized classifier to produce UP/FLAT/DOWN probabilities. The official classification documentation explicitly separates feature extraction and downstream fitting. [Official classification workflow](https://nx-ai.github.io/tirex/how-to/classification).

This is not zero-shot financial classification simply because the feature extractor is frozen. The supervised head, scaling and calibration learn from market labels and must remain inside their eligible partitions. Also, benchmark whole-series classification differs from forecasting the direction of a future return. Register that task transfer as a hypothesis, not an already demonstrated capability.

## Modern routing and state-space research: useful distinctions

**Time-MoE (ICLR 2025)** uses sparse internal experts within a forecasting backbone. **Timer-S1 (2026 preprint)** scales a sparse forecasting architecture and serial prediction; its reported model has 8.3B total parameters and 0.75B active per token. These are relevant recent architectures, but routing neural submodules inside one pretrained forecaster is a different scientific question from selecting among XGBoost, LSTM and text classifiers using market regimes. Their published forecasting findings do not reproduce the proposal's external gate. [Time-MoE](https://arxiv.org/abs/2409.16040), [Timer-S1](https://arxiv.org/abs/2603.04791).

Keep these as related work initially. Timer-S1's size is an engineering reason to question its fit to this project's budget, not proof it is inaccurate. A frozen small model, when available and qualified, is more tractable than reproducing large-scale pretraining. Neither sparse activation nor a paper's GPU throughput establishes local Mac latency.

**TSMamba (2024 preprint)** proposes forward/backward Mamba encoders and transfer from pretrained Mamba weights. The abstract still describes code release in future tense; this review verified the paper but did not verify a ready-to-use released checkpoint. Do not silently classify it as installed or readily reproducible. Bidirectionality over an entirely observed context window is not itself future leakage; including post-decision observations is. [Paper](https://arxiv.org/abs/2411.02941).

A useful recent corrective is **Hu et al., “How to Train Your Mamba for Time Series Forecasting,” KDD 2026**, published 20 April 2026. It examines SSM mechanisms and configurations, and questions unsupported blanket efficiency/expressiveness claims. The primary publisher search record was accessible; direct DOI access returned HTTP 403. [Publisher record](https://doi.org/10.1145/3770854.3780279). For this FYP, require matched lookbacks, parameter/compute budgets and a simple lagged-feature model before claiming an architecture effect. The immediate controller context is short and tabular, so long-sequence asymptotic advantages may offer little practical benefit.

## Recent decision and risk research belongs in a separate extension

DFF introduces decision-focused fine-tuning constrained to stay near an existing predictive model; its evaluations include portfolio optimization. It is a useful limited-data research direction, not a demonstrated improvement to this FYP's gate. Changing the training objective and allocation stage simultaneously would obscure the primary routing comparison. [Yang et al., AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34891).

**Decision-focused Sparse Tangent Portfolio Optimization**, ICML 2026, couples prediction, selection and a differentiable convex layer. Its central object is a portfolio allocator. It is therefore later-scope related work rather than a replacement for the proposal's fixed policy. [Jeon et al., 2026](https://proceedings.mlr.press/v306/jeon26a.html).

There is also directly relevant negative/diagnostic evidence: a 2026 preprint investigates inflated predictions and excessive turnover in SPO-based portfolio learning, emphasizing realistic output and turnover constraints. This is a reason to retain probability calibration, turnover and exposure diagnostics if decision loss is later introduced; it does not prove all decision-focused methods fail. [Wang and Hasuike, version 4, 13 July 2026](https://arxiv.org/abs/2605.01176).

The defensible innovation is a carefully tested **regime-aware, delayed-feedback gate with a stable three-class interface and independent risk enforcement**. It need not invent a new backbone. Demonstrating exactly when the gate changes weights, what information was available, why risk rejected an increase and how the account reconciles is a concrete CS contribution. Its economic benefit remains an empirical question.

## Adapting a new forecaster without changing the proposal

For the proposal's target return `R = Open[t+6] / Open[t+1] - 1` and cutoff-known threshold `tau = max(cost, lambda * past_horizon_volatility)`, a valid target CDF `F_R` implies:

```
p_DOWN = F_R((-tau)-)       # strict R < -tau
p_FLAT = F_R(tau) - F_R((-tau)-)
p_UP   = 1 - F_R(tau)
```

The left-limit notation handles atoms/ties; for a continuous distribution it reduces to the usual CDF evaluation. This is a proposed adapter formula, not evidence that any listed model supplies the required target distribution.

- A median forecast provides no calibrated class probabilities. Sparse quantiles require registered monotone interpolation, tail handling and separate calibration; values outside the reported quantile grid must not create invented certainty.
- Separate marginal quantiles of two future opening prices do **not** determine the distribution of their ratio. Use a justified joint path distribution, model the exact target distribution, or use frozen forecast/embedding features with a chronological classifier head.
- The next opening price is unknown at the decision cutoff. Using its realized value as an adapter input would leak execution information.
- The saved research screen used a different contract: next-close-to-close five-session outcomes with a fixed 1% threshold. It cannot validate next-open, volatility-scaled proposal labels or certify any new forecaster adapter.
- Preserve the same expert pool across static/gated comparisons, with and without HMM context. An improvement after adding a new expert is not by itself an improvement from routing.

## A bounded next comparison

First resolve the proposal-versus-current label/execution contract. Then register one modern tabular expert—TabICLv2 or the precisely pinned TabPFN-3.5 base—as a challenger beside the existing simple classifiers. Only after the fixed-pool controller comparison is coherent should one frozen sequence candidate be added, with TiRex features or one of Chronos-2/TimesFM-3/Moirai 2.0 as alternatives. Do not fit all of them merely because they are listed here.

Evaluate paired changes in log loss/Brier, class support, calibration, expert disagreement and then net account outcomes under the same frozen policy. Report date-block uncertainty, model size, CPU/GPU requirements, local latency and failures. Register every checkpoint, head, calibration method and selected threshold as part of its trial. Pretraining disclosure, synthetic-prior assumptions and inherited benchmark overlap remain explicit limitations. No later price window becomes an untouched holdout after it has been inspected.

All recommendations remain **reviewed, not implemented or tested** in this continuation. Public source availability was checked; no weights were downloaded, API account created, paid service used or external order authorized.
