import {
  Card, CardBody, CardHeader,
  Divider, Grid, H1, H2, H3,
  Pill, Row, Stack, Stat, Table, Text,
  useHostTheme,
} from 'cursor/canvas';

const PHASES = [
  {
    num: '01',
    name: 'Data Layer',
    duration: '1–2 weeks',
    status: 'Foundation',
    tone: 'info' as const,
    goal: 'Ingest multi-asset OHLCV + macro data into a clean, versioned local store.',
    tasks: [
      'Pull daily/hourly OHLCV via yfinance (equities) + ccxt (crypto)',
      'Fetch FRED macro data (VIX, yield curve, CPI) via pandas-datareader',
      'Persist to Parquet partitioned by asset/date for fast column reads',
      'Build a DataLoader class with caching, gap-fill, and adjust-for-splits logic',
      'Unit-test data pipeline with pytest — assert no lookahead leaks',
    ],
    libs: ['yfinance', 'ccxt', 'pandas-datareader', 'pyarrow', 'pytest'],
  },
  {
    num: '02',
    name: 'Feature Engineering',
    duration: '1 week',
    status: 'Signals',
    tone: 'info' as const,
    goal: 'Generate 30+ TA + statistical features; select the most predictive subset.',
    tasks: [
      'Compute RSI, MACD, Bollinger Bands, ATR, ADX, OBV via pandas-ta',
      'Add cross-asset features: BTC dominance, SPY correlation, sector ETF relative strength',
      'Statistical features: rolling z-score, Hurst exponent, realized volatility',
      'Regime features: HMM-detected market state (bull/bear/ranging) via hmmlearn',
      'Feature importance via XGBoost SHAP values; drop low-importance cols',
      'Store feature matrix as Parquet alongside raw OHLCV',
    ],
    libs: ['pandas-ta', 'hmmlearn', 'shap', 'scipy', 'numpy'],
  },
  {
    num: '03',
    name: 'ML Modeling',
    duration: '2 weeks',
    status: 'Alpha',
    tone: 'warning' as const,
    goal: 'Train a 3-class directional classifier (up/down/flat) with proper temporal CV.',
    tasks: [
      'Label generation: forward return over N bars, threshold into 3 classes',
      'Walk-forward cross-validation (PurgedGroupTimeSeriesSplit) — no leakage',
      'Model 1 — XGBoost classifier; tune via Optuna (100+ trials, VectorBT speed)',
      'Model 2 — LightGBM for ensemble diversity',
      'Model 3 — LSTM (PyTorch) on 60-bar sequences for regime-aware signals',
      'Ensemble: soft-vote probability average across all 3 models',
      'Track MLflow experiments: params, metrics, artifacts',
    ],
    libs: ['xgboost', 'lightgbm', 'torch', 'optuna', 'mlflow', 'scikit-learn'],
  },
  {
    num: '04',
    name: 'Backtesting',
    duration: '1–2 weeks',
    status: 'Validation',
    tone: 'warning' as const,
    goal: 'Simulate ML signals on 5+ years of out-of-sample data; validate edge is real.',
    tasks: [
      'VectorBT: vectorized portfolio simulation; sweep 10k+ param combos in seconds',
      'PyBroker: ML-native backtest with walk-forward validation in one framework',
      'Compute: Sharpe ratio, Sortino ratio, max drawdown, CAGR, Calmar ratio, win rate',
      'Monte Carlo simulation: 1000 bootstrap paths to get confidence intervals on metrics',
      'Transaction cost model: realistic slippage + commission per asset class',
      'Benchmark comparison: Buy & hold SPY, 60/40, trend-following CTA index',
    ],
    libs: ['vectorbt', 'pybroker', 'quantstats', 'numpy'],
  },
  {
    num: '05',
    name: 'Risk Management',
    duration: '1 week',
    status: 'Guard Rails',
    tone: 'danger' as const,
    goal: 'Size positions intelligently; hard-stop runaway losses before they compound.',
    tasks: [
      'Kelly Criterion fractional sizing (half-Kelly) per signal confidence score',
      'ATR-based stop-loss: 2× ATR trailing stop per position',
      'Portfolio heat limit: max 20% of capital at risk simultaneously',
      'Max drawdown circuit breaker: pause trading if equity drops >15% from peak',
      'Correlation filter: block new positions that increase portfolio correlation above 0.7',
      'VaR / CVaR at 95% confidence computed daily via historical simulation',
    ],
    libs: ['numpy', 'scipy', 'vectorbt'],
  },
  {
    num: '06',
    name: 'Live Signal Engine',
    duration: '1 week',
    status: 'Production',
    tone: 'success' as const,
    goal: 'Run the trained model daily; push signals to paper trading via Alpaca.',
    tasks: [
      'Cron job (GitHub Actions or Cloud Scheduler) fires daily at market close',
      'Fetch last N bars → run feature pipeline → model inference → signal output',
      'Alpaca paper trading API: submit bracket orders with ATR-based stop + limit',
      'Signal log stored in SQLite (prod) or Supabase (cloud)',
      'Alert via email/Slack webhook when signal fires or circuit breaker trips',
    ],
    libs: ['alpaca-trade-api', 'schedule', 'sqlite3', 'supabase-py', 'requests'],
  },
  {
    num: '07',
    name: 'Dashboard',
    duration: '1 week',
    status: 'Deliverable',
    tone: 'success' as const,
    goal: 'Interactive Streamlit app showing model health, equity curve, and live signals.',
    tasks: [
      'Equity curve chart (Plotly): strategy vs benchmark overlaid',
      'Rolling Sharpe + max drawdown rolling window chart',
      'Live signal table: asset, direction, confidence score, position size, stop level',
      'Model feature importance bar chart (SHAP values)',
      'Trade log with P&L per trade, win/loss streak',
      'Deploy to Streamlit Community Cloud or GCP Cloud Run (containerized)',
    ],
    libs: ['streamlit', 'plotly', 'pandas', 'docker', 'gcp'],
  },
];

const STRATEGIES = [
  ['Trend Following', 'EMA crossover + ADX filter', 'Daily', 'XGBoost classifier'],
  ['Mean Reversion', 'Bollinger Band squeeze + RSI extremes', 'Hourly', 'LightGBM'],
  ['Momentum', 'Cross-sectional 12-1 month momentum, sector rotation', 'Weekly', 'Rank-based scoring'],
  ['Regime-Aware', 'HMM market state gates all other signals', 'Daily', 'hmmlearn + ensemble'],
  ['Volatility Breakout', 'ATR expansion + volume surge entry', 'Daily', 'XGBoost + threshold'],
  ['Macro Factor', 'Yield curve, VIX term structure, dollar index as features', 'Weekly', 'LightGBM'],
  ['Statistical Arb (pairs)', 'Cointegrated pair z-score entry/exit (crypto)', 'Hourly', 'OLS residuals + rule'],
  ['LSTM Sequence', '60-bar temporal pattern recognition on OHLCV+vol', 'Daily', 'PyTorch LSTM'],
];

const TECH_STACK = [
  ['Data Ingestion', 'yfinance, ccxt, pandas-datareader', 'OHLCV equity + crypto + macro'],
  ['Feature Store', 'pandas-ta, scipy, hmmlearn', 'TA indicators, stats, HMM regime'],
  ['ML Framework', 'XGBoost, LightGBM, PyTorch, scikit-learn', 'Tree ensembles + deep learning'],
  ['Hyperparameter Tuning', 'Optuna', 'Bayesian search, 100+ trial sweeps'],
  ['Experiment Tracking', 'MLflow', 'Params, metrics, model artifacts, registry'],
  ['Backtesting', 'VectorBT, PyBroker', 'Vectorized speed + ML-native walk-forward'],
  ['Performance Analytics', 'QuantStats, numpy', 'Sharpe, Sortino, max drawdown, CAGR'],
  ['Risk Engine', 'scipy, numpy', 'VaR, CVaR, Kelly sizing, correlation filter'],
  ['Live Execution', 'alpaca-trade-api', 'Paper trading API, bracket orders'],
  ['Scheduling', 'GitHub Actions / GCP Cloud Scheduler', 'Daily cron, signal generation'],
  ['Storage', 'Parquet + SQLite / Supabase', 'Feature store + trade log'],
  ['Dashboard', 'Streamlit + Plotly', 'Equity curve, signals, SHAP explainability'],
  ['Deployment', 'Docker + GCP Cloud Run / Streamlit Cloud', 'Containerized, scalable'],
  ['Interpretability', 'SHAP', 'Feature attribution, model transparency'],
  ['Testing', 'pytest', 'Data pipeline, feature, signal validation'],
];

const METRICS = [
  { value: '7', label: 'Build Phases', tone: undefined },
  { value: '8', label: 'Quant Strategies', tone: undefined },
  { value: '3', label: 'ML Models (Ensemble)', tone: 'info' as const },
  { value: '30+', label: 'Engineered Features', tone: undefined },
  { value: '1000×', label: 'Monte Carlo Paths', tone: undefined },
  { value: '10–12 wks', label: 'MVP Timeline', tone: 'warning' as const },
];

export default function MLTradeEngineMVP() {
  const theme = useHostTheme();

  const mutedStyle = { color: theme.tokens.text.tertiary };

  return (
    <Stack gap={28} style={{ padding: '24px 28px', maxWidth: 1100 }}>

      {/* Header */}
      <Stack gap={6}>
        <H1>ML Trade Engine — MVP Blueprint</H1>
        <Text tone="secondary">
          A production-grade, Python-native algorithmic trading system using ML ensembles,
          multi-strategy signals, rigorous backtesting, and a live paper trading pipeline.
          No Pine Script. No TradingView lock-in.
        </Text>
      </Stack>

      {/* KPI strip */}
      <Grid columns={6} gap={12}>
        {METRICS.map(m => (
          <Stat key={m.label} value={m.value} label={m.label} tone={m.tone} />
        ))}
      </Grid>

      <Divider />

      {/* Architecture overview */}
      <Stack gap={12}>
        <H2>System Architecture</H2>
        <Text tone="secondary" size="small">
          Seven sequential layers — each layer's output feeds the next. Build and validate one layer before moving on.
        </Text>
        <Grid columns={7} gap={8} align="stretch">
          {['Data', 'Features', 'ML Model', 'Backtest', 'Risk', 'Execution', 'Dashboard'].map((layer, i) => (
            <Card key={layer} variant="default">
              <CardHeader trailing={<Text size="small" style={mutedStyle}>0{i + 1}</Text>}>
                {layer}
              </CardHeader>
              <CardBody>
                <Text size="small" tone="secondary">
                  {[
                    'Parquet store',
                    'pandas-ta + HMM',
                    'XGB + LGBM + LSTM',
                    'VectorBT + PyBroker',
                    'Kelly + ATR + VaR',
                    'Alpaca paper API',
                    'Streamlit + Plotly',
                  ][i]}
                </Text>
              </CardBody>
            </Card>
          ))}
        </Grid>
      </Stack>

      <Divider />

      {/* Phase breakdown */}
      <Stack gap={16}>
        <H2>Build Phases</H2>
        {PHASES.map(phase => (
          <Card key={phase.num} collapsible defaultOpen={parseInt(phase.num) <= 3}>
            <CardHeader trailing={
              <Row gap={6}>
                <Pill size="sm" tone={phase.tone} active>{phase.status}</Pill>
                <Text size="small" style={mutedStyle}>{phase.duration}</Text>
              </Row>
            }>
              Phase {phase.num} — {phase.name}
            </CardHeader>
            <CardBody>
              <Stack gap={12}>
                <Text tone="secondary" size="small">{phase.goal}</Text>
                <Stack gap={4}>
                  {phase.tasks.map((task, i) => (
                    <Row key={i} gap={8} align="start">
                      <Text size="small" style={mutedStyle} as="span">—</Text>
                      <Text size="small" as="span">{task}</Text>
                    </Row>
                  ))}
                </Stack>
                <Row gap={6} wrap>
                  {phase.libs.map(lib => (
                    <Pill key={lib} size="sm">{lib}</Pill>
                  ))}
                </Row>
              </Stack>
            </CardBody>
          </Card>
        ))}
      </Stack>

      <Divider />

      {/* Quant strategies */}
      <Stack gap={12}>
        <H2>Quantitative Strategies</H2>
        <Text tone="secondary" size="small">
          All strategies feed into the ensemble — the model learns which signals are regime-appropriate.
        </Text>
        <Table
          headers={['Strategy', 'Signal Logic', 'Timeframe', 'ML Layer']}
          rows={STRATEGIES}
          striped
          columnAlign={['left', 'left', 'center', 'left']}
        />
      </Stack>

      <Divider />

      {/* Risk management detail */}
      <Stack gap={12}>
        <H2>Risk Management Framework</H2>
        <Grid columns={3} gap={12}>
          <Card>
            <CardHeader trailing={<Pill size="sm" tone="danger" active>Hard</Pill>}>
              Circuit Breakers
            </CardHeader>
            <CardBody>
              <Stack gap={6}>
                <Text size="small">Max drawdown limit: 15% from equity peak</Text>
                <Text size="small">Max portfolio heat: 20% capital at risk</Text>
                <Text size="small">Correlation cap: block trades raising portfolio correlation above 0.7</Text>
              </Stack>
            </CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm" tone="warning" active>Sizing</Pill>}>
              Position Sizing
            </CardHeader>
            <CardBody>
              <Stack gap={6}>
                <Text size="small">Half-Kelly on each signal's predicted probability</Text>
                <Text size="small">ATR-based stop: 2× ATR trailing per position</Text>
                <Text size="small">Max single position: 5% of portfolio NAV</Text>
              </Stack>
            </CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm" tone="info" active>Metrics</Pill>}>
              Risk Reporting
            </CardHeader>
            <CardBody>
              <Stack gap={6}>
                <Text size="small">Daily VaR (95%) via historical simulation</Text>
                <Text size="small">CVaR / Expected Shortfall at 99%</Text>
                <Text size="small">Sharpe, Sortino, Calmar — rolling 90-day window</Text>
              </Stack>
            </CardBody>
          </Card>
        </Grid>
      </Stack>

      <Divider />

      {/* Full tech stack */}
      <Stack gap={12}>
        <H2>Full Technology Stack</H2>
        <Table
          headers={['Layer', 'Libraries / Tools', 'Purpose']}
          rows={TECH_STACK}
          striped
          columnAlign={['left', 'left', 'left']}
          stickyHeader
        />
      </Stack>

      <Divider />

      {/* Backtesting targets */}
      <Stack gap={12}>
        <H2>Performance Targets (Out-of-Sample)</H2>
        <Text tone="secondary" size="small">
          These are the minimum thresholds before considering live paper deployment.
        </Text>
        <Grid columns={5} gap={12}>
          <Stat value="> 1.0" label="Sharpe Ratio" tone="success" />
          <Stat value="< 20%" label="Max Drawdown" tone="warning" />
          <Stat value="> 55%" label="Win Rate" tone="success" />
          <Stat value="> 15%" label="Annual Return (CAGR)" tone="success" />
          <Stat value="> 0.5" label="Calmar Ratio" tone="info" />
        </Grid>
      </Stack>

      <Divider />

      {/* Roadmap */}
      <Stack gap={12}>
        <H2>10-Week MVP Roadmap</H2>
        <Table
          headers={['Week', 'Milestone', 'Key Deliverable', 'Gate to Proceed']}
          rows={[
            ['1–2', 'Data Layer complete', 'Parquet store for 5 assets, 5 years', 'Zero NaN rows, split-adjusted'],
            ['3', 'Feature Engineering complete', '30+ feature matrix per asset', 'No lookahead bias in any feature'],
            ['4–5', 'ML Models trained', 'XGB + LGBM + LSTM with MLflow runs', 'Val accuracy > random baseline'],
            ['6–7', 'Backtest validated', 'VectorBT + PyBroker equity curves', 'Sharpe > 1.0 out-of-sample'],
            ['8', 'Risk engine integrated', 'Kelly sizing + circuit breakers live', 'Max drawdown < 20% in sim'],
            ['9', 'Live signal engine running', 'Alpaca paper orders daily', 'Orders execute, logs clean'],
            ['10', 'Dashboard deployed', 'Streamlit app on Cloud Run', 'Public URL, all charts functional'],
          ]}
          striped
          columnAlign={['center', 'left', 'left', 'left']}
        />
      </Stack>

      <Divider />

      {/* Extensions */}
      <Stack gap={8}>
        <H2>Post-MVP Extensions</H2>
        <Grid columns={2} gap={12}>
          <Stack gap={6}>
            <H3>Strategy Expansion</H3>
            <Text size="small" tone="secondary">Options flow sentiment via unusual-whales API</Text>
            <Text size="small" tone="secondary">NLP sentiment from earnings call transcripts (FinBERT)</Text>
            <Text size="small" tone="secondary">Order book imbalance features from Level 2 data</Text>
            <Text size="small" tone="secondary">Reinforcement learning agent (Stable-Baselines3) for dynamic allocation</Text>
          </Stack>
          <Stack gap={6}>
            <H3>Infrastructure Upgrades</H3>
            <Text size="small" tone="secondary">Migrate to NautilusTrader for institutional-grade event-driven execution</Text>
            <Text size="small" tone="secondary">Real-time feature streaming via Kafka + Faust</Text>
            <Text size="small" tone="secondary">Feature store on Feast (GCP) for low-latency serving</Text>
            <Text size="small" tone="secondary">A/B test live models with shadow deployment framework</Text>
          </Stack>
        </Grid>
      </Stack>

      <Divider />

      <Text tone="tertiary" size="small">
        ML Trade Engine MVP — built on pandas-ta + XGBoost/LightGBM/PyTorch + VectorBT/PyBroker + Streamlit.
        No TradingView. No Pine Script. Full Python control.
      </Text>
    </Stack>
  );
}
