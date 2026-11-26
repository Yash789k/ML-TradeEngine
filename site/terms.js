const paper = 'https://arxiv.org/abs/2402.06698';
const factors = 'https://jkpfactors.com/data';
const filings = 'https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/how-read';
const finbert = 'https://huggingface.co/ProsusAI/finbert';

export function coverageTable(d) {
  const number = value => value.toLocaleString('en-US');
  const days = d.bars / d.symbols;
  const rows = [
    ['Daily prices and trading volume (OHLCV)', number(d.bars) + ' instrument-days', 'Each of the ' + d.symbols + ' stocks and exchange-traded funds (ETFs) has ' + number(days) + ' trading days. Adjusted open, high, low and close prices measure price changes. Reported volume counts shares traded.'],
    ['Global Factor Data (JKP)', d.factorThemes.length + ' US themes across ' + number(days) + ' trading days', 'Jensen, Kelly and Pedersen group stocks by characteristics such as value and momentum. The models use each theme’s 21-session return with a 21-session delay.', factors],
    ['US Securities and Exchange Commission (SEC) filings', number(d.filings) + ' annual reports from 10 companies', 'The study scores the tone of excerpts from each company’s annual filing, known as Form 10-K.', filings],
    ['Financial News and Stock Price Integration Dataset (FNSPID)', number(d.headlines) + ' headlines from 6 companies', 'The study samples the first distinct headline per company each week when news is available. Its tone contributes to forecasts for seven days.', paper],
    ['Federal Reserve policy statements', number(d.statements) + ' statements', 'Text models score the language of each interest-rate decision after publication.'],
    ['OpenAI monthly forecasts', '107 monthly batches, 2,354 instrument forecasts', 'Each batch covers 22 stocks and funds. The model assigns probabilities to a price decline, little change or a price gain.']
  ];
  const table = document.createElement('table');
  table.className = 'data-table coverage-table';
  table.innerHTML = '<thead><tr><th scope="col">Source</th><th scope="col">Sample</th><th scope="col">How the study uses it</th></tr></thead>';
  const body = table.createTBody();
  for (const [name, sample, use, url] of rows) {
    const row = body.insertRow();
    const header = document.createElement('th');
    header.scope = 'row';
    if (url) {
      const link = document.createElement('a');
      link.href = url;
      link.className = 'link';
      link.textContent = name;
      header.append(link);
    } else header.textContent = name;
    row.append(header);
    row.insertCell().textContent = sample;
    row.insertCell().textContent = use;
  }
  return table;
}

const terms = [
  ['10-K', 'An annual report filed with the US Securities and Exchange Commission. It describes a company’s business, risks and financial results.', filings],
  ['ATR', 'Average true range. This measures how far a price typically travels per session, including gaps from the previous close. It sets the distance for the simulated trailing stop.'],
  ['Basis points (bps)', 'One basis point is 0.01 percentage points. A five-basis-point fee on a $10,000 trade costs $5.'],
  ['Block bootstrap', 'A way to estimate uncertainty by resampling runs of consecutive trading days together. This preserves some dependence between nearby returns.'],
  ['CECI2', 'The project code for the adaptive ensemble trading study.'],
  ['Controller or gate', 'A model that decides how much influence each expert forecast has on the combined probabilities.'],
  ['Drawdown', 'The percentage decline from a portfolio’s previous peak. Maximum drawdown is its largest such decline during the selected period.'],
  ['ETF', 'An exchange-traded fund. Its shares trade on an exchange and can give investors exposure to a basket of stocks, bonds or other assets.'],
  ['Exposure', 'The share of the simulated portfolio invested in securities. The balance stays in cash.'],
  ['F1 score', 'A measure combining precision and recall. Precision asks how often predictions of a class are correct. Recall asks how many actual cases of that class the model finds. Macro F1 gives each class equal weight.'],
  ['FinBERT', 'A financial version of BERT (Bidirectional Encoder Representations from Transformers). It estimates whether the tone of a statement is positive, neutral or negative.', finbert],
  ['FNSPID', 'Financial News and Stock Price Integration Dataset. The study uses a weekly sample of company headlines from this collection.', paper],
  ['GPU', 'A graphics processing unit. It can run many calculations in parallel when training a neural network.'],
  ['GRU', 'Gated recurrent unit. This network learns which parts of previous market observations to remember when reading the next one.'],
  ['HMM', 'Hidden Markov model. It uses observed returns and volatility to estimate the probability of each underlying market state. The controller uses these probabilities when combining forecasts.'],
  ['JKP', 'Theis Ingerslev Jensen, Bryan Kelly and Lasse Heje Pedersen. Their Global Factor Data records returns for portfolios formed from stock characteristics.', factors],
  ['L2 regularization', 'A penalty on squared model coefficients during training. It discourages the model from assigning very large coefficients to individual inputs.'],
  ['LightGBM and XGBoost', 'Light Gradient Boosting Machine and Extreme Gradient Boosting. Each builds a sequence of decision trees that learn from errors in earlier trees.'],
  ['LLM', 'Large language model. The study uses recorded language-model forecasts derived from financial information available in the input.'],
  ['Log loss', 'A score for predicted probabilities. Assigning a low probability to the outcome that happens carries a large penalty. Lower values are better.'],
  ['LSTM', 'Long short-term memory. A recurrent neural network that learns which observations to retain or forget across a sequence.'],
  ['ML', 'Machine learning. These models fit relationships in historical data and use them to make forecasts.'],
  ['MoE', 'Mixture of experts. Several models each produce a forecast. A gate assigns weights and combines their forecasts.'],
  ['OHLCV', 'Open, high, low, close and volume. Each daily record contains the first, highest, lowest and last prices, plus shares traded. The study adjusts the four prices for corporate actions.'],
  ['Regime', 'A market state inferred from past returns and volatility. The state can change as new observations arrive.'],
  ['RL', 'Reinforcement learning. The offline policy learns from historical outcomes to choose among holding cash, holding SPY or holding the stocks selected by a model.'],
  ['SEC', 'US Securities and Exchange Commission. This agency receives the company reports used for the annual filing signals.', filings],
  ['Session', 'One trading day. A forecast horizon of five sessions counts trading days when the exchange is open.'],
  ['Sharpe ratio', 'A measure of return relative to volatility. Here it uses the mean daily net return divided by its standard deviation, scaled to 252 trading days, with a zero cash rate.'],
  ['S&P 500', 'The Standard & Poor’s 500 Index tracks a broad group of large US companies. SPY seeks to track this index.'],
  ['TCN', 'Temporal convolutional network. It looks for patterns in successive windows of market observations.'],
  ['Ticker', 'The exchange symbol identifying a security. Each instrument below has its own price history.'],
  ['Turnover', 'The value bought and sold relative to the portfolio’s value. Higher turnover generally incurs more trading fees.'],
  ['USD', 'United States dollars, the currency of the simulated portfolios.']
];

const tickers = [
  ['AAPL', 'Apple'], ['AMZN', 'Amazon'], ['BAC', 'Bank of America'],
  ['DIA', 'State Street SPDR Dow Jones Industrial Average ETF Trust, which tracks the Dow Jones Industrial Average.'],
  ['GLD', 'SPDR Gold Shares, a trust that holds gold bullion.'], ['GOOGL', 'Alphabet Class A shares'],
  ['GS', 'Goldman Sachs Group'], ['IWM', 'iShares Russell 2000 ETF, a fund tracking US small-company stocks.'],
  ['JPM', 'JPMorgan Chase'], ['META', 'Meta Platforms'], ['MSFT', 'Microsoft'], ['NVDA', 'NVIDIA'],
  ['QQQ', 'Invesco QQQ Trust, a fund tracking the Nasdaq-100 Index.'],
  ['SPY', 'State Street SPDR S&P 500 ETF Trust. This fund tracks the S&P 500 Index and serves as the broad US stock benchmark in the study.'],
  ['TLT', 'iShares 20+ Year Treasury Bond ETF, a fund holding long-dated US Treasury bonds.'], ['TSLA', 'Tesla'],
  ['XLE', 'State Street Energy Select Sector SPDR ETF'], ['XLF', 'State Street Financial Select Sector SPDR ETF'],
  ['XLI', 'State Street Industrial Select Sector SPDR ETF'], ['XLK', 'State Street Technology Select Sector SPDR ETF'],
  ['XLP', 'State Street Consumer Staples Select Sector SPDR ETF'], ['XLV', 'State Street Health Care Select Sector SPDR ETF']
];

function definitions(entries) {
  const list = document.createElement('dl');
  list.className = 'definitions';
  for (const [term, meaning, source] of entries) {
    const row = document.createElement('div');
    const title = document.createElement('dt');
    title.textContent = term;
    const detail = document.createElement('dd');
    detail.textContent = meaning;
    if (source) {
      const cite = document.createElement('a');
      cite.href = source;
      cite.className = 'link';
      cite.textContent = 'Source';
      cite.setAttribute('aria-label', 'Source for ' + term);
      detail.append(' ', cite);
    }
    row.append(title, detail);
    list.append(row);
  }
  return list;
}

export function glossaryView(root) {
  root.innerHTML = '<div class="head"><h1>Glossary</h1></div>';
  root.append(definitions(terms));
  const heading = document.createElement('h2');
  heading.className = 'bottom';
  heading.textContent = 'Instrument tickers';
  root.append(heading, definitions(tickers));
}
