const sources = [
  ['news', 'Company headlines'],
  ['filing', 'Annual filing excerpts'],
  ['fed', 'Federal Reserve statements'],
];
const chartPortfolios = [
  ['Adaptive router', 'Adjust the votes'],
  ['Classic MoE', 'Train the models together'],
  ['SPY buy and hold', 'Hold the market fund'],
];

function barChart(host, rows, format) {
  const largest = Math.max(...rows.map(([, value]) => Math.abs(value)), 1e-6);
  host.replaceChildren();
  for (const [label, value] of rows) {
    const row = document.createElement('div');
    row.className = 'pipeline-bar';
    const name = document.createElement('span');
    const number = document.createElement('strong');
    const track = document.createElement('div');
    const bar = document.createElement('i');
    name.textContent = label;
    number.textContent = format(value);
    track.className = 'track';
    track.setAttribute('aria-hidden', 'true');
    bar.style.width = 100 * Math.abs(value) / largest + '%';
    bar.classList.toggle('loss', value < 0);
    track.append(bar);
    row.append(name, number, track);
    host.append(row);
  }
}

export function pipelineView(root, guide, coverage, data, head) {
  root.innerHTML = head('From raw data to results', 'The original proposal asks whether changing whose forecast counts can improve returns after allowing for risk.') +
    '<div class="pipeline-jumps"><button data-jump=".pipeline-stages">Data and decisions</button><button data-jump=".pipeline-metrics">How scores work</button><button data-jump=".tested-combinations">Models and results</button></div><div class="pipeline-layout"><ol class="pipeline-stages"></ol><aside class="pipeline-metrics"><h2>How to read the scores</h2><dl></dl><a class="link" href="#comparison">See the historical comparisons</a></aside></div>';
  const stages = root.querySelector('.pipeline-stages');
  for (const [title, explanation] of guide.stages) {
    const row = document.createElement('li');
    const heading = document.createElement('h2');
    const text = document.createElement('p');
    heading.textContent = title;
    text.textContent = explanation;
    row.append(heading, text);
    stages.append(row);
  }
  const metrics = root.querySelector('.pipeline-metrics dl');
  for (const [title, explanation] of guide.metrics) {
    const label = document.createElement('dt');
    const text = document.createElement('dd');
    label.textContent = title;
    text.textContent = explanation;
    metrics.append(label, text);
  }

  stages.children[0].insertAdjacentHTML('beforeend',
    '<figure class="pipeline-chart"><figcaption>Text records by publication year</figcaption><label class="field" for="pipelineSource">Source<select id="pipelineSource"></select></label><div id="textBars"></div><p>A zero means this dataset contributed no documents of that type that year. The price dataset is separate.</p><a class="link" href="#data">See price data and all sources</a></figure><details class="protocol pipeline-universe"><summary>Which 22 instruments can the simulation trade?</summary><dl><dt>10 company stocks</dt><dd>Apple (AAPL), Amazon (AMZN), Bank of America (BAC), Alphabet (GOOGL), Goldman Sachs (GS), JPMorgan Chase (JPM), Meta (META), Microsoft (MSFT), NVIDIA (NVDA), Tesla (TSLA).</dd><dt>10 stock funds</dt><dd>Dow Jones (DIA), US small companies (IWM), Nasdaq-100 (QQQ), S&amp;P 500 (SPY), energy (XLE), financials (XLF), industrials (XLI), technology (XLK), consumer staples (XLP), health care (XLV).</dd><dt>2 other instruments</dt><dd>Gold (GLD) and long-term US Treasury bonds (TLT).</dd></dl></details>');
  const source = root.querySelector('#pipelineSource');
  for (const [key, label] of sources) source.add(new Option(label, key));
  function updateCoverage() {
    const rows = coverage.annualText.map(year => [year.year, year[source.value]]);
    barChart(root.querySelector('#textBars'), rows, value => value.toLocaleString());
  }
  source.onchange = updateCoverage;
  updateCoverage();

  stages.children[3].insertAdjacentHTML('beforeend',
    '<p><a class="link" href="#controller">See the recorded forecast and next-open target for one instrument</a></p>');

  stages.children[4].insertAdjacentHTML('beforeend',
    '<figure class="pipeline-chart"><figcaption>How much the same portfolios earned and lost</figcaption><label class="field" for="pipelineMeasure">Measure<select id="pipelineMeasure"><option value="net_return">Return after costs</option><option value="max_drawdown">Largest fall from a high</option><option value="sharpe">Sharpe ratio</option></select></label><div id="scoreBars"></div><p>Same simulated dates for all three portfolios, 2022–2025.</p></figure>');
  const results = data.groups.find(group => group.id === 'five').periods.find(period => period.id === 'all-2022-2025');
  const measures = root.querySelector('#pipelineMeasure');
  function updateScores() {
    const measure = measures.value;
    const rows = chartPortfolios.map(([model, label]) => [
      label, results.rows.find(row => row.model === model)[measure],
    ]);
    barChart(root.querySelector('#scoreBars'), rows,
      value => measure === 'sharpe' ? value.toFixed(2) : (value * 100).toFixed(1) + '%');
  }
  measures.onchange = updateScores;
  updateScores();
  testedCombinations(root, guide, data);
  for (const button of root.querySelectorAll('[data-jump]')) {
    button.onclick = () => {
      const heading = root.querySelector(button.dataset.jump).querySelector('h2');
      heading.tabIndex = -1;
      heading.focus({preventScroll:true});
      heading.scrollIntoView({block:'start'});
    };
  }
}

function testedCombinations(root, guide, data) {
  const period = data.groups.find(group => group.id === 'gpu').periods.find(row => row.id === 'all-2024-2025');
  const returns = new Map(period.rows.map(row => [row.model, row.net_return]));
  const format = value => (100 * value).toFixed(1) + '%';
  const section = document.createElement('section');
  section.className = 'tested-combinations bottom';
  section.innerHTML = '<h2>Which models worked together?</h2><p class="note">Each row uses the same forecasts and trading rules across 480 trading days in 2024–2025. Only the way the system uses those forecasts changes across the three results.</p><details class="protocol pipeline-universe"><summary>What does each controller optimize?</summary><dl><dt>Classic mixture of experts</dt><dd>Three classifiers and their vote counter train together on shared stock inputs. They learn to assign high probability to the direction that happened. The inputs include prices, investment factors and scores from financial text.</dd><dt>Context adapter</dt><dd>Takes forecasts from models that are already trained. It learns how to divide the vote by looking at earlier forecast errors, using information such as volatility, recent market returns and text scores.</dd><dt>Reinforcement learning model router</dt><dd>Picks one forecaster per stock and day. It learns from the probability that forecaster gave to the direction that happened, with a small penalty for changing forecasters.</dd><dt>Portfolio reinforcement learning</dt><dd>Chooses where the money goes: cash, the S&amp;P 500 fund or the basket picked by the classic mixture. It learns to grow the account after trading costs. The Sharpe ratio evaluates the resulting portfolio and plays no role in either reinforcement learning reward.</dd></dl></details>';
  for (const [label, key, models] of guide.pools) {
    const group = document.createElement('figure');
    group.className = 'pipeline-chart';
    const caption = document.createElement('figcaption');
    caption.textContent = label;
    const members = document.createElement('p');
    members.textContent = models.join(', ') + '.';
    const chart = document.createElement('div');
    group.append(caption, members, chart);
    barChart(chart, [
      ['Keep the same voting shares', returns.get('Fixed mixture / ' + key)],
      ['Adjust the votes for each stock', returns.get('Context adapter / ' + key)],
      ['Pick one model per stock', returns.get('RL model router / ' + key)],
    ], format);
    section.append(group);
  }
  const joint = document.createElement('figure');
  joint.className = 'pipeline-chart';
  joint.innerHTML = '<figcaption>Three models trained together</figcaption><p>One model reads price history, one uses scores from financial text, and one uses broad market conditions. Both portfolios below use the same three forecasts.</p><div></div>';
  barChart(joint.lastElementChild, [
    ['Give each forecast a third of the vote', returns.get('Joint experts uniform')],
    ['Learn how much each vote counts', returns.get('Joint neural mixture')],
  ], format);
  section.append(joint);
  const note = document.createElement('p');
  note.className = 'note';
  note.textContent = 'Over the same dates the market fund returned ' + format(returns.get('SPY buy and hold')) +
    '. Splitting the money equally across the 22 instruments and rebalancing daily returned ' + format(returns.get('Equal weight')) + '. The study had already examined this period during development.';
  section.append(note);
  root.append(section);
}
