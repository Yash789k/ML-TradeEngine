import {methodLabels} from './method-labels.js';

const portfolios = [
  ['Adaptive router', methodLabels.recent, 'Thirteen models make forecasts. Those with a better recent record in similar markets get a larger vote.'],
  ['Uniform vote', methodLabels.equal, 'Uses the same 13 forecasts and trading rule. Each forecast has the same influence.'],
  ['SPY buy and hold', methodLabels.market, 'Buy a fund that tracks the S&P 500 index of large US companies. Hold it through each annual test.'],
];

const money = new Intl.NumberFormat('en-US', {style:'currency', currency:'USD', maximumFractionDigits:0});

export function overview(root, data) {
  const periods = data.groups.find(group => group.id === 'five').periods;
  root.innerHTML = `<article class="overview">
    <h1>Does weighting forecasts by market conditions improve performance?</h1>
    <p class="intro">The proposal tests whether a learned gate that weights a fixed set of stock-direction classifiers using market conditions improves risk-adjusted performance over single classifiers and a static ensemble. A separate experiment evaluates periodic language-model-guided adaptation. <a class="link" href="#pipeline">See how the test works</a>.</p>
    <ol class="study-flow" aria-label="How the study works">
      <li><a href="#data">Input data</a><span>Prices, text and investment factors</span></li>
      <li><a href="#models">Direction forecasts</a><span>Chance of a rise, little change or a fall</span></li>
      <li><a href="#controller">Forecast weights</a><span>Combine the forecasts using market conditions</span></li>
      <li><a href="#comparison">Simulated portfolios</a><span>Apply the trading rule and measure outcomes</span></li>
    </ol>
    <div class="overview-study"><div class="overview-story"><h2>What happened to $10,000?</h2>
    <p>Both voting methods use the same 13 forecasters and trading rules. The chart shows cumulative returns in dollars. The broader results also compare drawdown, which measures the largest fall from a previous high, and Sharpe, which measures return relative to daily volatility. Buying the market provides an investment benchmark.</p>
    <p id="periodFinding"></p><a class="link" href="#comparison/pipeline">Compare the original pipeline with alternatives</a></div>
    <figure class="sheet outcome"><label class="field">Historical period<select id="overviewPeriod"><option value="all-2022-2025">2022–2025</option><option>2022</option><option>2023</option><option>2024</option><option>2025</option></select></label>
    <figcaption>Ending value after trading costs</figcaption><div id="outcomes"></div></figure></div>
    <p class="note">The researchers had already examined these years during development. The results describe those historical markets. Future performance is still an open question.</p>
    <div class="overview-links"><a class="link" href="#controller">See how the models voted on one stock</a><a class="link" href="#simulation">Replay a simulated portfolio</a><a class="link" href="assets/ensemble-study.pdf">Read the illustrated report</a></div></article>`;
  const select = root.querySelector('#overviewPeriod');
  const chart = root.querySelector('#outcomes');
  function update() {
    const period = periods.find(p => p.id === select.value);
    const rows = portfolios.map(([id, label, description]) => ({
      label, description, value: period.rows.find(r => r.model === id).net_return,
    }));
    const values = rows.map(r => 10000 * (1 + r.value));
    const scale = Math.max(10000, ...values);
    chart.replaceChildren();
    rows.forEach(({label, description}, i) => {
      const row = document.createElement('div');
      row.className = 'outcome-row';
      row.innerHTML = '<div class="outcome-label"><strong></strong><span></span></div><div class="track" aria-hidden="true"><i></i></div><p></p>';
      row.querySelector('strong').textContent = label;
      row.querySelector('span').textContent = money.format(values[i]);
      row.querySelector('i').style.width = 100 * values[i] / scale + '%';
      row.querySelector('p').textContent = description;
      chart.append(row);
    });
    const winner = values.indexOf(Math.max(...values));
    const names = ['The model that rewarded recent accuracy', 'The equal-vote portfolio', 'The market fund'];
    root.querySelector('#periodFinding').textContent = select.value === 'all-2022-2025'
      ? 'The adaptive router beat equal votes on this path. The market fund earned more and had a higher Sharpe ratio. The uncertainty interval for the router’s advantage over equal votes includes zero.'
      : (rows[winner].value < 0 ? 'All three portfolios lost money in '+select.value+'. ' : 'In '+select.value+', the starting investment was $10,000. ')
        + names[winner]+' finished highest at '+money.format(values[winner])+'.';
  }
  select.onchange = update;
  update();
}
