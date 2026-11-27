const stages = [
  ['Forecast the stock', 'Price history, trading volume, financial text or investment factors, depending on the model.', 'Probabilities of a rise, little change and a fall.', 'Assign more probability to the direction that happens.'],
  ['Train a mixture together', 'The classic mixture gives all three forecasters the same stock features. The later joint experiment gives its three networks separate price, text and market inputs.', 'One forecast formed by combining three votes.', 'Train the forecasts and their vote counter together to reduce forecast error.'],
  ['Adjust the votes', 'Forecasts from models that have already been trained, plus context such as market returns and volatility.', 'A share of the vote for each model. The shares add up to 100%.', 'Reduce forecast error. The fixed mixture learns shares that stay constant for the test. The context adapter lets the shares vary.'],
  ['Pick one model', 'Forecast context, earlier forecast errors and which model was picked before.', 'One forecaster per stock and date.', 'Use reinforcement learning to reduce forecast error and penalize switching.'],
  ['Choose what to hold', 'Signals from the classic mixture, recent market returns and volatility, and the portfolio’s previous choice.', 'Cash, the S&P 500 fund or the basket selected by the classic mixture.', 'Use reinforcement learning to grow portfolio value after fees.'],
];

const measures = [
  ['net_return', 'Return after costs'],
  ['max_drawdown', 'Largest fall from a high'],
  ['sharpe', 'Sharpe ratio'],
];

const nonLearning = new Set(['Cash', 'SPY buy and hold', 'Equal weight', 'Uniform vote', '21-session momentum', 'Training prior', 'Market experts vote']);

const candidates = [
  ['Adaptive router', 'Adaptive router'],
  ['Classic MoE', 'Classic mixture'],
  ['HMM gate', 'Market-state gate'],
  ['Uniform vote', 'Equal forecast votes'],
  ['MoE + offline RL', 'Portfolio allocator'],
  ['Context adapter / sequence networks', 'Adaptive votes among four price-history models'],
  ['Fixed mixture / sequence networks', 'Fixed votes among four price-history models'],
  ['RL model router / sequence networks', 'Pick one of four price-history models'],
  ['Joint neural mixture', 'Three trained models with learned votes'],
  ['Joint experts uniform', 'Three trained models with equal votes'],
  ['SPY buy and hold', 'S&P 500 fund'],
  ['Equal weight', 'Equal money across the 22 instruments'],
];

export function comparisonSummary(host, group, period, state) {
  host.innerHTML = `<section class="comparison-summary">
    <details class="protocol" id="trainingSummary"><summary>Training inputs, outputs and goals</summary><dl class="training-steps"></dl></details>
    <h2>Which approach led, and when?</h2>
    <label class="field" for="reviewMeasure">Measure<select id="reviewMeasure"></select></label>
    <p class="note" id="periodLeader"></p><p class="note" id="periodDifference"></p>
    <details class="protocol" id="annualSummary"><summary>Compare these approaches year by year</summary><div class="annual-matrix"></div></details>
  </section>`;
  const training = host.querySelector('#trainingSummary');
  training.open = state.training;
  training.ontoggle = () => { state.training = training.open; };
  const annual = host.querySelector('#annualSummary');
  annual.open = state.annual;
  annual.ontoggle = () => { state.annual = annual.open; };
  const steps = host.querySelector('.training-steps');
  for (const [title, input, output, goal] of stages) {
    const name = document.createElement('dt');
    const detail = document.createElement('dd');
    name.textContent = title;
    for (const [label, description] of [['Inputs',input],['Output',output],['Goal',goal]]) {
      const line = document.createElement('p');
      const key = document.createElement('strong');
      key.textContent = label + '. ';
      line.append(key, description);
      detail.append(line);
    }
    steps.append(name, detail);
  }

  const selector = host.querySelector('#reviewMeasure');
  for (const [key,label] of measures) selector.add(new Option(label,key,false,key===state.measure));
  const summary = host.querySelector('#periodLeader');
  const matrix = host.querySelector('.annual-matrix');
  const years = group.periods.filter(item => /^\d{4}$/.test(item.id));
  function update() {
    const metric = selector.value;
    state.measure = metric;
    const prefer = metric === 'max_drawdown' ? -1 : 1;
    const format = value => !Number.isFinite(value) ? 'Unavailable' :
      metric === 'sharpe' ? value.toFixed(2) : (100*value).toFixed(2)+'%';
    const ranked = period.rows.filter(row => Number.isFinite(row[metric]))
      .sort((a,b) => prefer*(b[metric]-a[metric]));
    const leader = ranked[0];
    const learned = ranked.find(row => !nonLearning.has(row.model) && !/exposure/i.test(row.model));
    const market = period.rows.find(row => row.model === 'SPY buy and hold');
    const name = candidates.find(([key]) => key===leader.model)?.[1] ?? leader.model;
    const summaryText = (metric==='max_drawdown' ? 'Smallest fall from a high' : 'Highest result') +
      ' in '+period.label+' was '+format(leader[metric])+' for '+name+'.';
    summary.textContent = summaryText + (market && market!==leader ? ' The S&P 500 fund recorded '+format(market[metric])+'.' : '');
    if (learned && learned !== leader) {
      const label = candidates.find(([key]) => key===learned.model)?.[1] ?? learned.model;
      summary.textContent += ' The leading portfolio using trained forecasts was '+label+' at '+format(learned[metric])+'.';
    }
    const difference = host.querySelector('#periodDifference');
    const adaptive = period.rows.find(row => row.model==='Adaptive router');
    const equal = period.rows.find(row => row.model==='Uniform vote');
    difference.textContent = adaptive && equal
      ? 'Adaptive router '+format(adaptive[metric])+' · equal votes on the same forecasts '+format(equal[metric])+'.'
      : '';
    difference.hidden = !difference.textContent;
    matrix.replaceChildren();
    for (const [key,label] of candidates) {
      const values = years.map(year => year.rows.find(row => row.model===key)?.[metric]);
      if (!values.some(Number.isFinite)) continue;
      const row = document.createElement('div');
      const heading = document.createElement('strong');
      const results = document.createElement('dl');
      heading.textContent=label;
      years.forEach((year,i) => {
        const cell=document.createElement('div');
        const yearLabel=document.createElement('dt');
        const value=document.createElement('dd');
        yearLabel.textContent=year.id;
        value.textContent=format(values[i]);
        cell.append(yearLabel,value);
        results.append(cell);
      });
      row.append(heading,results);
      matrix.append(row);
    }
  }
  selector.onchange = update;
  update();
}
