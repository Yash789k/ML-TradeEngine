const portfolios = [
  ['Adaptive router', 'Give more weight to recently accurate models', 'Thirteen models make forecasts. Those with a better recent record in similar markets get a larger vote.'],
  ['Classic MoE', 'Train the models to work together', 'Three models learn alongside a fourth model that decides how much each forecast counts for each stock.'],
  ['SPY buy and hold', 'Buy and hold the market', 'Buy a fund that tracks the S&P 500 index of large US companies. Hold it through each annual test.'],
];

const money = new Intl.NumberFormat('en-US', {style:'currency', currency:'USD', maximumFractionDigits:0});

export function overview(root, data) {
  const periods = data.groups.find(group => group.id === 'five').periods;
  root.innerHTML = `<article class="overview">
    <h1>Can a model learn which stock forecasts to trust?</h1>
    <p class="intro">A model that reads prices can disagree with one that reads company news. This study asks whether learning how much to trust each forecast helps a simulated investor earn more or lose less when markets fall. <a class="link" href="#pipeline">Follow the data to the results</a>.</p>
    <div class="overview-study"><div class="overview-story"><h2>What happened to $10,000?</h2>
    <p>Each portfolio started with the same amount. Two used forecasts to choose investments. A third simply bought the market. The portfolios traded at simulated prices and paid trading costs.</p>
    <p id="periodFinding"></p><a class="link" href="#comparison">Compare the trading approaches</a></div>
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
    const names = ['The model that rewarded recent accuracy', 'The jointly trained models', 'The market fund'];
    root.querySelector('#periodFinding').textContent = select.value === 'all-2022-2025'
      ? 'Buying the market finished ahead. The model that rewarded recent accuracy had the smallest fall from a previous high during these four years. It also kept more money in cash on average.'
      : (rows[winner].value < 0 ? 'All three portfolios lost money in '+select.value+'. ' : 'In '+select.value+', the starting investment was $10,000. ')
        + names[winner]+' finished highest at '+money.format(values[winner])+'.';
  }
  select.onchange = update;
  update();
}
