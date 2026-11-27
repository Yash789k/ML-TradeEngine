export const votingMethods = {
  original: ['Original market-state gate','The proposal’s learned gate gives each of 13 existing forecasts a share of the vote.'],
  adapter: ['Context adapter','Learns how to change the blend for each instrument and date. It takes the forecasts as given and learns weights that reduce forecast error.'],
  fixed: ['Fixed mixture','Learns a set of voting shares for each annual test. The shares stay the same for each instrument and day within that year.'],
  joint: ['Joint neural mixture','The price, text and market forecasters train together with their vote counter.'],
  equal: ['Joint forecasters, equal votes','Uses the same three jointly trained forecasters and gives each one a third of the vote.'],
  router: ['Reinforcement-learning model router','Picks one forecaster for this instrument and date. That forecaster gets the whole vote.'],
  classic: ['Classic mixture','Three forecasters and their vote counter train together on the same price, factor and text inputs.'],
};

const pools = {
  'sequence networks': 'Four price-history models',
  'trees and linear': 'Three price models',
  'all experts': 'All 13 forecasts',
};
const labels = ['Fall','Little change','Rise'];

export async function inspectVotes(root, method, {load, active, head, picker, pct, escape, selection}) {
  const kind = method === 'adapter' || method === 'fixed' ? 'adapter-examples'
    : method === 'classic' ? 'classic-votes' : 'gpu-trace';
  const data = await load(kind);
  if (!active()) return;
  const isExample = Array.isArray(data.examples);
  const usesPool = kind === 'adapter-examples' || method === 'router';
  const dates = isExample ? [...new Set(data.examples.map(row=>row.day))] : data.dates;
  const symbols = isExample ? [...new Set(data.examples.map(row=>row.symbol))] : data.symbols;
  const [title, description] = votingMethods[method];
  root.innerHTML = head(title, description) +
    '<div class="controls vote-controls">'+picker('voteMethod','Voting method',Object.keys(votingMethods).map(key=>[key,votingMethods[key][0]]),method)+
    picker('inspectDate','Decision date',dates.map(date=>[date,date]),dates.includes(selection.date)?selection.date:dates.at(-1))+
    picker('inspectSymbol','Instrument',symbols.map(symbol=>[symbol,symbol]),symbols.includes(selection.symbol)?selection.symbol:symbols[0])+
    (usesPool ? picker('inspectPool','Forecasters',Object.entries(pools),selection.pool) : '')+'</div>'+
    '<div class="controllergrid vote-grid"><section class="sheet"><h2>Combined forecast</h2><div id="inspectorForecast"></div><p class="note" id="inspectorContext"></p></section>'+
    '<section class="sheet"><h2>How much each forecast counts</h2><div id="inspectorWeights"></div></section></div>'+
    '<p class="note"><a class="link" href="#models">See what each model reads</a> · <a class="link" href="#comparison">Compare portfolio results</a></p>';
  function update() {
    const date = root.querySelector('#inspectDate').value;
    const symbol = root.querySelector('#inspectSymbol').value;
    const pool = root.querySelector('#inspectPool')?.value;
    selection.date = date;
    selection.symbol = symbol;
    if(pool) selection.pool = pool;
    let forecast, weights;
    const context = root.querySelector('#inspectorContext');
    context.textContent = '';
    if (isExample) {
      const entry = data.examples.find(row=>row.day===date && row.symbol===symbol);
      if (kind==='adapter-examples') {
        const record = entry.results[pool][method==='adapter'?'Context adapter':'Fixed mixture'];
        ({forecast, weights} = record);
        const market = entry.context[data.contextNames.indexOf('market')];
        const volatility = entry.context[data.contextNames.indexOf('vol')];
        context.textContent = method==='adapter'
          ? 'The adapter saw '+pct(volatility,2)+' recent daily volatility and a '+pct(market,2)+' daily market return, along with price range, drawdown and text scores.'
          : 'The fixed mixture uses the same shares for every instrument and date in this evaluation year.';
      } else {
        forecast = entry.forecast;
        context.textContent = 'The archive saves this combined forecast.';
      }
    } else {
      const index = dates.indexOf(date)*symbols.length+symbols.indexOf(symbol);
      if (method==='router') {
        const router = data.routers['RL model router / '+pool];
        forecast = router.forecast[index];
        weights = Object.fromEntries(router.experts.map((expert,i)=>[expert, Number(router.selected[index]===i)]));
      } else {
        forecast = [0,0,0];
        weights = {};
        for (const [name,rows] of Object.entries(data.heads)) {
          const [learned,...prediction] = rows[index];
          const weight = method==='equal' ? 1/3 : learned;
          weights[name] = weight;
          prediction.forEach((value,k)=>{forecast[k] += weight*value;});
        }
        context.textContent = 'Both views use the same three saved forecasts for this instrument and date.';
      }
    }
    if (kind==='adapter-examples') {
      const comparison = document.createElement('a');
      comparison.className = 'link';
      comparison.href = method==='adapter' ? '#controller/fixed' : '#controller/adapter';
      comparison.textContent = method==='adapter' ? 'See the fixed mixture' : 'See the context adapter';
      context.append(' ', comparison);
    }
    const summary = root.querySelector('#inspectorForecast');
    summary.innerHTML = '<div class="probs">'+forecast.map((value,i)=>'<i style="width:'+100*value+'%;background:'+['#ac7744','#a3aba1','#197257'][i]+'"></i>').join('')+
      '</div><div class="probkey">'+forecast.map((value,i)=>'<span>'+labels[i]+' '+pct(value)+'</span>').join('')+'</div>';
    root.querySelector('#inspectorWeights').innerHTML = weights
      ? Object.entries(weights).sort((a,b)=>b[1]-a[1]).map(([name,weight])=>
        '<div class="weightrow"><span>'+escape(name)+'</span><span class="track"><i style="width:'+100*weight+'%"></i></span><strong>'+pct(weight)+'</strong></div>').join('')
      : '<p class="note">Per-model voting shares are unavailable in this archive.</p>';
  }
  root.querySelectorAll('#inspectDate, #inspectSymbol, #inspectPool').forEach(control=>{control.onchange=update;});
  update();
}
