import {plotPortfolios} from './portfolio-graph.js';
const explanations = {
  'Joint neural mixture': 'A learned gate combines three models trained alongside it from scratch: a recurrent price encoder, a neural text-sentiment expert and a linear market expert. Their weighted probabilities feed the shared portfolio rules.',
  'Joint experts uniform': 'Averages the predictions of exactly the same three jointly trained experts with a one-third weight each.',
  'Adaptive router': 'Combines 13 forecasts using weights fitted to forecast loss. It adjusts the weights as outcomes mature, using expert performance in similar market states.',
  'Classic MoE': 'Three classifiers train jointly with a gate. The gate assigns each classifier a weight for each instrument using price, factor and text signals.',
  'Classic MoE every five sessions': 'Uses the traditional mixture’s forecasts and holds positions for five sessions between rebalances.',
  'MoE without text': 'Retrains the three experts and their gate on the same market and factor data after removing the text and coverage inputs.',
  'MoE + offline RL': 'A reinforcement learner chooses cash, SPY or the traditional mixture’s stock basket. It learns from the previous year and accounts for trading costs.',
  'HMM gate': 'Combines separately trained experts using their past accuracy, current features and probabilities from a hidden Markov model of market conditions.',
  'HMM gate dynamic': 'Adds position and sector limits, a drawdown pause, volatility stops and volume limits to the regime gate.',
  'HMM gate every five sessions': 'Uses the regime gate’s forecasts and holds positions for five sessions between rebalances.',
  'Gate without HMM': 'Uses the same independently trained experts and a learned gate, omitting the market-state probabilities.',
  'Uniform vote': 'Gives the same weight to every expert forecast before applying the trading rules.',
  'Market experts vote': 'Averages the seven price-based classifiers before applying the trading rules.',
  'SPY buy and hold': 'Holds the S&P 500 fund throughout each annual evaluation window and exits at the end of the fold.',
  'Equal weight': 'Divides capital equally among the 22 stocks and funds and rebalances daily. Several instruments own overlapping securities.',
  'Cash': 'Holds uninvested dollars. The simulation credits zero cash interest.',
  'Training prior': 'Uses historical class frequencies from the training sample as its forecast for every instrument.',
  '21-session momentum': 'Ranks instruments by their trailing 21-session return and applies the shared score threshold and position limits.',
  'FinBERT logistic': 'A logistic classifier uses financial sentiment from company news, filings and central-bank statements.',
  'JKP factors': 'A logistic classifier uses lagged Jensen, Kelly and Pedersen factor themes.',
  'OpenAI Decisions': 'Uses recorded OpenAI directional forecasts from published Federal Reserve statements.',
  'OpenAI asset digest': 'Uses archived stock-specific OpenAI forecasts from monthly evidence packets.',
  'Granite MoE statement': 'Uses a Granite language model to assign direction probabilities from the latest eligible Fed statement.',
  'Qwen statement': 'Uses a Qwen language model to classify the latest eligible Federal Reserve statement.',
  'Logistic': 'Estimates class probabilities from a linear combination of price and volume features.',
  'XGBoost': 'Combines boosted decision trees trained on price and volume features.',
  'LightGBM': 'Uses leaf-wise gradient-boosted trees to estimate directional probabilities.',
  'LSTM': 'Processes 20-session market sequences with a long short-term memory network.',
  'GRU': 'Processes 20-session market sequences with a gated recurrent network.',
  'TCN': 'Uses temporal convolutions to extract patterns from the market sequence.',
  'Patch Transformer': 'Splits a 20-session market sequence into patches and learns relationships between patches using attention.',
  'Conventional review': 'Uses the conventional candidate-selection rule and applies the registered validation criteria before changing the gate.',
  'Random review': 'Tests a seeded random candidate using the same validation criteria as the research agent.',
  'LLM review': 'Uses a hosted research agent to propose a gate setting. The validation rule decides whether the candidate enters the next evaluation.',
};
function details(name) {
  if (explanations[name]) return explanations[name];
  if (/exposure|matched SPY/i.test(name)) return 'Holds SPY in the same daily proportion as the named strategy and keeps the balance in cash.';
  if (name.startsWith('RL model router')) return 'For each stock and date, a fitted Q model chooses one expert in the named pool. The selected expert supplies its full direction forecast. The reward uses forecast loss and a penalty for changing experts.';
  if (/context|adapter/i.test(name)) return 'Learns context-dependent weights over the named expert pool from earlier forecasts, then applies the fitted gate in the next year.';
  if (/constant|static|fixed/i.test(name)) return 'Fits one set of weights for the named expert pool using earlier forecasts and holds those weights fixed during evaluation.';
  return name;
}

function picker(id, label, pairs, value) {
  const field = document.createElement('label');
  field.className = 'field';
  field.textContent = label;
  const select = document.createElement('select');
  select.id = id;
  for (const [key, text] of pairs) select.add(new Option(text, key, false, key === value));
  field.append(select);
  return field;
}

const percent = value => value == null ? '—' : (100 * value).toFixed(2) + '%';
const cash = value => value == null ? '—' : '$' + Math.round(value).toLocaleString('en-US');
function metricTable(headings, rows) {
  const frame = document.createElement('div');
  frame.className = 'scroll';
  const table = document.createElement('table');
  table.className = 'data-table';
  const header = table.createTHead().insertRow();
  for (const heading of headings) {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = heading;
    header.append(th);
  }
  const body = table.createTBody();
  for (const values of rows) {
    const tr = body.insertRow();
    for (const value of values) tr.insertCell().textContent = value;
  }
  frame.append(table);
  return frame;
}

export function comparisonView(root, data, paths, {head, saveCSV, icon, load}) {
  let studyId = 'five', chosenPeriod = 'all-2024-2025';
  let approach = 'Joint neural mixture', benchmark = 'Joint experts uniform';
  function draw() {
    const group = data.groups.find(g => g.id === studyId);
    const period = group.periods.find(p => p.id === chosenPeriod) || group.periods[0];
    chosenPeriod = period.id;
    const names = period.rows.map(r => r.model);
    if (!names.includes(approach)) approach = names.includes('HMM gate') ? 'HMM gate' : names[0];
    if (!names.includes(benchmark) || benchmark === approach) benchmark = names.find(n => n === 'SPY buy and hold' && n !== approach) || names.find(n => n !== approach);
    root.innerHTML = head('Performance comparison', 'Historical portfolios · Net of simulated trading costs',
      '<button id="exportPeriods">'+icon('download')+'Export</button>')+
      '<details class="protocol"><summary>Experiment design</summary><div id="design"></div></details><div class="controls" id="periodControls"></div><section class="sheet"><div class="metrics"><div class="metric"><small>Approach return</small><strong id="cmpReturn"></strong></div><div class="metric"><small>Benchmark return</small><strong id="baseReturn"></strong></div><div class="metric"><small>Difference</small><strong id="cmpGap"></strong></div></div><div id="allPortfolioGraph"></div><div id="pairNotes"></div><div id="yearChart"></div></section><section class="bottom"><div class="sectionhead"><h2>All approaches</h2><span class="tagtext" id="sample"></span></div><div id="periodResults"></div><p class="note">Drawdown starts at the opening value of the selected period. Fees are scaled to $10,000 of starting capital. Each row covers the same decision dates. These periods had already been examined during development.</p><p class="note"><a class="link" href="https://colab.research.google.com/drive/1qjCtnMsCAxeps4P7_RBPiJtPpmHHSqQD?authuser=1">Open the GPU training notebook ↗</a></p></section>';
    const design=root.querySelector('#design');
    design.append(metricTable(['Experiment','What varies','Trading rule'],[
      ['Joint training','Weights of three new experts and their gate','Shared score threshold and sizing'],
      ['Context adapter','Weights assigned to previously trained models','Same trading rules as fixed mixture'],
      ['RL model router','Expert identity for each stock at each close','Same trading rules as fixed mixture'],
      ['Portfolio RL','Allocation to cash, SPY or the classic basket','Learned allocation rule'],
      ['Controller stress tests','Expert availability or tree forecast direction','Same policy within each scenario'],
    ]));
    const note=document.createElement('p');note.className='note';
    note.textContent='The custom controllers first train on 2023 outcomes and evaluate in 2024. The second fit includes matured 2024 outcomes and evaluates in 2025. Earlier studies also cover 2022 and 2023. Each annual portfolio begins in cash and pays 5 basis points per side.';
    design.append(note);
    const controls = root.querySelector('#periodControls');
    const pairs = names.map(n => [n,n]);
    controls.append(picker('studyGroup','Study',data.groups.map(g => [g.id,g.label]),studyId),
      picker('evalWindow','Evaluation window',group.periods.map(p => [p.id,p.label]),period.id),
      picker('cmpModel','Approach',pairs,approach),picker('cmpBenchmark','Benchmark',pairs.filter(([n]) => n !== approach),benchmark));
    const primary = period.rows.find(r => r.model === approach);
    const reference = period.rows.find(r => r.model === benchmark);
    root.querySelector('#cmpReturn').textContent = percent(primary.net_return);
    root.querySelector('#baseReturn').textContent = percent(reference.net_return);
    const diff = 100*(primary.net_return-reference.net_return);
    root.querySelector('#cmpGap').textContent = (diff>0?'+':'')+diff.toFixed(2)+' pp';
    for (const name of [approach,benchmark]) {
      const line = document.createElement('p');
      line.className = 'note';
      const bold = document.createElement('strong'); bold.textContent = name+'. ';
      line.append(bold,details(name)); root.querySelector('#pairNotes').append(line);
    }
    const count = root.querySelector('#sample');
    count.textContent = period.start+' – '+period.end+' · '+period.sessions+' sessions';
    const years = group.periods.filter(p => /^\d{4}$/.test(p.id) && p.rows.some(r=>r.model===approach) && p.rows.some(r=>r.model===benchmark));
    const chart = root.querySelector('#yearChart');
    const title = document.createElement('h2'); title.textContent = 'Annual returns'; chart.append(title);
    const entries = years.map(p => [p.id,...[approach,benchmark].map(n=>p.rows.find(r=>r.model===n).net_return)]);
    const scale = Math.max(.01,...entries.flatMap(e=>e.slice(1).map(Math.abs)));
    for (const [year,a,b] of entries) {
      const line = document.createElement('div'); line.className='pairYear';
      const label=document.createElement('span'); label.textContent=year; line.append(label);
      for(const [i,v] of [a,b].entries()){
        const cell=document.createElement('div');cell.className='pairCell';
        const strip=document.createElement('span');strip.className='pairStrip';
        const bar=document.createElement('i');bar.style.width=(50*Math.abs(v)/scale)+'%';
        bar.style[v<0?'right':'left']='50%';bar.style.background=i?'#ac7744':'#197257';strip.append(bar);
        const val=document.createElement('span');val.textContent=percent(v);cell.append(strip,val);line.append(cell);
      }
      chart.append(line);
    }
    const key=document.createElement('p');key.className='note';key.textContent='Annual columns, left to right: '+approach+' and '+benchmark+'.';chart.prepend(key);chart.prepend(title);
    const graph=plotPortfolios(root.querySelector('#allPortfolioGraph'),paths[studyId],period,approach,benchmark);
    root.oncomparisonresize=graph.draw;
    const headings=['Approach','Net return','Max drawdown','Average invested','Sharpe','Fees per $10k'];
    const rows = [...period.rows].sort((a,b)=>b.net_return-a.net_return);
    root.querySelector('#periodResults').append(metricTable(headings,rows.map(r=>[r.model,percent(r.net_return),percent(r.max_drawdown),percent(r.mean_exposure),r.sharpe==null?'—':r.sharpe.toFixed(2),cash(r.fees)])));
    for (const [i,tr] of Array.from(root.querySelectorAll('#periodResults tbody tr')).entries()){
      const name=rows[i].model;
      tr.tabIndex=0;
      const swatch=document.createElement('i');
      swatch.className='lineKey';swatch.style.background=graph.colorFor(name);
      tr.cells[0].prepend(swatch);
      tr.onpointerenter=tr.onfocus=()=>graph.highlight(name);
      tr.onpointerleave=tr.onblur=()=>graph.highlight(null);
    }
    root.querySelector('#exportPeriods').onclick=()=>saveCSV('ensemble-'+studyId+'-'+period.id+'.csv',[['Study','Window','Sessions',...headings],...rows.map(r=>[group.label,period.label,r.sessions,r.model,r.net_return,r.max_drawdown,r.mean_exposure,r.sharpe,r.fees])]);
    if (approach.startsWith('Joint ') || approach.startsWith('RL model router')) {
      const panel = document.createElement('details');
      panel.className = 'bottom modelTrace';
      const summary = document.createElement('summary');
      summary.textContent = 'Inspect a model decision';
      panel.append(summary);
      root.querySelector('.sheet').after(panel);
      panel.ontoggle = async () => {
        if (!panel.open || panel.children.length > 1) return;
        try {
          const trace = await load(studyId==='gpu'?'gpu-trace':'model-trace');
          if (!panel.isConnected || panel.children.length > 1) return;
          const dates = trace.dates.filter(d => d >= period.start && d <= period.end);
          const inputs = document.createElement('div');
          inputs.className = 'controls';
          inputs.append(picker('traceDate','Decision date',dates.map(d=>[d,d]),dates[0]),
            picker('traceSymbol','Instrument',trace.symbols.map(n=>[n,n]),'AAPL'));
          const view = document.createElement('div');
          panel.append(inputs,view);
          function updateTrace() {
            const i = trace.dates.indexOf(inputs.querySelector('#traceDate').value)*trace.symbols.length
              + trace.symbols.indexOf(inputs.querySelector('#traceSymbol').value);
            if (trace.routers[approach]) {
              const router = trace.routers[approach];
              view.replaceChildren(metricTable(['Selected model','Down','Flat','Up'],
                [[router.experts[router.selected[i]],...router.forecast[i].map(percent)]]));
            } else {
              const rows = Object.entries(trace.heads).map(([name,values])=>{
                const [w,...p]=values[i];
                return [name,percent(approach==='Joint experts uniform'?1/3:w),...p.map(percent)];
              });
              view.replaceChildren(metricTable(['Underlying model','Vote weight','Down','Flat','Up'],rows));
            }
          }
          inputs.onchange=updateTrace;
          updateTrace();
        } catch(error) {
          const note=document.createElement('p');note.textContent=error.message;panel.append(note);
        }
      };
    }
    controls.onchange = event => {
      const {id,value}=event.target;
      if(id==='studyGroup')studyId=value;
      if(id==='evalWindow')chosenPeriod=value;
      if(id==='cmpModel')approach=value;
      if(id==='cmpBenchmark')benchmark=value;
      draw();
      root.querySelector('#'+id).focus();
    };
  }
  draw();
}
