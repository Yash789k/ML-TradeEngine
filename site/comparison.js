import {comparisonSummary} from './comparison-summary.js';
import {plotPortfolios, portfolioColors, initialGraphSelection, displayPortfolio, proposalComparison} from './portfolio-graph.js';
const explanations = {
  'Joint neural mixture': 'Trains three forecasters together. One reads recent prices, one reads signals from financial text, and one measures the broad market’s return, volatility and share of instruments rising. A fourth model learns how much of each vote to use for each stock.',
  'Joint experts uniform': 'Averages the predictions of exactly the same three jointly trained experts with a one-third weight each.',
  'Adaptive router': 'Combines 13 forecasts. As earlier predictions come due, it gives more influence to models that did well in similar market conditions.',
  'Classic MoE': 'Trains three forecasters together on price, factor and text inputs. A fourth model, called a gate, learns how much influence each forecaster gets for each stock.',
  'Classic MoE every five sessions': 'Uses the traditional mixture’s forecasts and holds positions for five sessions between rebalances.',
  'MoE without text': 'Retrains the three experts and their gate on the same market and factor data after removing the text and coverage inputs.',
  'MoE + offline RL': 'Learns from past portfolio outcomes and trading costs. Each day it chooses one of three investments from cash, the S&P 500 fund and the basket selected by the classic mixture.',
  'HMM gate': 'Estimates the chances of three market conditions from historical prices. It uses those chances and past expert accuracy to vary the influence of independently trained models.',
  'HMM gate dynamic': 'Adds position and sector limits, a drawdown pause, volatility stops and volume limits to the regime gate.',
  'HMM gate every five sessions': 'Uses the regime gate’s forecasts and holds positions for five sessions between rebalances.',
  'Gate without HMM': 'Uses the same independently trained experts and a learned gate, omitting the market-state probabilities.',
  'Uniform vote': 'Gives the same weight to every expert forecast before applying the trading rules.',
  'Market experts vote': 'Averages the seven price-based classifiers before applying the trading rules.',
  'SPY buy and hold': 'Buys SPY, a fund that tracks the S&P 500. It holds the fund until the end of each annual evaluation.',
  'Equal weight': 'Divides capital equally among the 22 stocks and funds and rebalances daily. Several instruments own overlapping securities.',
  'Cash': 'Holds uninvested dollars. The simulation credits zero cash interest.',
  'Training prior': 'Counts how often prices rose, fell or stayed roughly flat in training. It assigns those same three probabilities to every stock.',
  '21-session momentum': 'Ranks instruments by their trailing 21-session return and applies the shared score threshold and position limits.',
  'FinBERT logistic': 'Uses a language model trained on financial writing to score the tone of news, filings and central-bank statements. A second model turns those scores into a forecast.',
  'JKP factors': 'Uses earlier returns from published groups of investment strategies, such as value and momentum, to forecast the direction of each instrument.',
  'OpenAI Decisions': 'Uses recorded OpenAI directional forecasts from published Federal Reserve statements.',
  'OpenAI asset digest': 'Uses archived stock-specific OpenAI forecasts from monthly evidence packets.',
  'Granite MoE statement': 'Uses a Granite language model to assign direction probabilities from the latest eligible Fed statement.',
  'Qwen statement': 'Uses a Qwen language model to classify the latest eligible Federal Reserve statement.',
  'Logistic': 'Adds together price and volume signals with weights learned from historical examples, then estimates the chances of a rise, fall or flat market.',
  'XGBoost': 'Builds a series of decision trees from price and volume signals. Each new tree learns from errors the earlier trees made.',
  'LightGBM': 'Builds decision trees from price and volume signals. Each tree expands first where a new split most improves the training forecast.',
  'LSTM': 'Reads 20 sessions of prices and volume in order. Its internal memory learns which earlier observations still matter for the next forecast.',
  'GRU': 'Reads 20 sessions of prices and volume in order. It learns how much of its earlier memory to keep as each new session arrives.',
  'TCN': 'Looks for patterns across nearby sessions with filters that slide along the price and volume history.',
  'Patch Transformer': 'Divides the price and volume history into short blocks. It learns which blocks deserve more attention when forecasting the next move.',
  'Conventional review': 'Proposes the same preset change to how strongly the expert weights are constrained each year. Historical performance and risk checks decide whether to use it.',
  'Random review': 'Draws one candidate setting at random each year and applies the same historical checks as the research agent.',
  'LLM review': 'A hosted language-model research agent reads past expert scores and proposes one setting for the next year. Historical checks decide whether to adopt it. Daily forecasts and trades still follow the portfolio rules.',
};
function details(name) {
  if (explanations[name]) return explanations[name];
  if (/exposure|matched SPY/i.test(name)) {
    const reference = name.includes('Classic') ? 'the classic mixture' : name.includes('RL') ? 'the portfolio reinforcement learner' : 'the market-state gate';
    return 'Invests the same fraction of capital as '+reference+' each day, buying the S&P 500 fund and leaving the rest in cash. This comparison measures how much stock selection contributed.';
  }
  const [method,scope] = name.split(' / ');
  const pool = {
    'all experts':'all 13 forecasters covering market prices, financial language and investment factors',
    'trees and linear':'three forecasters, one linear model and two sets of decision trees',
    'sequence networks':'four neural networks that read a sequence of prices and trading volumes',
  }[scope];
  if (method === 'RL model router') return 'Chooses a single forecast for each stock and date from '+pool+'. During training, more accurate forecasts earn a higher score and changing experts carries a penalty.';
  if (method === 'Context adapter') return 'Combines '+pool+'. The share of each vote changes with the information available about the stock and market that day.';
  if (method === 'Fixed mixture') return 'Combines '+pool+'. It learns each model’s share of the vote from earlier forecasts and keeps those shares fixed throughout the evaluation year.';
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

export function comparisonView(root, data, paths, {head, saveCSV, icon, load, example, proposal}) {
  let studyId = 'five', chosenPeriod = proposal ? 'all-2023-2025' : 'all-2022-2025';
  let pairOpen = false;
  const summaryState = {measure:'net_return', training:false, annual:false};
  let approach = proposal ? 'HMM gate' : 'Adaptive router', benchmark = 'Uniform vote';
  const namesAcrossStudies = data.groups.flatMap(group => group.models);
  const colorFor = portfolioColors(namesAcrossStudies);
  const graphState = {models:initialGraphSelection(proposal?proposalComparison:undefined), universe:new Set(namesAcrossStudies), metrics:new Set(['equity','drawdown']), open:false};
  function draw() {
    const group = data.groups.find(g => g.id === studyId);
    const period = group.periods.find(p => p.id === chosenPeriod) || group.periods[0];
    chosenPeriod = period.id;
    const names = period.rows.map(r => r.model);
    if (!names.includes(approach)) approach = names.includes('HMM gate') ? 'HMM gate' : names[0];
    if (!names.includes(benchmark) || benchmark === approach) benchmark = names.find(n => n === 'SPY buy and hold' && n !== approach) || names.find(n => n !== approach);
    root.innerHTML = head('Performance comparison', 'Each line shows what happened to the same starting investment. A higher ending balance means more money. A deeper drawdown means a larger fall from a previous high.',
      '<button id="exportPeriods">'+icon('download')+'Export</button>')+
      '<p class="note comparison-guide"><a class="link" href="#pipeline">Where the data and scores come from</a></p><details class="protocol"><summary>How the approaches differ, using one stock</summary><div id="design" class="experiment-example"></div></details><div class="controls" id="periodControls"></div><div id="comparisonSummary"></div><section class="sheet"><div id="allPortfolioGraph"></div><details class="protocol" id="pairDetail"><summary>Compare two portfolios by year</summary><div class="controls" id="pairControls"></div><div class="metrics"><div class="metric"><small>Approach return</small><strong id="cmpReturn"></strong></div><div class="metric"><small>Benchmark return</small><strong id="baseReturn"></strong></div><div class="metric"><small>Difference</small><strong id="cmpGap"></strong></div></div><div id="pairNotes"></div><div id="yearChart"></div></details></section><section class="bottom"><div class="sectionhead"><h2>All approaches</h2><span class="tagtext" id="sample"></span></div><div id="periodResults"></div><p class="note">Drawdown starts at the opening value of the selected period. Fees are scaled to $10,000 of starting capital. Each row covers the same decision dates. These periods had already been examined during development.</p><p class="note"><a class="link" href="https://colab.research.google.com/drive/1qjCtnMsCAxeps4P7_RBPiJtPpmHHSqQD?authuser=1">Open the GPU training notebook ↗</a></p></section>';
    const design=root.querySelector('#design');
    const exampleTitle = document.createElement('h2');
    exampleTitle.textContent = example.question;
    const intro = document.createElement('p');
    intro.textContent = example.example;
    design.append(exampleTitle, intro);
    function addExamples(parent, entries) {
      const list = document.createElement('dl');
      for (const [label, explanation] of entries) {
        const term = document.createElement('dt');
        const meaning = document.createElement('dd');
        term.textContent = label;
        meaning.textContent = explanation;
        list.append(term, meaning);
      }
      parent.append(list);
    }
    addExamples(design, example.methods);
    const other = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = 'How does a forecast turn into a trade? What else was tested?';
    const trading = document.createElement('p');
    trading.textContent = example.trade;
    other.append(summary, trading);
    addExamples(other, example.experiments);
    design.append(other);
    const controls = root.querySelector('#periodControls');
    comparisonSummary(root.querySelector('#comparisonSummary'),group,period,summaryState);
    const pairs = names.map(n => [n,n]);
    controls.append(picker('studyGroup','Study',data.groups.map(g => [g.id,g.label]),studyId),
      picker('evalWindow','Evaluation window',group.periods.map(p => [p.id,p.label]),period.id));
    const pairDetail = root.querySelector('#pairDetail');
    pairDetail.open = pairOpen;
    pairDetail.ontoggle = () => { pairOpen = pairDetail.open; };
    const pairControls = root.querySelector('#pairControls');
    pairControls.append(picker('cmpModel','Approach',pairs,approach),picker('cmpBenchmark','Benchmark',pairs.filter(([n]) => n !== approach),benchmark));
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
        bar.style[v<0?'right':'left']='50%';bar.style.background=colorFor(i?benchmark:approach);strip.append(bar);
        const val=document.createElement('span');val.textContent=percent(v);cell.append(strip,val);line.append(cell);
      }
      chart.append(line);
    }
    const key=document.createElement('p');key.className='note';key.textContent='Annual columns, left to right: '+approach+' and '+benchmark+'.';chart.prepend(key);chart.prepend(title);
    const graph=plotPortfolios(root.querySelector('#allPortfolioGraph'),paths[studyId],period,approach,benchmark,graphState,colorFor,details);
    root.oncomparisonresize=graph.draw;
    const headings=['Approach','Return after costs','Largest fall from a high','Average invested','Sharpe ratio','Fees per $10k'];
    const rows = [...period.rows].sort((a,b)=>b.net_return-a.net_return);
    root.querySelector('#periodResults').append(metricTable(headings,rows.map(r=>[displayPortfolio(r.model),percent(r.net_return),percent(r.max_drawdown),percent(r.mean_exposure),r.sharpe==null?'—':r.sharpe.toFixed(2),cash(r.fees)])));
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
    controls.onchange = pairControls.onchange = event => {
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
