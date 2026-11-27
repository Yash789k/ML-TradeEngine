import {methodLabels} from './method-labels.js';

const metrics = [
  ['equity', 'Portfolio value'], ['drawdown', 'Fall from previous high'], ['daily', 'Daily return'],
];

export const displayPortfolio = name => ({
  'Adaptive router':methodLabels.recent,
  'HMM gate':methodLabels.original,
  'Uniform vote':methodLabels.equal,
  'SPY buy and hold':methodLabels.market,
  'Classic MoE':methodLabels.classic,
  'MoE without text':'Classic mixture trained without financial text',
  'Gate without HMM':'Mixture of experts without market states',
  'Logistic':'Single logistic regression model',
  'LLM review':'Language-model proposal (OpenAI)',
  'Conventional review':'Fixed proposal',
  'Random review':'Random proposal',
  'Joint neural mixture':methodLabels.joint,
  'Joint experts uniform':methodLabels.jointEqual,
  'Equal weight':'Equal investment across 22 instruments',
}[name] ?? name.replace(/^Context adapter(?= \/ )/,methodLabels.adapter)
  .replace(/^Fixed mixture(?= \/ )/,methodLabels.fixed)
  .replace(/^RL model router(?= \/ )/,methodLabels.router).replace(/ \/ sequence networks$/, ' / four models that read price history'));

export function portfolioColors(names) {
  const hues = [158, 32, 202, 74, 12, 181, 51, 218];
  const colors = new Map([...new Set(names)].sort().map((name, i) => [
    name, 'hsl('+hues[i % hues.length]+' '+(38 + 9 * (Math.floor(i / 8) % 3))+'% '+(32 + 6 * (Math.floor(i / 8) % 5))+'%)',
  ]));
  return name => colors.get(name);
}

export const proposalComparison = 'Original pipeline and alternatives';
const presets = new Map([
  [proposalComparison, ['HMM gate','Gate without HMM','Uniform vote','SPY buy and hold']],
  ['Same forecasts, different votes', ['Adaptive router','SPY buy and hold','HMM gate','Uniform vote']],
  ['Choosing a price model', ['RL model router / sequence networks','Context adapter / sequence networks','Fixed mixture / sequence networks']],
  ['With and without financial text', ['Classic MoE','MoE without text','HMM gate','FinBERT logistic']],
  ['Jointly trained experts', ['Joint neural mixture','Joint experts uniform','Classic MoE','SPY buy and hold']],
  ['Single-model baseline', ['HMM gate','Logistic','SPY buy and hold']],
  ['Research agent', ['LLM review','Conventional review','Random review','HMM gate']],
  ['Trading benchmarks', ['Cash','SPY buy and hold','Equal weight']],
]);

const presetQuestions = {
  [proposalComparison]: 'The proposal pipeline uses learned weights and market-state data. The first three lines use the same forecasters. They compare learned weights with and without market-state inputs, and equal weights. The S&P 500 fund is the benchmark.',
  'Same forecasts, different votes': 'Three controllers start with the same 13 forecasts and use the same trading rule. Equal votes give each forecast the same say. The original market-state gate learns how to weight them. The adaptive router uses recent errors to update the votes. The S&P 500 fund is the investment benchmark.',
  'Choosing a price model': 'When four models read the same price history, does it pay to pick one for each stock? The reinforcement learner picks one, the context adapter blends their votes for each stock, and the fixed mixture keeps the same blend.',
  'With and without financial text': 'Did financial text help the classic mixture? Compare Classic MoE with its retrained version without text. FinBERT reads financial language, while the HMM gate puts text alongside other forecasts.',
  'Jointly trained experts': 'Does learning how much to trust the price, text and market models beat giving each one a third of the vote? The classic mixture uses a different set of experts, and SPY shows what holding the S&P 500 fund earned.',
  'Single-model baseline': 'Logistic regression is a single forecaster from the original pool. This check asks whether combining forecasts improves on that forecaster under the same trading rule.',
  'Research agent': 'Did letting a language model propose a change improve on a fixed suggestion, a random suggestion or the original HMM gate? Every proposed change faced the same historical checks before the following year.',
  'Trading benchmarks': 'How much did choosing investments matter during this period? Cash stayed in dollars, SPY held the S&P 500 fund, and equal weight split the money across all 22 instruments.',
};

function family(name) {
  if (/^(Context adapter|Fixed mixture|RL model router)/.test(name)) return 'Expert pool experiments';
  if (/^(Joint|Classic|MoE|Adaptive|HMM|Gate|Market experts|Uniform vote)/.test(name)) return 'Mixtures and controllers';
  if (/^(Cash|SPY|Equal weight|Exposure-matched|21-session)/.test(name)) return 'Trading benchmarks';
  if (/review$/.test(name)) return 'Research agent experiments';
  if (/FinBERT|Decisions|digest|statement/.test(name)) return 'Text models';
  return 'Market and factor models';
}

export function initialGraphSelection(preset='Same forecasts, different votes') { return new Set(presets.get(preset)); }

const format = (key, value) => key === 'equity'
  ? value.toLocaleString('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0})
  : (100*value).toFixed(2)+'%';

export function plotPortfolios(host, group, period, approach, benchmark, state, colorFor, describe) {
  const indices = group.dates.flatMap((date,i) => date >= period.start && date <= period.end ? [i] : []);
  const dates = ['Start', ...indices.map(i=>group.marks[i])];
  const models = period.rows.map(r=>r.model);
  const allTraces = models.map(model => {
    const equity = [10000], drawdown = [0], daily = [null];
    let peak = 10000;
    for (const i of indices) {
      const r = group.returns[model][i];
      const next = equity.at(-1)*(1+r);
      peak = Math.max(peak,next);
      equity.push(next); drawdown.push(next/peak-1); daily.push(r);
    }
    return {model,equity,drawdown,daily,color:colorFor(model)};
  });
  const panel = document.createElement('details');
  panel.className = 'portfolioPicker';
  panel.open = state.open;
  panel.ontoggle = () => { state.open = panel.open; };
  const summary = document.createElement('summary');
  const shortcuts = document.createElement('div');
  shortcuts.className = 'portfolioShortcuts';
  const modelGrid = document.createElement('div');
  modelGrid.className = 'portfolioOptions';
  panel.append(summary,modelGrid);
  shortcuts.setAttribute('role','group');shortcuts.setAttribute('aria-label','Approach presets');
  const shortcutSelections = new Map();
  const outcomeTable = document.createElement('table');outcomeTable.className='selected-results';
  const selections = new Map();
  function check(label, parent, value, update) {
    const node = document.createElement('label'), input = document.createElement('input');
    input.type = 'checkbox'; input.checked = value;
    input.onchange = () => update(input.checked);
    node.append(input,document.createTextNode(label)); parent.append(node);
    return {node,input};
  }
  function setModels(chosen) {
    state.models = new Set(chosen);
    for (const name of models) selections.get(name).checked = chosen.has(name);
    refresh();
  }
  const choices = [...presets,['Highlighted pair',[approach,benchmark]],['All',models]];
  for (const [text,names] of choices) {
    const btn = document.createElement('button'); btn.type = 'button'; btn.textContent = text;
    btn.disabled = !names.every(n=>models.includes(n));
    btn.onclick = () => setModels(new Set(text === 'All' ? state.universe : names));
    shortcutSelections.set(btn,new Set(names));shortcuts.append(btn);
  }
  const families = new Map();
  for (const name of models) {
    const title=family(name);
    if(!families.has(title)){
      const frame=document.createElement('fieldset'),label=document.createElement('legend');
      label.textContent=title;frame.append(label);modelGrid.append(frame);families.set(title,frame);
    }
    const {input,node} = check(displayPortfolio(name),families.get(title),state.models.has(name),on => {
      if (on) state.models.add(name); else state.models.delete(name);
      refresh();
    });
    const swatch=document.createElement('i');swatch.className='portfolioKey';swatch.style.background=colorFor(name);
    node.insertBefore(swatch,input.nextSibling);selections.set(name,input);
  }
  const toggles = document.createElement('div');toggles.className='portfolioMetrics';
  toggles.setAttribute('role','group');toggles.setAttribute('aria-label','Chart metrics');
  for (const [key,label] of metrics) check(label,toggles,state.metrics.has(key),on=>{
    if(on)state.metrics.add(key);else state.metrics.delete(key);
    refresh();
  });
  const plots = document.createElement('div'), caption = document.createElement('p');
  caption.className='note portfolioCaption';
  const question=document.createElement('p');question.className='note';
  const methods=document.createElement('details');methods.className='protocol';methods.open=false;
  const methodsTitle=document.createElement('summary');methodsTitle.textContent='How the selected approaches work';
  const descriptions=document.createElement('div');methods.append(methodsTitle,descriptions);
  host.replaceChildren(shortcuts,question,panel,outcomeTable,toggles,plots,caption,methods);

  let charts=[], selected=[], focused=null, index=null;
  function highlight(name,position=null) {
    focused = selected.find(t=>t.model===name) || null;
    index = focused ? position : null;
    if (focused) {
      const i=index??indices.length;
      const values=charts.filter(c=>focused[c.key][i]!==null)
        .map(c=>c.label+' '+format(c.key,focused[c.key][i]));
      caption.textContent=[dates[i],displayPortfolio(focused.model),...values].join(' · ');
    } else caption.textContent='';
    draw();
  }
  function draw() {
    for(const chart of charts) {
      const {canvas,low,high,key} = chart;
      const {width:w,height:h} = canvas.getBoundingClientRect();
      if(!w || !h) continue;
      const ratio = window.devicePixelRatio || 1;
      if(canvas.width!==Math.round(w*ratio) || canvas.height!==Math.round(h*ratio)){
        canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);
      }
      const ctx=canvas.getContext('2d');
      ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
      const left=w<500?54:66,right=w-12,top=9,bottom=h-24;
      const x=i=>left+i*(right-left)/indices.length, y=v=>bottom-(v-low)/(high-low)*(bottom-top);
      chart.geometry={left,right,y};
      ctx.font='12px system-ui';ctx.lineWidth=1;
      for(let i=0;i<=3;i++){
        const value=low+(high-low)*i/3,py=y(value);ctx.strokeStyle='#dce4df';
        ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(right,py);ctx.stroke();
        ctx.fillStyle='#51675a';ctx.textAlign='right';ctx.fillText(key==='equity'?'$'+(value/1000).toFixed(1)+'k':(value*100).toFixed(1)+'%',left-5,py+4);
      }
      const ranked = focused ? [...selected.filter(t=>t!==focused),focused] : selected;
      for(const t of ranked){
        ctx.globalAlpha=focused && t!==focused ? .22 : (focused || selected.length<6 || t.model===approach || t.model===benchmark ? 1 : .62);
        ctx.strokeStyle=t.color;ctx.lineWidth=t===focused || t.model===approach || t.model===benchmark?2.1:1.12;
        ctx.beginPath();let started=false;
        t[key].forEach((v,i)=>{if(v!==null){if(started)ctx.lineTo(x(i),y(v));else ctx.moveTo(x(i),y(v));started=true;}});
        ctx.stroke();
      }
      ctx.globalAlpha=1;
      if(index!==null){
        ctx.lineWidth=1;ctx.strokeStyle='#789084';ctx.beginPath();ctx.moveTo(x(index),top);ctx.lineTo(x(index),bottom);ctx.stroke();
        const v=focused[key][index];
        if(v!==null){ctx.fillStyle=focused.color;ctx.beginPath();ctx.arc(x(index),y(v),3.5,0,2*Math.PI);ctx.fill();}
      }
      ctx.fillStyle='#51675a';ctx.textAlign='left';ctx.fillText('Start',left,h-5);
      ctx.textAlign='right';ctx.fillText(dates.at(-1),right,h-5);
    }
  }
  function refresh() {
    selected=allTraces.filter(t=>state.models.has(t.model)).sort((a,b)=>
      Number(a.model===approach || a.model===benchmark)-Number(b.model===approach || b.model===benchmark));
    summary.textContent='Approaches ('+selected.length+' of '+models.length+')';
    let preset=null;
    for (const [button,names] of shortcutSelections) {
      if(!button.disabled && names.size===selected.length && selected.every(t=>names.has(t.model))) {
        preset ??= button.textContent;
      }
      button.setAttribute('aria-pressed',String(!button.disabled && names.size===selected.length && selected.every(t=>names.has(t.model))));
    }
    if (preset==='Highlighted pair') question.textContent='How did '+approach+' compare with '+benchmark+' on the same dates? The top chart shows what happened to $10,000 after costs. The second shows losses from earlier highs.';
    else if (presetQuestions[preset]) question.textContent=presetQuestions[preset];
    else question.textContent=selected.length ? 'Each line starts with $10,000 on the same date. The charts show the balance after trading costs, the fall from its previous high and the change each session.' : '';
    descriptions.replaceChildren();methods.hidden=!selected.length;
    for (const trace of selected) {
      const line=document.createElement('p'),name=document.createElement('strong');
      line.className='note';name.textContent=displayPortfolio(trace.model)+'. ';line.append(name,describe(trace.model));
      descriptions.append(line);
    }

    outcomeTable.replaceChildren();
    outcomeTable.hidden = !selected.length || selected.length>8;
    if (!outcomeTable.hidden) {
      const header=outcomeTable.createTHead().insertRow();
      for (const label of ['Selected portfolio','Return','Largest fall','Sharpe ratio']) {
        const cell=document.createElement('th');
        cell.scope='col';
        cell.textContent=label;
        header.append(cell);
      }
      const body=outcomeTable.createTBody();
      for (const trace of selected) {
        const data=period.rows.find(row=>row.model===trace.model);
        const row=body.insertRow();
        const label=row.insertCell();
        const swatch=document.createElement('i');
        swatch.className='portfolioKey';
        swatch.style.background=trace.color;
        label.append(swatch,displayPortfolio(trace.model));
        row.insertCell().textContent=format('percent',data.net_return);
        row.insertCell().textContent=format('percent',data.max_drawdown);
        row.insertCell().textContent=Number.isFinite(data.sharpe)?data.sharpe.toFixed(2):'—';
      }
    }
    plots.replaceChildren();charts=[];focused=null;index=null;caption.textContent='';
    if(!selected.length || !state.metrics.size) return;
    for(const [key,label] of metrics.filter(([k])=>state.metrics.has(k))) {
      const figure=document.createElement('figure'), title=document.createElement('figcaption'), canvas=document.createElement('canvas');
      title.textContent=label;canvas.className='chart portfolioChart';canvas.tabIndex=0;
      canvas.setAttribute('role','img');canvas.setAttribute('aria-label',label+' for '+selected.length+' approaches, '+period.start+' through '+period.end);
      figure.className='portfolioFigure';figure.append(title,canvas);plots.append(figure);
      let low=Infinity,high=-Infinity;
      for(const t of selected) for(const v of t[key]) if(v!==null){low=Math.min(low,v);high=Math.max(high,v);}
      const margin=(high-low)*.045 || (key==='equity'?100:.005);
      const chart={canvas,key,label,low:low-margin,high:key==='drawdown'?0:high+margin};
      charts.push(chart);
      canvas.onpointermove=e=>{
        caption.removeAttribute('aria-live');
        const {left,right,y}=chart.geometry,rect=canvas.getBoundingClientRect();
        const i=Math.max(key==='daily'?1:0,Math.min(indices.length,Math.round((e.clientX-rect.left-left)/(right-left)*indices.length)));
        const py=e.clientY-rect.top;
        const near=selected.reduce((a,b)=>Math.abs(y(a[key][i])-py)<Math.abs(y(b[key][i])-py)?a:b);
        highlight(near.model,i);
      };
      canvas.onpointerleave=canvas.onblur=()=>highlight(null);
      canvas.onkeydown=e=>{
        if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(e.key))return;
        e.preventDefault();
        caption.setAttribute('aria-live','polite');
        let i=index??indices.length,t=Math.max(0,selected.indexOf(focused));
        if(e.key==='ArrowLeft')i--;if(e.key==='ArrowRight')i++;
        if(e.key==='Home')i=0;if(e.key==='End')i=indices.length;
        if(e.key==='ArrowUp')t--;if(e.key==='ArrowDown')t++;
        highlight(selected[(t+selected.length)%selected.length].model,Math.max(key==='daily'?1:0,Math.min(indices.length,i)));
      };
    }
    draw();
  }
  refresh();
  return {draw,highlight,colorFor};
}
