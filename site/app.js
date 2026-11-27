import {reviewResults} from './review-results.js';
import {methodLabels} from './method-labels.js';
import {inspectVotes, votingMethods} from './vote-inspector.js';
import {modelMap} from './model-map.js';
import {sourceGuide} from './source-guide.js';
import {pipelineView} from './pipeline.js';
import {overview} from './overview.js';
import {coverageTable, glossaryView} from './terms.js';
import {simulate,drawChart} from './simulator.js';
import {comparisonView} from './comparison.js';
import {renderArchitecture} from './architecture.js';
const root=document.querySelector('#app'), dollars=new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0});
const pct=(v,n=1)=>(v*100).toFixed(n)+'%', num=(v,n=2)=>v==null?'—':Number(v).toFixed(n);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const files=new Map();function load(name){if(!files.has(name))files.set(name,fetch('assets/'+name+'.json').then(r=>{if(!r.ok)throw Error('Could not load '+name);return r.json()}).catch(e=>{files.delete(name);throw e}));return files.get(name);}
const icons={play:'<path d="m6 3 11 7-11 7z" fill="currentColor"/>',pause:'<path d="M5 3h4v14H5zm7 0h4v14h-4z" fill="currentColor"/>',next:'<path d="m4 3 10 7-10 7z" fill="currentColor"/><path d="M16 3v14" stroke="currentColor" stroke-width="2"/>',download:'<path d="M10 2v10m-4-4 4 4 4-4M3 14v3h14v-3" fill="none" stroke="currentColor" stroke-width="1.6"/>'};
const icon=id=>'<svg aria-hidden="true" viewBox="0 0 20 20">'+icons[id]+'</svg>';
let horizon=5,model='HMM gate',bench='Exposure-matched SPY',index=null,asset='AAPL',mode='Recorded',scenario='original',chartMode='Equity',capital=10000,fee=5,cap=15,threshold=5,timer=null,token=0;
let paint=()=>{},onStep=()=>{};
const option=(list,value)=>list.map(v=>'<option'+(v===value?' selected':'')+'>'+esc(v)+'</option>').join('');
const field=(id,label,inner)=>'<label for="'+id+'">'+label+inner+'</label>';
const select=(id,items,value)=>'<select id="'+id+'">'+option(items,value)+'</select>';
const metric=(label,id)=>'<div class="metric"><small>'+label+'</small><strong id="'+id+'">—</strong></div>';
const table=(cols,rows)=>'<div class="scroll"><table class="data-table"><thead><tr>'+cols.map(x=>'<th scope="col">'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
const head=(title,caption,button='')=>'<div class="head"><div><h1>'+title+'</h1><p class="meta">'+caption+'</p></div>'+button+'</div>';
const el=id=>document.getElementById(id);
const on=(id,event,cb)=>el(id)?.addEventListener(event,cb);
const displayDate=s=>new Date(s+'T12:00:00Z').toLocaleDateString('en-US',{year:'numeric',month:'short',day:'numeric',timeZone:'UTC'});
const colors=['#197257','#ac7744','#788b9f'];
const legend=items=>'<div class="legend">'+items.map(([name,color])=>'<span><i style="background:'+color+'"></i>'+esc(name)+'</span>').join('')+'</div>';
function stop(){clearInterval(timer);timer=null;if(el('play')){el('play').innerHTML=icon('play');el('play').setAttribute('aria-label','Play simulation');}}
function saveCSV(filename,rows){const csv=rows.map(row=>row.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\n');const u=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),a=document.createElement('a');a.href=u;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
function timeline(length){return '<div class="timeline"><button class="iconbtn primary" id="play" aria-label="Play simulation">'+icon('play')+'</button><button class="iconbtn" id="step" aria-label="Next session">'+icon('next')+'</button><input type="range" id="dateRange" aria-label="Replay session" min="0" max="'+(length-1)+'" value="'+index+'"><output id="asof"></output></div>';}
function bindTimeline(dates){
  function go(i){index=Math.max(0,Math.min(dates.length-1,i));el('dateRange').value=index;el('asof').textContent=displayDate(dates[index]);onStep();}
  on('dateRange','input',e=>{stop();go(+e.target.value)});on('step','click',()=>{stop();go(index+1)});
  on('play','click',()=>{if(timer){stop();return}if(index===dates.length-1)go(0);el('play').innerHTML=icon('pause');el('play').setAttribute('aria-label','Pause simulation');timer=setInterval(()=>{go(index+1);if(index>=dates.length-1)stop()},105);});
  go(index);
}
function bindChart(dates){const c=el('mainChart');on('mainChart','click',e=>{stop();const r=c.getBoundingClientRect(),pad=r.width<500?42:55;index=Math.round(Math.max(0,Math.min(1,(e.clientX-r.left-pad)/(r.width-pad-10)))*(dates.length-1));el('dateRange').value=index;el('asof').textContent=displayDate(dates[index]);onStep();});}
function dd(values,start){let peak=start,worst=0;for(const v of values){peak=Math.max(peak,v);worst=Math.max(worst,1-v/peak)}return worst;}
function modeButtons(){return '<div class="pilltab"><button id="recorded" class="'+(mode==='Recorded'?'on':'')+'">Recorded</button><button id="custom" class="'+(mode==='Custom'?'on':'')+'">What if</button></div>';}
function plotTabs(){return '<div class="pilltab"><button id="equity" class="'+(chartMode==='Equity'?'on':'')+'">Portfolio value</button><button id="drawdown" class="'+(chartMode==='Equity'?'':'on')+'">Fall from peak</button></div>';}
function tabsChart(){for(const [id,name] of [['equity','Equity'],['drawdown','Drawdown']])on(id,'click',()=>{chartMode=name;el('equity').classList.toggle('on',chartMode==='Equity');el('drawdown').classList.toggle('on',chartMode==='Drawdown');paint()});}
async function simulation(ticket){
 const [study,sim]=await Promise.all([load('study-'+horizon),load('simulation')]);if(ticket!==token)return;
 const custom=mode==='Custom';if(index==null)index=study.dates.length-1;index=Math.min(index,study.dates.length-1);
 const customFields=[['fee','Cost per side',fee,0,30,1,' bps'],['cap','Position cap',cap,1,25,1,'%'],['threshold','Signal threshold',threshold,0,35,1,'%']].map(([id,title,v,l,h,s,u])=>'<label><span class="rangeval">'+title+'<output id="'+id+'val">'+v+u+'</output></span><input id="'+id+'" type="range" min="'+l+'" max="'+h+'" step="'+s+'" value="'+v+'"></label>').join('');
 root.innerHTML=head('Portfolio replay','Follow a simulated portfolio through 2023–2025. It uses forecasts made after the market closes and trades at the next opening price.','<button id="export">'+icon('download')+'Export</button>')+
 '<div class="grid"><section class="plotarea"><div class="metrics">'+metric('Portfolio value','value')+metric('Total return','return')+metric('Largest fall from a peak','dd')+'</div><div class="tabrow">'+plotTabs()+'<span class="tagtext">USD</span></div><canvas class="chart" id="mainChart" role="img" aria-label="Portfolio and benchmark values over the selected sessions"></canvas><div id="legend"></div>'+timeline(study.dates.length)+'</section><aside class="settings"><h2>Simulation settings</h2>'+modeButtons()+
 (custom?field('capital','Starting capital','<input id="capital" type="number" step="1000" min="100" max="10000000" value="'+capital+'">')+customFields:
 field('strategy','Portfolio',select('strategy',study.models,model))+field('horizon','Forecast horizon',select('horizon',['5 sessions','1 session'],horizon+' session'+(horizon===5?'s':''))))+
 field('benchmark','Compare with',select('benchmark',[...(custom?[]:['Exposure-matched SPY']),'SPY buy and hold','Equal weight','Uniform vote','Cash','21-session momentum'],bench))+
 '<p class="note">'+(custom?'Five-session gate forecasts. Changes recalculate holdings, costs and cash at each opening.':'Each annual test starts and ends in cash. Buying or selling $1,000 costs 50 cents.')+'</p></aside></div>'+
 '<section class="bottom"><div class="sectionhead"><h2>'+(custom?'Positions at the next open':'Benchmark comparison')+'</h2><span class="tagtext" id="subdate"></span></div><div id="records"></div></section>';
 let calculated;let curve,benchmark,strategylab;
 function recalc(){calculated=custom?simulate(sim,{capital,fee,cap,threshold}):null;curve=custom?calculated.curve:study.equity[model];benchmark=study.equity[bench].map(v=>v*(custom?capital:10000)/10000);strategylab=custom?'Custom gate policy':model;el('legend').innerHTML=legend([[strategylab,colors[0]],[bench,colors[1]]]);}
 recalc();
 function update(){let initial=custom?capital:10000;el('value').textContent=dollars.format(curve[index]);el('return').textContent=pct(curve[index]/initial-1);el('dd').textContent=pct(dd(curve.slice(0,index+1),initial));paint();
 if(custom){el('subdate').textContent=displayDate(sim.entryDates[index]);const pos=calculated.positions[index];el('records').innerHTML=table(['Instrument','Target weight','Adjusted units'],[...pos.map(p=>[p.symbol,pct(p.weight,2),num(p.units,3)]),['Cash',pct(1-pos.reduce((s,p)=>s+p.weight,0),2),'—']]);}
 else{el('subdate').textContent='Full evaluation period';}}
 paint=()=>drawChart(el('mainChart'),study.marks,[{values:curve,color:colors[0],start:custom?capital:10000},{values:benchmark,color:colors[1],start:custom?capital:10000}],index,chartMode==='Drawdown');
 if(!custom){const names=[...new Set([model,bench,'Uniform vote','SPY buy and hold','Equal weight','Cash'])];el('records').innerHTML=table(['Portfolio','Net return','Max drawdown','Sharpe','Exposure'],names.map(n=>{const r=study.summary.find(x=>x.model===n);return [n,pct(r.net_return,2),pct(r.max_drawdown,2),num(r.sharpe),pct(r.mean_exposure)]}));}
 onStep=update;bindTimeline(study.marks);bindChart(study.marks);tabsChart();
 on('custom','click',()=>{mode='Custom';horizon=5;if(bench==='Exposure-matched SPY')bench='SPY buy and hold';render()});on('recorded','click',()=>{mode='Recorded';render()});on('strategy','change',e=>{model=e.target.value;render()});on('horizon','change',e=>{horizon=parseInt(e.target.value);index=null;render()});on('benchmark','change',e=>{bench=e.target.value;render()});
 for(const id of ['fee','cap','threshold','capital'])on(id,id==='capital'?'change':'input',e=>{let v=+e.target.value;if(id==='fee')fee=v;if(id==='cap')cap=v;if(id==='threshold')threshold=v;if(id==='capital'){capital=Math.max(100,Math.min(1e7,v||10000));e.target.value=capital;}if(el(id+'val'))el(id+'val').textContent=v+(id==='fee'?' bps':'%');recalc();update()});
 on('export','click',()=>saveCSV('portfolio-replay.csv',[['mark',strategylab,bench],...study.marks.slice(0,index+1).map((d,i)=>[d,curve[i],benchmark[i]])]));
}
const voteSelection={date:null,symbol:null,pool:'sequence networks'};
const voteOptions = Object.entries(votingMethods).map(([key,[name]])=>[key,name]);
const votePicker = (id,label,pairs,value)=>field(id,label,'<select id="'+id+'">'+
 pairs.map(([key,name])=>'<option value="'+esc(key)+'"'+(key===value?' selected':'')+'>'+esc(name)+'</option>').join('')+'</select>');
function bindVoteMethod(){
 on('voteMethod','change',event=>{
   const selected=event.target.value;
   location.hash=selected==='original'?'controller':'controller/'+selected;
 });
}
async function controller(ticket, method){
 if(method!=='original'){
   await inspectVotes(root,method,{load,active:()=>ticket===token,head,picker:votePicker,pct,escape:esc,selection:voteSelection});
   if(ticket===token)bindVoteMethod();
   return;
 }
 const sim=await load('simulation');if(ticket!==token)return;index=Math.min(index??sim.dates.length-1,sim.dates.length-1);
 root.innerHTML=head(methodLabels.original,'This is one recorded decision from the original mixture of experts with its scheduled annual refit.')+
 '<p class="note comparison-guide"><a class="link" href="assets/ensemble-study.pdf#page=5">Apple example in the report</a> · <a class="link" href="#comparison/pipeline">See this mixture in the results chart</a></p>'+
 '<div class="controls vote-controls">'+votePicker('voteMethod','Voting method',voteOptions,'original')+field('ticker','Instrument',select('ticker',sim.symbols,asset))+
 field('when','Decision date',select('when',sim.dates,sim.dates[index]))+'</div>'+
 '<ol class="model-flow adapter-flow" aria-label="This decision in the original pipeline"><li><strong>13 forecasts</strong><ul><li>Each predicts the stock’s direction</li></ul></li><li><strong>Mixture of experts</strong><ul><li>Combine them by vote share</li></ul></li><li><strong>Trading rule</strong><ul><li>Set the next-open share of the account</li></ul></li></ol>'+

 '<div class="controllergrid vote-grid"><section class="sheet"><h2>Combined forecast</h2><div id="classification"></div>'+
 '<hr class="rule"><h3>Next-open target</h3><p class="note" id="targetDecision"></p>'+
 '<details class="protocol vote-details"><summary>Market conditions the vote counter used</summary><div id="states"></div>'+
 '<p class="note">The vote counter also uses earlier model accuracy. Forecast outcomes enter its history after the forecast window closes.</p></details></section>'+
 '<section class="sheet"><h2>How much each forecast counts</h2><div id="bars"></div><p class="note" id="voteCalculation"></p></section></div>'+
 '<details class="protocol bottom vote-details"><summary>All forecasts and what each added to the chance of a rise</summary><div id="expertTable"></div></details>';
 const fixedTargets=simulate(sim).positions;
 let revision=0;
 async function show(){const n=++revision;const date=sim.dates[index],detail=await load('detail-'+date.slice(0,4));if(n!==revision||ticket!==token)return;
 const t=detail.dates.indexOf(date),j=sim.symbols.indexOf(asset),w=detail.weights[t][j],p=detail.probabilities[t][j];
 const ranks=detail.experts.map((name,k)=>({name,k,w:w[k]})).sort((a,b)=>b.w-a.w);
 el('bars').innerHTML=ranks.map(x=>'<div class="weightrow"><span>'+esc(x.name)+'</span><span class="track"><i style="width:'+100*x.w+'%"></i></span><strong>'+pct(x.w,1)+'</strong></div>').join('');
 const q=[sim.down[index][j],sim.flat[index][j],sim.up[index][j]],labs=['Fall','Little change','Rise'],cs=['#ac7744','#a3aba1','#197257'];
 el('classification').innerHTML='<div class="probs">'+q.map((s,k)=>'<i style="width:'+100*s+'%;background:'+cs[k]+'"></i>').join('')+'</div><div class="probkey">'+labs.map((l,k)=>'<span>'+l+' '+pct(q[k],1)+'</span>').join('')+'</div>';
 const top = ranks[0];
 const upContribution = (100*top.w*p[top.k][2]).toFixed(2);
 el('voteCalculation').textContent=top.name+' had '+pct(top.w,2)+' of the vote and predicted '+pct(p[top.k][2],2)+
   ' up. Multiplying them contributes '+upContribution+' percentage points to the combined '+pct(q[2],1)+' up forecast.';
 const target=fixedTargets[index].find(position=>position.symbol===asset);
 const score=q[2]-q[0];
 const points=(100*score).toFixed(1);
 const higher=sim.up[index].filter((up,k)=>{
   const other=up-sim.down[index][k];
   return other>score || (other===score && k<j);
 }).length;
 const decision=target ? 'The fixed rule targeted '+pct(target.weight,1)+' of the account at the next open.'
  : score>.05 ? 'The target was 0%. It cleared the minimum but ranked '+(higher+1)+' of '+sim.symbols.length+' by score. The rule selects only the five highest scores.'
  : 'The fixed rule set the next-open target to zero because the score did not exceed five percentage points.';
 el('targetDecision').textContent=asset+' scored '+points+' percentage points after subtracting its down probability from its up probability. '+decision;
 el('states').innerHTML=table(['State','Probability'],sim.posterior[index].map((v,k)=>[['Lower mean return','Middle mean return','Higher mean return'][k],pct(v,2)]));
 el('expertTable').innerHTML=table(['Forecaster','Vote','Down','Little change','Up','Adds to up (points)'],ranks.map(x=>[x.name,pct(x.w,2),...p[x.k].map(v=>pct(v,2)),(100*x.w*p[x.k][2]).toFixed(2)]));
 }
 bindVoteMethod();
 on('ticker','change',e=>{asset=e.target.value;show()});on('when','change',e=>{index=sim.dates.indexOf(e.target.value);show()});await show();
}
let stressPeriod='all';
const scenarios={original:'Recorded forecasts',language_outage:'Language experts offline',reversed_tree:'Reversed tree forecasts'};
async function stress(ticket){
 const data=await load('stress');if(ticket!==token)return;
 const all=data.summary.filter(x=>x.scenario===scenario&&x.period===stressPeriod);
 const marketName='SPY buy and hold';
 const market=data.summary.find(r=>r.scenario==='original'&&r.period===stressPeriod&&r.model===marketName);
 if(!all.some(r=>r.model===marketName))all.push(market);
 const dates=data.mark.filter((_,i)=>stressPeriod==='all'||data.decision[i].startsWith(stressPeriod));
 const ids=data.decision.map((d,i)=>stressPeriod==='all'||d.startsWith(stressPeriod)?i:-1).filter(i=>i>=0);
 const methods=[
  ['Adaptive router','Weights updated using recent forecast errors',colors[0]],
  ['HMM gate','Mixture of experts using market states',colors[1]],
  ['Uniform vote','Equal votes for the same forecasts',colors[2]],
  [marketName,'S&P 500 buy and hold','#a05251']
 ];
 const labels=new Map(methods.map(([key,label])=>[key,label]));
 const series=methods.map(([key,name,color])=>{
  const full=data.scenarios[scenario][key]??data.scenarios.original[key];
  const base=ids[0]===0?10000:full[ids[0]-1];
  return {name,color,values:ids.map(i=>full[i]/base*10000)};
 });
 const titles={
  original:'The three voting rules use the same 13 forecasts.',
  language_outage:'Each July, eight forecasts remain. The test removes FinBERT logistic, OpenAI Decisions, OpenAI asset digest, Qwen statement and Granite statement, then rescales the remaining weights to 100%.',
  reversed_tree:'All 13 forecasts remain. Each July, XGBoost and LightGBM swap their rise and fall probabilities. Their little-change probabilities stay the same.'
 };
 root.innerHTML=head('What if a forecast goes wrong?','Across 959 trading days in 2022–2025, these simulations compare what happened when news forecasts disappeared or two price models reversed their predictions.')+
 '<div class="controls">'+field('scenario','Scenario',select('scenario',Object.values(scenarios),scenarios[scenario]))+
 field('period','Evaluation window',select('period',['All years','2022','2023','2024','2025'],stressPeriod==='all'?'All years':stressPeriod))+'</div>'+
 '<div class="sheet"><div class="metrics">'+metric('Adaptive router return','routerReturn')+metric('Uniform vote return','voteReturn')+metric('Router drawdown','routerDD')+'</div><canvas class="chart" id="stressChart" role="img" aria-label="Return paths under the selected market stress"></canvas>'+
 legend(series.map(s=>[s.name,s.color]))+'<p class="note">'+titles[scenario]+'</p><p class="note" id="stressReading"></p><details class="protocol pipeline-universe"><summary>Why compare with the market-state gate?</summary><p class="note">It is the controller from the original proposal. Both controllers see the same forecasts and hidden Markov model estimates of market conditions. The original gate learned how to combine those inputs. The adaptive router updates voting shares using recent forecast errors in similar market conditions. Equal votes show how the same forecast set did with no learned voting rule. To test the market-state input itself, compare HMM gate with Gate without HMM in Compare results.</p></details><p class="note"><a class="link" href="assets/ensemble-study.pdf#page=8">Read the stress-test findings in the report</a></p></div><section class="bottom"><div class="sectionhead"><h2>Portfolio outcomes</h2></div>'+
 table(['Portfolio','Net return','Max drawdown','Sharpe','Log loss'],all.map(r=>[labels.get(r.model)??r.model,pct(r.net_return,2),pct(r.max_drawdown,2),num(r.sharpe),num(r.log_loss,4)]))+'</section><section class="bottom"><div class="sectionhead"><h2>Days around regime changes</h2></div>'+
 table(['Controller','Sessions','Mean daily return','5th percentile day','Log loss'],data.conditional.filter(x=>x.scenario===scenario&&x.group==='transition').map(r=>[r.model,r.sessions,pct(r.mean_net_daily_return,2),pct(r.daily_return_p05,2),num(r.log_loss,4)]))+
 '<p class="note">The transition table spans all four years and includes the five observed sessions beginning at each state change. 2022 was selected as a known bear market. Results describe historical resilience. The block-bootstrap intervals for the unaltered forecasts include zero.</p></section>';
 const a=all.find(x=>x.model==='Adaptive router'),v=all.find(x=>x.model==='Uniform vote');el('routerReturn').textContent=pct(a.net_return,2);el('voteReturn').textContent=pct(v.net_return,2);el('routerDD').textContent=pct(a.max_drawdown,2);paint=()=>drawChart(el('stressChart'),dates,series,dates.length-1,false);paint();
 const gate=all.find(r=>r.model==='HMM gate');
 el('stressReading').textContent='After costs, adaptive votes earned '+pct(a.net_return,2)+
   ', market-state votes earned '+pct(gate.net_return,2)+', and equal votes earned '+pct(v.net_return,2)+
   '. Holding the market fund returned '+pct(market.net_return,2)+' on the same dates.';
 on('scenario','change',e=>{scenario=Object.keys(scenarios).find(k=>scenarios[k]===e.target.value);render()});on('period','change',e=>{stressPeriod=e.target.value==='All years'?'all':e.target.value;render()});
}
async function architecture(ticket){
 const [sim,study]=await Promise.all([load('simulation'),load('study-5')]);if(ticket!==token)return;index=Math.min(index??sim.dates.length-1,sim.dates.length-1);
 root.innerHTML=head('System architecture','The forecast controller decides whose prediction to use. The portfolio decision turns the result into something the simulator can buy, sell or hold.')+'<div class="controls">'+field('architectureDate','Decision date',select('architectureDate',sim.dates,sim.dates[index]))+'</div><div id="flow"></div>';
 const refresh=()=>renderArchitecture(el('flow'),study,sim,index);on('architectureDate','change',e=>{index=sim.dates.indexOf(e.target.value);refresh()});refresh();
}
async function datasets(ticket){
 const [study,evidence]=await Promise.all([load('study-5'),load('source-evidence')]);if(ticket!==token)return;const d=study.coverage;
 root.innerHTML=head('Data sources','January 2017 to December 2025')+
 '<section><h2>Quantitative data</h2><div class="sourcesgrid source-charts">'+
 '<figure><figcaption>Yahoo Finance adjusted closing prices</figcaption><img class="sourceplot" src="assets/yahoo-finance-prices.png" alt="Weekly adjusted closing prices for five funds, indexed to 100 at the start of 2017"><p class="note"><span id="bars"></span> daily records across <span id="symbols"></span> instruments. The chart shows five funds as examples.</p></figure>'+
 '<figure><figcaption>Jensen, Kelly and Pedersen investment factors</figcaption><img class="sourceplot" src="assets/investment-factor-returns.png" alt="Annual returns by calendar year for 13 investment themes"><p class="note"><span id="themes"></span> market-wide themes. The models use trailing returns with a 21-session delay.</p></figure></div></section>'+
 '<section class="bottom"><h2>Qualitative data</h2><div class="sourcesgrid source-charts">'+
 '<figure><figcaption>Federal Reserve statement tone</figcaption><img class="sourceplot" src="assets/federal-reserve-text-sentiment.png" alt="Positive minus negative scores from FinBERT for Federal Reserve statements"></figure>'+
 '<div id="sourceEvidence"></div></div></section>'+
 '<details class="bottom protocol"><summary>Full source coverage</summary><div id="coverageTable"></div><p class="note">The selected universe includes surviving companies and overlapping funds. Model pretraining can include events from this historical evaluation.</p></details>'+
 '<p class="note"><a class="link" href="assets/ensemble-study.pdf#page=2">Quantitative and qualitative charts in the report</a></p>';
 sourceGuide(el('sourceEvidence'),evidence);
 el('coverageTable').append(coverageTable(d));
 el('bars').textContent=d.bars.toLocaleString();el('symbols').textContent=d.symbols;el('themes').textContent=d.factorThemes.length;
}
async function comparison(ticket, method, selection){
 const [data,paths,example]=await Promise.all([load("periods"),load("period-paths"),load("forecast-example")]);
 if(ticket===token){comparisonView(root,data,paths,{head,saveCSV,icon,load,example,proposal:selection==='pipeline'});paint=()=>root.oncomparisonresize?.();}
}
function training(){
 root.innerHTML=head('Training and forecast accuracy','Saved measurements from the Colab experiments')+
 '<div class="sourcesgrid source-charts"><figure><figcaption>Jointly trained mixture of experts</figcaption>'+
 '<img class="sourceplot" src="assets/joint-mixture-training.png" alt="Training objective falls across twelve epochs for the 2024 and 2025 models">'+
 '<p class="note">Training cutoff for the 2024 model was 29 December 2023. For the 2025 model it was 31 December 2024.</p></figure>'+
 '<figure><figcaption>Forecast accuracy on held-out dates</figcaption>'+
 '<img class="sourceplot" src="assets/forecast-comparison-held-out.png" alt="Held-out log loss for three voting methods across three groups of forecasters">'+
 '<p class="note">Lower log loss means the model assigned more probability to what happened. The portfolios use the same trading rule across these comparisons.</p></figure></div>'+
 '<p class="note"><a class="link" href="#comparison">See portfolio performance</a> · <a class="link" href="assets/ensemble-study.pdf#page=7">Training charts in the report</a></p>';
}
const areas={
 study:[['overview','Summary'],['pipeline','Pipeline'],['architecture','System diagram']],
 models:[['models','Model inputs and outputs'],['controller','Model votes']],
 results:[['comparison','Compare results'],['reviews','Research-agent review'],['training','Training charts'],['simulation','Portfolio replay'],['stress','Stress tests']],
 data:[['data','Data sources'],['glossary','Glossary']],
};

function renderNavigation(route, method) {
 const area=Object.keys(areas).find(key=>areas[key].some(([id])=>id===route)) ?? 'study';
 for(const link of document.querySelectorAll('.section-nav a')){
   const current=link.dataset.area===area;
   link.classList.toggle('active',current);
   if(current) link.setAttribute('aria-current','true'); else link.removeAttribute('aria-current');
 }
 const pages=areas[area].map(([id,label])=>[id,label]);
 if(area==='models') pages.push(['controller/adapter',methodLabels.adapter]);
 const selected=method==='adapter'?'controller/adapter':route;
 el('subnav').innerHTML=pages.map(([id,label])=>'<a href="#'+id+'"'+(id===selected?' class="active" aria-current="page"':'')+'>'+label+'</a>').join('');
}

async function render(){
 const ticket=++token;stop();paint=()=>{};onStep=()=>{};
 const [route,selection]=(location.hash.slice(1)||'overview').replace(/^datasets$/, 'data').split('/');
 const method=route==='controller'&&Object.hasOwn(votingMethods,selection)?selection:'original';
 const views={overview:async t=>{const data=await load('periods');if(t===token)overview(root,data);},pipeline:async t=>{const [guide,study,data]=await Promise.all([load('pipeline-guide'),load('study-5'),load('periods')]);if(t===token)pipelineView(root,guide,study.coverage,data,head);},models:async t=>{const guide=await load('model-map');if(t===token)modelMap(root,guide,head);},reviews:async t=>{const [periods,five,one]=await Promise.all([load('periods'),load('study-5'),load('study-1')]);if(t===token)reviewResults(root,periods,{five,one},{head,table,pct});},training,simulation,comparison,controller,stress,architecture,data:datasets,glossary:()=>glossaryView(root)};
 renderNavigation(route,method);
 try{await (Object.hasOwn(views,route)?views[route]:views.pipeline)(ticket,method,selection)}catch(e){if(ticket===token)root.innerHTML='<h1 class="bottom">Study unavailable</h1><p class="note">'+esc(e.message)+'</p><button class="bottom" id="retry">Retry</button>';on('retry','click',render);}
}
window.addEventListener('resize',()=>paint());window.addEventListener('hashchange',render);render();
