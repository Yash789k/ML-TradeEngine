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
async function controller(ticket){
 const sim=await load('simulation');if(ticket!==token)return;index=Math.min(index??sim.dates.length-1,sim.dates.length-1);
 root.innerHTML=head('Who gets a vote?','For one stock and date, each model predicts a rise, a fall or little change over five trading days. A larger weight gives its prediction more influence on the final forecast.')+
 '<div class="controls">'+field('ticker','Instrument',select('ticker',sim.symbols,asset))+field('when','Decision date',select('when',sim.dates,sim.dates[index]))+'</div><div class="controllergrid"><section class="sheet"><h2>Expert weights</h2><div id="bars"></div></section><section class="sheet"><h2>Combined forecast</h2><div id="classification"></div><hr class="rule"><h3>Market state probabilities</h3><div id="states"></div><hr class="rule"><p class="note">Weights use filtered market conditions and earlier expert accuracy. Outcomes enter memory only after their forecast window closes.</p></section></div><section class="bottom"><div class="sectionhead"><h2>Forecast contributions</h2></div><div id="expertTable"></div></section>';
 let revision=0;
 async function show(){const n=++revision;const date=sim.dates[index],detail=await load('detail-'+date.slice(0,4));if(n!==revision||ticket!==token)return;
 const t=detail.dates.indexOf(date),j=sim.symbols.indexOf(asset),w=detail.weights[t][j],p=detail.probabilities[t][j];
 const ranks=detail.experts.map((name,k)=>({name,k,w:w[k]})).sort((a,b)=>b.w-a.w);
 el('bars').innerHTML=ranks.map(x=>'<div class="weightrow"><span>'+esc(x.name)+'</span><span class="track"><i style="width:'+Math.min(100,100*x.w/Math.max(...w))+'%"></i></span><strong>'+pct(x.w,1)+'</strong></div>').join('');
 const q=[sim.down[index][j],sim.flat[index][j],sim.up[index][j]],labs=['DOWN','FLAT','UP'],cs=['#ac7744','#a3aba1','#197257'];
 el('classification').innerHTML='<div class="probs">'+q.map((s,k)=>'<i style="width:'+100*s+'%;background:'+cs[k]+'"></i>').join('')+'</div><div class="probkey">'+labs.map((l,k)=>'<span>'+l+' '+pct(q[k],1)+'</span>').join('')+'</div>';
 el('states').innerHTML=table(['State','Probability'],sim.posterior[index].map((v,k)=>[['Lower mean return','Middle mean return','Higher mean return'][k],pct(v,2)]));
 el('expertTable').innerHTML=table(['Expert','Weight','DOWN','FLAT','UP'],ranks.map(x=>[x.name,pct(x.w,2),...p[x.k].map(v=>pct(v,2))]));
 }
 on('ticker','change',e=>{asset=e.target.value;show()});on('when','change',e=>{index=sim.dates.indexOf(e.target.value);show()});await show();
}
let stressPeriod='all';
const scenarios={original:'Recorded forecasts',language_outage:'Language experts offline',reversed_tree:'Reversed tree forecasts'};
async function stress(ticket){
 const data=await load('stress');if(ticket!==token)return;
 const all=data.summary.filter(x=>x.scenario===scenario&&x.period===stressPeriod);
 const dates=data.mark.filter((_,i)=>stressPeriod==='all'||data.decision[i].startsWith(stressPeriod));
 const ids=data.decision.map((d,i)=>stressPeriod==='all'||d.startsWith(stressPeriod)?i:-1).filter(i=>i>=0);
 const methods=['Adaptive router','HMM gate','Uniform vote'];
 const series=methods.map((name,j)=>{const full=data.scenarios[scenario][name],base=ids[0]===0?10000:full[ids[0]-1];return {name,color:colors[j],values:ids.map(i=>full[i]/base*10000)}});
 const titles={original:'All 13 experts available.',language_outage:'Each July, the five text and language forecasts become unavailable. Remaining weights renormalize.',reversed_tree:'Each July, both boosted-tree experts exchange their UP and DOWN probabilities.'};
 root.innerHTML=head('What if a forecast goes wrong?','Across 959 trading days in 2022–2025, these simulations compare what happened when news forecasts disappeared or two price models reversed their predictions.')+
 '<div class="controls">'+field('scenario','Scenario',select('scenario',Object.values(scenarios),scenarios[scenario]))+
 field('period','Evaluation window',select('period',['All years','2022','2023','2024','2025'],stressPeriod==='all'?'All years':stressPeriod))+'</div>'+
 '<div class="sheet"><div class="metrics">'+metric('Adaptive router return','routerReturn')+metric('Uniform vote return','voteReturn')+metric('Router drawdown','routerDD')+'</div><canvas class="chart" id="stressChart" role="img" aria-label="Return paths under the selected market stress"></canvas>'+
 legend(series.map(s=>[s.name,s.color]))+'<p class="note">'+titles[scenario]+'</p></div><section class="bottom"><div class="sectionhead"><h2>Portfolio outcomes</h2></div>'+
 table(['Portfolio','Net return','Max drawdown','Sharpe','Log loss'],all.map(r=>[r.model,pct(r.net_return,2),pct(r.max_drawdown,2),num(r.sharpe),num(r.log_loss,4)]))+'</section><section class="bottom"><div class="sectionhead"><h2>Days around regime changes</h2></div>'+
 table(['Controller','Sessions','Mean daily return','5th percentile day','Log loss'],data.conditional.filter(x=>x.scenario===scenario&&x.group==='transition').map(r=>[r.model,r.sessions,pct(r.mean_net_daily_return,2),pct(r.daily_return_p05,2),num(r.log_loss,4)]))+
 '<p class="note">The transition table spans all four years and includes the five observed sessions beginning at each state change. 2022 was selected as a known bear market. Results describe historical resilience. The block-bootstrap intervals for the unaltered forecasts include zero.</p></section>';
 const a=all.find(x=>x.model==='Adaptive router'),v=all.find(x=>x.model==='Uniform vote');el('routerReturn').textContent=pct(a.net_return,2);el('voteReturn').textContent=pct(v.net_return,2);el('routerDD').textContent=pct(a.max_drawdown,2);paint=()=>drawChart(el('stressChart'),dates,series,dates.length-1,false);paint();
 on('scenario','change',e=>{scenario=Object.keys(scenarios).find(k=>scenarios[k]===e.target.value);render()});on('period','change',e=>{stressPeriod=e.target.value==='All years'?'all':e.target.value;render()});
}
async function architecture(ticket){
 const [sim,study]=await Promise.all([load('simulation'),load('study-5')]);if(ticket!==token)return;index=Math.min(index??sim.dates.length-1,sim.dates.length-1);
 root.innerHTML=head('System architecture','Follow one decision from published prices and news to a forecast, then to a simulated trade at the next market opening.')+'<div class="controls">'+field('architectureDate','Decision date',select('architectureDate',sim.dates,sim.dates[index]))+'</div><div id="flow"></div>';
 const refresh=()=>renderArchitecture(el('flow'),study,sim,index);on('architectureDate','change',e=>{index=sim.dates.indexOf(e.target.value);refresh()});refresh();
}
async function datasets(ticket){
 const study=await load('study-5');if(ticket!==token)return;const d=study.coverage;
 root.innerHTML=head('Research datasets','The models learned from prices and trading volume, published investment signals, company reports and financial news. The records span January 2017 to December 2025.')+
 '<div class="sheet"><div class="metrics">'+metric('Instrument-days','bars')+metric('Instruments','symbols')+metric('Factor themes','themes')+'</div></div><div class="sourcesgrid bottom"><img class="sourceplot" src="assets/datasets.png" alt="Adjusted prices of five benchmark exchange-traded funds and FinBERT sentiment for Federal Reserve statements"><img class="sourceplot" src="assets/sources.png" alt="Text counts and returns for thirteen US factor themes by year"></div><section class="bottom"><div class="sectionhead"><h2>Coverage</h2></div>'+
 '<div id="coverageTable"></div>'+
 '<p class="note">The selected universe includes surviving companies and overlapping ETFs. The headline sample covers 2019 through 2023. Model pretraining can include events that occurred during this retrospective evaluation.</p></section><section class="bottom"><a class="link" href="https://github.com/Yash789k/ML-TradeEngine/blob/main/ceci2_demo/README.md">Data sources and training protocol</a></section>';
 el('coverageTable').append(coverageTable(d));
 el('bars').textContent=d.bars.toLocaleString();el('symbols').textContent=d.symbols;el('themes').textContent=d.factorThemes.length;
}
async function comparison(ticket){
 const [data,paths]=await Promise.all([load("periods"),load("period-paths")]);
 if(ticket===token){comparisonView(root,data,paths,{head,saveCSV,icon,load});paint=()=>root.oncomparisonresize?.();}
}
async function render(){
 const ticket=++token;stop();paint=()=>{};onStep=()=>{};
 const route=(location.hash.slice(1)||'overview').replace(/^datasets$/, 'data'),views={overview:async t=>{const data=await load('periods');if(t===token)overview(root,data);},simulation,comparison,controller,stress,architecture,data:datasets,glossary:()=>glossaryView(root)};
 document.querySelectorAll('nav a').forEach(a=>{const yes=a.hash==='#'+route;a.classList.toggle('active',yes);if(yes)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')});
 try{await (views[route]||views.overview)(ticket)}catch(e){if(ticket===token)root.innerHTML='<h1 class="bottom">Study unavailable</h1><p class="note">'+esc(e.message)+'</p><button class="bottom" id="retry">Retry</button>';on('retry','click',render);}
}
window.addEventListener('resize',()=>paint());window.addEventListener('hashchange',render);render();
