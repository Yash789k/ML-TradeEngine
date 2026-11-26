// Match the study's fractional, long-only next-open accounting and annual exits.
export function simulate(data,{capital=10000,fee=5,cap=15,threshold=5}={}){
  let equity=capital,cash=capital,shares=Array(data.symbols.length).fill(0);
  const curve=[],positions=[],cost=fee/10000;
  for(let i=0;i<data.dates.length;i++){
    const prices=data.entry[i],marks=data.exit[i];
    const invested=shares.map((q,j)=>q*prices[j]);
    const nav=cash+invested.reduce((a,b)=>a+b,0);
    const candidates=data.symbols.map((_,j)=>({j,signal:data.up[i][j]-data.down[i][j]}))
      .filter(x=>x.signal>threshold/100).sort((a,b)=>b.signal-a.signal||a.j-b.j).slice(0,5);
    const inverse=candidates.reduce((sum,x)=>sum+1/data.vol[i][x.j],0);
    const weights=shares.map(()=>0);
    for(const {j} of candidates)weights[j]=Math.min(cap/100,1/data.vol[i][j]/inverse);
    let after=nav;
    for(let k=0;k<6;k++)after=nav-cost*weights.reduce((sum,w,j)=>sum+Math.abs(w*after-invested[j]),0);
    const fills=weights.map((w,j)=>w*after-invested[j]);
    cash-=fills.reduce((sum,v)=>sum+v+cost*Math.abs(v),0);
    shares=shares.map((q,j)=>q+fills[j]/prices[j]);
    positions.push(weights.map((w,j)=>({symbol:data.symbols[j],weight:w,units:shares[j]})).filter(x=>x.weight>0));
    const marked=shares.reduce((sum,q,j)=>sum+q*marks[j],0);
    equity=cash+marked;
    if(i===data.dates.length-1||data.dates[i+1].slice(0,4)!==data.dates[i].slice(0,4)){
      cash+=marked*(1-cost);equity=cash;shares.fill(0);
    }
    curve.push(equity);
  }
  return {curve,positions};
}

export function drawChart(canvas, dates, series, day, drawdown=false){
  const box=canvas.getBoundingClientRect();if(!box.width)return;
  const dpr=window.devicePixelRatio||1, w=box.width,h=box.height;
  canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
  const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);
  const left=w<500?42:55,right=10,top=12,bottom=28,ow=w-left-right,oh=h-top-bottom;
  const all=series.map(s=>{
    let peak=s.start||10000;
    return s.values.map(v=>{peak=Math.max(peak,v);return drawdown?(v/peak-1)*100:v});
  });
  const flat=all.flat();let lo=Math.min(...flat),hi=Math.max(...flat);
  if(drawdown)hi=0;else {lo=Math.min(lo,series[0].start||10000);hi=Math.max(hi,series[0].start||10000);}
  const spread=hi-lo||Math.max(1,hi*.05);lo-=spread*.06;hi+=spread*.05;
  const x=i=>left+i*ow/Math.max(1,dates.length-1),y=v=>top+(hi-v)*oh/(hi-lo);
  ctx.font='12px system-ui';ctx.fillStyle='#52685a';ctx.lineWidth=1;
  for(let k=0;k<5;k++){
    const v=lo+k*(hi-lo)/4,yy=y(v);ctx.strokeStyle='#e3e9e3';ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(w-right,yy);ctx.stroke();
    ctx.textAlign='right';ctx.fillText(drawdown?v.toFixed(0)+'%':'$'+(v/1000).toFixed(1)+'k',left-7,yy+4);
  }
  const count=w<500?3:5;
  for(let k=0;k<count;k++){const i=Math.round(k*(dates.length-1)/(count-1));ctx.textAlign=k===0?'left':k===count-1?'right':'center';ctx.fillText(dates[i].slice(0,7),x(i),h-5);}
  for(let j=all.length-1;j>=0;j--){
    const values=all[j].slice(0,day+1);if(!values.length)continue;
    ctx.beginPath();values.forEach((v,i)=>i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v)));ctx.strokeStyle=series[j].color;ctx.lineWidth=j===0?2.8:1.8;ctx.lineJoin='round';ctx.stroke();
    if(j===0&&!drawdown){ctx.lineTo(x(day),y(lo));ctx.lineTo(left,y(lo));ctx.closePath();const g=ctx.createLinearGradient(0,top,0,h-bottom);g.addColorStop(0,'rgba(63,147,108,.17)');g.addColorStop(1,'rgba(63,147,108,0)');ctx.fillStyle=g;ctx.fill();}
    ctx.beginPath();ctx.arc(x(day),y(values[values.length-1]),3.3,0,Math.PI*2);ctx.fillStyle=series[j].color;ctx.fill();
  }
  if(day<dates.length-1){ctx.strokeStyle='#8b9b92';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(day),top);ctx.lineTo(x(day),h-bottom);ctx.stroke();}
}
