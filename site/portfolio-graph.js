// Each portfolio compounds from the same starting value for the selected window.
export function plotPortfolios(host, group, period, approach, benchmark) {
  const cols = group.dates.map((date,i)=>date>=period.start && date<=period.end ? i : -1).filter(i=>i>=0);
  const traces = period.rows.map(({model},i) => {
    let equity=10000;
    const values=[equity, ...cols.map(j=>equity *= 1+group.returns[model][j])];
    const color = model===approach?'#197257':model===benchmark?'#ac7744':`hsl(${(i*137.5)%360} 38% 42%)`;
    return {model,values,color};
  });
  const canvas = document.createElement('canvas');
  canvas.className='chart';canvas.setAttribute('role','img');
  canvas.setAttribute('aria-label',`Equity curves for all ${traces.length} approaches with data from ${period.start} through ${period.end}`);
  const caption=document.createElement('p');caption.className='note';
  const emptyCaption = `${traces.length} portfolios · $10,000 at the start of the window`;
  caption.textContent=emptyCaption;
  host.replaceChildren(canvas,caption);
  const size=() => {
    const {width:w,height:h}=canvas.getBoundingClientRect();
    const left=w<500?46:62,top=12,bottom=h-30,right=w-12;
    const low=Math.min(...traces.map(t=>Math.min(...t.values)))*.985;
    const high=Math.max(...traces.map(t=>Math.max(...t.values)))*1.015;
    return {w,h,left,right,top,bottom,x:i=>left+i*(right-left)/cols.length,
      y:v=>bottom-(v-low)/(high-low)*(bottom-top),low,high};
  };
  let hover=null,hoverIndex=null;
  function draw() {
    if(!canvas.isConnected)return;
    const g=size(),ratio=window.devicePixelRatio||1;
    canvas.width=Math.round(g.w*ratio);canvas.height=Math.round(g.h*ratio);
    const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.font='12px system-ui';
    for(let i=0;i<=4;i++){
      const val=g.low+(g.high-g.low)*i/4,y=g.y(val);ctx.strokeStyle='#e0e6e1';
      ctx.beginPath();ctx.moveTo(g.left,y);ctx.lineTo(g.right,y);ctx.stroke();
      ctx.fillStyle='#51675a';ctx.textAlign='right';ctx.fillText('$'+(val/1000).toFixed(1)+'k',g.left-5,y+4);
    }
    const order=t=>t.model===hover?3:t.model===approach?2:t.model===benchmark?1:0;
    for(const t of [...traces].sort((a,b)=>order(a)-order(b))){
      const rank=order(t);ctx.globalAlpha=hover ? (t.model===hover?1:.28) : rank ? 1 : .62;
      ctx.strokeStyle=t.color;ctx.lineWidth=rank?2.4:1.05;ctx.beginPath();
      t.values.forEach((v,i)=>i?ctx.lineTo(g.x(i),g.y(v)):ctx.moveTo(g.x(i),g.y(v)));
      ctx.stroke();
      if(t.model===hover&&hoverIndex!=null){
        ctx.fillStyle=t.color;ctx.beginPath();ctx.arc(g.x(hoverIndex),g.y(t.values[hoverIndex]),4,0,2*Math.PI);ctx.fill();
      }
    }
    ctx.globalAlpha=1;ctx.fillStyle='#51675a';ctx.textAlign='left';ctx.fillText('Start',g.left,g.h-6);
    ctx.textAlign='right';ctx.fillText(group.marks[cols.at(-1)],g.right,g.h-6);
  }
  function highlight(model,index=null){
    hover=model;hoverIndex=index;
    if(model){
      const t=traces.find(t=>t.model===model),k=index??cols.length;
      caption.textContent=model+' · '+(k?group.marks[cols[k-1]]:'Start')+' · '+t.values[k].toLocaleString('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0});
    }else caption.textContent=emptyCaption;
    draw();
  }
  canvas.onpointermove=e=>{
    const g=size(),r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
    const i=Math.max(0,Math.min(cols.length,Math.round((x-g.left)/(g.right-g.left)*cols.length)));
    const t=traces.reduce((a,b)=>Math.abs(g.y(a.values[i])-y)<=Math.abs(g.y(b.values[i])-y)?a:b);
    highlight(t.model,i);
  };
  canvas.onpointerleave=()=>highlight(null);
  draw();
  return {highlight,draw,colorFor:name=>traces.find(t=>t.model===name).color};
}
