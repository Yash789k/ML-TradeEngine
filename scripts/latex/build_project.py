"""Create a native LaTeX derivative of the frozen research-v2 manuscript.

No experiment is rerun. Text/tables come from the verified manuscript; all 13
figures are TikZ/PGFPlots source generated from the same frozen statistics.
"""
from pathlib import Path
import hashlib,json,re,shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/overleaf/ml-trade-engine-research-latex'
STATS=ROOT/'data/research-v2/statistics.json'
MD=ROOT/'docs/research-v2/MANUSCRIPT.md'
S=json.loads(STATS.read_text())
for d in ['sections','tables','figures','data','evidence']:(OUT/d).mkdir(parents=True,exist_ok=True)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw_escape(s):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in str(s)).replace('→',r'$\rightarrow$').replace('−','-').replace('–','--').replace('—','---')

def tex(s):
    tokens={}
    def hold(value):
        key=f'ZZTOKEN{len(tokens)}ZZ';tokens[key]=value;return key
    s=re.sub(r'`([^`]+)`',lambda m:hold(r'\path{'+m[1]+'}'),str(s))
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:hold(r'\href{'+m[2]+'}{'+raw_escape(m[1])+'}'),s)
    s=re.sub(r'\[([0-9]+(?:,\s*[0-9]+)*)\]',lambda m:hold(r'\citep{'+','.join(S['references'][int(n)-1]['id'] for n in re.findall(r'\d+',m[1]))+'}'),s)
    s=re.sub(r'\*\*([^*]+)\*\*',lambda m:hold(r'\textbf{'+raw_escape(m[1])+'}'),s)
    s=raw_escape(s)
    for k,v in tokens.items():s=s.replace(k,v)
    return s

def df(name):return pd.DataFrame(S['tables'][name])
def num(x):return f'{float(x):.12g}'
def coords(x,y):return ' '.join(f'({num(a)},{num(b)})' for a,b in zip(x,y) if np.isfinite([a,b]).all())
def axis(options,plots):return '\\begin{tikzpicture}\n\\begin{axis}['+options+']\n'+plots+'\n\\end{axis}\n\\end{tikzpicture}\n'
def line(x,y,color='teal',extra=''):
    return r'\addplot+['+color+',mark=none,'+extra+'] coordinates {'+coords(x,y)+'};\n'
def categories(values):return ','.join('{'+raw_escape(v)+'}' for v in values)
def savefig(key,body):
    (OUT/'figures'/f'{key}.tex').write_text('% Native editable figure; frozen statistics '+sha(STATS)+'\n'+body)
def panels(plots,width='.48'):
    return '\n\\hfill\n'.join(r'\begin{minipage}[t]{'+width+r'\linewidth}\centering'+'\n'+p+r'\end{minipage}' for p in plots)
def errorplot(g,color='teal',ylabel=None):
    rows=[]
    for i,(_,r) in enumerate(g.iterrows()):rows.append(f"{num(r.effect)} {i} {num(r.effect-r.ci_low)} {num(r.ci_high-r.effect)}")
    return r'\addplot+['+color+r',only marks,mark=*,error bars/.cd,x dir=both,x explicit] table[x=x,y=y,x error minus=lo,x error plus=hi,row sep=\\]{x y lo hi\\'+'\\\\\n'.join(rows)+r'\\};'+'\n'

def figures():
    # Dataset flow and class support.
    c=df('diag_class_counts');periods=['development','selection','audit'];plots=''
    for klass,color in zip(['DOWN','FLAT','UP','UNMATURED'],['blue','gray','teal','orange']):
        vals=[c.loc[(c.period==p)&(c.class_name==klass),'n'].sum() for p in periods]
        plots+=r'\addplot+['+color+',fill='+color+',draw='+color+'] coordinates {'+coords(range(3),vals)+'};\n'+r'\addlegendentry{'+klass+'}\n'
    a=axis(r'width=\linewidth,height=5.2cm,ybar stacked,bar width=12pt,xtick={0,1,2},xticklabels={Development,Selection,Audit},ylabel={Instrument-date rows},legend columns=2,legend style={at={(0.5,1.05)},anchor=south},enlarge x limits=.25',plots)
    d=df('diag_dataset_flow').set_index('stage');keys=['raw_symbol_sessions','feature_complete_symbol_sessions','decision_symbol_sessions','mature_labeled_symbol_sessions']
    b=axis(r'width=.85\linewidth,height=5.2cm,xbar,bar width=11pt,ytick={0,1,2,3},yticklabels={Raw rows,Feature complete,Forecasts,Mature forecasts},xlabel={Rows (thousands)},enlarge y limits=.18,xtick={0,20000,40000,60000},xticklabels={0,20,40,60},xmax=60000,scaled x ticks=false,x tick label style={/pgf/number format/fixed},xmin=0',r'\addplot+[teal] coordinates {'+coords([d.loc[k,'n'] for k in keys],range(4))+'};')
    savefig('diag_dataset',panels([a,r'\resizebox{\linewidth}{!}{'+b+'}']))
    # System diagram entirely native TikZ.
    savefig('system_architecture',r'''\begin{tikzpicture}[node distance=9mm and 7mm,box/.style={draw,rounded corners=1mm,align=center,text width=40mm,minimum height=15mm,font=\footnotesize},>=Stealth]
\node[box] (data) {Completed public bars\\Raw response and adjustments\\Session/freshness checks};
\node[box,right=of data] (model) {Frozen causal model\\Past-only fit/calibration/gate\\Registered refit anchor};
\node[box,right=of model] (plan) {Immutable daily plan\\Probabilities and target deltas\\Input/model/source hashes};
\node[box,below=of data] (research) {Versioned historical evidence\\Registered trials and forecasts\\Exploratory diagnostics};
\node[box,below=of model] (account) {Shared execution contract\\Cash, reservations, fills and fees\\Deployment cash; trend shadow};
\node[box,below=of plan] (state) {Durable state and receipts\\Lock/recovery/backup\\Calendar scheduler and watchdog};
\draw[->] (data)--(model);\draw[->] (model)--(plan);\draw[->] (plan)--(state);\draw[->] (model)--(account);\draw[->] (state)--(account);\draw[->] (account)--(research);
\end{tikzpicture}''')
    savefig('information_timeline',r'''\begin{tikzpicture}[x=1cm,y=1cm,>=Stealth,font=\small]
\draw[->,thick] (0,0)--(13.5,0);
\foreach \x/\t in {1/t,3/{t+1},11/{t+6}}{\draw (\x,-.15)--(\x,.15);\node[above] at (\x,.2) {$\t$};}
\node[align=center,text width=30mm,below] at (1,-.4) {Completed close;\\features available};
\node[align=center,text width=33mm,below] at (4,-.4) {First permitted fill;\\target denominator};
\node[align=center,text width=38mm,below] at (11,-.4) {Outcome endpoint;\\label becomes observed};
\draw[<->,teal] (3,1.2)--(11,1.2);\node[above] at (7,1.3) {Five-session target return; adjacent labels overlap};
\node[align=center] at (6.7,-2.0) {Model fitting uses labels whose endpoints strictly precede the decision.};
\end{tikzpicture}''')
    # Confusion counts as native shaded matrix cells, with exact counts.
    plots=[];classes=['DOWN','FLAT','UP'];c=df('diag_confusion')
    for model in ['hybrid','prior','always_up']:
        g=c[(c.period=='audit')&(c.model==model)];code=r'\begin{tikzpicture}[x=1.05cm,y=.95cm,font=\scriptsize]'+'\n'
        for _,r in g.iterrows():
            x=classes.index(r.predicted_class);y=2-classes.index(r.true_class);shade=int(10+70*r['count']/g['count'].max())
            code+=f'\\fill[blue!{shade}] ({x},{y}) rectangle ++(1,1);\\node[text={"white" if shade>48 else "black"}] at ({x+.5},{y+.5}) {{{int(r["count"])}}};\n'
        for i,k in enumerate(classes):code+=f'\\node[below] at ({i+.5},0) {{{k}}};\\node[left] at (0,{2-i+.5}) {{{k}}};\n'
        code+=r'\node[above,font=\small] at (1.5,3.1) {'+raw_escape(model)+', $n='+str(int(g['count'].sum()))+'$};\n'+r'\node[below] at (1.5,-.45) {Predicted class};\end{tikzpicture}'
        plots.append(code)
    savefig('diag_confusion',panels(plots,'.32'))
    # Reliability and confidence-coverage from original full precision.
    rel=df('diag_reliability');cover=df('diag_coverage');a='';b=''
    for model,color in [('raw_mix','violet'),('hybrid','teal'),('prior','black')]:
        g=rel[(rel.period=='audit')&(rel.model==model)&(rel.n>0)]
        rows='\\\\\n'.join(f'{num(r.confidence)} {num(r.observed_accuracy)} {num(r.observed_accuracy-r.ci_low)} {num(r.ci_high-r.observed_accuracy)}' for _,r in g.iterrows())
        a+=r'\addplot+['+color+r',mark=*,error bars/.cd,y dir=both,y explicit] table[x=x,y=y,y error minus=lo,y error plus=hi,row sep=\\]{x y lo hi\\'+rows+r'\\};'+ '\n'+r'\addlegendentry{'+raw_escape(model)+'}\n'
        g=cover[(cover.period=='audit')&(cover.model==model)&(cover.n>0)]
        b+=line(g.coverage,g.accuracy,color,'mark=*')+r'\addlegendentry{'+raw_escape(model)+'}\n'
    a+=line([0,1],[0,1],'gray','dashed')
    savefig('diag_calibration',panels([axis(r'width=\linewidth,height=5.4cm,xmin=0,xmax=1,ymin=0,ymax=1,xlabel={Mean top probability},ylabel={Observed accuracy},legend pos=north west',a),axis(r'width=\linewidth,height=5.4cm,xmin=0,xmax=1,ymin=0,ymax=1,xlabel={Retained fraction},ylabel={Accuracy},legend pos=north east',b)]))
    # Forecast effects and ablations.
    plots=[];inf=df('diag_paired_inference');order=['linear','tree','raw_mix','hybrid']
    for period in periods:
        g=inf[(inf.period==period)&(inf.endpoint=='log_loss')&(inf.block_length==10)].set_index('model').loc[order]
        plots.append(axis(r'width=.94\linewidth,height=4.8cm,title={'+period+r'},ytick={0,1,2,3},yticklabels={Linear,Tree,Raw,Hybrid},xlabel={Loss minus prior},xmin=-.02,xmax=.15,xtick={0,.05,.10,.15},scaled x ticks=false,x tick label style={/pgf/number format/fixed,/pgf/number format/precision=2},extra x ticks={0},extra x tick style={grid=major}',errorplot(g)))
    savefig('diag_forecast_inference',panels(plots,'.32'))
    g=df('exp_contrasts');g=g[g.block==10].iloc[::-1]
    savefig('exp_contrasts',axis(r'width=.86\linewidth,height=9cm,ytick={'+','.join(map(str,range(len(g))))+r'},yticklabels={'+categories(g.id)+r'},xlabel={Daily log-loss difference},scaled x ticks=false,xtick={-.04,0,.04,.08,.12},x tick label style={/pgf/number format/fixed,/pgf/number format/precision=2},extra x ticks={0},extra x tick style={grid=major}',errorplot(g)))
    g=df('exp_learning').groupby('variant')[['train_log_loss','validation_log_loss']].mean().loc[['fraction_0.5','fraction_0.75','base','history_504']]
    code=''
    for k,color,label in [('train_log_loss','teal','Training'),('validation_log_loss','orange','Outer validation')]:code+=r'\addplot+['+color+'] coordinates {'+coords(range(4),g[k])+'};\n'+r'\addlegendentry{'+label+'}\n'
    savefig('exp_learning_v2',axis(r'width=.88\linewidth,height=5.6cm,ybar,bar width=13pt,ymin=0,xtick={0,1,2,3},xticklabels={50\% fitting rows,75\% fitting rows,Full 1008 window,504 window},xlabel={Chronological training design},ylabel={Log loss},legend columns=2,legend style={at={(.5,1.02)},anchor=south},enlarge x limits=.18',code))
    # Generalization matrix retains the complete22-instrument panel.
    g=df('diag_instrument');a=g[g.model=='hybrid'].pivot(index='symbol',columns='period',values='log_loss')[periods];b=g[g.model=='prior'].pivot(index='symbol',columns='period',values='log_loss')[periods]
    heatmap('diag_generalization',a-b,'Hybrid minus prior log loss',-0.2,.2,False,8.2)
    # Every observed account date retained in editable data files.
    g=df('diag_portfolio_curves');strategies=['trend','ml','adaptive_hybrid','equal_weight','cash'];colors=['blue','orange','teal','black','gray']
    for strategy in strategies:
        q=g[g.strategy==strategy].copy();q['session_index']=np.arange(len(q));q.to_csv(OUT/'data'/f'portfolio-{strategy}.csv',index=False)
    code=r'\begin{tikzpicture}\begin{groupplot}[group style={group size=1 by 3,vertical sep=9mm},width=.90\linewidth,height=4.1cm,scaled y ticks=false,y tick label style={/pgf/number format/fixed,/pgf/number format/precision=2},xmin=0,xmax=685,xtick={0,125,251,376,502,625,685},xticklabels={Jan24,Jul24,Jan25,Jul25,Jan26,Jul26,Sep26}]'+'\n'
    for i,(col,label) in enumerate([('equity','Equity (USD)'),('drawdown','Drawdown (fraction)'),('exposure','Gross exposure')]):
        code+=r'\nextgroupplot[ylabel={'+label+'}'+(r',legend columns=3,legend style={at={(.5,1.06)},anchor=south},xticklabels=\empty' if i==0 else r',xticklabels=\empty' if i==1 else '')+']\n'
        for strategy,color in zip(strategies,colors):
            code+=r'\addplot+['+color+r',mark=none,thin] table[col sep=comma,x=session_index,y='+col+'] {data/portfolio-'+strategy+'.csv};\n'
            if i==0:code+=r'\addlegendentry{'+raw_escape(strategy)+'}\n'
    code+=r'\end{groupplot}\end{tikzpicture}'
    savefig('diag_portfolio',code)
    g=df('diag_portfolio');g=g[(g.period=='full')&(g.strategy!='cash')].set_index('strategy').loc[strategies[:-1]]
    a=''
    for col,color,label in [('commission_usd','blue','Commission'),('slippage_proxy_usd','orange','Slippage proxy')]:a+=r'\addplot+['+color+'] coordinates {'+coords(range(4),g[col])+'};\n'+r'\addlegendentry{'+label+'}\n'
    opts=r'width=\linewidth,height=5.2cm,xtick={0,1,2,3},xticklabels={Trend,ML,Hybrid,Equal weight},enlarge x limits=.18'
    savefig('diag_costs',panels([axis(opts+r',ybar stacked,bar width=14pt,ylabel={Cost (USD)},legend style={at={(.5,1.05)},anchor=south}',a),axis(opts+r',ybar,bar width=14pt,ylabel={Executed notional / initial capital}',r'\addplot+[teal] coordinates {'+coords(range(4),g.turnover_initial_capital)+'};')]))
    g=df('diag_stress');g=g[g.status=='completed'].pivot(index='scenario',columns='strategy',values='net_return');delta=(g-g.loc['base'])*100
    heatmap('diag_stress',delta,'Return change (percentage points)',-5,5,True,5.6)
    # Boxplot distribution computed from same848 timing observations.
    g=df('exp_runtime');groups=list(g.groupby('variant'));code=''
    for i,(name,q) in enumerate(groups):
        vals=q.fit_seconds.to_numpy();q1,med,q3=np.quantile(vals,[.25,.5,.75]);iqr=q3-q1;low=vals[vals>=q1-1.5*iqr].min();high=vals[vals<=q3+1.5*iqr].max()
        code+=r'\addplot+[boxplot prepared={draw position='+str(i)+',lower whisker='+num(low)+',lower quartile='+num(q1)+',median='+num(med)+',upper quartile='+num(q3)+',upper whisker='+num(high)+r'},blue] coordinates {};'+ '\n'
    savefig('exp_runtime',axis(r'width=.9\linewidth,height=6cm,boxplot/draw direction=y,xtick={'+','.join(map(str,range(len(groups))))+r'},xticklabels={'+categories([k for k,q in groups])+r'},x tick label style={rotate=35,anchor=east},ylabel={Fit seconds},ymin=0',code))

def heatmap(key,data,label,low,high,nodes,height):
    # Explicit rectangles avoid automatic colorbar/annotation placement and
    # retain an editable, vector representation of every source cell.
    cellw=2.2 if nodes else 3.0
    cellh=.49 if nodes else .29
    width=len(data.columns)*cellw
    out=[r'\begin{tikzpicture}[x=1cm,y=1cm,font=\scriptsize]']
    def color(v):
        u=float(np.clip((v-low)/(high-low),0,1));a,b,t=([33,102,172],[247,247,247],u*2) if u<=.5 else ([247,247,247],[178,24,43],u*2-1)
        rgb=[round(x+(y-x)*t) for x,y in zip(a,b)]
        return '{rgb,255:red,'+str(rgb[0])+';green,'+str(rgb[1])+';blue,'+str(rgb[2])+'}'
    for i,idx in enumerate(data.index):
        y=(len(data)-i-1)*cellh
        out.append(r'\node[anchor=east] at (-.12,'+num(y+cellh/2)+') {'+raw_escape(idx)+'};')
        for j in range(len(data.columns)):
            x=j*cellw;v=float(data.iloc[i,j])
            out.append(r'\fill[fill='+color(v)+'] ('+num(x)+','+num(y)+') rectangle ++('+num(cellw)+','+num(cellh)+');')
            if nodes:out.append(r'\node at ('+num(x+cellw/2)+','+num(y+cellh/2)+') {'+f'{v:.2f}'+'};')
    for j,col in enumerate(data.columns):
        out.append(r'\node[anchor=north,align=center] at ('+num((j+.5)*cellw)+',-.12) {'+raw_escape(col).replace(r'\_',r'\\')+'};')
    # A separate horizontal key cannot overlap cell labels or axis ticks.
    start=(width-5)/2
    for i in range(100):out.append(r'\fill[fill='+color(low+(high-low)*(i+.5)/100)+'] ('+num(start+i*.05)+',-1.18) rectangle ++(.051,.18);')
    for t,value in [(0,low),(.5,0),(1,high)]:out.append(r'\node[anchor=north] at ('+num(start+t*5)+',-1.2) {'+num(value)+'};')
    out.extend([r'\node[anchor=north,font=\footnotesize] at ('+num(width/2)+',-1.65) {'+raw_escape(label)+'};',r'\end{tikzpicture}'])
    savefig(key,'\n'.join(out))

def table_tex(number,title,source,rows):
    n=len(rows[0]);length=[max(len(r[j]) for r in rows) for j in range(n)]
    weights=np.array([min(3.8,max(.65,l/15)) for l in length]);weights/=weights.sum()
    # Exact sum accounts for2(n-1) intercolumn pads; outer padding removed.
    numeric=[all(re.fullmatch(r'[-+0-9.,% ]+|N/A|nan|--',r[j]) for r in rows[1:]) for j in range(n)]
    widths=[r'>{'+(r'\raggedleft' if numeric[j] else r'\raggedright')+r'\arraybackslash}p{\dimexpr '+f'{v:.5f}'+r'\textwidth-'+f'{2*(n-1)*3/n:.3f}'+r'pt\relax}' for j,v in enumerate(weights)]
    header=' & '.join(r'\textbf{'+tex(c)+'}' for c in rows[0])+r' \\'
    lines=[r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.14}',r'\begin{longtable}{@{}'+''.join(widths)+'@{}}',r'\caption{'+tex(title)+r'}\label{tab:'+str(number)+r'}\\',r'\toprule',header,r'\midrule\endfirsthead',r'\multicolumn{'+str(n)+r'}{l}{\small\tablename\ \thetable\ (continued)}\\',r'\toprule',header,r'\midrule\endhead',r'\midrule\multicolumn{'+str(n)+r'}{r}{\footnotesize Continued on the next page}\\\endfoot',r'\bottomrule\endlastfoot']
    def cell(c):
        return tex(c).replace(r'\_',r'\_\allowbreak{}').replace('/',r'/\allowbreak{}').replace('HistGradientBoostingClassifier',r'Hist\allowbreak{}Gradient\allowbreak{}Boosting\allowbreak{}Classifier')
    for row in rows[1:]:lines.append(' & '.join(cell(c) for c in row)+r' \\')
    lines.extend([r'\end{longtable}',r'\noindent\textit{Source:} '+tex(source)+r'\par\endgroup',r'\medskip'])
    if number!=21:
        # Short tables remain whole; only the long evidence inventory spans pages.
        lines=[r'\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.14}',r'\caption{'+tex(title)+r'}\label{tab:'+str(number)+r'}',r'\begin{tabular}{@{}'+''.join(widths)+'@{}}',r'\toprule',header,r'\midrule']+[' & '.join(cell(c) for c in row)+r' \\' for row in rows[1:]]+[r'\bottomrule\end{tabular}\par\smallskip',r'\raggedright\textit{Source:} '+tex(source),r'\end{table}']
    (OUT/'tables'/f'table-{number:02}.tex').write_text('\n'.join(lines)+'\n')

def bibliography():
    # Bibliographic metadata is already verified; omit internal audit notes from
    # the printed bibliography but preserve that original verification in evidence.
    original=(ROOT/'docs/research-v2/references.bib').read_text()
    original=re.sub(r',?\n  note = \{[^\n]+\}', '', original)
    original=original.replace('author = {New York Stock Exchange}', 'author = {{New York Stock Exchange}}').replace('author = {GitHub}', 'author = {{GitHub}}')
    # Structure already-verified journal metadata instead of printing all
    # publications as miscellaneous documents. No new citation is introduced.
    journal_pattern=r'@misc\{([^,]+),([^@]*?)  howpublished = \{([^\n]+?) (\d+)\((\d+)\), (\d+)-(\d+)(?:; [^\n]+)?\},'
    def article(m):
        return '@article{'+m[1]+','+m[2]+'  journal = {'+m[3]+'},\n  volume = {'+m[4]+'},\n  number = {'+m[5]+'},\n  pages = {'+m[6]+'--'+m[7]+'},'
    original=re.sub(journal_pattern,article,original,flags=re.S)
    (OUT/'references.bib').write_text(original)

PREAMBLE=r'''% Compiler: pdfLaTeX. BibTeX is run automatically by Overleaf/latexmk.
\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,microtype}
\usepackage[margin=25mm,headheight=14pt]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,array,ragged2e}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,justification=justified,singlelinecheck=false}
\usepackage{tikz,pgfplots}
\usetikzlibrary{arrows.meta,positioning,calc}
\usepgfplotslibrary{groupplots,statistics,colormaps}
\pgfplotsset{compat=1.18,every axis/.append style={font=\footnotesize,label style={font=\footnotesize},tick label style={font=\scriptsize},legend style={font=\scriptsize,draw=none},axis lines=left,grid=major,grid style={gray!18},legend cell align=left},colormap={diverging}{rgb255(0cm)=(33,102,172);rgb255(1cm)=(247,247,247);rgb255(2cm)=(178,24,43)}}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{xurl}
\usepackage[hidelinks,bookmarksnumbered]{hyperref}
\usepackage{bookmark}
\usepackage[section]{placeins}
\usepackage{fancyvrb,fvextra}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small ML-TradeEngine research}
\fancyhead[R]{\small Historical study; prospective interim}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\setlength{\parindent}{1.3em}\setlength{\parskip}{2pt}
\setlength{\emergencystretch}{2em}
\raggedbottom
\renewcommand{\topfraction}{.9}
\renewcommand{\bottomfraction}{.85}
\renewcommand{\textfraction}{.08}
\renewcommand{\floatpagefraction}{.65}
\clubpenalty=10000\widowpenalty=10000
\setcounter{tocdepth}{2}
\hypersetup{pdftitle={Causal Forecasts, Constrained Portfolios, and Prospective Operation},pdfauthor={Yash Kataria}}
\begin{document}
\pagenumbering{roman}
\begin{titlepage}
\centering
\vspace*{8mm}
{\LARGE\bfseries Causal Forecasts, Constrained Portfolios, and Prospective Operation\par}
\vspace{5mm}
{\large An auditable empirical study of ML-TradeEngine\par}
\vspace{8mm}
{\Large Yash Kataria\par}
\vspace{6mm}
{28 September 2026\par}
\vspace{6mm}
{\small Historical evidence through 25 September 2026.\\Prospective sections remain interim; external broker orders are disabled.\par}
\vspace{10mm}
\begin{minipage}{.94\textwidth}
\section*{Abstract}
\input{sections/abstract}
\par\medskip
\noindent\textbf{Keywords:} quantitative trading; probability calibration; walk-forward evaluation; dependent-sample inference; paper trading; reproducibility.
\end{minipage}
\vfill
{\small Research report. Journal-specific formatting can be applied when a publication venue is selected.\par}
\end{titlepage}
\setcounter{page}{2}
\tableofcontents
\clearpage
\pagenumbering{arabic}
'''

def build():
    for rel,expected in S['sources'].items():
        if sha(ROOT/rel)!=expected:raise ValueError('Frozen statistics source changed: '+rel)
    figures();bibliography()
    lines=MD.read_text().splitlines();parts={};order=[];current=None;i=0;body=[];in_refs=False;appendix=False;table_no=0;figure_no=0
    def flush():
        nonlocal body
        if current:(OUT/'sections'/f'{current}.tex').write_text('\n\n'.join(body)+'\n')
        body=[]
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('## Abstract'):
            current='abstract';i+=1;continue
        if line.startswith('## References'):
            flush();current=None;in_refs=True;i+=1;continue
        if line.startswith('## Appendix '):
            flush();in_refs=False;appendix=True
            m=re.match(r'## Appendix ([A-Z])\. (.+)',line);current='appendix-'+m[1].lower();order.append((current,m[2],True));body=([r'\clearpage'] if current=='appendix-c' else [])+[r'\section{'+tex(m[2])+r'}\label{sec:'+current+'}'];i+=1;continue
        if in_refs:i+=1;continue
        if re.match(r'## \d+\.',line):
            flush();m=re.match(r'## (\d+)\. (.+)',line);current='section-'+m[1].zfill(2);order.append((current,m[2],False));body=([r'\clearpage'] if current=='appendix-c' else [])+[r'\section{'+tex(m[2])+r'}\label{sec:'+current+'}'];i+=1;continue
        if current is None:i+=1;continue
        if line.startswith('### '):
            body.append(r'\subsection{'+tex(re.sub(r'^### \d+\.\d+\s+','',line))+'}');i+=1;continue
        m=re.match(r'\*\*Table (\d+)\. (.+)\*\* Source: (.+)',line)
        if m:
            number=int(m[1]);table_no+=1;title=m[2];source=m[3];rows=[];i+=1
            while i<len(lines) and not lines[i].strip():i+=1
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[s.strip() for s in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',s) for s in cells):rows.append(cells)
                i+=1
            table_tex(number,title,source,rows)
            body.extend([r'\noindent Table~\ref{tab:'+str(number)+'} '+tex('reports '+title[0].lower()+title[1:]+'.'),r'\input{tables/table-'+str(number).zfill(2)+'}'])
            continue
        m=re.match(r'!\[Figure (\d+): (.+)\]\(([^)]+)\)',line)
        if m:
            number=int(m[1]);figure_no+=1;key=Path(m[3]).stem;caption=m[2]
            if not (OUT/'figures'/f'{key}.tex').exists():raise ValueError('Missing native figure '+key)
            body.extend([r'\noindent Figure~\ref{fig:'+str(number)+'} '+tex('shows '+caption[0].lower()+caption[1:]),r'\begin{figure}[htbp]\centering'+'\n'+r'\input{figures/'+key+'}\n'+r'\caption{'+tex(caption)+r'}\label{fig:'+str(number)+'}\n'+r'\end{figure}'])
            i+=1
            while i<len(lines) and (not lines[i].strip() or lines[i].startswith('Figure ')):i+=1
            continue
        if line.startswith('```'):
            chunk=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):chunk.append(lines[i]);i+=1
            body.append(r'\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\scriptsize]'+'\n'+'\n'.join(chunk)+'\n'+r'\end{Verbatim}');i+=1;continue
        if line.startswith('`r_target'):
            body.append(r'\begin{equation}\label{eq:target}r^{\mathrm{target}}_t=\frac{C_{t+6}}{C_{t+1}}-1.\end{equation}');i+=1;continue
        if line.startswith('`p_calibrated'):
            body.append(r'\begin{equation}\label{eq:temperature}p_T(c\mid x)=\frac{\exp\{\log p_{\mathrm{raw}}(c\mid x)/T\}}{\sum_k\exp\{\log p_{\mathrm{raw}}(k\mid x)/T\}}.\end{equation}');i+=1;continue
        if line.startswith('`Equity_t'):
            body.append(r'\begin{equation}\label{eq:equity}E_t=B_t+\sum_s q_{s,t}M_{s,t},\end{equation}\noindent where $E_t$ is equity, $B_t$ is cash, $q_{s,t}$ is the held quantity and $M_{s,t}$ is the mark for instrument $s$.');i+=1;continue
        body.append(tex(line));i+=1
    flush()
    main=[PREAMBLE]+[r'\input{sections/'+name+'}' for name,title,app in order if not app]
    main+=[r'\clearpage\phantomsection\addcontentsline{toc}{section}{References}',r'\bibliographystyle{unsrtnat}',r'\bibliography{references}',r'\clearpage\appendix']
    main += [r'\input{sections/'+name+'}' for name,title,app in order if app]
    main+=[r'\end{document}'];(OUT/'main.tex').write_text('\n'.join(main)+'\n')
    (OUT/'latexmkrc').write_text("$pdf_mode = 1;\n$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error %O %S';\n")
    for p in [STATS,ROOT/'data/research-v2/claim-map.json',ROOT/'docs/research-v2/references.json']:shutil.copy2(p,OUT/'evidence'/p.name)
    meta=dict(source_manuscript_sha256=sha(MD),source_statistics_sha256=sha(STATS),table_count=table_no,figure_count=figure_no,figure_format='13 native editable TikZ/PGFPlots files; no raster images or includegraphics',empirical_change='none; formatting derivative of frozen evidence',source_files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.tex','.bib','.csv']})
    (OUT/'evidence/latex-provenance.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:meta[k] for k in ['table_count','figure_count','figure_format']}))

if __name__=='__main__':build()
