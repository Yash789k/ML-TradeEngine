"""Build the model-controller report and its figures from the archived research."""
from pathlib import Path
import shutil
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter, FuncFormatter

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
DATA=ROOT/"data/ceci2-demo"
def read(path): return json.loads(path.read_text())
studies={g["id"]:g for g in read(ROOT/"site/assets/periods.json")["groups"]}
def period(study,key):
    return next(p for p in studies[study]["periods"] if p["id"]==key)
def rows(study,key):
    return pd.DataFrame(period(study,key)["rows"]).set_index("model")
def p(v): return f"{v:.1%}"
def ret(study,key,name): return p(rows(study,key).loc[name,"net_return"])
four=rows("five","all-2022-2025")
gpu=rows("gpu","all-2024-2025")
gdir=DATA/"colab-t4-v1"
joint=read(gdir/"joint-experts/result.json")
soft=read(gdir/"custom-gates/receipt.json")
hard=read(gdir/"expert-selector/receipt.json")
ablation=read(DATA/"moe-v2/result.json")
coverage=read(ROOT/"site/assets/study-5.json")["coverage"]
green="#16684e"; ink="#172e26"; muted="#4e6259"; ochre="#a4642d"; blue="#506f9a"; brick="#aa5856"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#cbd5cf","axes.labelcolor":ink,"xtick.color":muted,"ytick.color":muted,"text.color":ink,"pdf.fonttype":42})
models=["Adaptive router","Classic MoE","MoE + offline RL","SPY buy and hold"]
names=["Adaptive model weights","Traditional mixture","Portfolio allocator","S&P 500 fund"]
colors=[green,ochre,blue,brick]
def annual(ax):
    for i,(model,name,color) in enumerate(zip(models,names,colors)):
        values=[rows("five",str(y)).loc[model,"net_return"] for y in range(2022,2026)]
        ax.bar(np.arange(4)+(i-1.5)*.185,values,width=.17,label=name,color=color,zorder=3)
    ax.set_xticks(range(4),range(2022,2026));ax.yaxis.set_major_formatter(PercentFormatter(1,0))
    ax.axhline(0,lw=.8,color=muted);ax.grid(axis="y",color="#e3e8e4")
    ax.legend(ncol=2,frameon=False,loc="upper right",fontsize=9);ax.set_ylabel("Annual net return")
def pools(ax):
    prefix=["Fixed mixture","Context adapter","RL model router"]
    for i,(m,label,color) in enumerate(zip(prefix,["Fixed learned weights","Context-dependent weights","Select one model"],[blue,green,ochre])):
        vals=[gpu.loc[f"{m} / {s}","net_return"] for s in ["all experts","trees and linear","sequence networks"]]
        bars=ax.bar(np.arange(3)+(i-1)*.248,vals,width=.225,label=label,color=color,zorder=3)
        ax.bar_label(bars,labels=[p(v) for v in vals],padding=3,fontsize=8)
    ax.set_xticks(range(3),["All 13 experts","Linear model and trees","Four sequence networks"])
    ax.yaxis.set_major_formatter(PercentFormatter(1,0));ax.set_ylim(-.055,.32);ax.set_ylabel("Cumulative net return")
    ax.axhline(0,lw=.8,color=muted);ax.grid(axis="y",color="#e3e8e4");ax.legend(ncol=3,frameon=False,fontsize=8)
def curves(ax):
    paths=read(ROOT/"site/assets/period-paths.json")["gpu"]
    dates=pd.to_datetime(paths["marks"])
    for model,label,c in zip(["Joint experts uniform","Joint neural mixture","SPY buy and hold"],["Same experts, equal votes","Same experts, learned votes","S&P 500 fund"],[ochre,green,blue]):
        eq=10000*np.cumprod(1+np.array(paths["returns"][model]));ax.plot(dates,eq,label=label,color=c,lw=1.8)
    ax.grid(axis="y",color="#e3e8e4");ax.legend(frameon=False,fontsize=9);ax.set_ylabel("Portfolio value, US dollars");ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"${v/1000:.0f}k"))
    import matplotlib.dates as md
    ax.xaxis.set_major_locator(md.MonthLocator(bymonth=[1,4,7,10]));ax.xaxis.set_major_formatter(md.DateFormatter("%b %y"))
def standalone(name,title,draw):
    f,ax=plt.subplots(figsize=(9,3.9));f.subplots_adjust(left=.1,right=.98,bottom=.15,top=.84)
    f.text(.1,.94,title,fontsize=16,weight="bold",va="top");draw(ax);f.savefig(OUT/(name+".png"),dpi=175);plt.close(f)
standalone("annual","How four portfolios handled the same four years",annual)
standalone("pools","Changing the underlying models changed the outcome",pools)
standalone("joint","Giving the same three trained experts different votes",curves)

pages=[]
def page(title,subtitle):
    f=plt.figure(figsize=(8.5,11));pages.append(f)
    f.text(.078,.941,title,size=23,weight="bold",va="top")
    f.text(.08,.885,subtitle,size=11,color=muted,va="top")
    f.add_artist(plt.Line2D([.08,.92],[.858,.858],color="#b8c9bf",lw=.8))
    f.text(.08,.035,"CECI2 ensemble study   /   27 November 2026",size=8,color=muted)
    f.text(.92,.035,str(len(pages)),size=8,ha="right",color=muted)
    return f
def para(f,y,text,width=91):
    wrapped=textwrap.wrap(text,width=width)
    f.text(.08,y,"\n".join(wrapped),fontsize=10.5,va="top",linespacing=1.48)
    return y-len(wrapped)*.0202-.025
def section(f,y,title,text):
    f.text(.08,y,title,size=13,weight="bold",va="top",color=green)
    return para(f,y-.036,text)
def table(f,box,columns,body,widths=None,font=9):
    ax=f.add_axes(box);ax.axis("off")
    t=ax.table(cellText=body,colLabels=columns,colWidths=widths,cellLoc="right",colLoc="right",bbox=[0,0,1,1]);t.auto_set_font_size(False);t.set_fontsize(font)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("white");cell.set_facecolor("#dce9e1" if r==0 else "#f0f4f1" if r%2 else "white")
        if r==0:cell.get_text().set_weight("bold")
        if c==0:cell.get_text().set_ha("left")
f=page("Who should control the models?","Historical evidence from 22 US stocks and exchange-traded funds")
y=para(f,.825,"A forecasting expert estimates whether a price will rise, stay flat or fall. The controller decides how much of a vote to give each expert. A separate trading rule converts their combined forecast into stock positions. This report asks when the controller earns its place.")
annual(f.add_axes([.13,.46,.77,.235]))
y=section(f,.405,"Where adaptation helped",f"The adaptive router earned {p(four.loc[models[0],'net_return'])} over 2022–2025 with a {p(four.loc[models[0],'max_drawdown'])} maximum drawdown. The traditional mixture earned {p(four.loc[models[1],'net_return'])} and fell {p(four.loc[models[1],'max_drawdown'])} from its peak. The adaptive system held up better in the 2022 decline and captured less of the 2023 recovery.")
section(f,y,"Where the controller struggled",f"On the Tesla T4 graphics processor, equal votes from three newly trained experts earned {p(gpu.loc['Joint experts uniform','net_return'])} across 2024–2025. The learned gate over those same forecasts earned {p(gpu.loc['Joint neural mixture','net_return'])}. The two years had already been inspected. These results support a historical comparison, and future performance remains unmeasured.")

f=page("From a forecast to a trade","The model, the controller and the portfolio answer different questions")
y=section(f,.82,"1   What information was available?",f"The market panel contains {coverage['bars']:,} daily observations across {coverage['symbols']} instruments. Each record includes opening, high, low and closing prices plus trading volume. The study also uses 13 lagged investment-factor themes, {coverage['statements']} Federal Reserve statements, {coverage['filings']} annual filing excerpts and {coverage['headlines']} sampled company headlines.")
y=section(f,y,"2   Which model supplies the forecast?","A traditional mixture trains its experts and their gate together. A contextual adapter receives probabilities from models already trained on earlier data and changes their weights with conditions. A reinforcement learner can instead select one expert by name. A portfolio allocator acts one level later and chooses among cash, an index fund and a stock basket.")
y=section(f,y,"3   How does the forecast change the portfolio?","Each evening, the baseline policy subtracts the predicted chance of a fall from the chance of a rise. If the difference exceeds five percentage points, up to five stocks qualify. The rule favors less volatile names and caps each position at 15%. It fills at the next adjusted opening price and charges five basis points per side, or $0.05 per $100 traded. Annual folds liquidate before the next window.")
y=section(f,y,"4   Which comparisons are fair?","A five-session forecast predicts direction over five trading sessions, while the default portfolio can rebalance daily. The one-session study uses a different label and has more eligible dates. The site compares strategies only on common decision dates within each study. A portfolio starts at $10,000; uninvested cash earns zero.")
section(f,y,"5   What did the coding agent do?","The coding agent implemented offline experiments. In a separate historical test, a hosted research agent proposed changes to gate settings. Candidates had to pass validation before activation. The hosted agent retained the incumbent across the five-session reviews.")

f=page("Performance across market periods","Annual returns include simulated fees; drawdowns are measured from a prior peak")
body=[]
for name,label in zip(models,names):
    body.append([label,*[ret("five",str(y),name) for y in range(2022,2026)],p(four.loc[name,"net_return"]),p(four.loc[name,"max_drawdown"])])
table(f,[.08,.62,.84,.19],["Portfolio","2022","2023","2024","2025","Total","Drawdown"],body,[.29,.113,.113,.113,.113,.115,.143],8.5)
y=section(f,.59,"The bear-market test","The classic mixture and reinforcement policy were rerun for 2022 using expert training outcomes through 2019, calibration through 2020 and policy training through 2021. The bear market was deliberately selected after its direction was known. The adaptive router reduced the loss, although its lower average market exposure also matters.")
y=section(f,y,"Risk has more than one meaning","Maximum drawdown measures the largest loss from a prior portfolio peak. Average invested capital measures how much money was exposed to the market. A smaller drawdown can come from holding more cash. An exposure-matched comparison holds the index fund in the same daily proportion and keeps the rest in cash.")
z=rows("five","all-2023-2025")
y=section(f,y,"Trading frequency and costs",f"Across 2023–2025 the daily classic mixture earned {p(z.loc['Classic MoE','net_return'])}. Rebalancing every fifth session earned {p(z.loc['Classic MoE every five sessions','net_return'])} and reduced fees on the $10,000 account from ${z.loc['Classic MoE','fees']:,.0f} to ${z.loc['Classic MoE every five sessions','fees']:,.0f}. This change also altered which positions were held between rebalances.")
section(f,y,"How the benchmark is constructed","SPY is the ticker for the exchange-traded fund tracking the Standard & Poor’s 500 Index. Buy and hold keeps SPY throughout each evaluation fold. Equal investment across all 22 instruments is a separate benchmark; the selected universe contains surviving companies and overlapping fund holdings.")

f=page("Which expert pool benefits?","Tesla T4 experiment   /   480 matched sessions in 2024 and 2025")
pools(f.add_axes([.13,.52,.77,.27]))
y=section(f,.45,"Read the comparison within each group","The blue portfolio learns one weight per expert for the year. The green portfolio changes weights with volatility, price range, drawdown, market return and published sentiment. The brown portfolio uses reinforcement learning to pick one model per instrument and date. All three send forecasts to the same trading rule.")
y=section(f,y,"Forecast quality and portfolio return","Log loss penalizes confident wrong predictions. A lower value is better. Returns also depend on whether a forecast crosses the trading threshold, which stocks qualify, and the cost of changing holdings. A model can earn more while assigning worse probabilities.")
section(f,y,"What the uncertainty interval says",f"In the full pool, the contextual gate increased average log loss by {soft['context_minus_fixed_log_loss']:.4f} relative to fixed weights. The paired 95% interval was {soft['paired_95_interval'][0]:.4f} to {soft['paired_95_interval'][1]:.4f}. For the expert-selecting reinforcement learner the increase was {hard['primary_loss_change']:.4f}. Resampling consecutive 21-session blocks preserves some dependence between nearby dates. The sequence-pool returns are descriptive.")

f=page("New experts trained together","GPU run   /   A direct test of the value of the learned gate")
curves(f.add_axes([.14,.53,.77,.255]))
y=section(f,.46,"The three votes","A recurrent neural network reads a short price sequence. A second network reads sentiment features from statements, headlines and filings. A linear model estimates direction from market context. All three experts train alongside a small gating network. For the control, we take the same trained expert outputs and give each one-third of the vote.")
ci=joint["paired_95_interval"]
y=section(f,y,"The test held the experts fixed",f"The joint gate returned {p(gpu.loc['Joint neural mixture','net_return'])} with {p(gpu.loc['Joint neural mixture','max_drawdown'])} drawdown. Equal votes returned {p(gpu.loc['Joint experts uniform','net_return'])} with {p(gpu.loc['Joint experts uniform','max_drawdown'])} drawdown. The forecast-loss difference was {joint['context_minus_uniform_log_loss']:.5f}; its paired 95% interval ran from {ci[0]:.4f} to {ci[1]:.4f}. The ranking of portfolio returns is clearer than the difference in forecast quality.")
section(f,y,"How the folds were separated","The 2024 evaluation used only training labels that matured before its first decision. The 2025 refit added eligible 2024 outcomes. Each fold fitted feature scaling on training data. The GPU and local CPU runs used the same protocol; minibatch order and numerical kernels can produce different fitted weights.")

f=page("Text, stress and interpretation","What changed, what coverage allowed, and how far the result travels")
y=section(f,.82,"What did qualitative data add?",f"The 2023–2025 traditional mixture earned {p(z.loc['Classic MoE','net_return'])} with text and {p(z.loc['MoE without text','net_return'])} after retraining without it. Eligible recent company news appeared on {p(ablation['text_coverage']['news_available'])} of stock-dates; eligible annual filings appeared on {p(ablation['text_coverage']['filing_available'])}. The text model invested a larger share of capital on average. The return difference cannot identify a particular news source as its cause.")
y=section(f,y,"What happens if an expert breaks?","The stress study suppresses the five language channels each July or exchanges the up and down probabilities of the two tree models. The table gives full-period returns, including subsequent decisions. The two interventions diagnose specific failures in a historical market path.")
table(f,[.08,y-.15,.84,.15],["What changed","Adaptive weights","Regime gate","Equal vote"],[[label,*[ret(g,"all-2022-2025",n) for n in ["Adaptive router","HMM gate","Uniform vote"]]] for g,label in [("five","Recorded forecasts"),("language_outage","Language unavailable"),("reversed_tree","Reversed tree forecasts")]], [.36,.24,.20,.20])
y=section(f,y-.19,"What remains uncertain?","The periods were already examined during research. The universe contains overlapping funds and surviving stocks. Current data vintages and language-model pretraining can include information unavailable historically. The simulations use adjusted opening prices, fixed fees and fractional positions. They do not establish real fill quality or prospective profitability.")
section(f,y,"The strongest next test","Freeze the model pool, training budget, trading rule and benchmark before a new evaluation window. Archive the evidence available at each decision and compare both forecast loss and portfolio results against the same experts under fixed weights.")

allrows=rows("five","all-2024-2025").sort_values("net_return",ascending=False)
for idx,chunk in enumerate([allrows.iloc[:24],allrows.iloc[24:]]):
    f=page("All approaches on shared dates",f"Local research ledger   /   2024–2025   /   Rows {1 if idx==0 else 25}–{24 if idx==0 else len(allrows)} of {len(allrows)}")
    body=[[n,p(r.net_return),p(r.max_drawdown),p(r.mean_exposure),"—" if pd.isna(r.sharpe) else f"{r.sharpe:.2f}",f"${r.fees:,.0f}"] for n,r in chunk.iterrows()]
    table(f,[.08,.16,.84,.66],["Archived portfolio name","Return","Drawdown","Invested","Sharpe","Fees"],body,[.36,.13,.15,.13,.105,.125],8.25)
    para(f,.13,"Sharpe divides mean daily excess return by daily volatility and annualizes the ratio using 252 sessions. Fees assume $10,000 at the start. The Colab rerun is reported separately on pages 4 and 5.",width=96)

f=page("Model names and source records","A reading key for the charts, portfolio names and downloadable evidence")
y=section(f,.82,"Forecasting networks","LSTM means long short-term memory. GRU means gated recurrent unit. Both read an ordered sequence and remember earlier observations. TCN means temporal convolutional network; it searches a sequence for local patterns. A patch transformer divides the sequence into short blocks and learns which blocks relate to one another.")
y=section(f,y,"Routing and market context","MoE means mixture of experts. RL means reinforcement learning, where a controller learns from a reward tied to a decision. HMM means hidden Markov model; it estimates probabilities of unobserved market states. A regime gate uses these probabilities to inform how much each expert contributes.")
y=section(f,y,"Prices, text and factors","ETF means exchange-traded fund. OHLCV stands for open, high, low, close and volume. SEC is the US Securities and Exchange Commission. JKP refers to researchers Jensen, Kelly and Pedersen, whose factor themes summarize groups of related investment signals. FinBERT is a language model adapted to financial text.")
y=section(f,y,"Reproduce the figures","This report uses the matched-period website data and downloaded Colab training receipts from Git revision 378734f69514. The build_report.py file beside the PDF regenerates its tables and charts. Annual and quarterly breakdowns are available on the live site. The comparisons use the recorded model names so that each row can be traced back to its ledger.")
for label,url in [
("Interactive portfolio comparison","https://ceci2-ensemble-lab.breezytoad67.chatgpt.site/#comparison"),
("Executed Colab notebook","https://colab.research.google.com/drive/1qjCtnMsCAxeps4P7_RBPiJtPpmHHSqQD?authuser=1"),
("Versioned data, protocols and fitted models","https://github.com/Yash789k/ML-TradeEngine/tree/378734f695143dcd7a41087b8dd88cc431b9f075/data/ceci2-demo"),
("Study methods and research literature","https://github.com/Yash789k/ML-TradeEngine/blob/378734f695143dcd7a41087b8dd88cc431b9f075/ceci2_demo/moe_v2/README.md")]:
    f.text(.08,y,label,fontsize=10,color=green,url=url);y-=.032
with PdfPages(OUT/"ensemble-study.pdf",metadata={"Title":"Who should control the models?","Subject":"Historical evaluation of expert routing for trading forecasts"}) as pdf:
    for f in pages:pdf.savefig(f);plt.close(f)

shutil.copyfile(OUT/"ensemble-study.pdf", ROOT/"site/assets/ensemble-study.pdf")
