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
from matplotlib.figure import Figure
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
    f=Figure(figsize=(8.5,11));pages.append(f)
    f.text(.078,.941,title,size=23,weight="bold",va="top")
    f.text(.08,.885,subtitle,size=11,color=muted,va="top")
    f.add_artist(plt.Line2D([.08,.92],[.858,.858],color="#b8c9bf",lw=.8))
    f.text(.08,.035,"CECI2 ensemble study   /   27 November 2026",size=8,color=muted)
    f.text(.92,.035,str(len(pages)),size=8,ha="right",color=muted)
    return f
def para(f,y,text,width=91,gap=.025):
    wrapped=textwrap.wrap(text,width=width)
    f.text(.08,y,"\n".join(wrapped),fontsize=10.5,va="top",linespacing=1.48)
    return y-len(wrapped)*.0202-gap
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

SITE="https://ceci2-ensemble-lab.breezytoad67.chatgpt.site/#"
def link(f,y,label,route):
    f.text(.08,y,label,fontsize=9,color=green,url=SITE+route)

def flow(f,y,nodes):
    from matplotlib.patches import FancyArrowPatch
    centers=np.linspace(.173,.827,len(nodes))
    for x,(title,detail) in zip(centers,nodes):
        f.text(x,y+.012,title,ha="center",va="center",fontsize=10,weight="bold",color=green)
        f.text(x,y-.025,detail,ha="center",va="center",fontsize=8.5,color=muted,linespacing=1.4)
    for a,b in zip(centers[:-1],centers[1:]):
        f.add_artist(FancyArrowPatch((a+.083,y),(b-.083,y),transform=f.transFigure,
                                    arrowstyle="->",color=muted,mutation_scale=11))

def style_chart(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y",color="#e3e8e4")
    ax.tick_params(labelsize=8)

sim=read(ROOT/"site/assets/simulation.json")
detail=read(ROOT/"site/assets/detail-2025.json")
date="2025-12-22"
i=sim["dates"].index(date)
t=detail["dates"].index(date)
j=sim["symbols"].index("AAPL")
weights=detail["weights"][t][j]
votes=detail["probabilities"][t][j]
down,flat,up=[sim[k][i][j] for k in ("down","flat","up")]
sources=read(ROOT/"site/assets/forecast-sources.json")
receipts=ROOT/"ceci2_demo/receipts"
records=pd.read_csv(receipts/"company-scores.csv")
companies=(records.groupby(["symbol","kind"]).size().unstack(fill_value=0)
           .reset_index().to_dict("records"))
packet=next(r for r in reversed(read(receipts/"monthly-decisions.json")) if r["date"]<=date)
digest=detail["experts"].index("OpenAI asset digest")
apple=dict(decision=date,packet=packet["date"],rawUp=packet["forecasts"]["AAPL"]["UP"],
           calibratedUp=votes[digest][2],weight=weights[digest])
evidence={**sources,"companies":companies,"apple":apple}
(ROOT/"site/assets/source-evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")


f=page("Market conditions and portfolio risk", "A historical test of combining forecasts and turning them into simulated trades")
flow(f,.747,[("Observe","Prices and volume\nPublic text and factors"),("Forecast","Models give chances\nof up, flat and down"),("Choose votes","Blend forecasts\nor pick one model"),("Simulate","Turn the result into\nnext-open positions")])
y=section(f,.622,"What the original proposal tests","Can a learned gate that weights a fixed set of stock-direction classifiers under changing market conditions deliver more robust risk-adjusted performance than single classifiers or a static ensemble? The proposal separately asks whether periodic language-model-guided adaptation improves on scheduled retraining or conventional search.")
y=section(f,y,"How the tests line up","The fixed blend, context adapter and model router share 480 evaluation sessions in 2024–2025. The stress study covers 959 sessions in 2022–2025. Models can have different training inputs and histories. Within each test, we match dates and trading costs.")
y=section(f,y,"Two ways to score the result","Forecast log loss penalizes putting little probability on the outcome that happened. Portfolio return measures gains after fees. Drawdown measures the worst fall from a previous high. Sharpe measures average daily return relative to how much it fluctuates, using a zero cash reference here.")
section(f,y,"What we found",f"On the four-year historical run, the original gate’s Sharpe ratio was {four.loc['HMM gate','sharpe']:.2f} versus {four.loc['Uniform vote','sharpe']:.2f} for equal votes and {four.loc['SPY buy and hold','sharpe']:.2f} for buying the market. The later adaptive router reached {four.loc['Adaptive router','sharpe']:.2f}. The five-session research agent retained the existing gate setting.")
link(f,.077,"Explore the data-to-results pipeline","pipeline")

f=page("The data behind the forecasts", "Yahoo Finance prices, public company text, Fed statements and academic factors")
for n,(key,title,color) in enumerate([
    ("news","Sampled company headlines",green),
    ("filing","Annual filing excerpts from the SEC",ochre),
    ("fed","Federal Reserve statements",blue),
]):
    ax=f.add_axes([.12,.712-.116*n,.78,.068])
    years=[r["year"] for r in coverage["annualText"]]
    counts=[r[key] for r in coverage["annualText"]]
    ax.bar(years,counts,color=color,width=.62)
    ax.set_title(f"{title}  ·  {sum(counts)} records",loc="left",fontsize=9)
    ax.set_xticks(years,labels=years if n==2 else [""]*len(years))
    ax.yaxis.set_major_locator(plt.MaxNLocator(2,integer=True))
    style_chart(ax)
ax=f.add_axes([.12,.259,.78,.117])
company_bars=ax.bar([r["symbol"] for r in companies],[r["news"] for r in companies],color=green)
ax.set_title("Sampled headlines by company",loc="left",fontsize=9)
ax.bar_label(company_bars,fontsize=7,padding=2)
ax.set_ylim(0,230)
style_chart(ax)
y=para(f,.222,"The Financial News and Stock Price Integration Dataset sample ends in 2023. Filings and Fed statements continue through 2025. Yahoo Finance supplies 49,764 daily records across 22 instruments. Jensen, Kelly and Pedersen supply 13 market-wide investment factors, used with a delay.",width=99,gap=.009)
para(f,y,"FinBERT scores text sentiment. OpenAI turns Fed statements and monthly instrument summaries into forecasts. The run used OpenAI Decisions for the role originally proposed for Jev.",width=99,gap=.009)
link(f,.061,"Open company coverage, fund prices and source charts","data")

f=page("What goes into each model?", "All 13 forecasts enter the original vote counter in the same three-probability format")
table(f,[.08,.46,.84,.34],["Forecasters","Input","Output"],[
    ["Logistic, XGBoost, LightGBM","One row of price, volume and market features","Up, flat, down"],
    ["LSTM, GRU, TCN, Patch Transformer","20 sessions of those same features","Up, flat, down"],
    ["Factor model","13 delayed investment-theme signals","Up, flat, down"],
    ["FinBERT logistic","Fed, filing and headline sentiment scores","Up, flat, down"],
    ["OpenAI Decisions, Qwen, Granite","Latest eligible Fed statement","Market forecast"],
    ["OpenAI asset digest","Recent returns, volatility, text scores and Fed statement","Instrument forecast"],
],[.33,.49,.18],8)
y=section(f,.421,"Where the market-state model fits","It reads daily returns and 21-session volatility of the S&P 500 fund. It reports the chance of lower-, middle- and higher-return market states. The original gate uses those probabilities, earlier forecast accuracy and stock context to estimate each model’s voting share.")
y=section(f,y,"Two ways of training models together","The classic mixture trains three forecasters on shared price, text and factor inputs. The joint experiment gives three models separate inputs: an eight-session price series, text sentiment, and broad-market returns, volatility and breadth. Both learn how to combine the forecasts.")
link(f,.079,"Open the model-by-model input and output map","models")

f=page("One decision, from votes to a trade", "Apple after the market closed on 22 December 2025")
ranked=sorted(range(len(weights)),key=lambda k:weights[k],reverse=True)
top=ranked[:4]
ax=f.add_axes([.31,.64,.59,.14])
bars=ax.barh(range(4),[weights[k] for k in top],color=green)
ax.set_yticks(range(4),[detail["experts"][k] for k in top],fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0,.50)
ax.xaxis.set_major_formatter(PercentFormatter(1,0))
ax.bar_label(bars,labels=[p(weights[k]) for k in top],fontsize=8,padding=3)
ax.set_title("Share of the vote",loc="left",fontsize=10)
k=top[0]
y=section(f,.589,"Multiply each forecast by its vote",f"The long short-term memory model (LSTM) had {p(weights[k])} of the vote and predicted {p(votes[k][2])} up. Its contribution was {100*weights[k]*votes[k][2]:.1f} percentage points. Adding all 13 votes gave {p(up)} up, {p(flat)} flat and {p(down)} down. The monthly OpenAI forecast held {p(apple['weight'])} of the vote.")
y=section(f,y,"The fixed rule turns 19.5 points into a position",f"Up minus down gave {100*(up-down):.1f} points. Apple placed second. The rule selected up to five instruments over five points, sized by volatility with a 15% cap, and targeted 15% for Apple at the next open. Buying or selling 1,000 dollars cost 50 cents.")
y=section(f,y,"Where reinforcement learning fits","The model router replaces blending with one 100% vote. It rewards a high probability for the outcome that occurred and charges 0.01 for switching models. The separate portfolio allocator picks cash, the S&P 500 fund or the classic mixture’s basket. Its reward is account growth after trading costs.")
link(f,.077,"See the recorded votes and trade target","controller")

f=page("Which model combinations paid off?", "Each group uses the same 480 trading sessions in 2024–2025")
pools(f.add_axes([.13,.55,.77,.235]))
y=para(f,.482,"The adapter adjusts voting shares for each stock. It earned 24.2% with four price-history models and lost 1.4% with all 13 forecasts. The same trading rule and costs apply to the three methods within each group.",gap=.009)
y=para(f,y,f"With three separately designed forecasters trained together, equal votes earned {p(gpu.loc['Joint experts uniform','net_return'])} and learned votes earned {p(gpu.loc['Joint neural mixture','net_return'])}. The market fund earned {p(gpu.loc['SPY buy and hold','net_return'])} on those dates.",gap=.02)
f.text(.08,y,"How the original strategies did across years",weight="bold",fontsize=11,color=green)
body=[[label,*[ret("five",str(year),name) for year in range(2022,2026)],p(four.loc[name,"net_return"])]
      for name,label in [("Adaptive router","Adaptive votes"),("HMM gate","Market-state gate"),("Uniform vote","Equal forecast votes"),("SPY buy and hold","S&P 500 fund")]]
table(f,[.08,y-.182,.84,.155],["Portfolio","2022","2023","2024","2025","Total"],body,[.30,.14,.14,.14,.14,.14],8.5)
link(f,.076,"Explore all approaches by return, drawdown and Sharpe","comparison")

f=page("How the voting rules held up", "The 2022–2025 stress tests change forecasts while keeping the trading rule fixed")
stress=read(ROOT/"site/assets/stress.json")
ax=f.add_axes([.13,.52,.77,.278])
marks=pd.to_datetime(stress["mark"])
for name,color in [("Adaptive router",green),("HMM gate",ochre),("Uniform vote",blue)]:
    ax.plot(marks,stress["scenarios"]["original"][name],label=name,color=color,lw=1.35)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"${v/1000:.0f}k"))
ax.legend(loc="upper left",frameon=False,fontsize=8,ncol=3)
ax.set_title("Portfolio value with the recorded forecasts",fontsize=10,loc="left")
style_chart(ax)
body=[[label,*[ret(group,"all-2022-2025",n) for n in ["Adaptive router","HMM gate","Uniform vote"]]]
      for group,label in [("five","Recorded"),("language_outage","Language forecasts removed"),("reversed_tree","Two price forecasts reversed")]]
table(f,[.08,.325,.84,.125],["Scenario","Adaptive","Market-state gate","Equal votes"],body,[.43,.17,.23,.17],8.5)
y=para(f,.285,"Adaptive votes beat equal votes in all three simulations. The original gate finished first among the three when language forecasts were removed. With recorded forecasts, the S&P 500 fund earned 55.9% and had a larger peak-to-trough loss than the adaptive router.")
para(f,y,"The 95% interval for the adaptive router’s return advantage over equal votes on recorded forecasts includes zero. These periods had already been examined. A future comparison can test whether the advantage repeats.")
link(f,.070,"Open all three scenario charts and the daily replay","stress")

with PdfPages(OUT/"ensemble-study.pdf",metadata={"Title":"Market conditions and portfolio risk"}) as pdf:
    for f in pages:pdf.savefig(f)
shutil.copyfile(OUT/"ensemble-study.pdf",ROOT/"site/assets/ensemble-study.pdf")
