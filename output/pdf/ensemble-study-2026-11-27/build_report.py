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
    for i,(m,label,color) in enumerate(zip(prefix,["Fixed learned weights","Weights learned from\nstock and market data","Pick one forecaster with\nreinforcement learning"],[blue,green,ochre])):
        vals=[gpu.loc[f"{m} / {s}","net_return"] for s in ["all experts","trees and linear","sequence networks"]]
        bars=ax.bar(np.arange(3)+(i-1)*.248,vals,width=.225,label=label,color=color,zorder=3)
        ax.bar_label(bars,labels=[p(v) for v in vals],padding=3,fontsize=8)
    ax.set_xticks(range(3),["All 13 forecasts","Linear model and trees","Four models that\nread price history"])
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
para(f,.635,"The proposal tests whether using market states to weight a fixed set of classifiers improves risk-adjusted performance over individual models and a static ensemble. A separate experiment tests periodic language-model-guided adaptation.")
proposal=rows("five","all-2023-2025")
proposal_names=["HMM gate","Gate without HMM","Uniform vote","Logistic","SPY buy and hold"]
proposal_labels=["Learned weights using\nmarket states","Learned weights without\nmarket states","Equal votes for the\nsame forecasts","Single logistic\nregression model","Hold the S&P 500 fund"]
proposal_colors=[green,"#a1b6a8",blue,ochre,brick]

def comparison_bars(axes):
    for col,(ax,field,title) in enumerate(zip(axes,["net_return","max_drawdown","sharpe"],["Return after costs","Largest fall","Sharpe ratio"])):
        values=[proposal.loc[name,field] for name in proposal_names]
        bars=ax.barh(range(len(values)),values,color=proposal_colors)
        ax.set_yticks(range(len(values)),proposal_labels if col==0 else [""]*len(values),fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.43)
        ax.set_xticks([])
        ax.tick_params(axis="y",length=0)
        ax.bar_label(bars,labels=[f"{v:.2f}" if field=="sharpe" else p(v) for v in values],padding=3,fontsize=8)
        ax.set_title(title,fontsize=9,loc="left",weight="bold")

comparison_bars([f.add_axes([.325+.197*i,.295,.17,.217]) for i in range(3)])
f.text(.08,.543,"Original pipeline and alternatives  ·  719 matched sessions, 2023–2025",fontsize=10,weight="bold",color=green)
y=para(f,.23,"The market-state inputs reduced the largest loss compared with the same gate without those inputs. Equal votes had a slightly smaller largest loss, while the single logistic model and market fund had higher Sharpe ratios.",gap=.009)
para(f,y,"Sharpe compares average daily return with volatility. The separate research-agent experiment kept the existing setting.",gap=.009)
link(f,.077,"Open the matched balance and drawdown charts","comparison/pipeline")
chart,axes=plt.subplots(1,3,figsize=(10.4,3.3),gridspec_kw={"wspace":.11})
chart.subplots_adjust(left=.24,right=.97,bottom=.07,top=.75)
chart.text(.037,.93,"The original pipeline and alternatives",weight="bold",fontsize=15)
chart.text(.037,.84,"719 matched trading sessions, 2023–2025",fontsize=10,color=muted)
comparison_bars(axes)
chart.savefig(OUT/"pipeline-comparison.png",dpi=150)
plt.close(chart)

ASSETS=ROOT/"site/assets"
fund_names={"SPY":"S&P 500","QQQ":"Nasdaq-100","IWM":"US small companies","GLD":"Gold","TLT":"Long-term US Treasuries"}
raw=ROOT/"data/raw/ceci2-v1"
price_file=ASSETS/"yahoo-fund-prices.json"
if all((raw/f"{ticker}.json").is_file() for ticker in fund_names):
    adjustments=pd.read_csv(receipts/"adjustments.csv.gz",index_col="date",parse_dates=True)
    paths={}
    for ticker in fund_names:
        quote=read(raw/f"{ticker}.json")["chart"]["result"][0]
        dates=pd.to_datetime(quote["timestamp"],unit="s",utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
        if not dates.equals(adjustments.index):
            raise ValueError(f"Quote dates for {ticker} differ from the frozen study dates")
        close=pd.Series(quote["indicators"]["quote"][0]["close"],index=dates)*adjustments[ticker]
        indexed=close/close.iloc[0]*100
        path=pd.concat([indexed.iloc[:1],indexed.groupby(indexed.index.to_period("W-FRI")).tail(1)])
        paths[ticker]={d.strftime("%Y-%m-%d"):round(v,4) for d,v in path.items()}
    price_file.write_text(json.dumps(paths,separators=(",",":"))+"\n")
price_paths=read(price_file)

def yahoo_prices(ax):
    for (ticker,series),color in zip(price_paths.items(),[green,blue,ochre,"#76507c",brick]):
        ax.plot(pd.to_datetime(list(series)),list(series.values()),label=f"{fund_names[ticker]} ({ticker})",color=color,lw=1.35)
    ax.set_xlim(pd.Timestamp("2017-01-01"),pd.Timestamp("2025-12-31"))
    ax.set_ylabel("First session = 100",fontsize=9)
    ax.legend(frameon=False,fontsize=7.5,ncol=2,loc="upper left")
    style_chart(ax)

def factor_chart(ax):
    factors=pd.DataFrame(coverage["annualFactors"]).set_index("year")
    colors=ax.imshow(factors.T,aspect="auto",cmap="RdBu",vmin=-.35,vmax=.35)
    ax.set_yticks(range(len(factors.columns)),[name.replace("_"," ") for name in factors.columns],fontsize=7.5)
    ax.set_xticks(range(len(factors)),factors.index,fontsize=8)
    colorbar=ax.figure.colorbar(colors,ax=ax,pad=.02,ticks=[-.35,0,.35],format=PercentFormatter(1,0),extend="both")
    colorbar.ax.tick_params(labelsize=8)

fed=pd.read_csv(receipts/"finbert-scores.csv")
def fed_chart(ax):
    dates=pd.to_datetime(fed.date.astype(str))
    tone=fed.positive-fed.negative
    ax.bar(dates,tone,width=18,color=green)
    ax.axhline(0,color=muted,lw=.65)
    ax.set_xlim(pd.Timestamp("2017-01-01"),pd.Timestamp("2025-12-31"))
    ax.set_ylabel("Positive minus negative",fontsize=8)
    style_chart(ax)

def save_source_plot(stem,draw,left=.10):
    fig,ax=plt.subplots(figsize=(8.5,3.5))
    fig.subplots_adjust(left=left,right=.96,top=.96,bottom=.13)
    draw(ax)
    fig.savefig(ASSETS/f"{stem}.png",dpi=140)
    plt.close(fig)

save_source_plot("yahoo-finance-prices",yahoo_prices)
save_source_plot("investment-factor-returns",factor_chart,left=.20)
save_source_plot("federal-reserve-text-sentiment",fed_chart)

f=page("Quantitative data", "Yahoo Finance market records and Jensen, Kelly and Pedersen investment factors")
f.text(.08,.808,"Yahoo Finance  ·  "+f"{coverage['bars']:,} daily records across {coverage['symbols']} instruments",size=12,weight="bold",color=green)
yahoo_prices(f.add_axes([.15,.553,.75,.218]))
para(f,.516,"Five funds from the 22-instrument universe. The lines show weekly snapshots of adjusted closing prices, scaled to 100 on the first day. The models also use daily opening, high and low prices and trading volume.",gap=.008)
f.text(.08,.412,"Investment factors  ·  13 market-wide themes",size=12,weight="bold",color=green)
factor_chart(f.add_axes([.212,.17,.69,.208]))
para(f,.129,"Each square shows one theme’s compounded return in one calendar year. Forecast models use trailing 21-session theme returns with a further 21-session delay.",gap=.005)
link(f,.061,"Open the charts and complete source coverage","data")

f=page("Qualitative data", "Sampled company headlines, annual filing excerpts and Federal Reserve statements")
years=[r["year"] for r in coverage["annualText"]]
ax=f.add_axes([.12,.68,.34,.12])
ax.bar(years,[r["news"] for r in coverage["annualText"]],color=green,width=.64)
ax.set_title(f"Sampled company headlines  ·  {coverage['headlines']}",loc="left",fontsize=9)
ax.set_xticks(years[::2])
style_chart(ax)
ax=f.add_axes([.55,.68,.34,.12])
for offset,key,label,color in [(-.18,"filing",f"SEC filing excerpts ({coverage['filings']})",ochre),(.18,"fed",f"Fed statements ({coverage['statements']})",blue)]:
    ax.bar(np.array(years)+offset,[r[key] for r in coverage["annualText"]],color=color,width=.34,label=label)
ax.set_title("Filings and Fed statements",loc="left",fontsize=9)
ax.legend(frameon=False,fontsize=7.5,bbox_to_anchor=(0,-.29),loc="upper left")
ax.set_xticks(years[::2])
style_chart(ax)
ax=f.add_axes([.12,.425,.77,.126])
company_bars=ax.bar([r["symbol"] for r in companies],[r["news"] for r in companies],color=green)
ax.set_title("Sampled headlines by company",loc="left",fontsize=9)
ax.bar_label(company_bars,fontsize=7,padding=2)
ax.set_ylim(0,230)
style_chart(ax)
ax=f.add_axes([.12,.213,.77,.139])
ax.set_title("Federal Reserve statement tone from FinBERT",loc="left",fontsize=9)
fed_chart(ax)
y=para(f,.161,"The company headline sample ends in 2023. Filings and Fed statements continue through 2025. Zero headlines means the sample has none for that company.",width=98,gap=.005)
para(f,y,"FinBERT scores text tone. OpenAI builds forecasts from Fed statements or monthly instrument summaries. The run used OpenAI Decisions for the role proposed for Jev.",width=98,gap=.005)
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
f.text(.08,.411,"Where market conditions enter",weight="bold",color=green,fontsize=11)
flow(f,.345,[("Market data","S&P 500 daily return\nand 21-session volatility"),("Market-state model","Chances of three\nmarket states"),("Forecast controller","Weights the direction\nforecasts")])
para(f,.233,"In the jointly trained experiment, three models specialize in recent prices, text sentiment and market context. Each supplies a forecast to a vote counter trained alongside them.")
f.text(.08,.159,"How three kinds of price model read the data",size=11,weight="bold",color=green)
para(f,.132,"Logistic regression adds weighted features. Tree models split on thresholds. Four price-history models read 20 sessions in order and predict the same three outcomes.",gap=.005)
link(f,.069,"Open the model diagrams and full input map","models")

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
y=para(f,.581,f"{p(weights[k])} vote share × {p(votes[k][2])} chance of a rise = {100*weights[k]*votes[k][2]:.1f} percentage points from LSTM. Add the weighted forecasts from all 13 models.",gap=.008)
ax=f.add_axes([.09,.426,.82,.045])
left=0.
for name,value,color in [("Fall",down,ochre),("Flat",flat,blue),("Rise",up,green)]:
    ax.barh([0],[value],left=left,color=color,height=.78)
    ax.text(left+value/2,0,f"{name}  {p(value)}",va="center",ha="center",color="white",fontsize=9,weight="bold")
    left+=value
ax.set_xlim(0,1)
ax.set_title("Combined forecast",loc="left",fontsize=10,weight="bold")
ax.axis("off")
flow(f,.334,[("Trading score",f"{p(up)} up minus {p(down)} down\n= {100*(up-down):.1f} points"),("Rank","Apple ranked second\namong the five selected"),("Set the position","15% target for Apple\nat the next open")])
para(f,.226,"The fixed rule selects up to five instruments above five points. It sizes positions by volatility, caps each at 15% and charges 50 cents per 1,000 dollars traded.")
para(f,.143,"In separate tests, the model router picks one forecast and learns from forecast accuracy. The portfolio allocator chooses cash, the market fund or the selected basket and learns from return after costs.",gap=.008)
link(f,.077,"See the recorded votes and trade target","controller")

f=page("Which model combinations paid off?", "Each group uses the same 480 trading sessions in 2024–2025")
pools(f.add_axes([.13,.55,.77,.235]))
y=para(f,.482,"Weights learned from stock and market data earned 24.2% with four price-history models and lost 1.4% with all 13 forecasts. The same trading rule and costs apply to the three methods within each group.",gap=.009)
y=para(f,y,f"With three separately designed forecasters trained together, equal votes earned {p(gpu.loc['Joint experts uniform','net_return'])} and learned votes earned {p(gpu.loc['Joint neural mixture','net_return'])}. The market fund earned {p(gpu.loc['SPY buy and hold','net_return'])} on those dates.",gap=.02)
f.text(.08,y,"How the original strategies did across years",weight="bold",fontsize=11,color=green)
body=[[label,*[ret("five",str(year),name) for year in range(2022,2026)],p(four.loc[name,"net_return"])]
      for name,label in [("Adaptive router","Weights updated using\nrecent forecast errors"),("HMM gate","Learned weights using\nmarket states"),("Uniform vote","Equal votes for the\nsame forecasts"),("SPY buy and hold","Hold the S&P 500 fund")]]
table(f,[.08,y-.182,.84,.155],["Portfolio","2022","2023","2024","2025","Total"],body,[.30,.14,.14,.14,.14,.14],8.5)
link(f,.076,"Explore all approaches by return, drawdown and Sharpe","comparison")

f=page("How the voting rules held up", "The 2022–2025 stress tests change forecasts while keeping the trading rule fixed")
stress=read(ROOT/"site/assets/stress.json")
ax=f.add_axes([.13,.52,.77,.278])
marks=pd.to_datetime(stress["mark"])
for name,label,color in [("Adaptive router","Weights updated using\nrecent forecast errors",ochre),("HMM gate","Learned weights using\nmarket states",green),("Uniform vote","Equal votes for the\nsame forecasts",blue)]:
    ax.plot(marks,stress["scenarios"]["original"][name],label=label,color=color,lw=1.35)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"${v/1000:.0f}k"))
ax.legend(loc="upper left",frameon=False,fontsize=8,ncol=3)
ax.set_title("Portfolio value with the recorded forecasts",fontsize=10,loc="left")
style_chart(ax)
body=[[label,*[ret(group,"all-2022-2025",n) for n in ["Adaptive router","HMM gate","Uniform vote"]]]
      for group,label in [("five","Recorded"),("language_outage","Language forecasts removed"),("reversed_tree","Two price forecasts reversed")]]
table(f,[.08,.325,.84,.125],["Scenario","Weights updated using\nrecent forecast errors","Learned weights using\nmarket states","Equal votes for the\nsame forecasts"],body,[.34,.24,.23,.19],7.7)
y=para(f,.285,"Adaptive votes beat equal votes in all three simulations. The original gate finished first among the three when language forecasts were removed. With recorded forecasts, the S&P 500 fund earned 55.9% and had a larger peak-to-trough loss than the adaptive router.")
para(f,y,"The 95% interval for the adaptive router’s return advantage over equal votes on recorded forecasts includes zero. These periods had already been examined. A future comparison can test whether the advantage repeats.")
link(f,.070,"Open all three scenario charts and the daily replay","stress")

with PdfPages(OUT/"ensemble-study.pdf",metadata={"Title":"Market conditions and portfolio risk"}) as pdf:
    for f in pages:pdf.savefig(f)
shutil.copyfile(OUT/"ensemble-study.pdf",ROOT/"site/assets/ensemble-study.pdf")
