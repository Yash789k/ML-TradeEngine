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
guide=read(ROOT/"site/assets/pipeline-guide.json")
f=page("From raw data to results", "How the original proposal becomes a historical simulation")
y=para(f,.824,"The proposal asks whether changing whose forecast counts can improve returns after allowing for risk. Follow the stages below in order. The first five follow a decision from its inputs to its later evaluation. The last happens between tests.")
for number,(title,explanation) in enumerate(guide["stages"],1):
    f.text(.08,y,f"{number}. {title}",size=12,weight="bold",va="top",color=green)
    y=para(f,y-.028,explanation,width=98,gap=.012)

f=page("What data went into the forecasts?", "Count each type of document by the year it was published")
for i,(key,title,color) in enumerate([
    ("news","Company headlines",green),("filing","Annual filing excerpts",ochre),("fed","Federal Reserve statements",blue),
]):
    ax=f.add_axes([.13,.68-i*.175,.77,.106])
    years=[r["year"] for r in coverage["annualText"]]
    values=[r[key] for r in coverage["annualText"]]
    ax.bar(years,values,color=color,width=.64)
    ax.set_title(title,loc="left",fontsize=10,weight="bold")
    ax.set_xticks(years,labels=years if i==2 else [""]*len(years),fontsize=8)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3,integer=True))
    ax.tick_params(axis="y",labelsize=8)
y=para(f,.285,"The charts use separate scales. This headline sample ends in 2023. Filings and Federal Reserve statements continue through 2025. A document can inform more than one decision.")
y=section(f,y,"What could the simulator buy?",f"The price file covers {coverage['symbols']} instruments and contains {coverage['bars']:,} instrument-days, one record for each instrument on each trading day.")
para(f,y,"Company stocks: AAPL, AMZN, BAC, GOOGL, GS, JPM, META, MSFT, NVDA and TSLA. Stock funds: DIA, IWM, QQQ, SPY, XLE, XLF, XLI, XLK, XLP and XLV. Gold and Treasury funds: GLD and TLT.")

f=page("How to read the scores", "Forecast scores and investment results answer different questions")
y=para(f,.824,"The same stock forecast can be scored as a prediction and used to simulate a trade. Forecast scores use the direction that eventually happened. Portfolio scores use the account balance after the trading rule and costs.")
for title,explanation in guide["metrics"]:
    y=section(f,y,title,explanation)
section(f,y,"Where to look next","Pages 4–5 show headline balances and one forecast decision. Page 6 compares years. Pages 7–9 test different ways of combining forecasts and changing inputs. Pages 10–12 contain all results and definitions.")

f=page("Can a model pick the right forecaster?", "A trading experiment with historical prices and news")
y=para(f,.82,"Suppose one model reads stock prices and another reads the news. They can disagree about what happens next. We tested whether a computer could learn whose forecast to trust, then simulated buying and selling stocks using its decisions.")
y=section(f,y,"What happened to $10,000?", "From 2022 through 2025, we compared two ways of combining forecasts with simply holding a fund that tracks large US companies. Each portfolio paid simulated trading costs.")
body=[[label, f"${10000*(1+four.loc[name,'net_return']):,.0f}"] for name,label in [
    ("Adaptive router","Give recently accurate models a larger vote"),
    ("Classic MoE","Train the forecasters and their vote counter together"),
    ("SPY buy and hold","Buy and hold the S&P 500 fund")]]
for row,name in zip(body,["Adaptive router","Classic MoE","SPY buy and hold"]):
    row.extend([p(four.loc[name,"max_drawdown"]),f"{four.loc[name,'sharpe']:.2f}"])
table(f,[.08,y-.13,.84,.13],["What the portfolio did","Ending balance","Largest fall","Sharpe"],body,[.57,.17,.14,.12],8.8)
y=section(f,y-.17,"Why consider the first method?",f"In 2022 all three lost money. The system that rewarded recent accuracy lost {p(abs(rows('five','2022').loc['Adaptive router','net_return']))}, while the jointly trained models lost {p(abs(rows('five','2022').loc['Classic MoE','net_return']))}. The first system also held more cash on average. Keeping money out of stocks can soften a loss and leave less money invested during a recovery.")
section(f,y,"What can we conclude?","Holding the fund earned the most over these four years. The next page follows one recorded forecast for Apple from the model votes to the simulated position. The later pages give the historical comparisons and the full results. These years were already familiar to the researchers. A future test would be needed to see whether the pattern repeats.")

f=page("One decision for Apple", "Recorded model votes after the market closed on 22 December 2025")
sim=read(ROOT/"site/assets/simulation.json")
detail=read(ROOT/"site/assets/detail-2025.json")
date="2025-12-22"
i=sim["dates"].index(date)
t=detail["dates"].index(date)
j=sim["symbols"].index("AAPL")
weights=detail["weights"][t][j]
votes=detail["probabilities"][t][j]
ranked=sorted(range(len(weights)),key=lambda k:weights[k],reverse=True)
top=[(detail["experts"][k],weights[k],votes[k][2]) for k in ranked[:4]]
ax=f.add_axes([.33,.65,.55,.132])
ax.barh(range(4),[w for _,w,_ in top],color=green)
ax.set_yticks(range(4),[name for name,_,_ in top],fontsize=8)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(PercentFormatter(1,0))
ax.set_xlim(0,.50)
ax.set_title("Share of the vote for the four most influential models",loc="left",fontsize=10)
ax.bar_label(ax.containers[0],labels=[p(w) for _,w,_ in top],fontsize=8,padding=3)

down,flat,up=[sim[k][i][j] for k in ("down","flat","up")]
model,weight,vote=top[0]
y=section(f,.59,"How the votes combine",f"The price-history model called {model} gave Apple a {p(vote)} chance of rising and held {p(weight)} of the vote. Multiplying these numbers gives {100*weight*vote:.1f} percentage points toward the final chance of a rise. Adding all 13 contributions gives {p(up)} up, {p(flat)} little change and {p(down)} down.")
state=max(sim["posterior"][i])
y=section(f,y,"What the market estimate tells us",f"The market-state model assigned {state:.2%} to the state with the highest average past return. The vote counter uses market conditions and past accuracy. The market estimate describes the state of the market. Apple’s own forecast still assigns only {p(up)} to a rise.")
scores=np.array(sim["up"][i])-np.array(sim["down"][i])
chosen=[k for k in np.argsort(-scores,kind="stable") if scores[k]>.05][:5]
target=min(.15,(1/sim["vol"][i][j])/sum(1/sim["vol"][i][k] for k in chosen)) if j in chosen else 0
y=section(f,y,"What the simulator did with the forecast",f"Up minus down gave {100*(up-down):.1f} percentage points. Apple ranked {chosen.index(j)+1} among the five selected instruments. The fixed rule gave less volatile holdings more room, capped each at 15% and targeted {p(target)} for Apple at the next opening price. The simulation charged 50 cents for each 1,000 dollars bought or sold.")
f.text(.08,y,"Open this recorded decision in Model votes",fontsize=10,color=green,url="https://ceci2-ensemble-lab.breezytoad67.chatgpt.site/#controller")

f=page("Performance across market periods","A negative return means the portfolio ended the year with less money than it started with")
body=[]
for name,label in zip(models,names):
    body.append([label,*[ret("five",str(y),name) for y in range(2022,2026)],p(four.loc[name,"net_return"]),p(four.loc[name,"max_drawdown"])])
table(f,[.08,.62,.84,.19],["Portfolio","2022","2023","2024","2025","Total","Drawdown"],body,[.29,.113,.113,.113,.113,.115,.143],8.5)
y=section(f,.59,"The bear-market test","The classic mixture and reinforcement policy were rerun for 2022 using expert training outcomes through 2019, calibration through 2020 and policy training through 2021. The bear market was deliberately selected after its direction was known. The adaptive router reduced the loss, although its lower average market exposure also matters.")
y=section(f,y,"Risk has more than one meaning","Maximum drawdown measures the largest loss from a prior portfolio peak. Average invested capital measures how much money was exposed to the market. A smaller drawdown can come from holding more cash. An exposure-matched comparison holds the index fund in the same daily proportion and keeps the rest in cash.")
z=rows("five","all-2023-2025")
y=section(f,y,"Trading frequency and costs",f"Across 2023–2025 the daily classic mixture earned {p(z.loc['Classic MoE','net_return'])}. Rebalancing every fifth session earned {p(z.loc['Classic MoE every five sessions','net_return'])} and reduced fees on the $10,000 account from ${z.loc['Classic MoE','fees']:,.0f} to ${z.loc['Classic MoE every five sessions','fees']:,.0f}. This change also altered which positions were held between rebalances.")
section(f,y,"How the benchmark is constructed","SPY is the ticker for the exchange-traded fund tracking the Standard & Poor’s 500 Index. Buy and hold keeps SPY throughout each evaluation fold. Equal investment across all 22 instruments is a separate benchmark; the selected universe contains surviving companies and overlapping fund holdings.")

f=page("Which models worked well together?","Three ways of counting votes across the same 480 trading days")
pools(f.add_axes([.13,.52,.77,.27]))
y=section(f,.45,"Which models were in each group?","The three price models were Logistic, XGBoost and LightGBM. The four price-history models were LSTM, GRU, TCN and Patch Transformer. The group of 13 added FinBERT, published investment factors and four language-model forecasts to those seven price models.")
y=para(f,y,"Blue keeps voting shares fixed. Green adjusts the shares to reduce forecast errors. Brown picks one model using past errors and a penalty for switching. All use the same trading rule.",gap=.012)
y=section(f,y,"Forecast quality and portfolio return","Log loss penalizes confident wrong forecasts. Portfolio returns also depend on which stocks cross the trading threshold and the cost of changing holdings.")
section(f,y,"What the uncertainty interval says",f"In the full pool, the contextual gate increased average log loss by {soft['context_minus_fixed_log_loss']:.4f} relative to fixed weights. The paired 95% interval was {soft['paired_95_interval'][0]:.4f} to {soft['paired_95_interval'][1]:.4f}. Picking one model increased loss by {hard['primary_loss_change']:.4f}. The interval groups nearby days into blocks.")

f=page("New experts trained together","Does a fourth model combine three forecasts better than a simple average?")
curves(f.add_axes([.14,.53,.77,.255]))
y=section(f,.46,"The three votes","A recurrent neural network reads a short price sequence. A second network reads sentiment features from statements, headlines and filings. A linear model estimates direction from market context. All three experts train alongside a small gating network. For the control, we take the same trained expert outputs and give each one-third of the vote.")
ci=joint["paired_95_interval"]
y=section(f,y,"The test held the experts fixed",f"The joint gate returned {p(gpu.loc['Joint neural mixture','net_return'])} with {p(gpu.loc['Joint neural mixture','max_drawdown'])} drawdown. Equal votes returned {p(gpu.loc['Joint experts uniform','net_return'])} with {p(gpu.loc['Joint experts uniform','max_drawdown'])} drawdown. The forecast-loss difference was {joint['context_minus_uniform_log_loss']:.5f}; its paired 95% interval ran from {ci[0]:.4f} to {ci[1]:.4f}. The ranking of portfolio returns is clearer than the difference in forecast quality.")
section(f,y,"How the folds were separated","The 2024 evaluation used only training labels that matured before its first decision. The 2025 refit added eligible 2024 outcomes. Each fold fitted feature scaling on training data. The GPU and local CPU runs used the same protocol; minibatch order and numerical kernels can produce different fitted weights.")

f=page("Text, stress and interpretation","What changed, what coverage allowed, and how far the result travels")
y=section(f,.82,"What did qualitative data add?",f"The 2023–2025 traditional mixture earned {p(z.loc['Classic MoE','net_return'])} with text and {p(z.loc['MoE without text','net_return'])} after retraining without it. Eligible recent company news appeared on {p(ablation['text_coverage']['news_available'])} of stock-dates; eligible annual filings appeared on {p(ablation['text_coverage']['filing_available'])}. The text model invested a larger share of capital on average. The return difference cannot identify a particular news source as its cause.")
y=section(f,y,"What happens if an expert breaks?","The market-state gate is the proposal’s original vote counter. Both it and the newer router see the same forecasts and market-state estimates. The router updates votes from recent errors in similar conditions. Each July the stress tests remove five language forecasts or reverse two price models.")
table(f,[.08,y-.15,.84,.15],["What changed","Adaptive weights","Regime gate","Equal vote"],[[label,*[ret(g,"all-2022-2025",n) for n in ["Adaptive router","HMM gate","Uniform vote"]]] for g,label in [("five","Recorded forecasts"),("language_outage","Language unavailable"),("reversed_tree","Reversed tree forecasts")]], [.36,.24,.20,.20])
y=section(f,y-.19,"What remains uncertain?","The periods were already examined during research. The universe contains overlapping funds and surviving stocks. Current data vintages and language-model pretraining can include information unavailable historically. The simulations use adjusted opening prices, fixed fees and fractional positions. They do not establish real fill quality or prospective profitability.")
stress=read(ROOT/"site/assets/stress.json")
interval=next(r for r in stress["uncertainty"] if r["scenario"]=="original" and r["endpoint"]=="net_return")
section(f,y,"What did adaptive voting improve?",f"With the recorded forecasts, it beat equal votes by {100*interval['difference']:.1f} percentage points. The historical uncertainty interval ran from {100*interval['low']:.1f} to +{100*interval['high']:.1f} points. The market fund earned {p(four.loc['SPY buy and hold','net_return'])} over the same dates. Removing the language forecasts gave the market-state gate the highest return among the three controllers.")

allrows=rows("five","all-2024-2025").sort_values("net_return",ascending=False)
for idx,chunk in enumerate([allrows.iloc[:24],allrows.iloc[24:]]):
    f=page("All approaches on shared dates",f"Local research ledger   /   2024–2025   /   Rows {1 if idx==0 else 25}–{24 if idx==0 else len(allrows)} of {len(allrows)}")
    body=[[n,p(r.net_return),p(r.max_drawdown),p(r.mean_exposure),"—" if pd.isna(r.sharpe) else f"{r.sharpe:.2f}",f"${r.fees:,.0f}"] for n,r in chunk.iterrows()]
    table(f,[.08,.16,.84,.66],["Archived portfolio name","Return","Drawdown","Invested","Sharpe","Fees"],body,[.36,.13,.15,.13,.105,.125],8.25)
    para(f,.13,"Sharpe divides mean daily excess return by daily volatility and annualizes the ratio using 252 sessions. Fees assume $10,000 at the start. The Colab rerun is reported separately on pages 7 and 8.",width=96)

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
with PdfPages(OUT/"ensemble-study.pdf",metadata={"Title":"Can a model pick the right forecaster?","Subject":"Historical evaluation of expert routing for trading forecasts"}) as pdf:
    for f in pages:pdf.savefig(f);plt.close(f)

shutil.copyfile(OUT/"ensemble-study.pdf", ROOT/"site/assets/ensemble-study.pdf")
