"""Read-only dashboard for the registered outcome-only continuation."""
from pathlib import Path
import json
import subprocess
import pandas as pd
import streamlit as st

ROOT=Path('/Users/a/Downloads/ml-trade-engine')
RELEASE=ROOT/'data/releases/outcomes-r7-20261027'
WORKSPACE=ROOT/'data/paper/research-outcomes-r7'
PYTHON=ROOT/'data/research-v2/clean-env/bin/python'


def main():
    st.set_page_config(page_title='ML TradeEngine | Outcome collection',layout='wide')
    st.title('ML TradeEngine — outcome collection')
    st.caption('Five preserved predictions · six newly scheduled outcome checks · external broker orders disabled')
    try:
        raw=subprocess.run([str(PYTHON),'-B',str(RELEASE/'collector.py'),'status','--release',str(RELEASE),'--workspace',str(WORKSPACE)],capture_output=True,text=True,timeout=120,check=True)
        state=json.loads(raw.stdout);protocol=json.loads((RELEASE/'protocol.json').read_text())
    except Exception as exc:
        st.error('Evidence or collector integrity check failed: '+str(exc));return
    a,b,c=st.columns(3)
    a.metric('Original decision dates',state['decision_dates_preserved'])
    b.metric('New outcome reads',f"{state['completed_outcome_reads']}/6")
    c.metric('Decision outcomes in new reads',f"{state['mature_decision_dates']}/5")
    st.info('Restarted for US 27–30 October and 2–3 November. First collection: 28 October 04:30 HKT. Last collection: 4 November 05:30 HKT. All timestamps below use Hong Kong time.')
    st.warning('The earlier outcome reads on US 23 and 26 October were missed. The original account tail remains incomplete. New price downloads can measure the original predictions, but cannot recreate missed trades or account marks.')
    if state['halted']:st.error('Outcome collector halted. Inspect the saved failure before taking action.')
    if state['missing_new_sessions']:st.error('Missing newly scheduled reads: '+', '.join(state['missing_new_sessions']))
    hb=state.get('heartbeat')
    if hb:
        age=(pd.Timestamp.now(tz='UTC')-pd.Timestamp(hb['at'])).total_seconds()
        if age>300:st.error(f'Collector heartbeat is stale: {age/60:.1f} minutes.')
        else:st.caption(f'Last collector heartbeat: {pd.Timestamp(hb["at"]).tz_convert("Asia/Hong_Kong"):%d %b %H:%M:%S HKT}')
    tabs=st.tabs(['Schedule','Preserved accounts','Evidence and paper'])
    with tabs[0]:
        rows=[]
        for row in protocol['schedule']:
            rows.append({'US session':row['session'],'Collection HKT':pd.Timestamp(row['collection_hkt']).strftime('%d %b %H:%M'),
                         'Review HKT':pd.Timestamp(row['observer_close']).tz_convert('Asia/Hong_Kong').strftime('%d %b %H:%M')})
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        st.write('Keep the Mac plugged in, lid open, logged in and online overnight. The sleep guard runs through 4 November 07:00 HKT. It prevents idle sleep but cannot override a closed lid or shutdown.')
        st.caption('No new predictions, strategy targets, refits, fills or account changes occur in this continuation. Six successful scheduled downloads and five original-date outcomes are required for its final dataset.')
        with st.expander('Verified status'):st.json(state)
    with tabs[1]:
        st.caption('Each account stops at its last actual receipt. These are separate, incomplete account segments; their returns are never combined.')
        rows=[]
        for seg in protocol['segments']:
            for name,account in seg['accounts_at_last_actual_receipt'].items():
                rows.append({'Segment':seg['segment'],'Account':name,'Equity at last actual mark':account['equity'],'Cash':account['cash'],'Fees':account['fees']})
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        st.write('All prior failed trials and halts are preserved. Five interrupted dates do not establish profitability or sustained operational reliability.')
    with tabs[2]:
        st.caption('The existing manuscript and interim update remain available; the restarted collection is not yet a completed result.')
        for label,path,mime in [('Manuscript PDF',ROOT/'output/pdf/ML_TradeEngine_Research_Week1.pdf','application/pdf'),
                                ('Editable LaTeX',ROOT/'output/overleaf/week1/main.tex','text/plain'),
                                ('Interim LaTeX update',ROOT/'output/overleaf/week1-r6-update-20261023/operational-update.tex','text/plain'),
                                ('Restart registration',RELEASE/'protocol.json','application/json')]:
            if path.exists():st.download_button(label,path.read_bytes(),path.name,mime)
        doc=ROOT/'docs/research-outcomes-r7/STATUS.md'
        if doc.exists():st.markdown(doc.read_text())


if __name__=='__main__':main()
