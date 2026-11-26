"""Research v2 product console; all account actions call the sealed collector CLI."""
from pathlib import Path
import json
import os
import subprocess
import sys
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
RELEASE=ROOT/'data/releases/research-v2-runtime-20260928-r2'
STATE=Path(os.environ.get('MLTE_UI_STATE',str(ROOT/'data/paper/research-v2')))
PYTHON=Path(os.environ.get('MLTE_RUNTIME_PYTHON',str(ROOT/'data/research-v2/clean-env/bin/python')))


def collector(command):
    if command not in {'status','tick','commission','backup','verify'}:
        raise ValueError('Unsupported local collector action')
    result=subprocess.run([str(PYTHON),'-B',str(RELEASE/'scripts/collector_v2.py'),command,
                           '--release',str(RELEASE),'--workspace',str(STATE)],
                          cwd=RELEASE,capture_output=True,text=True,timeout=600)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-5000:])
    return json.loads(result.stdout)


def read_json(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    st.set_page_config(page_title='ML TradeEngine | Research v2',layout='wide')
    st.title('ML TradeEngine')
    st.caption('Prospective local paper collection · Historical research · External orders disabled')
    if not (RELEASE/'manifest.json').exists():
        st.info('The isolated runtime is being qualified. No pilot decisions are enabled by this page.')
        return
    try:
        status=collector('status')
    except Exception as exc:
        st.error('Collector integrity check failed: '+str(exc));return
    st.sidebar.header('Local collector')
    st.sidebar.caption('Deployment: cash · Research shadow: trend · Initial capital: $10,000 each')
    for label,action in [('Check due events','tick'),('Run isolated commissioning check','commission'),('Verify backup and restore','backup')]:
        if st.sidebar.button(label,key=action):
            try:
                with st.spinner('Running the sealed collector…'):
                    receipt=collector(action)
                st.session_state['last_action']=dict(action=action,receipt=receipt)
                st.sidebar.success('Collector action completed. Receipt saved.')
            except Exception as exc:st.sidebar.error(str(exc))
    a,b,c,d=st.columns(4)
    a.metric('Valid pilot sessions',status['valid_observed_sessions'])
    b.metric('Required sessions',status['required_sessions'])
    c.metric('Mature decision dates',status['label_complete_decisions'])
    d.metric('Missed sessions',len(status['missing_sessions']))
    st.warning('Historical edge and profitable-trader readiness are not approved. Prospective evidence is still accumulating.')
    if status.get('next_event'):
        nxt=status['next_event'];st.info(f"Next event: {nxt['kind']} · {nxt['hong_kong']} (Hong Kong)")
    tabs=st.tabs(['Accounts','Operations','Historical results','Manuscript & evidence'])
    with tabs[0]:
        latest=read_json(STATE/'latest.json')
        preview=False
        if latest is None:
            latest=read_json(STATE/'commissioning/latest.json');preview=True
        if latest:
            st.subheader('Commissioning preview' if preview else 'Prospective accounts')
            st.caption('Commissioning never counts as a pilot observation.' if preview else f"Session {latest['session']} · {latest['trigger_mode']}")
            for name in ['deployment','shadow']:
                acc=latest[name]
                cols=st.columns(3);cols[0].metric(name.capitalize()+' equity',f"${acc['equity']:,.2f}")
                cols[1].metric('Cash',f"${acc['cash']:,.2f}");cols[2].metric('Reconciliation residual',f"${acc['reconciliation_error']:.2e}")
            st.dataframe(pd.DataFrame(latest['signals']),hide_index=True,width='stretch')
            st.caption('Quantities represent adjusted total-return proxy units, not broker shares.')
        else:st.info('No account receipt is available yet.')
    with tabs[1]:
        st.json(status)
        heartbeat=read_json(STATE/'heartbeat.json')
        if heartbeat:
            age=(pd.Timestamp.now(tz='UTC')-pd.Timestamp(heartbeat['at'])).total_seconds()
            if age>300:st.error(f'Collector heartbeat is stale ({age/60:.1f} minutes). The host may have slept or the job stopped.')
            st.json(heartbeat)
        failures=[]
        for p in sorted((STATE/'events').glob('*.json')):
            value=read_json(p)
            if value.get('action_required'):failures.append(value)
        if failures:st.error('Actionable collector events');st.dataframe(pd.DataFrame(failures),hide_index=True)
        st.write('The local schedule depends on an awake, logged-in host. Missing sessions halt decisions; they are never backfilled.')
        if 'last_action' in st.session_state:
            with st.expander('Last action receipt'):st.json(st.session_state['last_action'])
    with tabs[2]:
        stats=read_json(ROOT/'data/research-v2/statistics.json')
        if not stats:
            st.info('The experiment package is being finalized.')
        else:
            period=st.selectbox('Historical period',['audit','selection','development'])
            st.caption('All listed periods have been examined. These analyses are exploratory.')
            for key,label in [('diag_forecast_summary','Forecast quality'),('diag_portfolio','Capital-constrained accounts')]:
                st.subheader(label)
                rows=stats['tables'].get(key,[])
                frame=pd.DataFrame(rows)
                if 'period' in frame:frame=frame[frame.period==period]
                if key=='diag_portfolio':frame=frame[frame.scope=='matched_fresh_10000']
                st.dataframe(frame,hide_index=True,width='stretch')
            st.caption('Predictive log loss is lower-is-better. Portfolio comparisons use equal starting capital, dates and costs. Full metric definitions appear in the manuscript.')
    with tabs[3]:
        for label,path,mime in [('Current manuscript PDF',ROOT/'output/pdf/ML_TradeEngine_Research_v2.pdf','application/pdf'),
                                ('Editable manuscript',ROOT/'docs/research-v2/MANUSCRIPT.md','text/markdown'),
                                ('Claim map',ROOT/'data/research-v2/claim-map.json','application/json')]:
            if path.exists():st.download_button(label,path.read_bytes(),path.name,mime)
        status_doc=ROOT/'docs/research-v2/STATUS.md'
        if status_doc.exists():st.markdown(status_doc.read_text())

if __name__=='__main__':main()
