"""Real Streamlit-to-sealed-CLI-to-isolated-ledger commissioning test."""
from pathlib import Path
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.integration
def test_dashboard_commissioning_roundtrip(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    release=ROOT/'data/releases/research-v2-runtime-20260928-r2'
    assert (release/'manifest.json').exists(),'Qualified runtime required for UI evidence'
    monkeypatch.setenv('MLTE_UI_STATE',str(tmp_path/'state'))
    app=AppTest.from_file(str(ROOT/'app_v2.py'),default_timeout=300).run()
    assert not app.exception
    app.sidebar.button(key='commission').click().run(timeout=300)
    assert not app.exception
    assert len(app.sidebar.error)==0
    receipt=json.loads((tmp_path/'state/commissioning/latest.json').read_text())
    assert receipt['mode']=='commissioning_preview'
    assert receipt['external_orders'] is False
    assert len(receipt['signals'])==22
    for name in ['deployment','shadow']:
        assert abs(receipt[name]['reconciliation_error'])<1e-6
        assert (tmp_path/f'state/commissioning/{name}.sqlite').exists()
    assert app.metric[0].value=='0'
    (ROOT/'data/research-v2/tests/ui-evidence.json').write_text(json.dumps({'status':'passed','mode':receipt['mode'],'signals':len(receipt['signals']),'deployment':receipt['deployment'],'shadow':receipt['shadow'],'completed_at':receipt['completed_at'],'source':'Streamlit AppTest button -> sealed CLI -> public data -> dual SQLite ledger -> receipt; isolated state, zero pilot observations'},indent=2))
