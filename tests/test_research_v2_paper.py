"""Paper provenance must fail closed when evidence is stale or missing."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
from research_v2.paper import verify_statistics,Manuscript,read,DATA


def test_changed_or_missing_source_rejected(tmp_path):
    p=tmp_path/'metric.csv';p.write_text('value\n0.4\n')
    stats={'sources':{'metric.csv':hashlib.sha256(p.read_bytes()).hexdigest()},'summaries':{'experiments_summary':{'status':'complete'}}}
    assert verify_statistics(stats,tmp_path)
    p.write_text('value\n0.9\n')
    with pytest.raises(ValueError,match='changed evidence'):verify_statistics(stats,tmp_path)
    p.unlink()
    with pytest.raises(ValueError,match='Missing'):verify_statistics(stats,tmp_path)


def test_partial_experiments_cannot_build_final_statistics():
    with pytest.raises(ValueError,match='incomplete'):
        verify_statistics({'sources':{},'summaries':{'experiments_summary':{'status':'partial'}}})


def test_abstract_updates_when_metric_changes():
    stats=read(DATA/'statistics.json')
    changed=deepcopy(stats)
    row=next(r for r in changed['tables']['diag_forecast_summary'] if r['period']=='audit' and r['model']=='hybrid')
    row['accuracy']=.987654
    text=Manuscript(changed).result('abstract')
    assert '98.77% accuracy' in text
    assert '37.59% accuracy' not in text


def test_missing_figure_fails_build():
    stats=read(DATA/'statistics.json');changed=deepcopy(stats)
    del changed['figures']['diag_confusion']
    with pytest.raises(KeyError):Manuscript(changed).render_figure('confusion')


def test_unresolved_bibliography_fails_instead_of_silent_plaintext():
    m=Manuscript(read(DATA/'statistics.json'))
    with pytest.raises(ValueError,match='Unresolved bibliography'):
        m.resolve_citations('An unsupported assertion [missing2026].')
    assert m.resolve_citations('A verified citation [gu2020].')=='A verified citation [1].'


def test_empirical_claim_resolves_to_unique_source_row():
    stats=read(DATA/'statistics.json');m=Manuscript(stats);m.result('abstract')
    claims=[c for c in m.claims if c['kind']=='empirical_value']
    assert len(claims)>5
    for c in claims:
        assert c['row'] in stats['tables'][c['table']]
        assert c['value']==c['row'][c['column']]
