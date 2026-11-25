"""Smoke tests for the review dashboard: every view renders from the committed evidence.

These run the real ``app_review.py`` through Streamlit's test harness. They need
only the committed files under ``data/review-v1`` (not the local sealed evidence),
so they also pass on a fresh clone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from review_v1 import nav
from review_v1.paths import OUT, RECONSTRUCTED, RECORDED, SIMULATED, read_json

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app_review.py")

pytestmark = pytest.mark.skipif(not (OUT / "manifest.json").exists(), reason="run `python -m review_v1.build all` first")


def run(**query):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP, default_timeout=120)
    for key, value in query.items():
        app.query_params[key] = str(value)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def html(app) -> str:
    return "\n".join(str(getattr(e, "body", "") or getattr(e.proto, "body", "")) for e in app.get("html"))


def text(app) -> str:
    parts = [e.value for kind in ("markdown", "caption", "header", "subheader", "info", "warning", "error", "success") for e in app.get(kind)]
    return "\n".join(str(p) for p in parts)


@pytest.mark.parametrize("view", nav.VIEW_IDS)
def test_every_view_renders_and_marks_itself_ready(view):
    app = run(view=view)
    marker = html(app)
    assert 'id="review-ready"' in marker
    assert f'data-view="{view}"' in marker
    assert "has not been generated yet" not in text(app), "a section is missing its evidence file"


def test_unknown_view_falls_back_to_overview():
    assert 'data-view="overview"' in html(run(view="nonsense"))


@pytest.mark.parametrize("step", [s["n"] for s in nav.STEPS])
def test_default_case_renders_every_step_with_a_provenance_label(step):
    app = run(view="walkthrough", case=nav.DEFAULT_CASE, symbol=nav.DEFAULT_SYMBOL, step=step)
    marker = html(app)
    assert f'data-case="{nav.DEFAULT_CASE}"' in marker and f'data-symbol="{nav.DEFAULT_SYMBOL}"' in marker and f'data-step="{step}"' in marker
    assert any(f'data-provenance="{kind}"' in marker for kind in (RECORDED, RECONSTRUCTED, SIMULATED))
    caption = read_json(OUT / "narration.json")["walkthrough"][str(step)]["caption"]
    assert caption[:40] in marker


def test_every_case_and_every_kind_of_instrument_renders():
    index = read_json(OUT / "walkthrough/index.json")
    for case in index["cases"]:
        traded = case["traded_symbols"]
        idle = next(s for s in case["symbols"] if s not in traded)
        for symbol in ([traded[0]] if traded else []) + [idle]:
            for step in (4, 5, 6, 7):
                run(view="walkthrough", case=case["case_id"], symbol=symbol, step=step)


def test_simulated_results_are_never_labelled_recorded():
    """The replay section must carry the simulated label and say it is not recorded trading."""
    app = run(view="performance")
    marker, body = html(app), text(app)
    assert f'data-provenance="{SIMULATED}"' in marker
    assert "far too few to interpret a Sharpe ratio" in body
    # The loss sentence is derived from the saved results, so it must agree with them.
    import pandas as pd

    metrics = pd.read_csv(OUT / "sandbox/metrics.csv")
    invested = metrics.loc[(metrics.period == "post_release") & (metrics.arm != "cash"), "net_return"]
    assert ("Every invested portfolio lost money" in body) == bool((invested < 0).all())


def test_live_view_offers_the_run_button_and_never_claims_a_fill():
    app = run(view="live")
    assert [b.label for b in app.button] == ["Run the pipeline now"]
    assert f'data-provenance="{SIMULATED}"' in html(app)
    assert "simulates no fill and sends nothing to a broker" in text(app)
    # The summary at the top describes the run shown below it, not an older one.
    from review_v1 import live

    assert live.caption(live.last_run()) in html(app)


def test_live_caption_reads_the_saved_run():
    from review_v1 import live

    assert live.caption(None) == "No demo run has been saved yet."
    run_ = dict(session="2026-11-24", seconds=18.6, decision=dict(trend_count=8), signals=[{}] * 22, account=dict(orders=8), regime=dict(state="turbulent"))
    sentence = live.caption(run_)
    assert "session 2026-11-24, 8 of 22 instruments pass the trend rule, 8 pending orders" in sentence
    assert "regime turbulent" in sentence and "Completed in 19 seconds" in sentence
    assert "No fill is simulated and nothing is sent to a broker" in sentence


def test_overview_states_the_negative_result_plainly():
    body = text(run(view="overview"))
    assert "class-frequency prior" in body
    assert "no measurable information" in body
