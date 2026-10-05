"""Real Streamlit session smoke test; no browser or production database access."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from engine_lab.storage import EngineStore


def widget_with_label(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_create_preview_save_review_and_overhaul(tmp_path, monkeypatch):
    path = tmp_path / "ui.sqlite3"
    monkeypatch.setenv("ENGINE_LAB_DB", str(path))
    app = AppTest.from_file(str(Path(__file__).parent / "engine_health_lab.py"), default_timeout=30)
    app.run()
    assert not app.exception
    widget_with_label(app.button, "Create persistent engine").click().run()
    assert not app.exception
    store = EngineStore(path)
    engine = store.list_engines()[0]
    assert engine["operating_hours"] == 0
    widget_with_label(app.multiselect, "Add scheduled faults").set_value(["bearing_degradation"]).run()
    widget_with_label(app.button, "Generate preview — no persistence").click().run()
    assert not app.exception
    assert store.list_tests(engine["engine_id"]) == []
    assert len(app.dataframe) >= 3
    widget_with_label(app.button, "Save test & advance engine hours / wear").click().run()
    assert not app.exception
    tests = store.list_tests(engine["engine_id"])
    assert len(tests) == 1
    widget_with_label(app.slider, "Inspect sample").set_value(180).run()
    widget_with_label(app.selectbox, "Disposition").set_value("confirmed_fault")
    widget_with_label(app.text_input, "Cause / finding (do not copy an injection label without inspection)").set_value("Observed pitting")
    widget_with_label(app.text_area, "Inspection / repair / independent measurement evidence").set_value("Synthetic inspection example")
    widget_with_label(app.text_input, "Technician name or identifier").set_value("test-technician")
    widget_with_label(app.button, "Append review record").click().run()
    assert not app.exception
    feedback = store.list_feedback(tests[0]["test_id"])
    assert len(feedback) == 1 and feedback[0]["sample_index"] == 180
    assert store.get_test(tests[0]["test_id"])["analyses"][180]["root_cause"].startswith("Not confirmed")
    hours_before = store.get_engine(engine["engine_id"])["operating_hours"]
    widget_with_label(app.text_area, "Overhaul / component work notes").set_value("Partial bearing replacement")
    widget_with_label(app.slider, "Bearing wear removed").set_value(0.8)
    widget_with_label(app.button, "Record new overhaul & apply repair").click().run()
    assert not app.exception
    assert len(store.list_overhauls(engine["engine_id"])) == 2
    assert store.get_engine(engine["engine_id"])["operating_hours"] == hours_before
    # A new session reopens the saved state, not an ephemeral session-only engine.
    reopened = AppTest.from_file(str(Path(__file__).parent / "pages/1_Engine_Evidence_Lab.py"), default_timeout=30).run()
    assert not reopened.exception
    assert len(store.list_tests(engine["engine_id"])) == 1
    assert any("RUL unavailable" in warning.value for warning in reopened.warning)
