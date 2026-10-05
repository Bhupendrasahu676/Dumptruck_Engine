"""Export isolation and real-session smoke tests for the enhanced page."""
import csv
import importlib
import io
import json
from pathlib import Path
import sys

import pytest

from enhanced_dashboard import download_payloads, generate_preview, observation_inputs, sensor_rows


def sample():
    return {"elapsed_s": 0.0, "timestamp": "2026-01-01T00:00:00Z", "engine_id": "virtual-001",
            "episode_id": "episode-001", "base_trajectory_id": "base-001",
            "context": {"rpm": 1200, "mode": "haul"},
            "observations": {"engine.oil_condition_pct": 82, "engine.oil_pressure_kpa": None},
            "expected": {"engine.oil_condition_pct": 95, "engine.oil_pressure_kpa": 400},
            "quality": {"engine.oil_pressure_kpa": "missing"},
            "truth": {"active_faults": ["secret_fault"], "fault_levels": {"secret_fault": 0.8},
                      "event_phases": {"event-001": "active"}, "latent_wear": {"oil": 0.2}}}


def test_generation_never_passes_truth_schedule_or_ids_to_diagnostics():
    row = sample()
    observed = []
    def analyse(rows):
        observed.extend(rows)
        assert "secret_fault" not in json.dumps(rows)
        return [{"overall_health": None}]
    result = generate_preview({"seed": 793}, [{"kind": "secret_fault"}],
                              simulator=lambda config, events: {"config": config, "events": events, "samples": [row]},
                              analyser=analyse)
    assert observed == observation_inputs([row])
    assert "truth" not in observed[0] and "engine_id" not in observed[0]
    assert result["samples"][0]["truth"] == row["truth"]


def test_separate_exports_and_null_measurements_without_disk_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    preview = {"samples": [sample()], "diagnostics": [{"overall_health": None, "root_cause_confirmed": False}],
               "config": {"seed": 793}, "events": [{"kind": "secret_fault"}], "metadata": {"source": "simulation"}}
    payloads = download_payloads(preview)
    assert len(payloads) == 4 and list(tmp_path.iterdir()) == []
    reading = next(csv.DictReader(io.StringIO(payloads["observations.csv"])))
    assert reading["engine.oil_condition_pct"] == "82"
    assert reading["engine.oil_pressure_kpa"] == ""
    assert reading["context.rpm"] == "1200"
    assert not any("truth" in key or "quality" in key or "expected" in key or "fault" in key for key in reading)
    assert "secret_fault" not in payloads["observations.csv"]
    assert "secret_fault" not in payloads["diagnostics.json"]
    truth = next(csv.DictReader(io.StringIO(payloads["truth_labels.csv"])))
    assert truth["fault_levels.secret_fault"] == "0.8"
    assert json.loads(payloads["config_events.json"])["config"] == preview["config"]
    assert json.loads(payloads["diagnostics.json"])["rul_validated"] is False


def test_empty_results_and_misaligned_diagnostics():
    result = generate_preview({}, [], simulator=lambda *_: {"samples": []},
                              analyser=lambda _: pytest.fail("No diagnostics needed for empty samples"))
    assert result["diagnostics"] == []
    assert download_payloads(result)["observations.csv"] == ""
    with pytest.raises(ValueError, match="sample count"):
        generate_preview({}, [], simulator=lambda *_: {"samples": [sample()]}, analyser=lambda _: [])


def test_quality_and_missing_values_remain_visible():
    rows = sensor_rows(sample(), {"residuals": {"engine.oil_condition_pct": -13}, "rates": {"engine.oil_condition_pct": -0.2}})
    assert rows[0]["residual"] == -13
    assert rows[0]["rate_per_s"] == -0.2
    assert rows[1]["quality"] == "missing" and rows[1]["reading"] is None


def test_import_has_no_streamlit_ui_side_effects(monkeypatch):
    class ForbiddenStreamlit:
        def __getattr__(self, key):
            raise AssertionError("Import-time Streamlit access: " + key)
    monkeypatch.setitem(sys.modules, "streamlit", ForbiddenStreamlit())
    import enhanced_dashboard
    importlib.reload(enhanced_dashboard)


def test_real_page_explicit_generation_replay_and_separate_truth():
    pytest.importorskip("enhanced_simulation")
    pytest.importorskip("enhanced_analytics")
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parent / "pages/2_Enhanced_Dumper_Simulator.py"), default_timeout=30).run()
    assert not app.exception
    assert "enhanced_preview" not in app.session_state
    def widget(widgets, label):
        return next(item for item in widgets if item.label == label)
    widget(app.number_input, "Episode duration (s)").set_value(30.0)
    widget(app.number_input, "Number of scheduled events").set_value(2).run()
    # Repeated same-kind schedules remain independent and support sensor modes.
    app.selectbox(key="enhanced_event_0_kind").set_value("sensor_fault").run()
    app.selectbox(key="enhanced_event_1_kind").set_value("sensor_fault").run()
    app.selectbox(key="enhanced_event_0_mode").set_value("missing")
    app.number_input(key="enhanced_event_0_onset").set_value(0.0)
    app.number_input(key="enhanced_event_1_onset").set_value(0.0)
    widget(app.button, "Generate synthetic episode").click().run()
    assert not app.exception
    assert len(app.session_state["enhanced_preview"]["events"]) == 2
    assert len(app.tabs) == 4
    assert len(app.dataframe) >= 2
    widget(app.slider, "Replay sample").set_value(3).run()
    assert not app.exception
    widget(app.number_input, "Episode seed").set_value(900).run()
    assert app.session_state["enhanced_preview"]["config"]["seed"] == 793
    assert any("Controls have changed" in item.value for item in app.info)


def test_page_handles_invalid_config_and_empty_generation(monkeypatch):
    pytest.importorskip("enhanced_simulation")
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import enhanced_dashboard
    def invalid(*_):
        raise ValueError("Invalid sampling interval")
    monkeypatch.setattr(enhanced_dashboard, "generate_preview", invalid)
    app = AppTest.from_file(str(Path(__file__).parent / "pages/2_Enhanced_Dumper_Simulator.py"), default_timeout=30).run()
    assert not app.exception
    next(item for item in app.button if item.label == "Generate synthetic episode").click().run()
    assert not app.exception
    assert any("Invalid sampling interval" in item.value for item in app.error)
    monkeypatch.setattr(enhanced_dashboard, "generate_preview", lambda *_: {"samples": [], "diagnostics": [], "config": {}})
    next(item for item in app.button if item.label == "Generate synthetic episode").click().run()
    assert not app.exception
    assert any("no samples" in item.value for item in app.info)


def test_page_replay_missing_reading_and_download_controls(monkeypatch):
    pytest.importorskip("enhanced_simulation")
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import enhanced_dashboard
    def preview(config, events):
        samples = [{**sample(), "elapsed_s": float(index)} for index in range(4)]
        diagnostic = {"elapsed_s": 0.0, "overall_health": None, "subsystem_health": {"engine": None},
                      "hypotheses": [{"fault": "possible_sensor_issue", "evidence": ["Missing pressure reading"],
                                      "alternatives": ["Connection failure"], "possible_cause": "Measurement unavailable",
                                      "mitigation": "Inspect sensor connection"}], "recommended_action": "Inspect sensor"}
        return {"config": config, "events": events, "samples": samples, "diagnostics": [diagnostic] * 4}
    monkeypatch.setattr(enhanced_dashboard, "generate_preview", preview)
    app = AppTest.from_file(str(Path(__file__).parent / "pages/2_Enhanced_Dumper_Simulator.py"), default_timeout=30).run()
    next(item for item in app.button if item.label == "Generate synthetic episode").click().run()
    assert not app.exception
    assert len(app.get("download_button")) == 4
    assert [tab.label for tab in app.tabs] == ["Live / replay sensors", "Observation-only RCA", "Injected truth (separate)", "Downloads"]
    assert any(metric.value == "Unavailable" for metric in app.metric)
    next(item for item in app.slider if item.label == "Replay sample").set_value(2).run()
    assert not app.exception
    assert app.session_state["enhanced_sample"] == 2
    next(item for item in app.checkbox if item.label.startswith("Autoplay replay")).set_value(True).run()
    assert not app.exception
    assert app.session_state["enhanced_sample"] == 3
    app.run()
    assert not app.exception and app.session_state["enhanced_sample"] == 3
