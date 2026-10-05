"""In-memory enhanced simulator UI; import is safe without Streamlit or a session.

Run ``streamlit run enhanced_dashboard.py`` or use the multipage entry. Export
helpers return strings only: generation never writes telemetry to disk.
"""
from __future__ import annotations

import csv
import io
import json
import math
from typing import Any


CONDITION_SIGNALS = (
    "engine.oil_condition_pct", "transmission.oil_condition_pct",
    "suspension.hydraulic_oil_condition_pct", "coolant_condition_pct",
)
SENSOR_TARGETS = (
    "engine.rpm", "engine.load_pct", "engine.block_temp_c", "engine.oil_temp_c",
    "engine.exhaust_temp_c", "engine.oil_pressure_kpa", "engine.block_vibration_mm_s",
    "engine.oil_condition_pct", "engine.oil_level_pct", "engine.coolant_temp_c",
    "engine.coolant_pump_temp_c", "engine.coolant_flow_kg_s", "engine.thermostat_open_pct",
    "transmission.oil_temp_c", "transmission.oil_pressure_kpa", "transmission.vibration_mm_s",
    "transmission.oil_condition_pct", "suspension.hydraulic_oil_condition_pct",
    "suspension.actuator_vibration_mm_s", "coolant_condition_pct", "brake.pad_wear_pct",
    "brake.stroke_pct", "brake.disc_temp_c",
)
_SAMPLE_INPUT_KEYS = ("elapsed_s", "timestamp", "context", "observations")
_ID_KEYS = ("elapsed_s", "timestamp", "engine_id", "episode_id", "base_trajectory_id")


def observation_inputs(samples):
    """Explicitly keep injected truth and schedule out of diagnostic calls."""
    return [{key: sample[key] for key in _SAMPLE_INPUT_KEYS if key in sample} for sample in samples]


def generate_preview(config, events, *, simulator=None, analyser=None):
    """Generate an ephemeral episode, validating its parallel diagnostic output."""
    if simulator is None:
        from enhanced_simulation import simulate_enhanced
        simulator = simulate_enhanced
    if analyser is None:
        from enhanced_analytics import analyse_episode
        analyser = analyse_episode
    result = simulator(config, events)
    samples = result.get("samples", [])
    diagnostics = list(analyser(observation_inputs(samples))) if samples else []
    if len(diagnostics) != len(samples):
        raise ValueError("Diagnostic output does not match the generated sample count.")
    return {**result, "diagnostics": diagnostics}


def _csv(rows):
    if not rows:
        return ""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def observations_csv(samples):
    """Readings/context/identity only; no truth, expected values or inference."""
    rows = []
    for sample in samples:
        row = {key: sample.get(key) for key in _ID_KEYS}
        row.update({"context." + key: value for key, value in sample.get("context", {}).items()})
        row.update(sample.get("observations", {}))
        rows.append(row)
    return _csv(rows)


def truth_csv(samples):
    rows = []
    for sample in samples:
        truth = sample.get("truth", {})
        row = {key: sample.get(key) for key in _ID_KEYS}
        row["active_faults"] = json.dumps(truth.get("active_faults", []), sort_keys=True)
        row["event_phases"] = json.dumps(truth.get("event_phases", {}), sort_keys=True)
        for group in ("fault_levels", "latent_wear"):
            row.update({group + "." + key: value for key, value in truth.get(group, {}).items()})
        for key, value in truth.items():
            if key not in ("active_faults", "event_phases", "fault_levels", "latent_wear"):
                row[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
        rows.append(row)
    return _csv(rows)


def download_payloads(preview):
    """Build four separate downloads, without file-system side effects."""
    def encode(value):
        return json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    return {
        "observations.csv": observations_csv(preview.get("samples", [])),
        "truth_labels.csv": truth_csv(preview.get("samples", [])),
        "diagnostics.json": encode({"source": "observation_only_heuristics", "trained_dl_model": False,
                                    "rul_validated": False, "diagnostics": preview.get("diagnostics", [])}),
        "config_events.json": encode({"config": preview.get("config", {}), "events": preview.get("events", []),
                                     "metadata": preview.get("metadata", {})}),
    }


def sensor_rows(sample, diagnostic):
    return [{"signal": key, "reading": value, "expected": diagnostic.get("expected", {}).get(key),
             "residual": diagnostic.get("residuals", {}).get(key),
             "rate_per_s": diagnostic.get("rates", {}).get(key),
              "quality": diagnostic.get("data_quality", {}).get(key, "missing" if value is None else "unknown"),
             "status": diagnostic.get("statuses", {}).get(key, "unknown")}
            for key, value in sample.get("observations", {}).items()]


def _number(value, suffix=""):
    return f"{value:.1f}{suffix}" if isinstance(value, (int, float)) and math.isfinite(value) else "Unavailable"


def _config_controls(st):
    with st.expander("Virtual truck and operating configuration", expanded=True):
        a, b = st.columns(2)
        engine_id = a.text_input("Virtual truck ID", "virtual-001")
        episode_id = b.text_input("Episode ID", "episode-001")
        a, b, c = st.columns(3)
        seed = a.number_input("Episode seed", 0, 2147483647, 793)
        engine_seed = b.number_input("Virtual truck reference seed", 0, 2147483647, 793)
        scenario = c.selectbox("Scenario", ["ideal", "stress", "overhaul"])
        a, b, c = st.columns(3)
        payload = a.selectbox("Fixed payload", ["empty", "light", "medium", "heavy", "overload"], index=2)
        start = b.selectbox("Start condition", ["warm", "cold"])
        ambient = c.number_input("Ambient temperature (°C)", -40.0, 60.0, 25.0)
        a, b, c = st.columns(3)
        duration = a.number_input("Episode duration (s)", 5.0, 3600.0, 1800.0, step=5.0)
        dt = b.number_input("Sampling interval dt (s)", 1.0, 60.0, 5.0)
        cycle = c.selectbox("Drive cycle", ["c-wtvc", "synthetic_route"])
        with st.expander("Synthetic wear and noise controls"):
            wear = st.slider("Initial wear", 0.0, 1.0, 0.0)
            noise = st.slider("Noise scale", 0.0, 3.0, 1.0)
            wear_scale = st.number_input("Wear time scale", 0.01, 100.0, 1.0)
            hours = st.number_input("Initial operating hours", 0.0, 1000000.0, 0.0)
            st.caption("Accelerated aging is a synthetic assumption, not a validated physical aging rate.")
        return dict(engine_id=engine_id, episode_id=episode_id, seed=seed, engine_seed=engine_seed,
                    scenario=scenario, payload_class=payload, start_condition=start, ambient_c=ambient,
                    duration_s=duration, timestep_s=dt, drive_cycle=cycle, initial_wear=wear,
                    noise_scale=noise, wear_time_scale=wear_scale, initial_operating_hours=hours)


def _event_controls(st, catalog, config):
    with st.expander("Injected fault schedule — simulator controls only", expanded=True):
        count = st.number_input("Number of scheduled events", 0, 16, 0, key="enhanced_event_count")
        st.caption("Events may overlap or repeat, including multiple events of the same kind. These controls are not diagnostic evidence.")
        events = []
        for index in range(count):
            prefix = f"enhanced_event_{index}_"
            with st.expander(f"Event {index + 1}", expanded=True):
                kind = st.selectbox("Catalogue fault", list(catalog), key=prefix + "kind")
                description = catalog[kind]
                st.caption(f"{description.get('subsystem', '')}: {description.get('possible_cause', '')}")
                a, b, c = st.columns(3)
                onset = a.number_input("Onset (s)", min_value=0.0, value=300.0, key=prefix + "onset")
                duration = b.number_input("Active duration (s)", min_value=0.1, value=600.0, key=prefix + "duration")
                severity = c.slider("Severity", 0.0, 1.0, 0.8, key=prefix + "severity")
                a, b, c = st.columns(3)
                progression = a.slider("Progression", 0.0, 1.0, 0.2, key=prefix + "progression")
                recovery = b.number_input("Recovery (s)", min_value=0.0, value=120.0, key=prefix + "recovery")
                recurrence = c.number_input("Recurrence (s; 0 = once)", min_value=0.0, value=0.0, key=prefix + "recurrence")
                event = dict(kind=kind, onset_s=onset, duration_s=duration, severity=severity,
                             progression=progression, recovery_s=recovery, recurrence_s=recurrence or None,
                             event_id=f"event-{index + 1:03d}")
                if kind == "sensor_fault":
                    a, b = st.columns(2)
                    targets = list(dict.fromkeys(SENSOR_TARGETS + tuple(st.session_state.get("enhanced_signal_names", []))))
                    event["target_signal"] = a.selectbox("Sensor target", targets, key=prefix + "target")
                    event["sensor_mode"] = b.selectbox("Sensor mode", ["biased", "missing", "frozen", "drifting", "noisy"], key=prefix + "mode")
                    event["bias"] = st.number_input("Bias / drift / noise magnitude (signal units)", value=20.0, key=prefix + "bias")
                if onset >= config["duration_s"]:
                    st.warning("This event starts at or after the episode ends.")
                if duration < config["timestep_s"]:
                    st.warning("The selected sampling interval may miss this event.")
                events.append(event)
        st.caption("Recovery removes active disturbance; permanent synthetic mechanical wear can remain.")
        return events


def _sensor_panel(st, preview, index):
    samples, diagnostics = preview["samples"], preview["diagnostics"]
    sample, diagnostic = samples[index], diagnostics[index]
    st.caption(f"Synthetic sample {index + 1}/{len(samples)} · t={sample['elapsed_s']:.1f} s · {sample.get('timestamp', '')}")
    a, b = st.columns(2)
    a.metric("Overall heuristic health", _number(diagnostic.get("overall_health"), "%"))
    b.metric("RUL", "Not validated / unavailable")
    columns = st.columns(2)
    for offset, signal in enumerate(CONDITION_SIGNALS):
        columns[offset % 2].metric(signal, _number(sample.get("observations", {}).get(signal), "%"))
    st.markdown("**Subsystem heuristic health**")
    st.dataframe([{"subsystem": key, "heuristic_health": value} for key, value in diagnostic.get("subsystem_health", {}).items()], use_container_width=True)
    st.markdown("**Numerical sensors, residuals, rates and data quality**")
    st.dataframe(sensor_rows(sample, diagnostic), use_container_width=True, hide_index=True)
    available = list(sample.get("observations", {}))
    selected = st.multiselect("Trend signals (compare compatible units)", available,
                              default=[key for key in CONDITION_SIGNALS if key in available], key="enhanced_trends")
    if selected:
        # Pandas comes with Streamlit; use a numeric elapsed-time index, not row numbers.
        import pandas as pd
        history = samples[:index + 1]
        values = pd.DataFrame([{key: row.get("observations", {}).get(key) for key in selected} for row in history],
                              index=[row["elapsed_s"] for row in history])
        values.index.name = "elapsed_s"
        st.line_chart(values, use_container_width=True)
        residuals = pd.DataFrame([{key: row.get("residuals", {}).get(key) for key in selected} for row in diagnostics[:index + 1]], index=values.index)
        st.caption("Observation-only residual trends: reading minus context-derived healthy prior (not hidden paired simulator readings).")
        st.line_chart(residuals, use_container_width=True)
    return sample, diagnostic


def _hypothesis_panel(st, diagnostic):
    st.info("Observation-only possible causes. Heuristic confidence is not a probability; root cause is not confirmed.")
    st.write("Recommended action:", diagnostic.get("recommended_action", "No action available."))
    hypotheses = diagnostic.get("hypotheses", [])
    if not hypotheses:
        st.caption("No heuristic fault hypothesis at this sample. This does not prove absence of a fault.")
    for hypothesis in hypotheses:
        with st.expander(str(hypothesis.get("fault", "Possible cause")), expanded=True):
            st.write("Possible cause:", hypothesis.get("possible_cause", "Unavailable"))
            st.write("Evidence:", hypothesis.get("evidence", []))
            st.write("Alternatives:", hypothesis.get("alternatives", []))
            st.write("Mitigation / inspection:", hypothesis.get("mitigation", "Unavailable"))
            st.caption("Confidence: heuristic_not_probability · independent inspection required for confirmation")


def _waveform_panel(st, sample):
    with st.expander("Optional bounded synthetic vibration waveform"):
        st.caption("One second at 4096 Hz. Assumed spectral components, not a measured waveform or trained diagnostic input.")
        if st.button("Generate waveform demonstration"):
            try:
                from enhanced_simulation import synthesize_vibration
                result = synthesize_vibration({key: value for key, value in sample.items() if key != "truth"},
                                              sample_rate_hz=4096, duration_s=1.0, seed=793)
                st.line_chart(result["waveform"])
                st.json(result.get("metadata", {}))
                from vibration_analysis import spectral_features
                st.dataframe([spectral_features(result)], hide_index=True, use_container_width=True)
            except (ImportError, ValueError, TypeError, KeyError) as exc:
                st.warning(f"Waveform demonstration unavailable: {exc}")


def _render_results(st, preview, mode, autoplay=False):
    last = len(preview["samples"]) - 1
    current = min(max(st.session_state.get("enhanced_sample", 0), 0), last)
    st.session_state["enhanced_sample"] = min(current + 1, last) if autoplay else current
    if mode == "Replay" and last:
        index = st.slider("Replay sample", 0, last, key="enhanced_sample")
    else:
        index = last
    sensors, predictions, truth, exports = st.tabs(["Live / replay sensors", "Observation-only RCA", "Injected truth (separate)", "Downloads"])
    with sensors:
        sample, diagnostic = _sensor_panel(st, preview, index)
        _waveform_panel(st, sample)
    with predictions:
        _hypothesis_panel(st, diagnostic)
    with truth:
        st.warning("Injected simulator truth, not a prediction or root-cause confirmation. Excluded from diagnostic inputs.")
        st.json(sample.get("truth", {}))
        st.markdown("**Generated event schedule**")
        st.json(preview.get("events", []))
    with exports:
        st.caption("Observations CSV contains readings, operating context and identifiers only. Truth labels and heuristic diagnostics are separate artifacts.")
        try:
            payloads = download_payloads(preview)
        except (TypeError, ValueError) as exc:
            st.error(f"Export unavailable: {exc}")
        else:
            for name, data in payloads.items():
                st.download_button("Download " + name, data, file_name=name,
                                   mime="text/csv" if name.endswith(".csv") else "application/json", key="enhanced_download_" + name)
        with st.expander("Reproducibility and synthetic limitations"):
            st.json(preview.get("metadata", {}))
            st.json(preview.get("config", {}))


def main():
    import streamlit as st
    st.set_page_config(page_title="Enhanced Dumper Simulator", page_icon="🚚", layout="wide")
    st.title("Enhanced Dumper Simulator")
    st.warning("Synthetic prototype · physics and diagnostics are not validated · no trained DL model · RUL is not validated.")
    st.caption("Generate an in-memory episode explicitly. Downloads are separate user actions; no telemetry is silently saved to disk.")
    try:
        from enhanced_simulation import FAULT_CATALOG
    except ImportError as exc:
        st.error(f"Enhanced simulation module unavailable: {exc}")
        return
    config = _config_controls(st)
    events = _event_controls(st, FAULT_CATALOG, config)
    if st.button("Generate synthetic episode", type="primary"):
        try:
            with st.spinner("Generating readings and independent heuristic diagnostics…"):
                preview = generate_preview(config, events)
            st.session_state["enhanced_preview"] = preview
            st.session_state["enhanced_generated_request"] = {"config": config, "events": events}
            st.session_state["enhanced_sample"] = 0
            st.session_state["enhanced_signal_names"] = list(preview["samples"][0]["observations"]) if preview["samples"] else []
            st.session_state.pop("enhanced_generation_error", None)
        except (ValueError, TypeError, OverflowError, ImportError) as exc:
            st.session_state["enhanced_generation_error"] = str(exc)
    if st.session_state.get("enhanced_generation_error"):
        st.error("Generation failed: " + st.session_state["enhanced_generation_error"])
    preview = st.session_state.get("enhanced_preview")
    if preview is None:
        st.info("Configure a virtual truck and optional fault events, then select Generate synthetic episode.")
        return
    if not preview.get("samples"):
        st.info("The simulation returned no samples. Adjust the duration and sampling interval, then generate again.")
        return
    if st.session_state.get("enhanced_generated_request") != {"config": config, "events": events}:
        st.info("Controls have changed. Display and downloads still refer to the last generated episode; generate again to apply changes.")
    st.caption(f"Displayed episode: {preview['config'].get('engine_id', '')} / {preview['config'].get('episode_id', '')}")
    mode = st.radio("View", ["Replay", "Latest synthetic snapshot"], horizontal=True)
    fragment = getattr(st, "fragment", getattr(st, "experimental_fragment", None))
    autoplay = st.checkbox("Autoplay replay (one sample per second)", disabled=mode != "Replay" or fragment is None)
    autoplay = autoplay and mode == "Replay"
    st.caption("Replay uses the generated synthetic episode; it is not connected to a physical truck.")
    if fragment is not None:
        @fragment(run_every="1s" if autoplay else None)
        def replay_panel():
            _render_results(st, preview, mode, autoplay)
        replay_panel()
    else:
        _render_results(st, preview, mode)


if __name__ == "__main__":
    main()
