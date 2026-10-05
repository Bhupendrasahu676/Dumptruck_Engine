"""CPAI-aligned synthetic engine evidence lab. Run with Streamlit on port 8503."""
import json
import os
from pathlib import Path

import streamlit as st

from engine_lab.simulation import FAULT_KINDS, PAYLOAD_CLASSES, SENSORS, WEAR_KEYS, default_schedule, expected_readings
from engine_lab.storage import DISPOSITIONS, EngineStore
from engine_lab.workflow import export_jsonl, export_observations_csv, preview_test, rul_for_engine

ROOT = Path(__file__).resolve().parent


def engine_controls(store):
    st.sidebar.header("Engine digital thread")
    with st.sidebar.expander("Register engine"):
        with st.form("register_engine"):
            name = st.text_input("Engine name", value="DT-793 / Engine 01")
            initial = st.number_input("Initial operating hours", min_value=0.0, max_value=1000000.0, value=0.0)
            st.caption("Hours do not infer wear. A new registration begins at a synthetic zero-wear baseline.")
            if st.form_submit_button("Create persistent engine"):
                try:
                    new = store.create_engine(name, initial)
                    st.session_state["engine_choice"] = new["engine_id"]
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    engines = store.list_engines()
    if not engines:
        st.info("Register an engine in the sidebar to start. Tests and repairs are saved only by explicit actions.")
        st.stop()
    names = {e["engine_id"]: e["name"] + " · " + e["engine_id"][:8] for e in engines}
    selected = st.sidebar.selectbox("Engine", list(names), format_func=names.get, key="engine_choice")
    engine = store.get_engine(selected)
    st.sidebar.caption(f"Engine_ID: {selected}")
    st.sidebar.caption(f"Overhaul_ID: {engine['current_overhaul_id']}")
    st.sidebar.metric("Saved operating hours", f"{engine['operating_hours']:.4f}")
    return engine


def setup_test(engine):
    with st.expander("Test setup & hidden fault schedule", expanded=True):
        st.caption("Simulator controls are ground truth, NOT diagnostic inputs. Fixed payload; RPM and engine demand vary independently.")
        a, b, c = st.columns(3)
        start = a.selectbox("Start condition", ["cold", "warm"])
        payload = b.selectbox("Fixed payload category", PAYLOAD_CLASSES, index=2)
        seed = c.number_input("Random seed", min_value=0, max_value=2147483647, value=793, step=1)
        a, b, c = st.columns(3)
        duration = a.number_input("Test duration (seconds)", min_value=10, max_value=36000, value=600, step=10)
        timestep = b.selectbox("Sampling interval (seconds)", [0.5, 1.0, 2.0, 5.0], index=1)
        ambient = c.number_input("Ambient temperature (°C)", min_value=-40.0, max_value=60.0, value=22.0)
        st.caption("Reduced-order repeating idle/ramp/load/coast profile; not a certified dynamometer protocol or raw vibration waveform.")
        kinds = st.multiselect("Add scheduled faults", FAULT_KINDS, format_func=lambda x: x.replace("_", " ").title())
        schedules = []
        for kind in kinds:
            defaults = default_schedule(kind)
            st.markdown("**" + kind.replace("_", " ").title() + "**")
            a, b, c = st.columns(3)
            onset = a.number_input("Onset (s)", min_value=0.0, value=defaults["onset_s"], key=kind + "_onset")
            active = b.number_input("Active duration (s)", min_value=0.5, value=defaults["duration_s"], key=kind + "_duration")
            severity = c.slider("Severity", 0.0, 1.0, defaults["severity"], key=kind + "_severity")
            a, b, c = st.columns(3)
            recurrence = a.number_input("Repeat every (s; 0 = once)", min_value=0.0, value=0.0, key=kind + "_repeat")
            progression = b.slider("Progression over event", 0.0, 1.0, defaults["progression"], key=kind + "_progression")
            recovery = c.number_input("Recovery after event (s)", min_value=0.0, value=defaults["recovery_s"], key=kind + "_recovery")
            schedule = dict(kind=kind, onset_s=onset, duration_s=active, severity=severity,
                            recurrence_s=recurrence, progression=progression, recovery_s=recovery)
            if kind == "thermostat_malfunction":
                schedule["subtype"] = st.selectbox("Thermostat behaviour", ["delayed", "stuck_open", "stuck_closed"])
            if kind == "sensor_fault":
                schedule["subtype"] = st.selectbox("Sensor behaviour", ["biased", "frozen", "missing"])
                schedule["sensor"] = st.selectbox("Affected sensor", SENSORS)
            if onset >= duration:
                st.warning("This fault starts after the current test ends.")
            if active < timestep or (recurrence and recurrence < timestep):
                st.warning("Sampling may miss this event; choose a finer interval.")
            schedules.append(schedule)
        st.caption("Recovery removes the active disturbance, not accumulated physical damage. Repair reductions require a recorded overhaul.")
        if st.button("Generate preview — no persistence", type="primary"):
            try:
                with st.spinner("Simulating observations, then running independent inference…"):
                    st.session_state["lab_preview"] = preview_test(engine, {
                        "seed": seed, "start_condition": start, "payload_class": payload,
                        "duration_s": duration, "timestep_s": timestep, "ambient_c": ambient}, schedules)
                st.session_state["view_" + engine["engine_id"]] = "preview"
            except (ValueError, OverflowError) as exc:
                st.error(str(exc))


def choose_test(store, engine):
    preview = st.session_state.get("lab_preview")
    if preview and preview["config"]["engine_id"] != engine["engine_id"]:
        preview = None
    tests = store.list_tests(engine["engine_id"])
    options = (["preview"] if preview else []) + [t["test_id"] for t in reversed(tests)]
    if not options:
        st.info("Generate a preview or select an engine with saved tests.")
        return None, False
    key = "view_" + engine["engine_id"]
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    selected = st.selectbox("Test to inspect", options, key=key,
                            format_func=lambda x: "Latest preview (explicit save required)" if x == "preview" else "Saved · " + x)
    result = preview if selected == "preview" else store.get_test(selected)
    saved_ids = {t["test_id"] for t in tests}
    saved = result["config"]["test_id"] in saved_ids
    st.caption("Test_ID: " + result["config"]["test_id"] + " · Overhaul_ID: " + result["config"]["overhaul_id"])
    if selected == "preview" and not saved:
        if st.button("Save test & advance engine hours / wear"):
            try:
                store.save_test(result)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc) + ". Generate a fresh preview if engine history changed.")
    else:
        st.success("Saved immutable test. Viewing, replaying and downloading do not advance operating hours.")
    return result, saved


def evidence_view(store, result, saved):
    samples, analyses = result["samples"], result["analyses"]
    test_id = result["config"]["test_id"]
    index = st.slider("Inspect sample", 0, len(samples) - 1, len(samples) - 1, key="sample_" + test_id)
    sample, analysis = samples[index], analyses[index]
    a, b, c, d = st.columns(4)
    score = analysis["health"]["score"]
    a.metric("Heuristic engine health", "Unknown" if score is None else f"{score:.1f} / 100")
    b.metric("Data quality", analysis["quality"]["status"])
    c.metric("Inferred state", analysis["state"].replace("_", " "))
    d.metric("Elapsed test time", f"{sample['elapsed_s']:.1f} s")
    st.caption("Synthetic, unvalidated residual-based score — not a service release or a remaining-life estimate. Evidence: " + analysis["evidence_strength"])
    st.info(analysis["action"])
    expected = expected_readings(sample["context"])
    rows = []
    for signal in SENSORS:
        if signal == "egt_c":
            for n, value in enumerate(sample["observations"][signal]):
                rows.append(dict(signal=f"EGT cylinder {n + 1} (°C)", observed=value,
                                 expected=expected[signal][n], residual=analysis["residuals"][signal][n]))
        else:
            rows.append(dict(signal=signal, observed=sample["observations"][signal],
                             expected=expected[signal], residual=analysis["residuals"][signal]))
    st.dataframe(rows, hide_index=True, use_container_width=True)
    if analysis["quality"]["issues"]:
        st.warning("Measurement problems are not confirmed mechanical damage.")
        st.dataframe(analysis["quality"]["issues"], hide_index=True, use_container_width=True)
    st.subheader("1 · Anomaly detected")
    if analysis["anomalies"]:
        st.dataframe(analysis["anomalies"], hide_index=True, use_container_width=True)
    else:
        st.write("No usable reading currently exceeds its state-conditioned residual limit. This does not prove fault-free operation.")
    st.subheader("2 · Probable fault — unconfirmed hypothesis")
    for diagnosis in analysis["diagnosis"]:
        st.markdown("**" + diagnosis["hypothesis"] + "**")
        st.write("Observed evidence: " + "; ".join(diagnosis["evidence"]))
        st.write("Alternatives: " + "; ".join(diagnosis["alternatives"]))
        st.write("Checks required: " + "; ".join(diagnosis["required_confirmation"]))
    if not analysis["diagnosis"]:
        st.write("Insufficient persistent evidence for a fault hypothesis.")
    st.subheader("3 · Root cause — technician confirmation")
    st.write(analysis["root_cause"])
    feedback = store.list_feedback(test_id) if saved else []
    current_feedback = [f for f in feedback if f["sample_index"] == index]
    if current_feedback:
        st.caption("Human review records for this sample (self-reported; not independent certification). The original inference remains unchanged.")
        st.dataframe(current_feedback, hide_index=True, use_container_width=True)
    if saved:
        with st.expander("Add technician disposition"):
            with st.form("feedback_" + test_id):
                disposition = st.selectbox("Disposition", sorted(DISPOSITIONS), index=sorted(DISPOSITIONS).index("uncertain"))
                cause = st.text_input("Cause / finding (do not copy an injection label without inspection)")
                evidence = st.text_area("Inspection / repair / independent measurement evidence")
                technician = st.text_input("Technician name or identifier")
                st.caption(f"Applies to sample {index} at {sample['elapsed_s']:.1f} s. Confirmed fault requires cause, evidence and technician.")
                if st.form_submit_button("Append review record"):
                    try:
                        store.record_feedback(test_id, index, disposition, cause, evidence, technician)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
    else:
        st.caption("Save this test before recording technician evidence.")
    with st.expander("Observed trajectories & event timeline", expanded=True):
        signal = st.selectbox("Trend channel", SENSORS[:-1])
        trend = [{"elapsed_s": s["elapsed_s"], "observed": s["observations"][signal],
                  "expected": expected_readings(s["context"])[signal]} for s in samples]
        st.line_chart(trend, x="elapsed_s", y=["observed", "expected"])
        timeline = [{"elapsed_s": s["elapsed_s"], "health_score": a["health"]["score"],
                     "state": a["state"], "quality": a["quality"]["status"],
                     "anomalies": len(a["anomalies"]), "hypotheses": len(a["diagnosis"])}
                    for s, a in zip(samples, analyses)]
        st.line_chart(timeline, x="elapsed_s", y="health_score")
        st.dataframe(timeline, hide_index=True, use_container_width=True)
    with st.expander("Download separate evidence layers"):
        st.caption("Identifiers are grouping metadata, not training inputs. Predictions and hidden truth must never be fed back as sensor features.")
        stem = test_id[:8]
        st.download_button("Observation CSV", export_observations_csv(result), stem + "_observations.csv", "text/csv")
        for layer in ("observations", "predictions"):
            st.download_button(layer.title() + " JSONL", export_jsonl(result, layer), stem + "_" + layer + ".jsonl", "application/x-ndjson")
        st.download_button("Human feedback JSON", json.dumps(feedback, indent=2), stem + "_feedback.json", "application/json")
    with st.expander("Simulator ground truth — evaluator only; never diagnosis evidence"):
        st.warning("These are injected labels and latent wear, NOT detected or confirmed causes. Hidden from the inference boundary.")
        st.json({"schedule": result.get("schedules", []), "sample_truth": sample["truth"]})
        st.download_button("Ground truth JSONL (separate)", export_jsonl(result, "truth"), test_id[:8] + "_truth.jsonl", "application/x-ndjson")
        replay = {key: result.get(key) for key in ("schema_version", "config", "schedules", "initial_wear", "provenance")}
        st.download_button("Reproduction configuration (includes hidden controls)", json.dumps(replay, indent=2), test_id[:8] + "_reproduction.json", "application/json")


def lifecycle_view(store, engine):
    st.subheader("Persistent component history & operating hours")
    st.caption("Engine / overhaul / test IDs are persistent UUIDs. Wear shown here is simulator truth, not a measured diagnosis.")
    st.json(engine["wear"])
    st.dataframe(store.list_overhauls(engine["engine_id"]), hide_index=True, use_container_width=True)
    history = store.health_history(engine["engine_id"])
    if history:
        st.caption("Final usable heuristic health per saved test. Compare matching operating conditions; do not interpret score as life fraction.")
        st.dataframe(history, hide_index=True, use_container_width=True)
        # Break the trajectory at maintenance resets rather than joining unrelated baselines.
        chart = [dict(row, segment=row["overhaul_id"][:8]) for row in history]
        st.line_chart(chart, x="operating_hours", y="health_score", color="segment")
    with st.expander("Record overhaul / component repair"):
        with st.form("repair_" + engine["engine_id"]):
            notes = st.text_area("Overhaul / component work notes")
            repairs = {key: st.slider(key.title() + " wear removed", 0.0, 1.0, 0.0, key="repair_" + key) for key in WEAR_KEYS}
            st.caption("0 = unrepaired; 1 = all simulated wear removed. Partial repair can leave abnormalities. Total engine hours never reset.")
            if st.form_submit_button("Record new overhaul & apply repair"):
                if not notes.strip():
                    st.error("Describe the component work before recording it.")
                else:
                    try:
                        store.create_overhaul(engine["engine_id"], notes, repairs)
                        st.session_state.pop("lab_preview", None)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))


def rul_view(store, engine):
    st.subheader("RUL feasibility & uncertainty gate")
    st.caption("Uses only saved history from the current overhaul. No health-to-hours conversion or fixed confidence percentages.")
    evidence = None
    uploaded = st.file_uploader("Optional independently prepared calibration evidence JSON", type="json")
    if uploaded is not None:
        try:
            if uploaded.size > 5_000_000:
                raise ValueError("Evidence file exceeds the 5 MB limit")
            evidence = json.loads(uploaded.getvalue())
            if not isinstance(evidence, dict):
                raise ValueError("Evidence must be a JSON object")
        except (ValueError, UnicodeDecodeError) as exc:
            st.error("Invalid evidence: " + str(exc))
            return
    result = rul_for_engine(store, engine, evidence)
    st.write("**Status:** " + result["status"])
    if result["estimate_hours"] is None:
        st.warning("RUL unavailable — feasibility requirements not met.")
    else:
        st.metric("Synthetic RUL prediction", f"{result['estimate_hours']:,.1f} h")
        low, high = result["interval_hours"]
        st.write(f"90% nominal empirical interval: {low:,.1f}–{high:,.1f} h (not an individual failure probability).")
    for reason in result["reasons"]:
        st.write("• " + reason)
    st.caption("Minimum lab policy: 5 degrading tests spanning 100 h, one overhaul, defined failure outcome, frozen external model, ≥19 independent calibration engines and ≥10 untouched test engines. Cohorts and current engine must be disjoint. These thresholds are project policy, not CPAI certification criteria.")
    st.caption("This application has synthetic history only. Field-labelled evidence is rejected. No trained prognostic model or real failure outcomes are bundled.")
    st.download_button("Download RUL gate receipt", json.dumps(result, indent=2), "rul_gate.json", "application/json")


def main():
    st.set_page_config(page_title="Engine Evidence Lab", page_icon="🔧", layout="wide")
    st.title("Engine evidence & lifecycle lab")
    st.warning("SYNTHETIC RESEARCH ONLY · not validated for maintenance decisions. Human safety procedures take precedence.")
    st.caption("Observe → check quality → compare operating-state baseline → detect anomaly → propose fault → request technician confirmation.")
    path = Path(os.environ.get("ENGINE_LAB_DB", str(ROOT / "telemetry_logs" / "engine_lab.sqlite3")))
    store = EngineStore(path)
    engine = engine_controls(store)
    test_tab, history_tab, rul_tab = st.tabs(["Simulation & diagnosis", "History & overhaul", "RUL feasibility"])
    with test_tab:
        setup_test(engine)
        result, saved = choose_test(store, engine)
        if result:
            evidence_view(store, result, saved)
    with history_tab:
        lifecycle_view(store, engine)
    with rul_tab:
        rul_view(store, engine)


if __name__ == "__main__":
    main()
