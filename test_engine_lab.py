"""Behavioral regression tests for the additive engine evidence workflow."""
from copy import deepcopy
import json

import pytest

from engine_lab.analytics import analyse_samples, assess_rul
from engine_lab.simulation import FAULT_KINDS, SENSORS, default_schedule, expected_readings, simulate_test
from engine_lab.storage import EngineStore
from engine_lab.workflow import export_jsonl, export_observations_csv, observation_input, preview_test, rul_for_engine


def schedule(kind, **kwargs):
    return dict(default_schedule(kind), onset_s=20.0, duration_s=40.0, recovery_s=20.0, **kwargs)


def analyse(result):
    return analyse_samples([observation_input(s) for s in result["samples"]])


@pytest.mark.parametrize("kind", FAULT_KINDS)
def test_all_faults_have_onset_active_recovery_and_finite_output(kind):
    planned = schedule(kind)
    result = simulate_test({"duration_s": 100}, [planned])
    assert result == simulate_test({"duration_s": 100}, [planned])
    assert result["samples"][19]["truth"]["scheduled_events"] == []
    assert result["samples"][20]["truth"]["scheduled_events"][0]["phase"] == "active"
    assert result["samples"][60]["truth"]["scheduled_events"][0]["phase"] == "recovery"
    assert result["samples"][80]["truth"]["scheduled_events"] == []
    active = result["samples"][45]
    healthy = simulate_test({"duration_s": 100}, [])["samples"][45]
    assert active["observations"] != healthy["observations"]
    json.dumps(result, allow_nan=False)


def test_recurrence_progression_and_linear_recovery():
    planned = dict(default_schedule("bearing_degradation"), onset_s=10, duration_s=20,
                   recovery_s=10, recurrence_s=50, severity=0.4, progression=0.5)
    result = simulate_test({"duration_s": 110}, [planned])
    events = lambda t: result["samples"][t]["truth"]["scheduled_events"]
    assert events(10)[0]["effective_severity"] == pytest.approx(0.4)
    assert events(20)[0]["effective_severity"] == pytest.approx(0.5)
    assert events(35)[0]["effective_severity"] == pytest.approx(0.3)
    assert not events(40)
    assert events(60)[0]["occurrence"] == 1
    assert events(110)[0]["occurrence"] == 2


@pytest.mark.parametrize("subtype", ["missing", "biased", "frozen"])
@pytest.mark.parametrize("sensor", SENSORS)
def test_sensor_faults_do_not_create_physical_damage(subtype, sensor):
    config = {"duration_s": 100}
    baseline = simulate_test(config, [])
    faulty = simulate_test(config, [schedule("sensor_fault", subtype=subtype, sensor=sensor)])
    assert faulty["final_wear"] == baseline["final_wear"]
    for healthy, sample in zip(baseline["samples"], faulty["samples"]):
        assert healthy["truth"]["latent"] == sample["truth"]["latent"]
    assert faulty["samples"][45]["observations"][sensor] != baseline["samples"][45]["observations"][sensor]
    assert faulty["samples"][80]["observations"] == baseline["samples"][80]["observations"]
    json.dumps(faulty, allow_nan=False)


@pytest.mark.parametrize("start", ["cold", "warm"])
def test_healthy_warmup_idle_and_transitions_are_not_faults(start):
    result = simulate_test({"duration_s": 600, "start_condition": start}, [])
    predictions = analyse(result)
    assert not any(p["anomalies"] or p["diagnosis"] for p in predictions)
    assert all(p["quality"]["status"] == "good" for p in predictions)
    states = {p["state"] for p in predictions}
    assert {"steady_idle", "steady_loaded", "transition"} <= states
    if start == "cold":
        assert "cold_warmup" in states
    idle = expected_readings(result["samples"][0]["context"])
    loaded = expected_readings(result["samples"][70]["context"])
    assert idle["oil_pressure_kpa"] < loaded["oil_pressure_kpa"]


def test_independent_hypotheses_and_persistence_not_injected_answers():
    result = simulate_test({"duration_s": 400}, [default_schedule("bearing_degradation")])
    predictions = analyse(result)
    assert any(p["diagnosis"] for p in predictions)
    assert not predictions[60]["diagnosis"]  # newly started residual is not persistent
    assert all(p["root_cause"].startswith("Not confirmed") for p in predictions)
    corrupted = deepcopy(result)
    for sample in corrupted["samples"]:
        sample["truth"] = {"confirmed_cause": "fake root cause", "health": 0}
        sample["context"]["fault_name"] = "fake root cause"
        sample["observations"]["target_rul"] = 1
    assert analyse(corrupted) == predictions
    assert all(d["alternatives"] and d["required_confirmation"] for p in predictions for d in p["diagnosis"])


def test_low_pressure_high_demand_detected_but_sensor_bias_ambiguous():
    result = simulate_test({"duration_s": 400, "start_condition": "warm"}, [default_schedule("sensor_fault")])
    predictions = analyse(result)
    hypotheses = [d["hypothesis"] for p in predictions for d in p["diagnosis"]]
    assert any("Pressure-only" in h and "unresolved" in h for h in hypotheses)
    assert not any("Possible lubrication impairment" in h for h in hypotheses)


def test_missing_and_frozen_measurements_mask_health_not_fake_failure():
    for subtype in ("missing", "frozen"):
        planned = dict(default_schedule("sensor_fault"), onset_s=0, duration_s=300, subtype=subtype)
        result = simulate_test({"duration_s": 300}, [planned])
        predictions = analyse(result)
        invalid = [p for p in predictions if any(i["type"] == subtype for i in p["quality"]["issues"])]
        assert invalid
        assert all(p["health"]["score"] is None for p in invalid)
        assert all(p["residuals"]["oil_pressure_kpa"] is None for p in invalid)
        assert all(not p["diagnosis"] for p in invalid)


@pytest.mark.parametrize("subtype", ["delayed", "stuck_open", "stuck_closed"])
def test_thermostat_variants_depart_from_healthy_prior(subtype):
    result = simulate_test({"duration_s": 300}, [dict(default_schedule("thermostat_malfunction"), subtype=subtype)])
    assert any(p["diagnosis"] for p in analyse(result))


def test_fractional_duration_and_invalid_inputs():
    result = simulate_test({"duration_s": 10.5, "timestep_s": 2, "initial_hours": 100}, [])
    assert result["samples"][-1]["elapsed_s"] == 10.5
    assert result["samples"][-1]["operating_hours"] == pytest.approx(100 + 10.5 / 3600)
    for config in ({"duration_s": 0}, {"timestep_s": 0}, {"seed": True}, {"duration_s": float("nan")}, {"duration_s": 1e8}):
        with pytest.raises(ValueError):
            simulate_test(config, [])
    with pytest.raises(ValueError):
        simulate_test({}, [dict(default_schedule("bearing_degradation"), severity=2)])


def test_persistence_idempotency_overhaul_and_immutable_feedback(tmp_path):
    path = tmp_path / "engines.sqlite3"
    store = EngineStore(path)
    engine = store.create_engine("Truck A", 1000)
    result = preview_test(engine, {"duration_s": 120}, [dict(default_schedule("bearing_degradation"), onset_s=0)])
    assert store.list_tests(engine["engine_id"]) == []
    assert store.get_engine(engine["engine_id"])["operating_hours"] == 1000
    store.save_test(result)
    store.save_test(deepcopy(result))
    fresh = EngineStore(path)
    loaded = fresh.get_engine(engine["engine_id"])
    assert loaded["operating_hours"] == pytest.approx(1000 + 120 / 3600)
    assert loaded["wear"] == result["final_wear"]
    assert fresh.get_test(result["config"]["test_id"]) == result
    assert len(fresh.list_tests(engine["engine_id"])) == 1
    altered = deepcopy(result)
    altered["config"]["seed"] += 1
    with pytest.raises(ValueError, match="different content"):
        fresh.save_test(altered)
    stale = preview_test(engine, {"duration_s": 120}, [])
    with pytest.raises(ValueError, match="Stale"):
        fresh.save_test(stale)
    test_id = result["config"]["test_id"]
    with pytest.raises(ValueError):
        fresh.record_feedback(test_id, 60, "confirmed_fault", "bearing")
    fresh.record_feedback(test_id, 60, "confirmed_fault", "bearing", "Inspection found bearing pitting", "tech-1")
    assert len(fresh.list_feedback(test_id)) == 1
    assert fresh.get_test(test_id)["analyses"] == result["analyses"]
    overhaul = fresh.create_overhaul(engine["engine_id"], "Bearing repair, cooling not replaced", {"bearing": 0.8})
    repaired = fresh.get_engine(engine["engine_id"])
    assert repaired["current_overhaul_id"] == overhaul["overhaul_id"]
    assert repaired["wear"]["bearing"] == pytest.approx(loaded["wear"]["bearing"] * 0.2)
    assert repaired["wear"]["cooling"] == loaded["wear"]["cooling"]
    assert repaired["operating_hours"] == loaded["operating_hours"]
    assert len(fresh.list_overhauls(engine["engine_id"])) == 2
    before_overhaul = preview_test(loaded, {"duration_s": 10}, [])
    with pytest.raises(ValueError, match="current overhaul"):
        fresh.save_test(before_overhaul)
    continuation = preview_test(repaired, {"duration_s": 10}, [])
    assert continuation["samples"][0]["truth"]["wear"] == repaired["wear"]
    fresh.save_test(continuation)
    assert fresh.health_history(engine["engine_id"])


def test_incomplete_saved_result_rejected_without_hour_update(tmp_path):
    store = EngineStore(tmp_path / "test.db")
    engine = store.create_engine("A")
    result = preview_test(engine, {"duration_s": 10}, [])
    result["samples"].pop()
    result["analyses"].pop()
    with pytest.raises(ValueError, match="endpoints"):
        store.save_test(result)
    assert store.get_engine(engine["engine_id"])["operating_hours"] == 0


def test_export_layers_cannot_leak_targets_into_observations(tmp_path):
    engine = EngineStore(tmp_path / "test.db").create_engine("A")
    result = preview_test(engine, {"duration_s": 10}, [dict(default_schedule("bearing_degradation"), onset_s=0)])
    observations = export_jsonl(result, "observations")
    assert not any(word in observations for word in ("bearing_degradation", "truth", "health_score", "root_cause", "schedules", "wear"))
    row = json.loads(observations.splitlines()[0])
    assert row["engine_id"] == engine["engine_id"]
    assert "prediction" in json.loads(export_jsonl(result, "predictions").splitlines()[0])
    assert "ground_truth_not_model_input" in json.loads(export_jsonl(result, "truth").splitlines()[0])
    csv = export_observations_csv(result)
    assert csv.startswith("Engine_ID,Overhaul_ID,Test_ID,")
    assert "health" not in csv and "wear" not in csv


def rul_fixture():
    history = [{"test_id": f"test-{n}", "overhaul_id": "overhaul", "operating_hours": n * 25,
                "health_score": 100 - n * 3} for n in range(5)]
    meta = dict(model_id="locked-model", domain_id="lab-domain", failure_definition="defined-threshold",
                source="simulation")
    def outcome(prefix, n):
        return dict(meta, engine_id=f"{prefix}-{n}", operating_hours=100, remaining_hours=100,
                    failure_hours=208, failure_observed=True, predicted_before_failure=True)
    evidence = dict(meta, current_engine_id="current", training_engine_ids=["train-1"],
                    model_frozen_before_calibration=True,
                    prediction=dict(meta, engine_id="current", operating_hours=100, overhaul_id="overhaul", remaining_hours=80),
                    calibration_outcomes=[outcome("cal", n) for n in range(19)],
                    test_outcomes=[outcome("test", n) for n in range(10)])
    return history, evidence


def test_rul_calibration_is_empirical_and_labelled_synthetic():
    history, evidence = rul_fixture()
    result = assess_rul(history, evidence)
    assert result["status"] == "simulation_only"
    assert result["estimate_hours"] == 80
    assert result["interval_hours"] == [72, 88]
    assert "90pct" in result["uncertainty_kind"]
    assert assess_rul(history)["estimate_hours"] is None
    assert assess_rul([], evidence)["interval_hours"] is None


@pytest.mark.parametrize("problem", ["duplicate_calibration", "training_leak", "test_leak", "current_leak",
                                    "censored", "domain", "few_outcomes", "unfrozen", "bad_coverage",
                                    "current_hours", "overhaul_reset", "no_degradation", "short_history"])
def test_rul_gate_refuses_unsupported_numbers(problem):
    history, evidence = rul_fixture()
    if problem == "duplicate_calibration":
        evidence["calibration_outcomes"][1] = evidence["calibration_outcomes"][0]
    elif problem == "training_leak":
        evidence["training_engine_ids"] = ["cal-0"]
    elif problem == "test_leak":
        evidence["test_outcomes"][0]["engine_id"] = "cal-0"
    elif problem == "current_leak":
        evidence["test_outcomes"][0]["engine_id"] = "current"
    elif problem == "censored":
        evidence["calibration_outcomes"][0]["failure_observed"] = False
    elif problem == "domain":
        evidence["calibration_outcomes"][0]["domain_id"] = "different"
    elif problem == "few_outcomes":
        evidence["calibration_outcomes"].pop()
    elif problem == "unfrozen":
        evidence["model_frozen_before_calibration"] = False
    elif problem == "bad_coverage":
        for row in evidence["test_outcomes"]:
            row["failure_hours"] = 9999
    elif problem == "current_hours":
        evidence["prediction"]["operating_hours"] = 120
    elif problem == "overhaul_reset":
        history[0]["overhaul_id"] = "other"
    elif problem == "no_degradation":
        for row in history:
            row["health_score"] = 100
    elif problem == "short_history":
        for row in history:
            row["operating_hours"] /= 100
    result = assess_rul(history, evidence)
    assert result["status"] == "insufficient_evidence"
    assert result["reasons"] and result["estimate_hours"] is None and result["interval_hours"] is None


def test_workflow_rul_cannot_mislabel_synthetic_history_as_field(tmp_path):
    store = EngineStore(tmp_path / "test.db")
    engine = store.create_engine("A")
    _, evidence = rul_fixture()
    evidence["current_engine_id"] = engine["engine_id"]
    evidence["source"] = "field"
    result = rul_for_engine(store, engine, evidence)
    assert result["estimate_hours"] is None
    assert "synthetic" in result["reasons"][0]


def test_incomplete_overhaul_improves_lubrication_but_not_all_abnormalities():
    planned = dict(default_schedule("incomplete_overhaul"), onset_s=0, duration_s=300, severity=0.6, progression=0)
    result = simulate_test({"duration_s": 300}, [planned])
    early, late = result["samples"][60], result["samples"][240]
    def pressure_fraction(sample):
        return sample["truth"]["latent"]["oil_pressure_kpa"] / expected_readings(sample["context"])["oil_pressure_kpa"]
    assert pressure_fraction(late) > pressure_fraction(early)
    assert late["truth"]["latent"]["vibration_mm_s"] - expected_readings(late["context"])["vibration_mm_s"] > 3


def test_starvation_has_deeper_transition_dips_and_load_is_not_payload():
    planned = dict(default_schedule("oil_starvation"), onset_s=0, duration_s=160, severity=0.8, progression=0)
    result = simulate_test({"duration_s": 180, "payload_class": "heavy"}, [planned])
    ratios = [s["truth"]["latent"]["oil_pressure_kpa"] / expected_readings(s["context"])["oil_pressure_kpa"] for s in result["samples"]]
    assert ratios[30] < ratios[70] - 0.25
    assert result["config"]["payload_class"] == "heavy"
    assert result["samples"][0]["context"]["demand_pct"] != result["samples"][70]["context"]["demand_pct"]


def test_time_gap_does_not_count_as_persistent_observed_evidence():
    planned = dict(default_schedule("bearing_degradation"), onset_s=0)
    result = simulate_test({"duration_s": 120}, [planned])
    predictions = analyse_samples([observation_input(result["samples"][n]) for n in (0, 1, 2, 100)])
    assert all(not p["diagnosis"] for p in predictions)
    assert any(i["type"] == "time_gap" for i in predictions[-1]["quality"]["issues"])


def test_bad_measurement_endpoint_blocks_stale_rul(tmp_path):
    store = EngineStore(tmp_path / "test.db")
    engine = store.create_engine("A")
    planned = dict(default_schedule("sensor_fault"), onset_s=8, subtype="missing")
    store.save_test(preview_test(engine, {"duration_s": 10}, [planned]))
    result = rul_for_engine(store, store.get_engine(engine["engine_id"]))
    assert result["estimate_hours"] is None
    assert "stale" in result["reasons"][0]
